"""Noise-conditional classifiers (PyTorch).

NoisyResNet is the census classifier of the paper (and, with n_out=30, the null classifier). NoisyVGG is the second
architecture used to check that results do not depend on the census classifier.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
VEHICLES = {0, 1, 8, 9}


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, emb, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x, e):
        g, b = self.film(e).chunk(2, 1)
        h = F.silu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h)) * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return F.silu(h + self.sc(x))


class NoisyResNet(nn.Module):
    """ResNet with FiLM noise conditioning on a sinusoidal embedding of log(sigma)/4."""

    def __init__(self, n_out=10, ch=64, emb=128):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(32, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(3, ch, 3, 1, 1)
        self.blocks = nn.ModuleList([FiLMBlock(ch, ch, emb), FiLMBlock(ch, 2 * ch, emb, 2), FiLMBlock(2 * ch, 2 * ch, emb),
                                     FiLMBlock(2 * ch, 4 * ch, emb, 2), FiLMBlock(4 * ch, 4 * ch, emb)])
        self.fc = nn.Linear(4 * ch, n_out)

    def forward(self, x, sigma):
        f = torch.exp(torch.arange(16, device=x.device) * -(math.log(100) / 15))
        t = 0.25 * torch.log(sigma)[:, None] * f
        e = self.emb(torch.cat([torch.sin(t), torch.cos(t)], 1))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, e)
        return self.fc(h.mean((2, 3)))


class NoisyVGG(nn.Module):
    """Plain VGG stack (no residuals) with GroupNorm; the input is rescaled by 1/sqrt(1+sigma^2) and log(sigma)/4 is
    appended as a constant input channel (no FiLM)."""

    def __init__(self, n_out=10, w=64):
        super().__init__()
        layers, cin = [], 4
        for cout, pool in [(w, False), (w, True), (2 * w, False), (2 * w, True), (4 * w, False), (4 * w, True)]:
            layers += [nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU()] + ([nn.MaxPool2d(2)] if pool else [])
            cin = cout
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_out)

    def forward(self, x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        x = x / torch.sqrt(1 + s ** 2)
        c = (0.25 * torch.log(s)).expand(-1, 1, x.shape[2], x.shape[3])
        return self.fc(self.body(torch.cat([x, c], 1)).mean((2, 3)))


def train_classifier(model, X, Y, seed=0, epochs=25, batch=256, sigma_lo=0.05, sigma_hi=10.0, device='cuda'):
    """Train on images X ([N,3,32,32] in [-1,1]) with labels Y, noise levels log-uniform in [sigma_lo, sigma_hi],
    horizontal flips, AdamW with a one-cycle schedule (lr 2e-3, weight decay 5e-4), mixed precision."""
    torch.manual_seed(seed)
    Xd = X.to(device)
    Yd = torch.from_numpy(np.asarray(Y)).long().to(device)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Xd) // batch))
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xd), device=device)
        for i in range(0, len(Xd) - batch + 1, batch):
            j = perm[i:i + batch]
            x0 = Xd[j]
            if torch.rand(1).item() < 0.5:
                x0 = x0.flip(-1)
            sigma = torch.exp(torch.empty(batch, device=device).uniform_(math.log(sigma_lo), math.log(sigma_hi)))
            xt = x0 + sigma.view(-1, 1, 1, 1) * torch.randn_like(x0)
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                loss = F.cross_entropy(model(xt, sigma), Yd[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    print(f"trained in {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def classifier_logits(model, x, sigma, batch=1000):
    """Logits of a noise-conditional classifier for noisy images x at noise level sigma."""
    out = []
    for xb in x.split(batch):
        with torch.autocast('cuda', dtype=torch.float16, enabled=(xb.device.type == 'cuda')):
            out.append(model(xb, torch.full((len(xb),), float(sigma), device=xb.device)).float())
    return torch.cat(out)


@torch.no_grad()
def logits_on_grid(model, X, sigmas, repeats, seed, transform=None, batch=1000, device='cuda'):
    """Logits at every noise level in `sigmas` for clean images X, with `repeats` noise draws per image.
    The same noise draw is used at every level. `transform` is applied after noising (e.g. a vertical flip).
    Returns float16 [len(sigmas), repeats*len(X), classes]."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = None
    for r in range(repeats):
        for i0 in range(0, len(X), batch):
            x0 = X[i0:i0 + batch].to(device)
            eps = torch.randn(x0.shape, generator=g, device=device)
            for gi, s in enumerate(sigmas):
                xt = x0 + float(s) * eps
                if transform is not None:
                    xt = transform(xt)
                l = classifier_logits(model, xt, s, batch)
                if out is None:
                    out = np.zeros((len(sigmas), repeats * len(X), l.shape[1]), np.float16)
                out[gi, r * len(X) + i0: r * len(X) + i0 + len(x0)] = l.cpu().numpy().astype(np.float16)
    return out
