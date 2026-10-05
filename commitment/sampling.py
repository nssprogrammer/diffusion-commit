"""Samplers and model wrappers (PyTorch).

A denoiser is any callable D(x, sigma) -> estimate of the clean image (EDM convention, pixels in [-1, 1]).
The reverse-time SDE below records a classifier's logits along the way, so that a model's trajectories can be
compared with noised data (Proposition 1 of the paper).
"""
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .classifiers import classifier_logits

EDM_URL = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/'
EDM_CHECKPOINTS = {'uncond': 'edm-cifar10-32x32-uncond-vp.pkl', 'cond': 'edm-cifar10-32x32-cond-vp.pkl'}


def edm_schedule(steps=256, sigma_max=80.0, sigma_min=0.05, rho=7.0):
    """EDM noise schedule (descending)."""
    i = np.arange(steps)
    return (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def load_edm(kind='uncond', device='cuda', fp16=True, workdir='.'):
    """Load an NVIDIA EDM CIFAR-10 checkpoint (downloaded at run time; licensed CC BY-NC-SA 4.0)."""
    repo = os.path.join(workdir, 'edm')
    if not os.path.exists(repo):
        subprocess.run(['git', 'clone', '-q', 'https://github.com/NVlabs/edm.git', repo], check=True)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import dnnlib
    with dnnlib.util.open_url(EDM_URL + EDM_CHECKPOINTS[kind]) as f:
        net = pickle.load(f)['ema'].to(device).eval()
    net.use_fp16 = fp16
    return net


def edm_denoiser(net, labels=None, kappa=1.0):
    """D(x, sigma) for an EDM network. labels: one-hot class labels for the conditional model.
    kappa: tell the network the noise level is kappa*sigma (the noise-mismatch corruption of the paper)."""
    @torch.no_grad()
    def D(x, s):
        sig = torch.full((len(x),), float(s) * kappa, device=x.device)
        if labels is None:
            return net(x, sig).float()
        return net(x, sig, labels[:len(x)]).float()
    return D


def ddpm_denoiser(model_id='google/ddpm-cifar10-32', device='cuda'):
    """Wrap the original DDPM checkpoint as an EDM-style denoiser: at noise level sigma, use the nearest training
    timestep t (in log sigma_t, with sigma_t = sqrt((1 - abar_t)/abar_t)) and D(x) = x - sigma_t eps(x sqrt(abar_t), t)."""
    from diffusers import UNet2DModel
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
    abar = torch.cumprod(1 - betas, 0).numpy()
    sig_t = np.sqrt((1 - abar) / abar)

    @torch.no_grad()
    def D(x, s):
        t = int(np.argmin(np.abs(np.log(sig_t) - np.log(s))))
        with torch.autocast('cuda', dtype=torch.float16, enabled=(x.device.type == 'cuda')):
            eps = unet(x * math.sqrt(float(abar[t])), torch.full((len(x),), t, device=x.device)).sample.float()
        return x - float(sig_t[t]) * eps
    return D


def recording_map(schedule, record_sigmas):
    """Map schedule step index -> recording level index (nearest step to each recording noise level)."""
    rec = {}
    for gi, s in enumerate(record_sigmas):
        rec.setdefault(int(np.argmin(np.abs(schedule - s))), gi)
    return rec


@torch.no_grad()
def reverse_sde(D, classifier, n, seed, schedule, record_sigmas, sigma_min=0.05, n_classes=10, score_scale=1.0,
                guidance=None, guidance_band=None, keep_steps=None, device='cuda'):
    """Reverse-time SDE (Euler-Maruyama) from noise to sigma_min, recording the classifier's logits.

    guidance: optional callable g(x, sigma) -> extra score term (see guidance.py); guidance_band: (lo, hi) limits.
    score_scale: multiply the learned score (the score-scaling corruption of the paper).
    keep_steps: set of step indices whose states are returned (for branching restarts).
    Returns (logits [levels, n, classes] float16, terminal class probabilities [n, classes], images uint8 [n,3,32,32],
    states dict)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g, device=device) * float(schedule[0])
    rec = recording_map(schedule, record_sigmas)
    L = np.zeros((len(record_sigmas), n, n_classes), np.float16)
    states = {}
    steps = len(schedule)
    for i in range(steps):
        s0 = float(schedule[i])
        s1 = float(schedule[i + 1]) if i + 1 < steps else sigma_min
        if i in rec:
            L[rec[i]] = classifier_logits(classifier, x, s0).cpu().numpy().astype(np.float16)
        if keep_steps is not None and i in keep_steps:
            states[i] = x.clone()
        if s1 >= s0:
            continue
        sc = score_scale * (D(x, s0) - x) / s0 ** 2
        if guidance is not None and (guidance_band is None or guidance_band[0] <= s0 <= guidance_band[1]):
            sc = sc + guidance(x, s0)
        x = x + (s1 - s0) * (-2 * s0 * sc) + math.sqrt(2 * s0 * (s0 - s1)) * torch.randn(x.shape, generator=g, device=device)
    W = F.softmax(classifier_logits(classifier, x, sigma_min), 1).cpu().numpy()
    img = ((D(x, sigma_min).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
    return L, W, img, states


@torch.no_grad()
def continue_from(D, x, start_step, schedule, seed, sigma_min=0.05, device='cuda'):
    """Restart the reverse SDE from states x paused at schedule[start_step]; returns finished images (uint8)."""
    g = torch.Generator(device=device).manual_seed(seed)
    steps = len(schedule)
    for i in range(start_step, steps):
        s0 = float(schedule[i])
        s1 = float(schedule[i + 1]) if i + 1 < steps else sigma_min
        if s1 >= s0:
            continue
        sc = (D(x, s0) - x) / s0 ** 2
        x = x + (s1 - s0) * (-2 * s0 * sc) + math.sqrt(2 * s0 * (s0 - s1)) * torch.randn(x.shape, generator=g, device=device)
    return ((D(x, sigma_min).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()


@torch.no_grad()
def heun_sampler(D, n, seed, steps=18, sigma_max=80.0, sigma_min=0.002, device='cuda'):
    """EDM's deterministic Heun sampler (18 steps = 35 network evaluations). Returns images uint8."""
    g = torch.Generator(device=device).manual_seed(seed)
    t = np.append(edm_schedule(steps, sigma_max, sigma_min), 0.0)
    x = torch.randn(n, 3, 32, 32, generator=g, device=device) * float(t[0])
    for i in range(steps):
        d = (x - D(x, t[i])) / t[i]
        xn = x + (t[i + 1] - t[i]) * d
        if t[i + 1] > 0:
            d2 = (xn - D(xn, t[i + 1])) / t[i + 1]
            xn = x + (t[i + 1] - t[i]) * 0.5 * (d + d2)
        x = xn
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
