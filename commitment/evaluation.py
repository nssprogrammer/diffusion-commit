"""Labelling finished images with an independent ViT, and FID (PyTorch)."""
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .classifiers import CIFAR10_CLASSES

VIT_ID = 'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'


def to_float(img, device='cuda'):
    """uint8 [N,3,32,32] -> float in [-1, 1]."""
    return torch.from_numpy(img).float().div(127.5).sub(1).to(device)


class ViTLabeller:
    """ViT-Base fine-tuned on CIFAR-10 (98.4% test accuracy in our runs), applied at 224x224."""

    def __init__(self, device='cuda'):
        from transformers import AutoModelForImageClassification
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained(VIT_ID).to(device).eval().half()
        names = [self.model.config.id2label[i].lower() for i in range(10)]
        self.map = np.array([CIFAR10_CLASSES.index(n) if n in CIFAR10_CLASSES else i for i, n in enumerate(names)])

    @torch.no_grad()
    def __call__(self, img, batch=128):
        out = []
        for i in range(0, len(img), batch):
            x = F.interpolate(to_float(img[i:i + batch], self.device), size=224, mode='bilinear', align_corners=False)
            out.append(self.model(pixel_values=x.half()).logits.argmax(1).cpu().numpy())
        return self.map[np.concatenate(out)]


class FID:
    """FID against reference images (Inception pool3 features from pytorch-fid). Compare only equal sample sizes."""

    def __init__(self, reference_uint8, device='cuda'):
        try:
            import pytorch_fid  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytorch-fid'], check=True)
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance
        self.device = device
        self.inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self._fd = calculate_frechet_distance
        self.mu, self.cov = self.stats(reference_uint8)

    @torch.no_grad()
    def stats(self, img, batch=250):
        f = np.concatenate([self.inc(torch.from_numpy(img[i:i + batch]).float().div(255).to(self.device))[0]
                            .squeeze(-1).squeeze(-1).cpu().numpy() for i in range(0, len(img), batch)])
        return f.mean(0), np.cov(f, rowvar=False)

    def __call__(self, img):
        mu, cov = self.stats(img)
        return float(self._fd(mu, cov, self.mu, self.cov))
