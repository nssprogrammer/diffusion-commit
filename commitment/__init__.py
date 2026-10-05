"""Measuring class commitment in diffusion models.

Companion code for "When Does a Diffusion Model Decide What to Draw?" (S. C. Khilar).

CPU-only modules (NumPy/SciPy):
    density      density transitions (sigma_bif) and lopsidedness h along a 1-D order parameter
    stats        class-block bootstrap, Benjamini-Hochberg, Spearman-Brown, class-imbalance error
    calibration  monotone, cross-fitted recalibration of classifier log-odds
    committor    committors, decision levels and fidelity from branching restarts
    testbed      the exactly solvable six-class Gaussian-mixture test bed

GPU modules (PyTorch), imported explicitly when needed:
    classifiers  noise-conditional classifiers (ResNet with FiLM, VGG)
    sampling     EDM schedule, reverse-SDE sampler with recording/branching, Heun sampler, DDPM wrapper
    guidance     class-proportion tilts (first-order and exact)
    evaluation   ViT labelling and FID
"""
from . import density, stats, calibration, committor, testbed  # noqa: F401

__version__ = "1.0.0"
