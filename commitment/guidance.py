"""Class-proportion corrections ("tilts") of the score, computed with a noise-conditional classifier (PyTorch).

Reweighting the classes of a distribution by factors r_c changes its score at every noise level by
grad_x log sum_c r_c P(c|x). To correct a MODEL whose classes have the data's shapes but proportions f_m towards
target proportions f_t, using a classifier trained on data with class prior f_d, the exact tilt is
    grad_x [ LSE_c(l_c + log f_t/f_d) - LSE_c(l_c + log f_m/f_d) ]          (Equation 4 of the paper).
The first-order version uses the data posterior directly: grad_x [ LSE_c(l_c + log r_c) - LSE_c(l_c) ], r = f_t/f_m.
"""
import numpy as np
import torch


def _grad(classifier, x, sigma, shift_num, shift_den):
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        l = classifier(xg, torch.full((len(xg),), float(sigma), device=x.device)).float()
        obj = (torch.logsumexp(l + shift_num, 1) - torch.logsumexp(l + shift_den, 1)).sum()
        return torch.autograd.grad(obj, xg)[0]


def first_order_tilt(classifier, f_target, f_model, device='cuda'):
    """Returns g(x, sigma) for the first-order class tilt (r = f_target / f_model)."""
    logr = torch.as_tensor(np.log(np.asarray(f_target) / np.asarray(f_model)), dtype=torch.float32, device=device)
    zero = torch.zeros_like(logr)
    return lambda x, s: _grad(classifier, x, s, logr, zero)


def exact_tilt(classifier, f_target, f_model, f_data, device='cuda'):
    """Returns g(x, sigma) for the exact class tilt of Equation 4."""
    lt = torch.as_tensor(np.log(np.asarray(f_target) / np.asarray(f_data)), dtype=torch.float32, device=device)
    lm = torch.as_tensor(np.log(np.asarray(f_model) / np.asarray(f_data)), dtype=torch.float32, device=device)
    return lambda x, s: _grad(classifier, x, s, lt, lm)


def second_round_target(f_target, f_achieved):
    """Target for the second round: raise each class by the ratio by which the first round fell short."""
    t = np.asarray(f_target) * np.asarray(f_target) / np.asarray(f_achieved)
    return t / t.sum()
