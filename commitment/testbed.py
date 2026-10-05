"""The exactly solvable six-class Gaussian-mixture test bed (Appendix "An Exactly Solvable Test Bed").

Six isotropic Gaussian classes in d=32: two superclasses separated by +-3 along the first axis, three classes per
superclass on a triangle of radius 1.6 in the next two axes, and a private offset of 0.6 per class. Under noise,
p_sigma(x|c) = N(mu_c, (s_c^2 + sigma^2) I), so the score, posterior and committor are closed-form. The "data" has
uniform class weights; the "model" has the same shapes with imbalanced weights (a pure class-mass error).
"""
import numpy as np

D = 32
N_CLASSES = 6
SUPER = np.array([0, 0, 0, 1, 1, 1])
S2 = np.array([0.30, 0.45, 0.60, 0.35, 0.50, 0.65]) ** 2
F_DATA = np.ones(N_CLASSES) / N_CLASSES
F_MODEL = np.array([0.22, 0.12, 0.16, 0.20, 0.10, 0.20])


def _means():
    mu = np.zeros((N_CLASSES, D))
    tri = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
    for c in range(N_CLASSES):
        mu[c, 0] = 3.0 * (1 if SUPER[c] == 0 else -1)
        mu[c, 1:3] = 1.6 * tri[c % 3]
        mu[c, 3 + c] = 0.6
    return mu


MU = _means()
MU2 = (MU ** 2).sum(1)


def logits(x, s, f):
    v = S2 + s ** 2
    d2 = (x * x).sum(1, keepdims=True) - 2 * x @ MU.T + MU2[None]
    return np.log(f)[None] - 0.5 * d2 / v[None] - 0.5 * D * np.log(v)[None]


def posterior(x, s, f):
    l = logits(x, s, f)
    l -= l.max(1, keepdims=True)
    p = np.exp(l)
    return p / p.sum(1, keepdims=True)


def _mix(W, x, s):
    v = S2 + s ** 2
    Wv = W / v[None]
    return Wv @ MU - x * Wv.sum(1, keepdims=True)


def score(x, s, f):
    """Exact score of the noised mixture with class weights f."""
    return _mix(posterior(x, s, f), x, s)


def tilt_grad(x, s, P, r):
    """grad_x log sum_c r_c P_c(x), for a posterior P computed with any class prior (the shapes are shared)."""
    q = r[None] * P
    q = q / q.sum(1, keepdims=True)
    return _mix(q - P, x, s)


def schedule(n=256, smax=40.0, smin=0.02, rho=7.0):
    i = np.arange(n)
    return (smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho


SCHEDULE = schedule()


def sample(N, seed, mode=None, band=None, split=None, x0=None, i0=0, keep=None):
    """Reverse SDE (Euler-Maruyama) of the MODEL mixture.

    mode: None (no correction) | 'exact' (model posterior) | 'first_order' (data posterior, r = f_t/f_m)
          | 'shifted' (data posterior shifted by log f_m/f_d, identical to exact here).
    band: None (whole trajectory) | 'high' (sigma >= split only) | 'low' (sigma < split only).
    keep: set of step indices at which to store the states (for branching). Returns final labels [and states]."""
    g = np.random.default_rng(seed)
    x = g.standard_normal((N, D)) * SCHEDULE[0] if x0 is None else x0.copy()
    states = {}
    r = F_DATA / F_MODEL
    for i in range(i0, len(SCHEDULE)):
        s0 = SCHEDULE[i]
        s1 = SCHEDULE[i + 1] if i + 1 < len(SCHEDULE) else 0.0
        if keep is not None and i in keep:
            states[i] = x.copy()
        if s1 <= 0:
            break
        sc = score(x, s0, F_MODEL)
        on = mode is not None and (band is None or (band == 'high' and s0 >= split) or (band == 'low' and s0 < split))
        if on:
            if mode == 'exact':
                sc = sc + tilt_grad(x, s0, posterior(x, s0, F_MODEL), r)
            elif mode == 'first_order':
                sc = sc + tilt_grad(x, s0, posterior(x, s0, F_DATA), r)
            elif mode == 'shifted':
                Pd = posterior(x, s0, F_DATA) * (F_MODEL / F_DATA)[None]
                sc = sc + tilt_grad(x, s0, Pd / Pd.sum(1, keepdims=True), r)
        x = x + (s1 - s0) * (-2 * s0 * sc) + np.sqrt(2 * s0 * (s0 - s1)) * g.standard_normal(x.shape)
    labels = posterior(x, SCHEDULE[-1], F_MODEL).argmax(1)
    return (labels, states) if keep is not None else labels


def class_imbalance(labels):
    f = np.maximum(np.bincount(labels, minlength=N_CLASSES) / len(labels), 1e-4)
    return float(np.mean(np.abs(np.log(f / F_DATA))))
