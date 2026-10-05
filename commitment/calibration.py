"""Monotone, cross-fitted recalibration of a classifier's log-odds towards a calibrated log-likelihood ratio.

A coordinate u is an affine function of the Bayes log-likelihood ratio exactly when log p(u|a) - log p(u|b) is linear
in u (the "tilting identity"). The calibration map below is smooth and strictly increasing by construction:
u(z) = b0 + w0 z + sum_k w_k s softplus((z - t_k)/s) - sum_k v_k s softplus(-(z - t_k)/s), with all weights >= 0,
fitted by maximum likelihood on class labels. Cross-fitting scores every image with a map fitted on the other half.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _basis(zs, knots, s):
    return np.column_stack([zs] + [s * np.logaddexp(0.0, (zs - t) / s) for t in knots]
                           + [-s * np.logaddexp(0.0, -(zs - t) / s) for t in knots])


def monotone_fit(z, y, n_knots=7, lam=1e-6):
    """Fit the monotone calibration map on (z, y). Returns a function z -> calibrated log-odds."""
    mu, sd = z.mean(), z.std() + 1e-9
    zs = (z - mu) / sd
    knots = np.quantile(zs, np.linspace(0.03, 0.97, n_knots))
    s = max(np.median(np.diff(knots)), 0.05)
    B = _basis(zs, knots, s)
    sgn = np.where(y, 1.0, -1.0)
    n, K = len(z), B.shape[1]

    def f(par):
        b0, w = par[0], par[1:]
        m = -sgn * (b0 + B @ w)
        loss = np.mean(np.logaddexp(0.0, m)) + lam * np.sum(w[1:] ** 2)
        gu = -sgn * expit(m) / n
        gw = B.T @ gu
        gw[1:] += 2 * lam * w[1:]
        return loss, np.concatenate([[gu.sum()], gw])

    par0 = np.concatenate([[0.0], [0.5], np.zeros(K - 1)])
    r = minimize(f, par0, jac=True, method='L-BFGS-B', bounds=[(None, None)] + [(0.0, None)] * K)
    b0, w = r.x[0], r.x[1:]
    return lambda zz: b0 + _basis((zz - mu) / sd, knots, s) @ w


def crossfit(z, y, image_id):
    """Cross-fitted calibration: images with even id are scored by a map fitted on odd ids, and vice versa.
    Returns (calibrated values, slope of the linear fit of u on z, nonlinearity of the map)."""
    A = (image_id % 2 == 0)
    u = np.empty_like(z, dtype=np.float64)
    u[~A] = monotone_fit(z[A], y[A])(z[~A])
    u[A] = monotone_fit(z[~A], y[~A])(z[A])
    lin = np.polyfit(z, u, 1)
    return u, lin[0], float(np.std(u - np.polyval(lin, z)) / max(np.std(u), 1e-9))


def tilting_slope(u, y, n_bins=12):
    """Slope and intercept of the binned class log-ratio against u (central 90%).
    For a calibrated log-likelihood ratio of balanced classes: slope 1, intercept 0."""
    q = np.quantile(u, np.linspace(0.05, 0.95, n_bins + 1))
    c = 0.5 * (q[1:] + q[:-1])
    lr, w = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        j = (u >= lo) & (u < hi)
        na, nb = (j & y).sum(), (j & ~y).sum()
        lr.append(np.log((na + 0.5) / (nb + 0.5)) - np.log((y.sum() + 0.5) / ((~y).sum() + 0.5)))
        w.append(j.sum())
    slope, intercept = np.polyfit(c, lr, 1, w=np.sqrt(w))
    return slope, intercept
