"""Density transitions and lopsidedness along a one-dimensional order parameter.

For a class pair (a, b) the order parameter is a classifier's log-odds z = l_a - l_b, standardised at each noise
level so that the (weighted) class means sit at +1 and -1. Its distribution is estimated with a Gaussian kernel
density on a fixed grid. The density transition sigma_bif is the first noise level, scanning from high to low
noise, at which the density has two peaks after at least one level with a single peak. The lopsidedness h is the
constant term of the cusp normal form f(m) = h + a m - b m^3 fitted to the slope of the log-density there.

These functions reproduce the notebook implementations exactly (tests/test_density_equivalence.py).
"""
import numpy as np
from scipy.signal import fftconvolve

GRID = np.linspace(-4.0, 4.0, 4001)
DGRID = GRID[1] - GRID[0]


def kde_slope(m, w=None, bw_factor=1.5):
    """Weighted Gaussian KDE of m on GRID. Returns (slope of the log-density, density).

    Bandwidth: bw_factor x Silverman's rule, using the effective sample size of the weights."""
    m = np.clip(np.asarray(m, dtype=np.float64), -8, 8)
    w = np.ones_like(m) if w is None else np.asarray(w, dtype=np.float64)
    neff = w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)
    mu = np.average(m, weights=w)
    sd = np.sqrt(np.average((m - mu) ** 2, weights=w))
    bw = max(bw_factor * 1.06 * sd * neff ** (-1 / 5), 1e-3)
    h, _ = np.histogram(m, bins=np.append(GRID - DGRID / 2, GRID[-1] + DGRID / 2), weights=w, density=True)
    k = np.exp(-GRID ** 2 / (2 * bw ** 2))
    k /= k.sum()
    p = np.maximum(fftconvolve(h, k, 'same'), 1e-300)
    return np.gradient(np.log(p), GRID), p


def _merge(pts, merge):
    changed = True
    while changed and len(pts) > 1:
        changed = False
        for i in range(len(pts) - 1):
            if abs(pts[i][0] - pts[i + 1][0]) < merge and pts[i][1] != pts[i + 1][1]:
                del pts[i:i + 2]
                changed = True
                break
    return pts


def stationary_points(slope, density, lo=-3.5, hi=3.5, merge=0.03, prominence=0.05):
    """Peaks ('max') and dips ('min') of the density, as a list of (position, kind).

    Peak/dip pairs closer than `merge`, or differing by less than `prominence` nats of log-density, are removed."""
    keep = (GRID > lo) & (GRID < hi) & (density > 1e-3 * density.max())
    idx = np.where((np.sign(slope[:-1]) != np.sign(slope[1:])) & keep[:-1] & keep[1:])[0]
    pts = [(float(GRID[i] - slope[i] * DGRID / (slope[i + 1] - slope[i])) if slope[i + 1] != slope[i] else float(GRID[i]),
            'max' if slope[i] > 0 else 'min') for i in idx]
    pts = _merge(pts, merge)
    logp = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * DGRID)])

    def lp(x):
        return float(np.interp(x, GRID, logp))

    changed = True
    while changed and len(pts) > 1:
        changed = False
        for i in range(len(pts) - 1):
            if pts[i][1] != pts[i + 1][1] and abs(lp(pts[i][0]) - lp(pts[i + 1][0])) < prominence:
                del pts[i:i + 2]
                changed = True
                break
    return pts


def n_peaks(points):
    """Number of peaks in a list returned by stationary_points."""
    return sum(kind == 'max' for _, kind in points)


def cusp_fit(slope, density, window=1.0, b_min=0.02, s_max=0.6):
    """Fit f(m) = h + a m - b m^3 (after removing the quadratic term by a shift) over |m| < window.

    Returns (h, a, b)."""
    ok = (np.abs(GRID) < window) & (density > 0)
    c3, c2, c1, c0 = np.polyfit(GRID[ok], slope[ok], 3, w=np.sqrt(density[ok]))
    b = max(-c3, b_min)
    c3 = -b
    s = float(np.clip(-c2 / (3 * c3), -s_max, s_max))
    return c0 + c1 * s + c2 * s ** 2 + c3 * s ** 3, c1 + 3 * c3 * s ** 2 + 2 * c2 * s, b


def standardisation(z, wa, wb):
    """(midpoint, half-separation, separation in pooled-SD units) of two weighted classes along z."""
    mua, mub = np.average(z, weights=wa), np.average(z, weights=wb)
    half = 0.5 * (mua - mub)
    sda = np.sqrt(np.average((z - mua) ** 2, weights=wa))
    sdb = np.sqrt(np.average((z - mub) ** 2, weights=wb))
    return 0.5 * (mua + mub), half, abs(half) / (0.5 * (sda + sdb))


def reference_standardisation(L_ref, W_ref, a, b):
    """Per-level standardisation of the pair log-odds, computed on the data reference.

    L_ref: [levels, N, classes] logits; W_ref: [N, classes] terminal class probabilities."""
    return [standardisation(L_ref[g, :, a].astype(np.float64) - L_ref[g, :, b], W_ref[:, a] + 1e-12, W_ref[:, b] + 1e-12)
            for g in range(L_ref.shape[0])]


def find_transition(L, W, a, b, idx, std_ref, sigmas, balance=False, reweight=None, start=0, slope=False,
                    min_sep=0.3, bw_factor=1.5):
    """Density transition of pair (a, b) for rows `idx` of a run (model or data).

    L: [levels, N, classes] logits recorded at noise levels `sigmas` (descending). W: [N, classes] terminal weights.
    std_ref: output of reference_standardisation (the same standardisation is applied to model and data).
    balance: rescale the two classes to equal total weight ("rebalancing").
    reweight: per-class factors multiplying the weights ("mass-matching" when applied to the data).
    start: first level to scan (a level with a single peak is assumed above it when start > 0).
    slope: also estimate s = -da/dlog(sigma) near the transition.
    Returns dict(sigma, gi, h, a, b[, s]), or None if no transition is found."""
    Wa = W[idx, a].astype(np.float64)
    Wb = W[idx, b].astype(np.float64)
    if reweight is not None:
        Wa = Wa * reweight[a]
        Wb = Wb * reweight[b]
    if balance:
        tot = 0.5 * (Wa.sum() + Wb.sum())
        Wa = Wa / Wa.sum() * tot
        Wb = Wb / Wb.sum() * tot

    def dist(g):
        mid, half, sep = std_ref[g]
        if sep < min_sep:
            return None
        z = L[g, idx, a].astype(np.float64) - L[g, idx, b]
        return kde_slope((z - mid) / half, Wa + Wb, bw_factor)

    prev = None if start == 0 else 1
    for g in range(start, len(sigmas)):
        d = dist(g)
        nm = 1 if d is None else n_peaks(stationary_points(*d))
        if nm >= 2 and prev is not None:
            h, a_, b_ = cusp_fit(*d)
            out = dict(sigma=float(sigmas[g]), gi=g, h=h, a=a_, b=b_)
            if slope:
                gs = [k for k in range(max(0, g - 4), min(len(sigmas), g + 3)) if dist(k) is not None]
                av = [cusp_fit(*dist(k))[1] for k in gs]
                if len(gs) >= 3:
                    out['s'] = -np.polyfit(np.log(np.asarray(sigmas)[gs]), av, 1)[0]
            return out
        if nm == 1:
            prev = 1
    return None


def bootstrap_transition(L, W, a, b, n, std_ref, sigmas, rng, B=100, balance=False, reweight=None, pool=None):
    """Bootstrap medians and 95% intervals of sigma_bif and h at sample size n.

    Resamples rows of `pool` (default: all rows). Returns None if fewer than B/3 resamples show a transition."""
    rows = np.arange(len(W)) if pool is None else np.asarray(pool)
    r0 = find_transition(L, W, a, b, rng.choice(rows, size=n, replace=len(rows) < n), std_ref, sigmas, balance, reweight)
    if r0 is None:
        return None
    out = []
    for _ in range(B):
        r = find_transition(L, W, a, b, rng.choice(rows, size=n, replace=True), std_ref, sigmas, balance, reweight,
                            start=max(0, r0['gi'] - 8))
        if r:
            out.append((r['sigma'], r['h']))
    if len(out) < B // 3:
        return None
    o = np.array(out)
    return dict(sigma=np.median(o[:, 0]), sigma_ci=np.percentile(o[:, 0], [2.5, 97.5]),
                h=np.median(o[:, 1]), h_ci=np.percentile(o[:, 1], [2.5, 97.5]), n_ok=len(o))


def census_transition(Z, is_a, sigmas, start=0, bw_factor=1.5, merge=0.03, prominence=0.05, persist=1, min_sep=0.3):
    """Census on labelled data (every row has a known class).

    Z: list over noise levels of raw log-odds; is_a: boolean array. The sample standardises itself.
    persist: number of consecutive two-peak levels required. Returns (sigma, h, level) or None."""
    seen = start > 0
    run = 0
    for g in range(start, len(sigmas)):
        z = Z[g]
        mua, mub = z[is_a].mean(), z[~is_a].mean()
        half = 0.5 * (mua - mub)
        sep = abs(half) / (0.5 * (z[is_a].std() + z[~is_a].std()))
        if sep < min_sep:
            seen = True
            run = 0
            continue
        sm, p = kde_slope((z - 0.5 * (mua + mub)) / half, None, bw_factor)
        nm = n_peaks(stationary_points(sm, p, merge=merge, prominence=prominence))
        if nm >= 2 and seen:
            run += 1
            if run >= persist:
                g0 = g - persist + 1
                z0 = Z[g0]
                m0, m1 = z0[is_a].mean(), z0[~is_a].mean()
                hf = 0.5 * (m0 - m1)
                sm0, p0 = kde_slope((z0 - 0.5 * (m0 + m1)) / hf, None, bw_factor)
                return float(sigmas[g0]), cusp_fit(sm0, p0)[0], g0
        elif nm == 1:
            seen = True
            run = 0
    return None


def equivalent_split(h):
    """Share of the larger class in the class split that would produce lopsidedness |h| near the transition
    (h ~ tanh(eta), eta = 0.5 log(w_a / w_b)). |h| = 0.08 gives 0.54."""
    r = np.exp(2 * np.arctanh(np.clip(np.abs(h), 0, 0.95)))
    return r / (1 + r)
