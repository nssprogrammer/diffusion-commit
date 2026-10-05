"""Statistics used throughout the paper.

Each CIFAR-10 class appears in nine of the 45 class pairs, so pair-level statistics are not independent. Intervals
for anything averaged or correlated over pairs therefore use a class-block bootstrap: the ten classes are resampled
with replacement and each pair is weighted by the product of its classes' multiplicities.
"""
import numpy as np


def class_block_ci(x, y, pairs, rng, B=1000, fn=None, n_classes=10):
    """95% class-block bootstrap interval.

    With fn=None: interval of the weighted Pearson correlation between x and y (one value per pair).
    Otherwise fn(weights) -> statistic, e.g. lambda w: np.exp(np.average(log_ratio, weights=w))."""
    P = np.asarray(pairs)
    out = []
    for _ in range(B):
        cls = rng.choice(n_classes, n_classes, replace=True)
        w = np.array([np.sum(cls == a) * np.sum(cls == b) for a, b in P], float)
        if w.sum() < 6:
            continue
        if fn is None:
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            den = np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))
            out.append(np.sum(w * (x - mx) * (y - my)) / den if den > 0 else np.nan)
        else:
            out.append(fn(w))
    out = np.array(out, float)
    out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5]) if len(out) > 20 else np.array([np.nan, np.nan])


def geometric_mean_ratio_ci(log_ratios, pairs, rng, B=1000):
    """Geometric-mean ratio over pairs, with its class-block interval."""
    lr = np.asarray(log_ratios, float)
    ci = class_block_ci(None, None, pairs, rng, B, fn=lambda w: np.exp(np.average(lr, weights=w)))
    return float(np.exp(lr.mean())), ci


def scatter_ci(log_ratios, pairs, rng, B=1000):
    """Standard deviation over pairs of a log ratio ("spread of timing error"), with its class-block interval."""
    lr = np.asarray(log_ratios, float)
    ci = class_block_ci(None, None, pairs, rng, B,
                        fn=lambda w: np.sqrt(np.average((lr - np.average(lr, weights=w)) ** 2, weights=w)))
    return float(lr.std()), ci


def benjamini_hochberg(pvalues, q=0.05):
    """Boolean mask of hypotheses rejected at false-discovery rate q."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    passed = np.zeros(m, bool)
    k = np.where(p[order] <= q * np.arange(1, m + 1) / m)[0]
    if len(k):
        passed[order[:k.max() + 1]] = True
    return passed


def spearman_brown(r_half):
    """Reliability of the full sample from a split-half correlation."""
    return 2 * r_half / (1 + r_half)


def reliability_ceiling(r_half_x, r_half_y):
    """Largest correlation two measurements with these split-half reliabilities can show."""
    return float(np.sqrt(spearman_brown(r_half_x) * spearman_brown(r_half_y)))


def class_imbalance(freqs, ref):
    """Mean |log(f / f_ref)| over classes: the class-imbalance (class-mass) error."""
    f = np.maximum(np.asarray(freqs, float), 1e-4)
    return float(np.mean(np.abs(np.log(f / np.asarray(ref, float)))))


def class_imbalance_noise_floor(n, rng, n_classes=10, sims=2000):
    """Class-imbalance error expected from multinomial sampling alone with perfectly balanced classes.
    Returns (mean, 95th percentile). n=2,500 gives about 0.048."""
    vals = [class_imbalance(np.bincount(rng.choice(n_classes, n), minlength=n_classes) / n, np.full(n_classes, 1 / n_classes))
            for _ in range(sims)]
    return float(np.mean(vals)), float(np.percentile(vals, 95))
