"""Committors from branching restarts.

Inputs follow the notebooks:
    END[l, j, k]  class label of restart k from the paused image of trajectory j at branching level l
    PAR[j]        final class label of trajectory j itself
    levels[l]     noise levels of the branching points, in descending order
"""
import numpy as np


def committor_estimates(END, n_classes=10):
    """Empirical committor q_c for every paused image: array [levels, trajectories, classes]."""
    L, M, K = END.shape
    out = np.zeros((L, M, n_classes))
    for l in range(L):
        for j in range(M):
            out[l, j] = np.bincount(END[l, j], minlength=n_classes) / K
    return out


def decided_level(agree_fraction, levels, threshold=11 / 12):
    """Per trajectory: the highest noise level from which the agreement fraction stays >= threshold.

    agree_fraction: [levels, trajectories], e.g. the fraction of restarts with the same class as the trajectory."""
    out = np.full(agree_fraction.shape[1], np.nan)
    for j in range(agree_fraction.shape[1]):
        tail = np.flip(np.cumprod(np.flip(agree_fraction[:, j] >= threshold))).astype(bool)
        if tail.any():
            out[j] = levels[np.argmax(tail)]
    return out


def crossover_rate(END, PAR, a, b, min_count=20):
    """Pooled cross-over rate of pair (a, b) at each level: among trajectories ending in a or b, the fraction of their
    restarts landing in {a, b} that land on the other class of the pair. NaN where fewer than min_count restarts
    landed in the pair."""
    J = np.where(np.isin(PAR, [a, b]))[0]
    X = np.full(END.shape[0], np.nan)
    for l in range(END.shape[0]):
        same = sum(np.sum(END[l, j] == PAR[j]) for j in J)
        cross = sum(np.sum(END[l, j] == (b if PAR[j] == a else a)) for j in J)
        if same + cross >= min_count:
            X[l] = cross / (same + cross)
    return X


def pair_decision(X, levels, threshold=0.1):
    """Highest noise level from which the cross-over rate stays <= threshold, log-interpolated at the crossing."""
    ok = np.isfinite(X)
    good = ok & (X <= threshold)
    for l in range(len(X)):
        if good[l] and all(good[k] or not ok[k] for k in range(l, len(X))):
            if l > 0 and ok[l - 1] and X[l - 1] > threshold:
                w = (X[l - 1] - threshold) / (X[l - 1] - X[l])
                return float(np.exp(np.log(levels[l - 1]) + w * (np.log(levels[l]) - np.log(levels[l - 1]))))
            return float(levels[l])
    return np.nan


def fidelity_excess(Q_model, Q_posterior, K, rng, idx=None):
    """Per level: mean total variation between the model's committor (from K restarts) and a posterior, minus the
    value expected from K-draw multinomial noise alone if the posterior were the truth."""
    idx = np.arange(Q_model.shape[1]) if idx is None else idx
    out = []
    for l in range(Q_model.shape[0]):
        tv = 0.5 * np.abs(Q_model[l, idx] - Q_posterior[l, idx]).sum(1).mean()
        sim = np.mean([0.5 * np.abs(np.bincount(rng.choice(len(q), K, p=q / q.sum()), minlength=len(q)) / K - q).sum()
                       for q in Q_posterior[l, idx]])
        out.append(tv - sim)
    return np.array(out)


def shift_by_class_mass(Q_posterior, f_model, f_data):
    """Posterior of a model whose classes have the data's shapes but proportions f_model:
    P_m(c|x) proportional to P_d(c|x) f_model_c / f_data_c."""
    Q = Q_posterior * (np.asarray(f_model) / np.asarray(f_data))[None, None, :]
    return Q / Q.sum(2, keepdims=True)
