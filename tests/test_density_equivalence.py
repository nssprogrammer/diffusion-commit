"""The package's density code must reproduce the notebook implementation exactly."""
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from commitment import density


def _fake_run(rng, n, prior, sig):
    y = rng.choice(10, size=n, p=prior)
    L = np.zeros((len(sig), n, 10), np.float16)
    for g, s in enumerate(sig):
        l = rng.standard_normal((n, 10))
        l[np.arange(n), y] += 3.5 / (1 + s ** 1.2)
        L[g] = l.astype(np.float16)
    W = np.exp(L[-1].astype(float) * 2)
    return L, W / W.sum(1, keepdims=True)


def _reference(L_TE, W_TE, SIG, PAIRS):
    ns = dict(np=np, fftconvolve=fftconvolve, L_TE=L_TE, W_TE=W_TE, SIG=SIG, PAIRS=PAIRS, rng=np.random.default_rng(0))
    exec((Path(__file__).parent / '_notebook_reference.py.txt').read_text(), ns)
    return ns


def test_find_transition_matches_notebook():
    rng = np.random.default_rng(1)
    sig = np.geomspace(7.95, 0.086, 24)
    L_TE, W_TE = _fake_run(rng, 2000, np.ones(10) / 10, sig)
    L_M, W_M = _fake_run(rng, 1500, np.array([.1, .128, .067, .081, .075, .094, .108, .129, .094, .125]) / 1.001, sig)
    pairs = list(combinations(range(10), 2))[::4]
    ref = _reference(L_TE, W_TE, sig, pairs)
    rw = np.exp(np.random.default_rng(2).uniform(-0.5, 0.5, 10))
    idx = np.arange(len(W_M))
    for a, b in pairs:
        std = density.reference_standardisation(L_TE, W_TE, a, b)
        for kw in [dict(), dict(balance=True), dict(rw=rw)]:
            r_ref = ref['analyse'](L_M, W_M, a, b, idx, **kw, slope=True)
            pk = dict(balance=kw.get('balance', False), reweight=kw.get('rw'))
            r_new = density.find_transition(L_M, W_M, a, b, idx, std, sig, slope=True, **pk)
            assert (r_ref is None) == (r_new is None)
            if r_ref is not None:
                for k in r_ref:
                    assert np.isclose(r_ref[k], r_new[k]), (a, b, k)


def test_equivalent_split():
    assert abs(density.equivalent_split(0.08) - 0.54) < 0.005
