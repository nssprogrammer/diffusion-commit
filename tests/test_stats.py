import numpy as np

from commitment import stats


def test_spearman_brown_ceiling():
    assert abs(stats.reliability_ceiling(0.969, 0.937) - 0.976) < 0.001


def test_benjamini_hochberg():
    mask = stats.benjamini_hochberg([0.001, 0.01, 0.03, 0.2, 0.9], q=0.05)
    assert mask.tolist() == [True, True, True, False, False]  # 0.03 <= 0.05*3/5


def test_noise_floor_matches_paper():
    mean, _ = stats.class_imbalance_noise_floor(2500, np.random.default_rng(0), sims=1000)
    assert abs(mean - 0.048) < 0.003


def test_class_block_ci_contains_point():
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    x = rng.standard_normal(len(pairs)); y = x + 0.3 * rng.standard_normal(len(pairs))
    lo, hi = stats.class_block_ci(x, y, pairs, rng, B=300)
    assert lo < np.corrcoef(x, y)[0, 1] < hi
