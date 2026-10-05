import numpy as np

from commitment import calibration


def test_recovers_known_warps():
    rng = np.random.default_rng(0)
    n = 8000
    y = rng.random(n) < 0.5
    z_true = 2 * (np.where(y, 1.0, -1.0) + rng.standard_normal(n))
    for warp in [lambda s: 0.7 * s + 0.3, lambda s: 3 * np.tanh(s / 3), lambda s: np.sign(s) * np.sqrt(np.abs(s))]:
        u, _, _ = calibration.crossfit(warp(z_true), y, np.arange(n))
        central = np.abs(z_true) < np.percentile(np.abs(z_true), 95)
        assert np.corrcoef(u[central], z_true[central])[0, 1] > 0.99
        assert abs(calibration.tilting_slope(u, y)[0] - 1) < 0.1
