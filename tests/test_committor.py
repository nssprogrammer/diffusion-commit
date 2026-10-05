import numpy as np

from commitment import committor


def test_pair_decision_interpolates():
    levels = np.array([8.0, 4.0, 2.0, 1.0, 0.5])
    X = np.array([0.4, 0.3, 0.2, 0.05, 0.0])
    d = committor.pair_decision(X, levels, 0.1)
    assert 1.0 < d < 2.0


def test_decided_level():
    levels = np.array([8.0, 4.0, 2.0, 1.0])
    frac = np.array([[0.5, 1.0], [0.95, 1.0], [1.0, 1.0], [1.0, 1.0]])
    out = committor.decided_level(frac, levels, 11 / 12)
    assert out.tolist() == [4.0, 8.0]


def test_mass_shift_normalised():
    Q = np.random.default_rng(0).dirichlet(np.ones(10), size=(3, 5))
    S = committor.shift_by_class_mass(Q, np.full(10, 0.1) * np.linspace(0.5, 1.5, 10), np.full(10, 0.1))
    assert np.allclose(S.sum(2), 1)
