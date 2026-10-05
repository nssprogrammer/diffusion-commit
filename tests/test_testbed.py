from commitment import testbed


def test_exact_tilt_removes_most_of_the_imbalance():
    before = testbed.class_imbalance(testbed.sample(4000, 1))
    after = testbed.class_imbalance(testbed.sample(4000, 1, 'exact'))
    assert before > 0.2 and after < 0.5 * before
