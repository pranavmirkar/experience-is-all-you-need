"""Hand-calculated checks. R rows are phases, columns are families."""

import pytest

from experience_learning import metrics

R = [
    [0.8, 0.2, 0.1],
    [0.6, 0.9, 0.2],
    [0.5, 0.7, 0.95],
]
R_INIT = [0.3, 0.1, 0.05]


def test_average_forgetting():
    # f0: max(0.8, 0.6) - 0.5 = 0.3 ; f1: 0.9 - 0.7 = 0.2 ; mean 0.25
    assert metrics.average_forgetting(R, 3) == pytest.approx(0.25)


def test_forgetting_can_be_negative_when_family_keeps_improving():
    R2 = [[0.2, 0.0], [0.5, 0.3]]
    assert metrics.average_forgetting(R2, 2) == pytest.approx(-0.3)


def test_average_accuracy():
    # t0: 0.8 ; t1: (0.6+0.9)/2 = 0.75 ; t2: (0.5+0.7+0.95)/3 ; mean of the three
    expected = (0.8 + 0.75 + (0.5 + 0.7 + 0.95) / 3) / 3
    assert metrics.average_accuracy(R, 3) == pytest.approx(expected)


def test_forward_transfer():
    # t1: R[0][1]-init[1] = 0.1 ; t2: R[1][2]-init[2] = 0.15 ; mean 0.125
    assert metrics.forward_transfer(R, R_INIT, 3) == pytest.approx(0.125)


def test_regression_rate():
    before = [True, True, False, True]
    after = [True, False, False, True]
    assert metrics.regression_counts(before, after) == (3, 1)
    assert metrics.regression_rate(before, after) == pytest.approx(1 / 3)
    assert metrics.regression_rate([False, False], [True, True]) == 0.0


def test_pass_rate_and_errors():
    assert metrics.pass_rate([True, False, False, True]) == 0.5
    with pytest.raises(ValueError):
        metrics.pass_rate([])
    with pytest.raises(ValueError):
        metrics.regression_counts([True], [True, False])
    with pytest.raises(ValueError):
        metrics.average_forgetting(R, 1)
