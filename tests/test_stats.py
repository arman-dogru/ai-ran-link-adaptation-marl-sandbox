import numpy as np
import pytest
from aran import stats


def test_signflip_exact():
    d = np.array([1.0, 2.0, 3.0])
    # 8 patterns; |mean| >= 2 only for all + or all -  -> 2/8
    assert stats.signflip_p(d, 10_000, np.random.default_rng(0)) == pytest.approx(0.25)
    assert stats.signflip_p(np.zeros(4), 100, np.random.default_rng(0)) == 1.0


def test_signflip_mc_two_sided_and_never_zero():
    rng = np.random.default_rng(1)
    d = np.full(40, 1.0)
    p = stats.signflip_p(d, 999, rng)
    assert p == pytest.approx(1 / 1000)
    assert stats.signflip_p(-d, 999, rng) == pytest.approx(1 / 1000)
    x = np.random.default_rng(2).normal(0, 1, 40)
    assert stats.signflip_p(x, 2000, rng) > 0.05


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(stats.holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.all(stats.holm([0.9, 0.9]) <= 1.0)


def test_boot_crossed_resamples_both_axes():
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(5.0)[:, None], 7, 1)  # varies by seed only
    cols = rows.T.copy()[:5] if False else np.repeat(np.arange(7.0)[None, :], 5, 0)  # varies by episode only
    assert np.std(stats.boot_crossed(rows, 300, rng)) > 0.1
    assert np.std(stats.boot_crossed(cols, 300, rng)) > 0.1
    assert np.std(stats.boot_crossed(np.ones((5, 7)), 50, rng)) == 0
    assert np.mean(stats.boot_crossed(rows, 2000, rng)) == pytest.approx(2.0, abs=0.1)


def test_ci_and_boot1d():
    b = stats.boot_1d(np.arange(100.0), 4000, np.random.default_rng(0))
    lo, hi = stats.ci(b)
    assert lo < 49.5 < hi and hi - lo == pytest.approx(2 * 1.96 * np.std(np.arange(100.0)) / 10, rel=0.15)
