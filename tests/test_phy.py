import numpy as np
import pytest
from aran import phy


def test_bler_is_ten_percent_at_threshold():
    for m in range(phy.N_MCS):
        assert phy.bler(phy.THR10[m], m) == pytest.approx(0.1, abs=1e-12)


def test_bler_monotone():
    for m in range(phy.N_MCS):
        s = np.linspace(phy.THR10[m] - 5, phy.THR10[m] + 5, 101)
        assert np.all(np.diff(phy.bler(s, m)) < 0)
        assert np.all(np.diff(phy.bler(np.linspace(-15, 35, 200), m)) <= 0)
    for x in (-5.0, 5.0, 15.0):
        assert np.all(np.diff(phy.bler(x, np.arange(phy.N_MCS))) >= 0)
    assert np.all(np.diff(phy.bler(phy.THR10, np.arange(phy.N_MCS))) == 0)  # 10% at each own threshold
    assert np.all(np.diff(phy.bler(10.0, np.arange(6, 11))) > 0)


def test_bler_slope():
    # 90% -> 10% BLER over 2 ln 9 / SLOPE dB
    m = 5
    lo = phy.THR10[m] - 2 * np.log(9) / phy.SLOPE
    assert phy.bler(lo, m) == pytest.approx(0.9, abs=1e-9)


def test_tables_increasing():
    assert np.all(np.diff(phy.EFF) > 0) and np.all(np.diff(phy.THR10) > 0)
    assert len(phy.EFF) == len(phy.THR10) == 15


def test_cqi_from_sinr_edges():
    assert phy.cqi_from_sinr(phy.THR10[0] - 1e-6) == -1
    for m in range(phy.N_MCS):
        assert phy.cqi_from_sinr(phy.THR10[m]) == m
        assert phy.cqi_from_sinr(phy.THR10[m] - 1e-6) == m - 1
    assert phy.cqi_from_sinr(60.0) == phy.N_MCS - 1


def test_best_mcs_is_argmax_of_expected_goodput():
    for s in np.linspace(-10, 30, 81):
        g = [phy.expected_goodput(s, m) for m in range(phy.N_MCS)]
        assert phy.best_mcs(s) == int(np.argmax(g))


def test_pathloss():
    assert phy.pathloss_db(1000.0) == pytest.approx(128.1)
    assert phy.pathloss_db(10.0) == phy.pathloss_db(35.0)
    assert phy.pathloss_db(500.0) == pytest.approx(128.1 + 37.6 * np.log10(0.5))


def test_noise_and_db():
    assert phy.noise_dbm(10e6, 9.0) == pytest.approx(-95.0)
    assert phy.lin2db(phy.db2lin(7.3)) == pytest.approx(7.3)


def test_jain():
    assert phy.jain([1, 1, 1, 1]) == pytest.approx(1.0)
    assert phy.jain([1, 0, 0, 0]) == pytest.approx(0.25)
    assert phy.jain([0, 0]) == 1.0
    assert phy.jain([1, 2]) == pytest.approx(9 / 10)


def test_gauss_markov_rho():
    assert phy.gauss_markov_rho(0, 3.5e9, 1e-3) == pytest.approx(1.0)
    r = [phy.gauss_markov_rho(v, 3.5e9, 1e-3) for v in (3, 30, 60)]
    assert r[0] > r[1] > r[2] > 0
