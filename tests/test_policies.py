import numpy as np
import pytest
from aran import phy, policies as P
from aran.env import Config, MultiCellLAEnv

CFG = Config()


def test_olla_update_steps():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=1)
    pol = P.OLLA(target=0.2, step_down=1.0, init=-2.0)
    pol.reset(env)
    assert np.all(pol.off == -2.0)
    pol.update(env, {"ue": np.array([0, 4, 8]), "ack": np.array([True, False, True])})
    assert pol.off[0] == pytest.approx(-2.0 + 0.25) and pol.off[4] == pytest.approx(-3.0)
    assert pol.off[1] == -2.0


def test_olla_offset_clipped():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=1)
    pol = P.OLLA(step_down=5.0)
    pol.reset(env)
    for _ in range(5):
        pol.update(env, {"ue": np.array([0]), "ack": np.array([False])})
    assert pol.off[0] == -10.0


def test_olla_mcs_mapping():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=2)
    pol = P.OLLA()
    pol.reset(env)
    assert np.array_equal(pol.mcs_for(env), np.clip(env.base_mcs(), 0, 14))  # offset 0 -> CQI index
    pol.off[:] = -100
    assert np.all(pol.mcs_for(env) == 0)


def test_olla_reaches_target_bler_on_slow_channel():
    # one UE with mid-range SINR (median ~9.5 dB; seed 2) under constant outer interference
    cfg = Config(n_cells=1, ues_per_cell=1, episode_tti=4000, speed_kmh=3.0, outer_load=1.0)
    env = MultiCellLAEnv(cfg)
    for target in (0.1, 0.3):
        k = P.run_episode(env, P.OLLA(target=target, step_down=0.5), seed=2)
        assert k["bler"] == pytest.approx(target, abs=0.03)


def test_fixed_and_cqi_only():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=3)
    m, p = P.FixedMCS(6).act(env, None)
    assert list(m) == [6, 6, 6] and list(p) == [0, 0, 0]
    m, p = P.CQIOnly().act(env, None)
    assert np.array_equal(m, env.base_mcs())


def test_genie_uses_true_sinr():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=4)
    for _ in range(10):
        m, p = P.Genie().act(env, None)
        s, _ = env.sinr_db(env.t + CFG.cqi_delay, np.zeros(3), env.sched)
        assert np.array_equal(m, phy.best_mcs(s))
        env.step_raw(m, p)


def test_backoff_only_at_top_cqi():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=5)
    pol = P.OLLA(backoff=True)
    pol.reset(env)
    for _ in range(100):
        m, p = pol.act(env, None)
        top = (env.rep[env.sched] >= 14) & (m >= 14)
        assert np.array_equal(p, np.where(top, 2, 0))
        _, _, _, _, info = env.step_raw(m, p)
        pol.update(env, info)


def test_genie_beats_causal_rules():
    env = MultiCellLAEnv(CFG)
    seeds = range(900, 906)
    g = np.mean([P.run_episode(env, P.Genie(), s)["se"] for s in seeds])
    for pol in (P.CQIOnly(), P.OLLA(), P.FixedMCS(7)):
        assert g > np.mean([P.run_episode(env, pol, s)["se"] for s in seeds])


def test_common_random_numbers_across_policies():
    env = MultiCellLAEnv(CFG)
    a = P.run_episode(env, P.FixedMCS(3), 42, trace=True)["trace"]
    b = P.run_episode(env, P.OLLA(), 42, trace=True)["trace"]
    # same channel: with every cell at full power the SINR of a given UE at a given TTI is identical
    both = a["ue"] == b["ue"]
    assert both.any() and np.allclose(a["sinr"][both], b["sinr"][both])


def test_mixed_policy_overrides_cell0_only():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=6)
    pol = P.Mixed(lambda e, o: (0, 2), P.CQIOnly())
    pol.reset(env)
    m, p = pol.act(env, None)
    assert m[0] == 0 and p[0] == 2
    assert np.array_equal(m[1:], env.base_mcs()[1:]) and np.all(p[1:] == 0)
