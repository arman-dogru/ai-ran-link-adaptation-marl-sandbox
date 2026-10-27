import numpy as np
import pytest
from aran import phy
from aran.env import Config, MultiCellLAEnv, OBS_DIM, decode_action, encode_action, draw_episode, cell_positions

CFG = Config()


def rollout(env, seed, act):
    obs, _ = env.reset(seed=seed)
    out, done = [], False
    while not done:
        obs, r, done, _, info = env.step(act(env, obs))
        out.append((r, info["ack"].copy(), info["mcs"].copy(), info["sinr"].copy()))
    return out


def test_reset_obs_shape_and_space():
    env = MultiCellLAEnv(CFG)
    obs, info = env.reset(seed=3)
    assert obs.shape == (CFG.n_cells, OBS_DIM) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (CFG.n_cells,)
    assert set(info) >= {"t", "sched", "cqi"}
    assert info["t"] == 0


def test_episode_length_and_terminal():
    env = MultiCellLAEnv(Config(episode_tti=37))
    obs, _ = env.reset(seed=1)
    n, done = 0, False
    while not done:
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        n += 1
        assert not trunc
        assert np.isfinite(r) and r >= 0
    assert n == 37 and env.t == 37
    assert np.all(obs == 0)


def test_seed_determinism_and_variation():
    env = MultiCellLAEnv(CFG)
    act = lambda e, o: np.full(CFG.n_cells, 12)
    a, b = rollout(env, 7, act), rollout(env, 7, act)
    assert all(np.array_equal(x[1], y[1]) and x[0] == y[0] for x, y in zip(a, b))
    c = rollout(env, 8, act)
    assert not all(np.array_equal(x[3], y[3]) for x, y in zip(a, c))


def test_draw_episode_independent_of_actions():
    e1, e2 = draw_episode(CFG, 11), draw_episode(CFG, 11)
    for k in ("gain_db", "fade", "outer_on", "cqi_noise", "u_harq", "ue"):
        assert np.array_equal(e1[k], e2[k])
    assert e1["fade"].shape == (CFG.episode_tti + CFG.cqi_delay, CFG.n_cells * CFG.ues_per_cell, CFG.n_cells + CFG.n_outer)


def test_ue_attached_to_strongest_site():
    for s in range(5):
        e = draw_episode(CFG, s)
        assert np.all(np.argmax(e["gain_db"], 1) == e["serving"])


def test_ue_dropped_within_cell_disc():
    e = draw_episode(CFG, 2)
    d = np.linalg.norm(e["ue"] - e["cells"][e["serving"]], axis=1)
    assert np.all(d <= CFG.ue_radius_frac * CFG.isd_m + 1e-9)


def test_cell_positions_triangle():
    p = cell_positions(CFG)
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    assert np.allclose(d[~np.eye(3, dtype=bool)], CFG.isd_m)


def test_snr_without_interference():
    cfg = Config(n_cells=1, n_outer=0)
    env = MultiCellLAEnv(cfg)
    env.reset(seed=4)
    e = env.ep
    s, inr = env.sinr_db(5, np.zeros(1))
    want = e["gain_db"][:, 0] + cfg.tx_dbm + phy.lin2db(e["fade"][5, :, 0]) - phy.noise_dbm(cfg.bw_hz, cfg.nf_db)
    assert np.allclose(s, want)


def test_sinr_hand_computed_with_interference():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=5)
    e, k, p = env.ep, 9, np.array([0.0, -3.0, -6.0])
    s, inr = env.sinr_db(k, p)
    rx = phy.db2lin(e["gain_db"] + CFG.tx_dbm) * e["fade"][k]
    pw = phy.db2lin(np.concatenate([p, np.zeros(CFG.n_outer)]))
    on = np.concatenate([np.ones(CFG.n_cells), e["outer_on"][k]])
    tot = rx * pw * on
    sig = tot[np.arange(len(s)), e["serving"]]
    n = phy.db2lin(phy.noise_dbm(CFG.bw_hz, CFG.nf_db))
    assert np.allclose(s, phy.lin2db(sig / (tot.sum(1) - sig + n)))
    assert np.allclose(inr, phy.lin2db((tot.sum(1) - sig) / n))


def test_power_coupling():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=6)
    full, _ = env.sinr_db(3, np.zeros(3))
    back, _ = env.sinr_db(3, np.array([0.0, -6.0, -6.0]))
    srv = env.ep["serving"]
    assert np.all(back[srv == 0] > full[srv == 0])  # neighbours backed off -> cell-0 UEs gain
    assert np.all(back[srv != 0] < full[srv != 0] + 1e-9)


def test_cqi_report_is_delayed_measurement():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=9)
    D = CFG.cqi_delay
    for t in range(12):
        assert np.array_equal(env.rep, phy.cqi_from_sinr(env.meas[t]))
        env.step(np.full(3, encode_action(CFG, 0, 0)))
    assert len(env.meas) == D + 12


def test_measurement_uses_full_serving_power_and_actual_neighbour_power():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=10)
    p = np.array([-6.0, 0.0, -3.0])
    m = env._measure(5, p) - env.ep["cqi_noise"][5]
    for c in range(3):
        q = p.copy()
        q[c] = 0.0
        idx = np.flatnonzero(env.ep["serving"] == c)
        assert np.allclose(m[idx], env.sinr_db(5, q, idx)[0])


def test_ack_rule_and_goodput():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=12)
    for _ in range(30):
        t, u = env.t, env.sched.copy()
        mcs = np.array([3, 8, 13])
        sinr, _ = env.sinr_db(t + CFG.cqi_delay, np.zeros(3), u)
        _, r, _, _, info = env.step_raw(mcs, np.zeros(3, int))
        want = env.ep["u_harq"][t] >= phy.bler(sinr, mcs)
        assert np.array_equal(info["ack"], want)
        assert np.allclose(info["cell_reward"], phy.EFF[mcs] * want)
        assert r == pytest.approx(float(np.mean(phy.EFF[mcs] * want)), rel=1e-6)


def test_pf_scheduler_serves_every_ue():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=13)
    seen = set()
    for _ in range(CFG.episode_tti):
        seen.update(env.sched.tolist())
        assert np.all(env.ep["serving"][env.sched] == np.arange(3))
        _, _, done, _, _ = env.step(np.full(3, encode_action(CFG, 0, 0)))
        if done:
            break
    assert seen == set(range(CFG.n_cells * CFG.ues_per_cell))


def test_pf_picks_max_metric():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=14)
    for _ in range(20):
        env.step(np.full(3, encode_action(CFG, 0, 0)))
    r = np.where(env.rep >= 0, phy.EFF[np.maximum(env.rep, 0)], 0.01) / env.avg_thr
    for c in range(3):
        idx = np.flatnonzero(env.ep["serving"] == c)
        assert env.sched[c] == idx[np.argmax(r[idx])]


def test_action_codec_roundtrip():
    for a in range(CFG.n_actions):
        d, p = decode_action(CFG, a)
        assert encode_action(CFG, d, p) == a
    assert CFG.n_actions == 21
    d, p = decode_action(CFG, np.arange(3))
    assert list(p) == [0, 1, 2] and list(d) == [-4, -4, -4]


def test_step_applies_delta_and_power():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=15)
    base = env.base_mcs().copy()
    _, _, _, _, info = env.step(np.array([encode_action(CFG, 2, 2), encode_action(CFG, -4, 1), encode_action(CFG, 0, 0)]))
    assert np.array_equal(info["mcs"], np.clip(base + np.array([2, -4, 0]), 0, 14))
    assert np.array_equal(env.power_db, [-6.0, -3.0, 0.0])


def test_outer_load_stationary():
    cfg = Config(episode_tti=20000, n_cells=1, ues_per_cell=1, outer_load=0.3)
    e = draw_episode(cfg, 1)
    assert e["outer_on"].mean() == pytest.approx(0.3, abs=0.05)
    on = e["outer_on"][:, 0]
    starts = np.sum(on[1:] & ~on[:-1])
    assert on.sum() / max(1, starts) == pytest.approx(cfg.outer_burst_tti, rel=0.25)


def test_fading_unit_power_and_correlation():
    cfg = Config(episode_tti=4000, speed_kmh=3.0)
    e = draw_episode(cfg, 2)
    assert e["fade"].mean() == pytest.approx(1.0, abs=0.1)
    f = e["fade"][:, 0, 0]
    assert np.corrcoef(f[:-1], f[1:])[0, 1] > 0.95


def test_kpis_consistent():
    env = MultiCellLAEnv(CFG)
    rs = rollout(env, 16, lambda e, o: np.full(3, encode_action(CFG, -1, 0)))
    k = env.kpis()
    assert k["se"] == pytest.approx(np.mean([r[0] for r in rs]), rel=1e-6)
    assert k["bler"] == pytest.approx(1 - np.mean([r[1].mean() for r in rs]), rel=1e-6)
    assert 1 / 12 <= k["jain"] <= 1.0
    assert k["edge"] == pytest.approx(np.sort(k["ue_thr"])[:3].mean())
    assert k["power_db"] == 0.0


def test_observation_features():
    env = MultiCellLAEnv(CFG)
    env.reset(seed=17)
    env.step(np.array([encode_action(CFG, 0, 2), encode_action(CFG, 0, 1), encode_action(CFG, 0, 0)]))
    o, u = env._obs(), env.sched
    assert np.allclose(o[:, 8], np.array([-6.0, -3.0, 0.0]) / 6)
    assert np.allclose(o[:, 9], np.array([-1.5, -3.0, -4.5]) / 6)  # mean of the two neighbours
    wb = env.wb_cqi[u] + 1
    assert np.allclose(o[:, 10], np.array([wb[1] + wb[2], wb[0] + wb[2], wb[0] + wb[1]]) / 2 / 15, atol=1e-6)
    assert np.allclose(o[:, 0], (env.rep[u] + 1) / 15)
    assert np.allclose(o[:, 11], env.rep[u] >= 14)
    assert np.allclose(o[:, 1], env.olla10[u] / 10, atol=1e-6)
