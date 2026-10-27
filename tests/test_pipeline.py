"""Wiring tests: the train / dev / eval splits and causality hold in the pipeline code that produces the results,
not only in the helpers (review r1, M1)."""
import json, os, random, subprocess, sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aran import agents as A, policies as P, scenarios as S, stats
from aran.env import Config, MultiCellLAEnv, encode_action
import aggregate as G
import run_grid as RG
import tune_rules as TR

SMALL = Config(episode_tti=20)
HP = dict(warmup=16, batch=8, sync=10, replay=500)
TRAIN = set()


def _fake_kpis(seed):
    r = random.Random(seed)
    v = r.random()
    return dict(se=1 + v, bler=0.3, jain=0.8, edge=0.1 + v / 10, power_db=0.0, inr_db=20.0, ue_thr=[0.1] * 12)


# ---------- splits ----------
@pytest.mark.parametrize("method", A.METHODS)
def test_training_resets_only_on_training_episodes(method, monkeypatch):
    seen = []
    orig = MultiCellLAEnv.reset

    def spy(self, *, seed=None, options=None):
        seen.append(seed)
        return orig(self, seed=seed, options=options)
    monkeypatch.setattr(MultiCellLAEnv, "reset", spy)
    A.train(method, SMALL, 2, 3, P.OLLA(), hp=HP)
    assert seen == [A.train_episode_seed(2, e) for e in range(3)]
    assert not set(seen) & (set(A.DEV_SEEDS) | set(A.EVAL_SEEDS))


def test_tune_rules_scores_only_dev_episodes_and_records_them(monkeypatch):
    seen = []
    monkeypatch.setattr(P, "run_episode", lambda env, pol, s, trace=False: (seen.append(int(s)), _fake_kpis(s))[1])
    rows, best = TR.tune(SMALL, "default", workers=1, targets=(0.1, 0.2), steps=(0.5,), inits=(0.0, -2.0))
    assert seen and set(seen) <= set(A.DEV_SEEDS)
    assert best["dev_seeds"] == sorted(set(seen)) == list(A.DEV_SEEDS)
    assert len(rows) == 15 + 4
    assert best["olla_se"] == max(r["se"] for r in rows if r["kind"] == "olla")


def _tuning_file(d, sc="default"):
    json.dump(dict(scenario=sc, fixed_mcs=7, olla_target=0.2, olla_step=0.1, olla_init=-2.0, dev_seeds=[5000]),
              open(os.path.join(d, f"dev_tuning_{sc}.json"), "w"))


def test_grid_job_evaluates_on_eval_episodes_and_tracks_provenance(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(P, "run_episode", lambda env, pol, s, trace=False: (seen.append(int(s)), _fake_kpis(s))[1])
    _tuning_file(tmp_path)
    raw = str(tmp_path / "raw")
    d, msg = RG.job(("default", "rule", "cqi-only", 0, raw, False))
    assert seen == list(A.EVAL_SEEDS)
    ev = pd.read_csv(os.path.join(d, "eval.csv"))
    assert list(ev.episode_seed) == list(A.EVAL_SEEDS)
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["tuning"] == dict(fixed_mcs=7, olla_target=0.2, olla_step=0.1, olla_init=-2.0)
    from aran import provenance as PV
    assert meta["config"] == PV.cfg_dict(S.scenario_config("default", os.path.join(ROOT, "config/default.json")))
    assert meta["eval_seeds"] == [10000, 10099]
    # explicit skip when up to date, error when stale
    assert RG.job(("default", "rule", "cqi-only", 0, raw, False))[1] == "skip (up to date)"
    json.dump(dict(json.load(open(os.path.join(d, "meta.json"))), code_hash="old"), open(os.path.join(d, "meta.json"), "w"))
    with pytest.raises(RG.StaleResult):
        RG.job(("default", "rule", "cqi-only", 0, raw, False))
    os.remove(os.path.join(d, "meta.json"))
    with pytest.raises(RG.StaleResult):
        RG.job(("default", "rule", "cqi-only", 0, raw, False))


def test_committed_results_use_the_right_episodes():
    """The committed artefacts themselves: every eval.csv is on EVAL_SEEDS, tuning used DEV_SEEDS only."""
    raw = os.path.join(ROOT, "results/raw")
    for rel in S.expected_main_files():
        if rel.endswith("eval.csv"):
            assert list(pd.read_csv(os.path.join(raw, rel)).episode_seed) == list(A.EVAL_SEEDS), rel
    for sc in S.SCENARIOS:
        t = json.load(open(os.path.join(ROOT, f"results/dev_tuning_{sc}.json")))
        assert t["dev_seeds"] == list(A.DEV_SEEDS)
    for rel in ("sweep/rule_olla-tuned.csv", "horizon/rule_olla-tuned.csv"):
        assert set(pd.read_csv(os.path.join(raw, rel)).episode_seed) == set(list(A.EVAL_SEEDS)[:30])


# ---------- causality ----------
def _rollout(cfg, seed, t0, perturb, acts):
    env = MultiCellLAEnv(cfg)
    obs, _ = env.reset(seed=seed)
    if perturb:
        K, r, e = t0 + cfg.cqi_delay, np.random.default_rng(123), env.ep
        e["fade"][K:] = r.exponential(size=e["fade"][K:].shape)
        if e["outer_on"].shape[1]:
            e["outer_on"][K:] = r.random(e["outer_on"][K:].shape) < 0.5
        e["cqi_noise"][K:] = 3 * r.standard_normal(e["cqi_noise"][K:].shape)
        e["u_harq"][t0:] = r.random(e["u_harq"][t0:].shape)
    O, R = [obs.copy()], []
    for t in range(cfg.episode_tti):
        obs, rew, done, _, _ = env.step(acts[t])
        O.append(obs.copy())
        R.append(rew)
        if done:
            break
    return np.array(O), np.array(R)


@pytest.mark.parametrize("sc,seed,t0", [("default", 10000, 0), ("default", 5000, 7), ("default", 10001, 50),
                                        ("coupled", 10000, 30)])
def test_observations_are_causal(sc, seed, t0):
    """Rewriting the channel from fading index t0 + D on and the HARQ draws from TTI t0 on must leave obs[0..t0]
    untouched (no genie / future leak) while later observations and rewards do change."""
    cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
    cfg = Config(**dict(cfg.to_dict(), episode_tti=80))
    acts = np.random.default_rng(0).integers(0, cfg.n_actions, (cfg.episode_tti, cfg.n_cells))
    Oa, Ra = _rollout(cfg, seed, t0, False, acts)
    Ob, Rb = _rollout(cfg, seed, t0, True, acts)
    assert np.array_equal(Oa[:t0 + 1], Ob[:t0 + 1])
    assert not np.array_equal(Oa[t0 + 1:], Ob[t0 + 1:])
    assert not np.array_equal(Ra[t0:], Rb[t0:])


def test_step_measures_cqi_with_the_neighbours_actual_power():
    env = MultiCellLAEnv(Config())
    env.reset(seed=21)
    for t in range(6):
        pidx = np.array([2, 1, 0]) if t % 2 else np.array([0, 2, 1])
        env.step(np.array([encode_action(env.cfg, 0, int(p)) for p in pidx]))
        assert np.allclose(env.meas[-1], env._measure(t + env.cfg.cqi_delay, env.levels[pidx]))
        assert not np.allclose(env.meas[-1], env._measure(t + env.cfg.cqi_delay, np.zeros(3)))


# ---------- learner wiring ----------
def test_replay_marks_only_the_last_step_terminal(monkeypatch):
    flags = []
    orig = A.Replay.add
    monkeypatch.setattr(A.Replay, "add", lambda self, o, a, r, o2, d: (flags.append(d), orig(self, o, a, r, o2, d))[1])
    A.train("vdn", SMALL, 0, 2, P.OLLA(), hp=HP)
    T = SMALL.episode_tti
    assert flags == ([0.0] * (T - 1) + [1.0]) * 2


class Pick(torch.nn.Module):
    def __init__(self, a, n=21):
        super().__init__()
        self.a, self.n = a, n

    def forward(self, x):
        q = torch.zeros(x.shape[:-1] + (self.n,))
        q[..., self.a] = 1.0
        return q


def test_deployed_policy_applies_learned_power_and_delta():
    env = MultiCellLAEnv(Config())
    env.reset(seed=3)
    a = encode_action(env.cfg, -2, 2)
    m, p = A.NetPolicy(Pick(a), "x").act(env, env._obs())
    assert list(p) == [2, 2, 2]
    assert np.array_equal(m, np.clip(env.base_mcs() - 2, 0, 14))


# ---------- aggregation end to end ----------
def _synthetic_grid(root, E=8):
    """Rules: r[e] varies by episode; learners = olla-tuned + per-seed delta; rows written in shuffled order."""
    rng = np.random.default_rng(0)
    ep = np.array(list(A.EVAL_SEEDS)[:E])
    base = 1.5 + 0.3 * rng.random(E)
    keys = ("se", "bler", "jain", "edge", "power_db", "inr_db")

    def write(d, vals):
        os.makedirs(d, exist_ok=True)
        order = rng.permutation(E)
        pd.DataFrame({"episode_seed": ep[order], **{k: vals[order] for k in keys}}).to_csv(os.path.join(d, "eval.csv"), index=False)
    off = {"fixed": -0.5, "cqi-only": -0.2, "olla-10": -0.05, "olla-tuned": 0.0, "olla-tuned-backoff": -0.1, "genie": 1.0}
    for sc in S.SCENARIOS:
        _tuning_file(root, sc)
        for r in S.RULES:
            write(os.path.join(root, "raw", sc, f"rule_{r}"), base + off[r] + 0.01 * rng.standard_normal(E) * (r != "olla-tuned"))
        for m, dlt in (("sa", 0.05), ("iql", 0.1), ("iql_team", -0.03), ("vdn", -0.04)):
            for s in S.SEEDS:
                d = dlt if not (m == "sa" and s == 0) else 0.0  # one exact tie
                write(os.path.join(root, "raw", sc, f"{m}_s{s}"), base + d)
    return os.path.join(root, "raw"), base


def test_summarise_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "B", 200)
    monkeypatch.setattr(G, "FLIPS_MC", 2000)
    raw, base = _synthetic_grid(str(tmp_path))
    out = G.summarise(raw, list(S.SEEDS))
    comps = out["comparisons"]
    assert len(comps) == out["holm_family_size"] == 34
    assert np.allclose([c["p_holm"] for c in comps], stats.holm([c["p"] for c in comps]))  # one family of 34
    c = next(x for x in comps if (x["scenario"], x["a"], x["b"], x["metric"]) == ("default", "iql", "olla-tuned", "se"))
    assert c["diff"] == pytest.approx(0.1) and c["ci"] == pytest.approx((0.1, 0.1))  # exact pairing by episode seed
    assert c["rel_pct"] == pytest.approx(100 * 0.1 / base.mean())
    assert c["wins"] == 15 and c["p"] == pytest.approx(2 / 2 ** 15)
    sa = next(x for x in comps if (x["scenario"], x["a"], x["b"], x["metric"]) == ("default", "sa", "olla-tuned", "se"))
    assert sa["wins"] == 14 and sa["losses"] == 0  # the tied seed is neither
    vdn = next(x for x in comps if (x["scenario"], x["a"], x["b"], x["metric"]) == ("coupled", "vdn", "iql", "edge"))
    assert vdn["diff"] == pytest.approx(-0.14) and vdn["losses"] == 15
    assert out["scenarios"]["default"]["methods"]["iql"]["se"]["mean"] == pytest.approx(base.mean() + 0.1)


def test_compare_refuses_unpaired_episodes():
    with pytest.raises(AssertionError):
        G.compare(np.ones((1, 3)), np.ones((1, 3)), False, False, np.random.default_rng(0), [1, 2, 3], [1, 3, 2])
