import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import aggregate as G
from aran import scenarios as S


def _touch_all(raw):
    for rel in S.expected_main_files() + G.expected_sweep_files() + G.expected_horizon_files():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_guard_refuses_missing(tmp_path, capsys):
    raw = tmp_path / "raw"
    _touch_all(raw)
    os.remove(raw / "default" / "vdn_s3" / "eval.csv")
    out = tmp_path / "out"
    assert G.main(["--raw", str(raw), "--out", str(out)]) == 2
    assert not out.exists()
    assert "incomplete" in capsys.readouterr().err


def test_guard_refuses_extra(tmp_path, capsys):
    raw = tmp_path / "raw"
    _touch_all(raw)
    (raw / "default" / "vdn_s99").mkdir()
    (raw / "default" / "vdn_s99" / "eval.csv").write_text("x")
    out = tmp_path / "out"
    assert G.main(["--raw", str(raw), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "unexpected files" in err and "incomplete" not in err
    assert not out.exists()


def test_partial_mode_is_gone_and_empty_raw_refused(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    assert G.main(["--raw", str(raw), "--out", str(out)]) == 2 and not out.exists()
    with pytest.raises(SystemExit):
        G.main(["--raw", str(raw), "--out", str(out), "--allow-partial"])


def _good_metas(root):
    import json
    from aran import provenance as PV
    raw = root / "raw"
    _touch_all(raw)
    for sc in S.SCENARIOS:
        json.dump(dict(fixed_mcs=7, olla_target=0.2, olla_step=0.1, olla_init=-2.0), open(root / f"dev_tuning_{sc}.json", "w"))
    tun = dict(fixed_mcs=7, olla_target=0.2, olla_step=0.1, olla_init=-2.0)
    for rel in S.expected_main_files() + G.expected_sweep_files() + G.expected_horizon_files():
        if rel.endswith("meta.json"):
            sub = rel.split("/")[0]
            sc = sub if sub in S.SCENARIOS else "default"
            cfg = PV.cfg_dict(S.scenario_config(sc, os.path.join(G.ROOT, "config/default.json")))
            extra = {"sweep": ("scripts/run_sweep.py",), "horizon": ("scripts/run_horizon.py",)}.get(sub, ())
            json.dump(dict(code_hash=PV.code_hash(files=PV.CODE_FILES + extra), git_dirty=False, config=cfg, tuning=tun),
                      open(raw / rel, "w"))
    return raw


def test_provenance_clean_grid_passes(tmp_path):
    assert G.check_provenance(str(_good_metas(tmp_path))) == []


@pytest.mark.parametrize("where", ["coupled/vdn_s4/meta.json", "sweep/iql.meta.json", "horizon/rule_olla-10.meta.json"])
@pytest.mark.parametrize("field,value,needle", [("code_hash", "other", "mixed provenance"), ("git_dirty", True, "dirty"),
                                                ("git_dirty", None, "unknown git state"),
                                                ("tuning", {"fixed_mcs": 6}, "tuned rule parameters"),
                                                ("config", {"n_cells": 2}, "config differs")])
def test_provenance_problems_refused(tmp_path, capsys, field, value, needle, where):
    import json
    raw = _good_metas(tmp_path)
    p = raw / where
    m = json.load(open(p))
    m[field] = value
    json.dump(m, open(p, "w"))
    probs = G.check_provenance(str(raw))
    assert any(needle in x for x in probs)
    out = tmp_path / "out"
    assert G.main(["--raw", str(raw), "--out", str(out)]) == 2 and not out.exists()
    assert "provenance" in capsys.readouterr().err


def test_check_grid_complete(tmp_path):
    raw = tmp_path / "raw"
    _touch_all(raw)
    assert G.check_grid(str(raw)) == ([], [])
    n = len(S.SCENARIOS) * (2 * len(S.RULES) + 4 * len(S.LEARNERS) * len(S.SEEDS))
    assert len(S.expected_main_files()) == n


def test_comparison_family_and_holm():
    assert len(G.COMPARISONS) == 17
    comps = [dict(p=0.01, ci=(0.1, 0.2)), dict(p=0.02, ci=(-0.1, 0.3)), dict(p=0.04, ci=(0.1, 0.2))]
    G.holm_family(comps)
    assert [c["p_holm"] for c in comps] == pytest.approx([0.03, 0.04, 0.04])
    assert [c["significant"] for c in comps] == [True, False, True]


def test_compare_learned_vs_rule_uses_seed_unit():
    rng = np.random.default_rng(0)
    La = 1.0 + 0.01 * rng.standard_normal((15, 20))
    Rb = np.zeros((1, 20))
    c = G.compare(La, Rb, True, False, np.random.default_rng(1))
    assert c["unit"] == "training seed" and c["n"] == 15 and c["wins"] == 15
    assert c["p"] == pytest.approx(2 / 2 ** 15)
    assert c["ci"][0] > 0.9


def test_compare_rule_vs_rule_uses_episode_unit():
    a = np.arange(30.0)[None]
    c = G.compare(a, a - 1, False, False, np.random.default_rng(1))
    assert c["unit"] == "evaluation episode" and c["n"] == 30 and c["diff"] == pytest.approx(1.0)
    assert c["p"] < 1e-4


def test_compare_learner_pairs_by_seed():
    a = np.tile(np.array([1.0, -1.0] * 7 + [1.0])[:, None], (1, 5))
    b = np.zeros((15, 5))
    c = G.compare(a, b, True, True, np.random.default_rng(0))
    assert c["wins"] == 8 and c["losses"] == 7 and c["p"] > 0.5


def test_load_matrix_orders_episodes(tmp_path):
    raw = tmp_path
    for s in (0, 1):
        d = raw / "default" / f"iql_s{s}"
        d.mkdir(parents=True)
        pd.DataFrame(dict(episode_seed=[3, 1, 2], se=[30.0 + s, 10.0 + s, 20.0 + s])).to_csv(d / "eval.csv", index=False)
    M, ep = G.load_matrix(str(raw), "default", "iql", "se", [0, 1])
    assert list(ep) == [1, 2, 3] and M.tolist() == [[10, 20, 30], [11, 21, 31]]


def test_provenance_refuses_a_uniform_but_stale_code_hash(tmp_path):
    """All metas agree with each other but not with the current tree (results from older code)."""
    import json
    raw = _good_metas(tmp_path)
    for dp, _, fs in os.walk(raw):
        for f in fs:
            if f.endswith("meta.json"):
                m = json.load(open(os.path.join(dp, f)))
                json.dump(dict(m, code_hash="old-but-uniform"), open(os.path.join(dp, f), "w"))
    probs = G.check_provenance(str(raw))
    assert any("differs from the current tree" in x for x in probs)
    assert not any("mixed provenance" in x for x in probs)
