"""Tune the rule baselines on the DEV episodes only (never the evaluation episodes): the best fixed MCS and the best
OLLA (BLER target, step, initial offset) by mean spectral efficiency. Each scenario is tuned on its own.
Writes results/dev_tuning_<scenario>.{json,csv}; the JSON records every episode seed actually used."""
import argparse, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from aran.env import MultiCellLAEnv
from aran import policies as P, agents as A
from aran import scenarios as S

TARGETS = (0.01, 0.05, 0.1, 0.2, 0.3)
STEPS = (0.1, 0.25, 0.5, 1.0)
INITS = (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0)


def _score(env, pol, seeds, used):
    out = []
    for s in seeds:
        used.add(int(s))
        out.append(P.run_episode(env, pol, s)["se"])
    return float(np.mean(out))


def _olla(args):
    cfg, t, st, i0, seeds = args
    used = set()
    se = _score(MultiCellLAEnv(cfg), P.OLLA(t, st, init=i0), seeds, used)
    return dict(kind="olla", mcs="", target=t, step=st, init=i0, se=se), used


def tune(cfg, scenario, seeds=None, workers=7, targets=TARGETS, steps=STEPS, inits=INITS):
    """Returns (rows, best). `best["dev_seeds"]` lists every episode seed any candidate was scored on."""
    seeds = list(A.DEV_SEEDS) if seeds is None else list(seeds)
    env, used, rows = MultiCellLAEnv(cfg), set(), []
    for m in range(15):
        rows.append(dict(kind="fixed", mcs=m, target="", step="", init="", se=_score(env, P.FixedMCS(m), seeds, used)))
    jobs = [(cfg, t, st, i0, seeds) for t in targets for st in steps for i0 in inits]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_olla, jobs)
    else:
        res = list(map(_olla, jobs))
    for r, u in res:
        rows.append(r)
        used |= u
    fixed = max((r for r in rows if r["kind"] == "fixed"), key=lambda r: r["se"])
    olla = max((r for r in rows if r["kind"] == "olla"), key=lambda r: r["se"])
    best = dict(scenario=scenario, fixed_mcs=int(fixed["mcs"]), olla_target=float(olla["target"]),
                olla_step=float(olla["step"]), olla_init=float(olla["init"]), dev_seeds=sorted(used),
                fixed_se=fixed["se"], olla_se=olla["se"])
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default", choices=list(S.SCENARIOS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    cfg = S.scenario_config(a.scenario, os.path.join(ROOT, "config/default.json"))
    rows, best = tune(cfg, a.scenario, workers=a.workers)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"dev_tuning_{a.scenario}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["kind", "mcs", "target", "step", "init", "se"])
        w.writeheader()
        w.writerows(rows)
    json.dump(best, open(os.path.join(a.out, f"dev_tuning_{a.scenario}.json"), "w"), indent=2)
    print({k: v for k, v in best.items() if k != "dev_seeds"}, f"{len(best['dev_seeds'])} dev seeds")


if __name__ == "__main__":
    main()
