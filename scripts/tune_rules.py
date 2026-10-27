"""Tune the rule baselines on the DEV episodes only (never the evaluation episodes): the best fixed MCS and the best
OLLA (BLER target, step) by mean spectral efficiency. Writes results/dev_tuning_<scenario>.{json,csv}."""
import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aran.env import MultiCellLAEnv
from aran import policies as P, agents as A
from aran import scenarios as S

TARGETS = (0.01, 0.05, 0.1, 0.2, 0.3)
STEPS = (0.1, 0.25, 0.5, 1.0)
INITS = (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0)


def _olla(args):
    cfg, t, st, i0 = args
    env = MultiCellLAEnv(cfg)
    return dict(kind="olla", mcs="", target=t, step=st, init=i0,
                se=float(np.mean([P.run_episode(env, P.OLLA(t, st, init=i0), s)["se"] for s in A.DEV_SEEDS])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default", choices=list(S.SCENARIOS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    cfg = S.scenario_config(a.scenario, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/default.json"))
    env = MultiCellLAEnv(cfg)
    rows = []
    for m in range(15):
        se = np.mean([P.run_episode(env, P.FixedMCS(m), s)["se"] for s in A.DEV_SEEDS])
        rows.append(dict(kind="fixed", mcs=m, target="", step="", init="", se=float(se)))
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        rows += pool.map(_olla, [(cfg, t, st, i0) for t in TARGETS for st in STEPS for i0 in INITS])
    fixed = max((r for r in rows if r["kind"] == "fixed"), key=lambda r: r["se"])
    olla = max((r for r in rows if r["kind"] == "olla"), key=lambda r: r["se"])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"dev_tuning_{a.scenario}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["kind", "mcs", "target", "step", "init", "se"])
        w.writeheader()
        w.writerows(rows)
    best = dict(scenario=a.scenario, fixed_mcs=int(fixed["mcs"]), olla_target=float(olla["target"]), olla_step=float(olla["step"]), olla_init=float(olla["init"]),
                dev_seeds=[min(A.DEV_SEEDS), max(A.DEV_SEEDS)], fixed_se=fixed["se"], olla_se=olla["se"])
    json.dump(best, open(os.path.join(a.out, f"dev_tuning_{a.scenario}.json"), "w"), indent=2)
    print(best)


if __name__ == "__main__":
    main()
