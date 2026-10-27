"""Short, seed-reproducible demo (a few minutes on a laptop CPU): tune OLLA on 10 dev episodes, train one learner
for --episodes (default 150) and compare it with the rules on 20 evaluation episodes. Prints a table and writes
<out>/demo_eval.csv and <out>/demo_curve.csv. This is a smoke run, not the experiment (see scripts/run_grid.py)."""
import argparse, csv, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from aran import agents as A, policies as P
from aran.config import load_config
from aran.env import MultiCellLAEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="iql", choices=A.METHODS)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(ROOT, "config/demo.json"))
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = load_config(a.config)
    env = MultiCellLAEnv(cfg)
    t0 = time.time()
    dev = list(A.DEV_SEEDS)[:10]
    best = max(((t, st, i0) for t in (0.1, 0.2, 0.3) for st in (0.1, 0.5) for i0 in (-2.0, 0.0)),
               key=lambda x: np.mean([P.run_episode(env, P.OLLA(x[0], x[1], init=x[2]), s)["se"] for s in dev]))
    olla = P.OLLA(*best[:2], init=best[2])
    net, curve = A.train(a.method, cfg, a.seed, a.episodes, P.OLLA(*best[:2], init=best[2]),
                         hp=dict(eps_frac=0.6), log=lambda m: print(m, flush=True))
    pols = [P.CQIOnly(), P.OLLA(0.1, 0.5), olla, A.NetPolicy(net, a.method), P.Genie()]
    ev = list(A.EVAL_SEEDS)[:20]
    rows = []
    for pol in pols:
        for s in ev:
            k = P.run_episode(env, pol, s)
            rows.append(dict(policy=pol.name, episode_seed=s, **{x: k[x] for x in ("se", "edge", "bler", "power_db")}))
    os.makedirs(a.out, exist_ok=True)
    for name, rr in (("demo_eval.csv", rows), ("demo_curve.csv", curve)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    print(f"\nOLLA tuned on 10 dev episodes: target={best[0]}, step={best[1]}, init={best[2]}")
    print(f"{'policy':28s} {'SE':>7s} {'edge':>7s} {'BLER':>6s} {'power':>6s}")
    for pol in pols:
        r = [x for x in rows if x["policy"] == pol.name]
        print(f"{pol.name:28s} {np.mean([x['se'] for x in r]):7.3f} {np.mean([x['edge'] for x in r]):7.3f} "
              f"{np.mean([x['bler'] for x in r]):6.3f} {np.mean([x['power_db'] for x in r]):6.2f}")
    print(f"done in {time.time() - t0:.0f} s (20 evaluation episodes; no CIs - use the full grid for claims)")


if __name__ == "__main__":
    main()
