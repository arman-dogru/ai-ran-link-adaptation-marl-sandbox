"""Run the main grid: rule baselines and learners (scenario x method x seed), each learner trained from scratch on
training episodes and evaluated greedily on the shared evaluation episodes (EVAL_SEEDS).
Output: results/raw/<scenario>/<run>/{eval.csv, curve.csv, model.pt, meta.json}.

Re-runs: a run is skipped only if its meta.json shows the same code hash, config and tuned rule parameters (it would
reproduce byte for byte). A run whose outputs exist but whose provenance is stale or missing is an error unless
--force is given, so code changes can never leave stale rows behind silently."""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from aran import agents as A, policies as P
from aran.env import MultiCellLAEnv
from aran import scenarios as S
from aran import provenance as PV

KEYS = ("episode_seed", "se", "bler", "jain", "edge", "power_db", "inr_db")


def rule_policy(name, tun):
    return {"fixed": lambda: P.FixedMCS(tun["fixed_mcs"]), "cqi-only": P.CQIOnly, "olla-10": lambda: P.OLLA(0.1, 0.5),
            "olla-tuned": lambda: P.OLLA(tun["olla_target"], tun["olla_step"], init=tun["olla_init"]),
            "olla-tuned-backoff": lambda: P.OLLA(tun["olla_target"], tun["olla_step"], backoff=True, init=tun["olla_init"]),
            "genie": P.Genie}[name]()


def write_csv(path, rows, keys):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def evaluate(env, pol, seeds):
    rows = []
    for s in seeds:
        k = P.run_episode(env, pol, s)
        k["episode_seed"] = s
        rows.append(k)
    return rows


class StaleResult(RuntimeError):
    pass


def eval_seeds():
    """The evaluation episodes every reported number uses."""
    return list(A.EVAL_SEEDS)


def job(args):
    sc, kind, name, seed, out, force = args
    cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
    tun = json.load(open(os.path.join(os.path.dirname(os.path.abspath(out)), f"dev_tuning_{sc}.json")))
    d = os.path.join(out, sc, f"rule_{name}" if kind == "rule" else f"{name}_s{seed}")
    meta = PV.run_meta(cfg, tun, scenario=sc, kind=kind, method=name, seed=int(seed) if kind != "rule" else None,
                       eval_seeds=[min(A.EVAL_SEEDS), max(A.EVAL_SEEDS)],
                       train_episodes=S.EPISODES if kind != "rule" else 0)
    if not force and os.path.exists(os.path.join(d, "eval.csv")):
        if PV.up_to_date(os.path.join(d, "meta.json"), meta):
            return d, "skip (up to date)"
        raise StaleResult(f"{d}: outputs exist but meta.json is missing or stale; re-run with --force")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(1)
    env = MultiCellLAEnv(cfg)
    if kind == "rule":
        pol = rule_policy(name, tun)
    else:
        net, curve = A.train(name, cfg, seed, S.EPISODES, rule_policy("olla-tuned", tun))
        write_csv(os.path.join(d, "curve.csv"), curve, list(curve[0]))
        torch.save(net.state_dict(), os.path.join(d, "model.pt"))
        pol = A.NetPolicy(net, name)
    write_csv(os.path.join(d, "eval.csv"), evaluate(env, pol, eval_seeds()), KEYS)
    PV.write_meta(os.path.join(d, "meta.json"), meta)
    return d, f"{time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--scenarios", nargs="*", default=list(S.SCENARIOS), choices=list(S.SCENARIOS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rules", nargs="*", default=list(S.RULES), choices=list(S.RULES), help="rule baselines to run")
    ap.add_argument("--no-learners", action="store_true")
    a = ap.parse_args()
    jobs = []
    for sc in a.scenarios:
        jobs += [(sc, "rule", r, 0, a.out, a.force) for r in a.rules]
        if not a.no_learners:
            jobs += [(sc, "learn", m, s, a.out, a.force) for s in S.SEEDS for m in S.LEARNERS]
    with Pool(a.workers) as pool:
        for i, (d, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.relpath(d, a.out)} {dt}", flush=True)


if __name__ == "__main__":
    main()
