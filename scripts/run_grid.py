"""Run the main grid: rule baselines and learners (scenario x method x seed), each learner trained from scratch on
training episodes and evaluated greedily on the shared evaluation episodes. Existing outputs are skipped unless
--force. Output: results/raw/<scenario>/<run>/{eval.csv, curve.csv, model.pt}."""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from aran import agents as A, policies as P
from aran.env import MultiCellLAEnv
from aran import scenarios as S

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


def job(args):
    sc, kind, name, seed, out, force = args
    cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
    tun = json.load(open(os.path.join(os.path.dirname(os.path.abspath(out)), f"dev_tuning_{sc}.json")))
    d = os.path.join(out, sc, f"rule_{name}" if kind == "rule" else f"{name}_s{seed}")
    if not force and os.path.exists(os.path.join(d, "eval.csv")):
        return d, 0.0
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
    write_csv(os.path.join(d, "eval.csv"), evaluate(env, pol, A.EVAL_SEEDS), KEYS)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--scenarios", nargs="*", default=list(S.SCENARIOS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rules", nargs="*", default=list(S.RULES), help="rule baselines to run")
    ap.add_argument("--no-learners", action="store_true")
    a = ap.parse_args()
    jobs = []
    for sc in a.scenarios:
        jobs += [(sc, "rule", r, 0, a.out, a.force) for r in a.rules]
        if not a.no_learners:
            jobs += [(sc, "learn", m, s, a.out, a.force) for s in S.SEEDS for m in S.LEARNERS]
    with Pool(a.workers) as pool:
        for i, (d, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {os.path.relpath(d, a.out)} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
