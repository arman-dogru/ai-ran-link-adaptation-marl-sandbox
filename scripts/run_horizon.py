"""Evaluation-only check of the episode-length caveat: OLLA starts every episode from its initial offset and gets
only ~50 transmissions per UE in a 200-TTI episode, while the learners carry a prior across episodes. Here the
tuned OLLA rule, CQI-only and every trained learner are re-evaluated (default scenario, first 30 evaluation episode
seeds) with episodes of 200, 1000 and 3000 TTIs. Output: results/raw/horizon/<policy>.csv."""
import argparse, json, os, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch

from aran import agents as A, policies as P, scenarios as S
from aran.env import MultiCellLAEnv, with_
from run_grid import rule_policy, write_csv

HORIZONS = (200, 1000, 3000)
H_RULES = ("cqi-only", "olla-tuned", "olla-10")
N_EP = 30
KEYS = ("episode_tti", "seed", "episode_seed", "se", "bler", "jain", "edge", "power_db", "inr_db")


def expected_horizon_files():
    return sorted([f"horizon/rule_{r}.csv" for r in H_RULES] + [f"horizon/{m}.csv" for m in S.LEARNERS])


def job(args):
    name, raw, force = args
    path = os.path.join(raw, "horizon", f"{name}.csv")
    if not force and os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    torch.set_num_threads(1)
    base = S.scenario_config("default", os.path.join(ROOT, "config/default.json"))
    tun = json.load(open(os.path.join(os.path.dirname(os.path.abspath(raw)), "dev_tuning_default.json")))
    if name.startswith("rule_"):
        pols = [(-1, rule_policy(name[5:], tun))]
    else:
        pols = []
        for s in S.SEEDS:
            net = A.QNet(base.n_actions)
            net.load_state_dict(torch.load(os.path.join(raw, "default", f"{name}_s{s}", "model.pt")))
            pols.append((s, A.NetPolicy(net, name)))
    rows = []
    for h in HORIZONS:
        env = MultiCellLAEnv(with_(base, episode_tti=h))
        for s, pol in pols:
            for e in list(A.EVAL_SEEDS)[:N_EP]:
                k = P.run_episode(env, pol, e)
                k.update(episode_tti=h, seed=s, episode_seed=e)
                rows.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_csv(path, rows, KEYS)
    return path, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = [(f"rule_{r}", a.raw, a.force) for r in H_RULES] + [(m, a.raw, a.force) for m in S.LEARNERS]
    with Pool(a.workers) as pool:
        for p, dt in pool.imap_unordered(job, jobs):
            print(f"[{time.strftime('%H:%M:%S')}] {os.path.relpath(p, a.raw)} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
