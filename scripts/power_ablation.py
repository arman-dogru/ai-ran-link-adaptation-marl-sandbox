"""Power ablation (review r1, m4): deploy every trained IQL / IQL-team / VDN policy (15 seeds, both scenarios) on the
first 30 evaluation episodes twice - as learned, and with its MCS choice kept but every power choice forced to 0 dB.
Output: results/power_ablation.csv (scenario, method, seed, se_learned, se_full_power, edge_learned, edge_full_power)."""
import os, sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import torch

from aran import agents as A, policies as P, scenarios as S
from aran.env import MultiCellLAEnv
from run_grid import write_csv

METHODS = ("iql", "iql_team", "vdn")
N_EP = 30


class FullPower(A.NetPolicy):
    def act(self, env, obs):
        m, p = super().act(env, obs)
        return m, np.zeros_like(p)


def job(args):
    sc, m, s = args
    torch.set_num_threads(1)
    cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
    env = MultiCellLAEnv(cfg)
    net = A.QNet(cfg.n_actions)
    net.load_state_dict(torch.load(os.path.join(ROOT, f"results/raw/{sc}/{m}_s{s}/model.pt")))
    eps = list(A.EVAL_SEEDS)[:N_EP]
    a = [P.run_episode(env, A.NetPolicy(net, m), e) for e in eps]
    b = [P.run_episode(env, FullPower(net, m), e) for e in eps]
    return dict(scenario=sc, method=m, seed=s, se_learned=np.mean([k["se"] for k in a]), se_full_power=np.mean([k["se"] for k in b]),
                edge_learned=np.mean([k["edge"] for k in a]), edge_full_power=np.mean([k["edge"] for k in b]),
                power_db_learned=np.mean([k["power_db"] for k in a]))


def main():
    jobs = [(sc, m, s) for sc in S.SCENARIOS for m in METHODS for s in S.SEEDS]
    with Pool(int(os.environ.get("WORKERS", "7"))) as pool:
        rows = pool.map(job, jobs)
    write_csv(os.path.join(ROOT, "results/power_ablation.csv"), rows, list(rows[0]))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
