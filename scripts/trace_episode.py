"""Per-TTI trace of one evaluation episode (default scenario, episode seed 10000) for the tuned OLLA rule and
seed-0 IQL / VDN / SA learners. Output: results/trace_default.csv (t, policy, cell, ue, sinr_db, mcs, ack,
power_db, thr10_db = 10%-BLER threshold of the chosen MCS)."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch

from aran import agents as A, phy, policies as P, scenarios as S
from aran.env import MultiCellLAEnv
from run_grid import rule_policy, write_csv

SEED = 10_000


def main():
    cfg = S.scenario_config("default", os.path.join(ROOT, "config/default.json"))
    tun = json.load(open(os.path.join(ROOT, "results/dev_tuning_default.json")))
    env = MultiCellLAEnv(cfg)
    pols = [("olla-tuned", rule_policy("olla-tuned", tun)), ("genie", rule_policy("genie", tun))]
    for m in ("sa", "iql", "vdn"):
        net = A.QNet(cfg.n_actions)
        net.load_state_dict(torch.load(os.path.join(ROOT, f"results/raw/default/{m}_s0/model.pt")))
        pols.append((m, A.NetPolicy(net, m)))
    rows = []
    for name, pol in pols:
        tr = P.run_episode(env, pol, SEED, trace=True)["trace"]
        for t in range(len(tr["sinr"])):
            for c in range(cfg.n_cells):
                m = int(tr["mcs"][t, c])
                rows.append(dict(t=t, policy=name, cell=c, ue=int(tr["ue"][t, c]), sinr_db=round(float(tr["sinr"][t, c]), 4),
                                 mcs=m, ack=int(tr["ack"][t, c]), power_db=float(tr["power"][t, c]), thr10_db=float(phy.THR10[m])))
    write_csv(os.path.join(ROOT, "results/trace_default.csv"), rows, list(rows[0]))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
