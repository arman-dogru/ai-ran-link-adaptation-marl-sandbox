"""What did the learners learn? Replays the evaluation episodes (default and coupled scenarios, first 30 episode
seeds) and counts, per reported CQI, the MCS offset (chosen MCS - reported CQI index) and the power level each
policy picks. Learners: all 15 seeds pooled. Output: results/policy_actions.csv (scenario, policy, cqi, delta,
power_db, count)."""
import json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch

from aran import agents as A, scenarios as S
from aran.env import MultiCellLAEnv
from run_grid import rule_policy, write_csv

N_EP = 30


def probe(env, pol, seeds, cnt):
    for e in seeds:
        obs, _ = env.reset(seed=e)
        pol.reset(env)
        done = False
        while not done:
            mcs, p = pol.act(env, obs)
            cqi = env.rep[env.sched]
            for c in range(env.cfg.n_cells):
                cnt[(int(cqi[c]), int(mcs[c]) - max(int(cqi[c]), 0), float(env.levels[int(p[c])]))] += 1
            obs, _, done, _, info = env.step_raw(mcs, p)
            pol.update(env, info)


def main():
    torch.set_num_threads(1)
    rows = []
    seeds = list(A.EVAL_SEEDS)[:N_EP]
    for sc in S.SCENARIOS:
        cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
        tun = json.load(open(os.path.join(ROOT, f"results/dev_tuning_{sc}.json")))
        env = MultiCellLAEnv(cfg)
        for name in ("olla-tuned",) + tuple(S.LEARNERS):
            cnt = Counter()
            if name == "olla-tuned":
                probe(env, rule_policy(name, tun), seeds, cnt)
            else:
                for s in S.SEEDS:
                    net = A.QNet(cfg.n_actions)
                    net.load_state_dict(torch.load(os.path.join(ROOT, f"results/raw/{sc}/{name}_s{s}/model.pt")))
                    probe(env, A.NetPolicy(net, name), seeds, cnt)
            for (q, d, p), n in sorted(cnt.items()):
                rows.append(dict(scenario=sc, policy=name, cqi=q, delta=d, power_db=p, count=n))
    write_csv(os.path.join(ROOT, "results/policy_actions.csv"), rows, list(rows[0]))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
