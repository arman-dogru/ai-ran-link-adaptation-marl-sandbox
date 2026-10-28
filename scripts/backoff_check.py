"""Evidence for the back-off protocol deviation (docs/method.md): how often the first, strict back-off rule (top CQI
index AND OLLA at the top MCS) fires, on the dev AND the evaluation episodes of both scenarios, with the SE of the
strict rule, plain tuned OLLA and the relaxed (top CQI index only) rule. Output: results/backoff_strict_check.csv."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np

from aran import agents as A, phy, policies as P, scenarios as S
from aran.env import MultiCellLAEnv
from run_grid import write_csv


class StrictBackoff(P.OLLA):
    """The first version of the rule (commit 4f2065d): -6 dB only if the UE reports CQI index 14 AND OLLA picks MCS 14."""

    def act(self, env, obs):
        mcs = self.mcs_for(env)
        top = (env.rep[env.sched] >= phy.N_MCS - 1) & (mcs >= phy.N_MCS - 1)
        self.fired += int(top.sum())
        return mcs, np.where(top, len(env.cfg.power_levels_db) - 1, 0)


def main():
    rows = []
    for sc in S.SCENARIOS:
        t = json.load(open(os.path.join(ROOT, f"results/dev_tuning_{sc}.json")))
        env = MultiCellLAEnv(S.scenario_config(sc, os.path.join(ROOT, "config/default.json")))
        args = (t["olla_target"], t["olla_step"])
        for split, seeds in (("dev", A.DEV_SEEDS), ("eval", A.EVAL_SEEDS)):
            strict = StrictBackoff(*args, backoff=True, init=t["olla_init"])
            strict.fired = 0
            se = {name: float(np.mean([P.run_episode(env, pol, s)["se"] for s in seeds])) for name, pol in
                  (("strict", strict), ("olla", P.OLLA(*args, init=t["olla_init"])),
                   ("relaxed", P.OLLA(*args, backoff=True, init=t["olla_init"])))}
            rows.append(dict(scenario=sc, split=split, episodes=len(seeds), strict_fired_ttis=strict.fired,
                             se_strict=se["strict"], se_olla_tuned=se["olla"], se_relaxed=se["relaxed"]))
            print(rows[-1], flush=True)
    write_csv(os.path.join(ROOT, "results/backoff_strict_check.csv"), rows, list(rows[0]))


if __name__ == "__main__":
    main()
