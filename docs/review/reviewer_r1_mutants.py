"""The peer reviewer's round-1 mutants R1-R16 (their `my_mutants.py`, mutant list verbatim), re-run against this tree
with three harness changes, so the record is meaningful:
  - results/ is symlinked into each throw-away copy (the original copied no results/, so its control failed on
    test_committed_results_use_the_right_episodes and that unrelated failure "killed" most mutants);
  - an unmutated control run goes first and must pass;
  - only pytest exit 1 counts as killed (0 = SURVIVED, anything else = ERROR).
R2 and R3 report BAD-PATTERN: their call sites were refactored in r1; adapted R2/R3 are in scripts/mutation_check.py.
Writes results/mutation_reviewer_r1.json."""
import json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = [
 ("aran/agents.py", "        obs, _ = env.reset(seed=train_episode_seed(seed, ep))", "        obs, _ = env.reset(seed=10_000 + ep % 100)", "R1 learners train on the evaluation episodes"),
 ("scripts/run_grid.py", "evaluate(env, pol, A.EVAL_SEEDS)", "evaluate(env, pol, A.DEV_SEEDS)", "R2 grid evaluates on dev seeds"),
 ("scripts/tune_rules.py", '["se"] for s in A.DEV_SEEDS])))', '["se"] for s in A.EVAL_SEEDS])))', "R3 OLLA tuned on evaluation episodes"),
 ("aran/env.py", "        o[:, 0] = (self.rep[u] + 1) / 15.0", "        o[:, 0] = (phy.cqi_from_sinr(self.sinr_db(self.t + cfg.cqi_delay, self.power_db, u)[0]) + 1) / 15.0", "R4 obs[0] = true current-channel CQI (genie leak)"),
 ("aran/env.py", "        o[:, 3] = (self.rep[u] - self.prev_rep[u]) / 5.0", "        o[:, 3] = (phy.cqi_from_sinr(self.sinr_db(self.t + cfg.cqi_delay, self.power_db, u)[0]) - self.rep[u]) / 5.0", "R5 trend feature leaks future CQI"),
 ("aran/env.py", "        rep = phy.cqi_from_sinr(self.meas[self.t])", "        rep = phy.cqi_from_sinr(self.meas[min(self.t + 1, len(self.meas) - 1)])", "R6 CQI delay off by one (D-1)"),
 ("scripts/aggregate.py", "            base = float(Mb.mean())", "            base = float(Ma.mean())", "R7 relative % uses the wrong base"),
 ("scripts/aggregate.py", "wins=int((per_seed > 0).sum())", "wins=int((per_seed >= 0).sum())", "R8 ties counted as wins"),
 ("scripts/aggregate.py", "        p = stats.signflip_p(per_seed, 2 ** len(per_seed), rng)", "        p = stats.signflip_p(per_seed, 2 ** len(per_seed), rng) / 2", "R9 one-sided p for learners"),
 ("scripts/aggregate.py", "    holm_family(out[\"comparisons\"])", "    [holm_family([c for c in out[\"comparisons\"] if c[\"scenario\"] == sc]) for sc in S.SCENARIOS]", "R10 Holm family per scenario (17, not 34)"),
 ("aran/agents.py", "            buf.add(lo, a, r, obs2[:n_learn], float(done))", "            buf.add(lo, a, r, obs2[:n_learn], 0.0)", "R11 replay never marks terminal"),
 ("aran/agents.py", "        return np.clip(env.base_mcs() + delta, 0, 14), pidx", "        return np.clip(env.base_mcs() + delta, 0, 14), np.zeros_like(pidx)", "R12 deployed policy ignores power action"),
 ("aran/policies.py", "            p = np.where(top, len(env.cfg.power_levels_db) - 1, 0)", "            p = np.where(top & (mcs >= phy.N_MCS - 1), len(env.cfg.power_levels_db) - 1, 0)", "R13 revert to strict back-off rule"),
 ("aran/env.py", "        self.meas.append(self._measure(t + D, pdb))", "        self.meas.append(self._measure(t + D, np.zeros(C)))", "R14 CQI ignores neighbours' actual power"),
 ("scripts/aggregate.py", "        D = Ma - Mb  # broadcasting a rule's (1, E) row over seeds", "        D = Ma - Mb[:, ::-1]  # broadcasting a rule's (1, E) row over seeds", "R15 rule row mis-paired with learner episodes"),
 ("aran/agents.py", "    if method == \"iql_team\":\n        return np.full_like(cell_r, cell_r.mean())", "    if method == \"iql_team\":\n        return np.full_like(cell_r, cell_r.sum())", "R16 IQL-team uses summed (not mean) team reward"),
]


def run_one(edit):
    with tempfile.TemporaryDirectory() as td:
        for d in ("aran", "scripts", "tests", "config"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(td, d), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), td)
        os.symlink(os.path.join(ROOT, "results"), os.path.join(td, "results"))
        if edit:
            f, a, b = edit
            p = os.path.join(td, f); s = open(p).read()
            if s.count(a) != 1:
                return "BAD-PATTERN", None, ""
            open(p, "w").write(s.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"], cwd=td,
                           capture_output=True, text=True, timeout=900)
        fail = [l for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]
        last = [l.strip() for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
        return {0: "SURVIVED", 1: "killed"}.get(r.returncode, "ERROR"), r.returncode, (fail[:1] or last[-1:] or [""])[0]


def main():
    st, rc, why = run_one(None)
    control = dict(status="passed" if rc == 0 else "FAILED", exit=rc, summary=why)
    print("control:", control, flush=True)
    if rc != 0:
        return 1
    res = []
    with ThreadPoolExecutor(int(os.environ.get("MUT_WORKERS", "4"))) as ex:
        for (f, a, b, what), (st, rc, why) in zip(M, ex.map(lambda m: run_one(m[:3]), M)):
            print(f"{st:11s} {what} :: {why}", flush=True)
            res.append(dict(what=what, status=st, exit=rc, why=why))
    killed = sum(r["status"] == "killed" for r in res)
    json.dump(dict(control=control, killed=killed, total=len(M), mutants=res),
              open(os.path.join(ROOT, "results/mutation_reviewer_r1.json"), "w"), indent=1)
    print(killed, "/", len(M), "killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
