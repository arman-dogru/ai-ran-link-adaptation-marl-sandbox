"""Single-edit mutation check: each mutant is applied to a throw-away copy of aran/, scripts/ and tests/ (the working
tree is never touched) and `pytest -x` must fail. Writes results/mutation_check.json; exit 1 if any mutant survives."""
import json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = [  # (file, original, mutant, what it breaks)
    ("aran/phy.py", "x = SLOPE * (np.asarray(sinr_db, float) - THR10[np.asarray(mcs)]) + LN9", "x = SLOPE * (np.asarray(sinr_db, float) - THR10[np.asarray(mcs)])", "BLER is 50% (not 10%) at the threshold"),
    ("aran/phy.py", "return 1.0 / (1.0 + np.exp(x))", "return 1.0 / (1.0 + np.exp(-x))", "BLER rises with SINR"),
    ("aran/phy.py", 'np.searchsorted(THR10, np.asarray(sinr_db, float), side="right") - 1', 'np.searchsorted(THR10, np.asarray(sinr_db, float), side="left") - 1', "CQI at exact threshold rounds down"),
    ("aran/phy.py", "return np.argmax(EFF * (1.0 - bler(s, np.arange(N_MCS))), axis=-1)", "return np.argmax(EFF * (1.0 - bler(s, np.arange(N_MCS))) * 0 + (1.0 - bler(s, np.arange(N_MCS))), axis=-1)", "genie maximises success, not goodput"),
    ("aran/phy.py", "return 128.1 + 37.6 * np.log10(np.maximum(np.asarray(d_m, float), 35.0) / 1000.0)", "return 128.1 + 37.6 * np.log10(np.asarray(d_m, float) / 1000.0)", "pathloss not clipped at 35 m"),
    ("aran/phy.py", "return -174.0 + 10.0 * np.log10(bw_hz) + nf_db", "return -174.0 + 10.0 * np.log10(bw_hz) - nf_db", "noise figure sign"),
    ("aran/phy.py", "return 1.0 if s2 == 0 else float(np.sum(x) ** 2 / (len(x) * s2))", "return 1.0 if s2 == 0 else float(np.sum(x) ** 2 / s2)", "Jain not normalised by n"),
    ("aran/phy.py", "return float(j0(2 * np.pi * fd * tti_s))", "return float(j0(np.pi * fd * tti_s))", "Doppler correlation off by 2x"),
    ("aran/env.py", "            if np.argmax(g) == serving[i]:", "            if True:", "UEs not attached to the strongest site"),
    ("aran/env.py", "        h[t] = rho * h[t - 1] + s * w[t]", "        h[t] = w[t]", "fading has no time correlation"),
    ("aran/env.py", "    s = np.sqrt(max(0.0, 1 - rho * rho))", "    s = np.sqrt(max(0.0, 1 - rho))", "fading power not preserved"),
    ("aran/env.py", "p_on = 1.0 if load >= 1 else (p_off * load / (1 - load))", "p_on = 1.0 if load >= 1 else (p_off * load)", "outer load not stationary at target"),
    ("aran/env.py", "        intf = ctrl.sum(1) - sig\n        if cfg.n_outer:\n            intf = intf + (rx[:, C:] * e[\"outer_on\"][k][None, :]).sum(1)\n        return", "        intf = ctrl.sum(1) - sig\n        if cfg.n_outer:\n            intf = intf + rx[:, C:].sum(1)\n        return", "outer cells always on"),
    ("aran/env.py", "        ctrl = rx[:, :C] * phy.db2lin(np.asarray(power_db, float))[None, :]", "        ctrl = rx[:, :C]", "power back-off ignored"),
    ("aran/env.py", "        P[np.arange(len(e[\"serving\"])), e[\"serving\"]] = 1.0", "        pass", "CQI measured at the data power, not reference power"),
    ("aran/env.py", "        rep = phy.cqi_from_sinr(self.meas[self.t])", "        rep = phy.cqi_from_sinr(self.meas[-1])", "CQI report not delayed"),
    ("aran/env.py", "        return out + self.ep[\"cqi_noise\"][k]", "        return out", "no CQI measurement noise"),
    ("aran/env.py", "        metric = r_hat / self.avg_thr", "        metric = r_hat", "max-rate instead of PF"),
    ("aran/env.py", "        self.avg_thr = (1 - 1 / cfg.pf_window) * self.avg_thr + served / cfg.pf_window", "        self.avg_thr = (1 - 1 / cfg.pf_window) * self.avg_thr", "PF average ignores service"),
    ("aran/env.py", "        ack = e[\"u_harq\"][t] >= phy.bler(sinr, mcs)", "        ack = e[\"u_harq\"][t] < phy.bler(sinr, mcs)", "ACK probability inverted"),
    ("aran/env.py", "        sinr, inr = self.sinr_db(t + D, pdb, u)", "        sinr, inr = self.sinr_db(t, pdb, u)", "transmission uses a stale channel"),
    ("aran/env.py", "    return np.asarray(cfg.mcs_deltas)[a // npow], a % npow", "    return np.asarray(cfg.mcs_deltas)[a % len(cfg.mcs_deltas)], a // len(cfg.mcs_deltas)", "action decoding swapped"),
    ("aran/env.py", "        mcs = np.clip(self.base_mcs() + delta, 0, phy.N_MCS - 1)", "        mcs = np.clip(delta, 0, phy.N_MCS - 1)", "MCS delta not relative to CQI"),
    ("aran/env.py", "            o[:, 9] = nb / 6.0", "            o[:, 9] = self.power_db / 6.0", "neighbour power feature is own power"),
    ("aran/env.py", "edge=float(np.sort(ue_thr)[:q].mean())", "edge=float(np.sort(ue_thr)[-q:].mean())", "edge KPI uses the best UEs"),
    ("aran/env.py", "se=float(self.cell_bits.sum() / (T * self.cfg.n_cells))", "se=float(self.cell_bits.sum() / T)", "SE not per cell"),
    ("aran/policies.py", "        up = self.step_down * self.target / (1.0 - self.target)", "        up = self.step_down * self.target", "OLLA up-step wrong ratio"),
    ("aran/policies.py", "np.where(ack, up, -self.step_down)", "np.where(ack, -self.step_down, up)", "OLLA steps inverted"),
    ("aran/policies.py", "        self.off = np.full(len(env.ep[\"serving\"]), self.init)", "        self.off = np.zeros(len(env.ep[\"serving\"]))", "OLLA initial offset ignored"),
    ("aran/policies.py", "est = np.where(cqi >= 0, phy.THR10[np.maximum(cqi, 0)], phy.THR10[0] - 2.0) + self.off[u]", "est = np.where(cqi >= 0, phy.THR10[np.maximum(cqi, 0)], phy.THR10[0] - 2.0) - self.off[u]", "OLLA offset sign"),
    ("aran/policies.py", "            top = env.rep[env.sched] >= phy.N_MCS - 1", "            top = env.rep[env.sched] >= 0", "back-off heuristic always backs off"),
    ("aran/policies.py", "            top = env.rep[env.sched] >= phy.N_MCS - 1", "            top = env.rep[env.sched] >= phy.N_MCS", "back-off heuristic never fires"),
    ("aran/policies.py", "        sinr, _ = env.sinr_db(env.t + env.cfg.cqi_delay, np.zeros(C), env.sched)", "        sinr, _ = env.sinr_db(env.t, np.zeros(C), env.sched)", "genie reads a stale channel"),
    ("aran/agents.py", "    if method == \"iql_team\":\n        return np.full_like(cell_r, cell_r.mean())", "    if method == \"iql_team\":\n        return cell_r", "IQL-team uses own reward"),
    ("aran/agents.py", "        y = r.sum(1) + gamma * (1 - d) * q2.sum(1)", "        y = r.mean(1) + gamma * (1 - d) * q2.sum(1)", "VDN target uses mean reward"),
    ("aran/agents.py", "        return nn.functional.smooth_l1_loss(q.sum(1), y)", "        return nn.functional.smooth_l1_loss(q.mean(1), y)", "VDN mixes by mean"),
    ("aran/agents.py", "        a2 = net(o2).argmax(-1, keepdim=True)", "        a2 = tgt(o2).argmax(-1, keepdim=True)", "not double DQN"),
    ("aran/agents.py", "    y = r + gamma * (1 - d).unsqueeze(1) * q2", "    y = r + gamma * q2", "terminal not masked (independent)"),
    ("aran/agents.py", "        y = r.sum(1) + gamma * (1 - d) * q2.sum(1)", "        y = r.sum(1) + gamma * q2.sum(1)", "terminal not masked (VDN)"),
    ("aran/agents.py", "    return TRAIN_SEED_BASE * (int(seed) + 1) + int(ep)", "    return TRAIN_SEED_BASE * int(seed) + int(ep)", "seed 0 trains on dev/eval episodes"),
    ("aran/agents.py", "    n_learn = 1 if method == \"sa\" else cfg.n_cells", "    n_learn = cfg.n_cells", "single agent learns in every cell"),
    ("aran/agents.py", "    torch.manual_seed(seed)\n", "    torch.manual_seed(0)\n", "training seed ignored by torch"),
    ("aran/agents.py", "                if updates % hp[\"sync\"] == 0:\n                    tgt.load_state_dict", "                if updates % hp[\"sync\"] == -1:\n                    tgt.load_state_dict", "target net never synced"),
    ("aran/agents.py", "        self.full = self.full or self.i == 0", "        self.full = False", "replay forgets wrap-around"),
    ("aran/stats.py", "        c = rng.integers(0, E, E)\n", "        c = np.arange(E)\n", "crossed bootstrap ignores episodes"),
    ("aran/stats.py", "        r = rng.integers(0, S, S)\n", "        r = np.arange(S)\n", "crossed bootstrap ignores seeds"),
    ("aran/stats.py", "        return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))", "        return float(np.mean((signs * d).mean(1) >= obs - 1e-12))", "exact sign-flip one-sided"),
    ("aran/stats.py", "    return float((np.sum(np.abs(null) >= obs - 1e-12) + 1) / (flips + 1))", "    return float(np.sum(np.abs(null) >= obs - 1e-12) / flips)", "MC sign-flip p can be 0"),
    ("aran/stats.py", "        run = max(run, min(1.0, (k - r) * p[i]))", "        run = min(1.0, (k - r) * p[i])", "Holm not monotone"),
    ("aran/stats.py", "def ci(boot, lo=2.5, hi=97.5):", "def ci(boot, lo=10, hi=90):", "80% CI"),
    ("scripts/aggregate.py", "        if not a.allow_partial:\n            return 2", "        if not a.allow_partial:\n            pass", "aggregate accepts partial grid"),
    ("scripts/aggregate.py", "        if os.path.abspath(a.out) == os.path.abspath(os.path.join(ROOT, \"results\")):", "        if False:", "--allow-partial overwrites results/"),
    ("scripts/aggregate.py", "        p = stats.signflip_p(per_seed, 2 ** len(per_seed), rng)", "        p = stats.signflip_p(D.ravel(), FLIPS_MC, rng)", "learned test on (seed, episode) cells"),
    ("scripts/aggregate.py", "    adj = stats.holm([c[\"p\"] for c in comps])", "    adj = [c[\"p\"] for c in comps]", "no multiple-comparison correction"),
    ("scripts/aggregate.py", "        c[\"significant\"] = bool(q < 0.05 and (c[\"ci\"][0] > 0 or c[\"ci\"][1] < 0))", "        c[\"significant\"] = bool(q < 0.05)", "significance ignores the CI"),
    ("scripts/aggregate.py", "        df = pd.read_csv(os.path.join(raw, sc, f\"{method}_s{s}\", \"eval.csv\")).sort_values(\"episode_seed\")", "        df = pd.read_csv(os.path.join(raw, sc, f\"{method}_s{s}\", \"eval.csv\"))", "episodes not aligned before pairing"),
    ("aran/scenarios.py", "    out = []\n    for sc in scenarios:", "    out = []\n    for sc in scenarios[:1]:", "grid check ignores the coupled scenario"),
]


def run_one(i, f, a, b):
    with tempfile.TemporaryDirectory() as td:
        for d in ("aran", "scripts", "tests", "config"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(td, d), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), td)
        p = os.path.join(td, f)
        s = open(p).read()
        if s.count(a) != 1:
            return i, "BAD-PATTERN", s.count(a)
        open(p, "w").write(s.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"], cwd=td,
                           capture_output=True, text=True, timeout=900)
        return i, "killed" if r.returncode != 0 else "SURVIVED", r.returncode


def main():
    res = [None] * len(M)
    with ThreadPoolExecutor(int(os.environ.get("MUT_WORKERS", "6"))) as ex:
        for i, st, rc in ex.map(lambda x: run_one(x[0], *x[1][:3]), enumerate(M)):
            res[i] = dict(file=M[i][0], what=M[i][3], status=st)
            print(f"{st:12s} {M[i][0]:22s} {M[i][3]}", flush=True)
    killed = sum(r["status"] == "killed" for r in res)
    json.dump(dict(killed=killed, total=len(M), mutants=res), open(os.path.join(ROOT, "results/mutation_check.json"), "w"), indent=1)
    print(f"{killed}/{len(M)} mutants killed")
    return 0 if killed == len(M) else 1


if __name__ == "__main__":
    sys.exit(main())
