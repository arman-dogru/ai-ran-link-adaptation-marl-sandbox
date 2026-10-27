"""One-off provenance backfill (review r1, m7). The committed raw results were produced before run_grid / run_sweep /
run_horizon wrote meta.json. This script
  1. regenerates, with the current committed code, every rule run of the main grid, one training seed of every
     learner in both scenarios (full training), every rule file of the sweep and horizon checks, and the IQL sweep
     and horizon files, in a temporary directory;
  2. requires each regenerated output to equal the committed one (CSV bytes; model tensors);
  3. only then writes meta.json / <name>.meta.json next to every committed output (marked "backfilled") and checks
     that run_grid / run_sweep / run_horizon now report every committed output as "skip (up to date)".
The record goes to results/provenance_check.json. It refuses to run on a dirty tree."""
import filecmp, json, os, shutil, sys, tempfile, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch

from aran import agents as A, provenance as PV, scenarios as S
import run_grid as G, run_sweep as SW, run_horizon as H

RAW = os.path.join(ROOT, "results/raw")
CHECK_SEED = 0


def _regen(args):
    kind, a = args
    if kind == "grid":
        return kind, a, G.job(a)[1]
    if kind == "sweep":
        return kind, a, SW.job(a)[1]
    return kind, a, H.job(a)[1]


def _same_model(p, q):
    x, y = torch.load(p), torch.load(q)
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def main():
    files = PV.CODE_FILES + ("scripts/run_sweep.py", "scripts/run_horizon.py")
    sha, dirty = PV.git_state(files=files)
    if dirty:
        sys.exit("refusing: code files are dirty; commit first")
    tmp = tempfile.mkdtemp(prefix="aran_backfill_")
    for sc in S.SCENARIOS:
        shutil.copy(os.path.join(ROOT, f"results/dev_tuning_{sc}.json"), tmp)
    g_raw = os.path.join(tmp, "grid", "raw")
    os.makedirs(g_raw)
    for sc in S.SCENARIOS:
        shutil.copy(os.path.join(ROOT, f"results/dev_tuning_{sc}.json"), os.path.join(tmp, "grid"))
    s_raw = os.path.join(tmp, "sv", "raw")
    os.makedirs(s_raw)
    shutil.copy(os.path.join(ROOT, "results/dev_tuning_default.json"), os.path.join(tmp, "sv"))
    os.symlink(os.path.join(RAW, "default"), os.path.join(s_raw, "default"))
    jobs = [("grid", (sc, "learn", m, CHECK_SEED, g_raw, True)) for sc in S.SCENARIOS for m in S.LEARNERS]
    jobs += [("sweep", ("iql", s_raw, True)), ("horizon", ("iql", s_raw, True))]
    jobs += [("grid", (sc, "rule", r, 0, g_raw, True)) for sc in S.SCENARIOS for r in S.RULES]
    jobs += [("sweep", (f"rule_{r}", s_raw, True)) for r in SW.SWEEP_RULES]
    jobs += [("horizon", (f"rule_{r}", s_raw, True)) for r in H.H_RULES]
    verified, bad = [], []
    with Pool(int(os.environ.get("WORKERS", "7"))) as pool:
        for kind, a, dt in pool.imap_unordered(_regen, jobs):
            if kind == "grid":
                sc, k, name, seed = a[:4]
                rel = os.path.join(sc, f"rule_{name}" if k == "rule" else f"{name}_s{seed}")
                pairs = [("eval.csv", filecmp.cmp)]
                if k != "rule":
                    pairs += [("curve.csv", filecmp.cmp), ("model.pt", _same_model)]
                ok = all(f(os.path.join(g_raw, rel, n), os.path.join(RAW, rel, n), **({"shallow": False} if f is filecmp.cmp else {}))
                         for n, f in pairs)
                rel_files = [os.path.join(rel, n) for n, _ in pairs]
            else:
                rel = os.path.join(kind, f"{a[0]}.csv")
                ok = filecmp.cmp(os.path.join(s_raw, rel), os.path.join(RAW, rel), shallow=False)
                rel_files = [rel]
            (verified if ok else bad).extend(rel_files)
            print(f"[{time.strftime('%H:%M:%S')}] {'identical' if ok else 'DIFFERENT'} {rel} ({dt})", flush=True)
    if bad:
        json.dump(dict(ok=False, different=sorted(bad), identical=sorted(verified)), open(os.path.join(ROOT, "results/provenance_check.json"), "w"), indent=1)
        sys.exit(f"regenerated outputs differ: {bad}")
    note = dict(backfilled=True, backfilled_at=sha, verified_by="results/provenance_check.json")
    n = 0
    for sc in S.SCENARIOS:
        cfg = S.scenario_config(sc, os.path.join(ROOT, "config/default.json"))
        tun = json.load(open(os.path.join(ROOT, f"results/dev_tuning_{sc}.json")))
        runs = [("rule", r, 0) for r in S.RULES] + [("learn", m, s) for m in S.LEARNERS for s in S.SEEDS]
        for kind, name, seed in runs:
            d = os.path.join(RAW, sc, f"rule_{name}" if kind == "rule" else f"{name}_s{seed}")
            meta = PV.run_meta(cfg, tun, scenario=sc, kind=kind, method=name, seed=int(seed) if kind != "rule" else None,
                               eval_seeds=[min(A.EVAL_SEEDS), max(A.EVAL_SEEDS)],
                               train_episodes=S.EPISODES if kind != "rule" else 0, **note)
            PV.write_meta(os.path.join(d, "meta.json"), meta)
            n += 1
    base = S.scenario_config("default", os.path.join(ROOT, "config/default.json"))
    tun = json.load(open(os.path.join(ROOT, "results/dev_tuning_default.json")))
    for kind, script, names in (("sweep", "scripts/run_sweep.py", [f"rule_{r}" for r in SW.SWEEP_RULES] + list(S.LEARNERS)),
                                ("horizon", "scripts/run_horizon.py", [f"rule_{r}" for r in H.H_RULES] + list(S.LEARNERS))):
        for name in names:
            meta = PV.run_meta(base, tun, extra_files=(script,), kind=kind, policy=name, **note)
            PV.write_meta(os.path.join(RAW, kind, f"{name}.meta.json"), meta)
            n += 1
    # the run scripts themselves must now consider every committed output up to date
    status = [G.job((sc, "rule", r, 0, RAW, False))[1] for sc in S.SCENARIOS for r in S.RULES]
    status += [G.job((sc, "learn", m, s, RAW, False))[1] for sc in S.SCENARIOS for m in S.LEARNERS for s in S.SEEDS]
    status += [SW.job((nm, RAW, False))[1] for nm in [f"rule_{r}" for r in SW.SWEEP_RULES] + list(S.LEARNERS)]
    status += [H.job((nm, RAW, False))[1] for nm in [f"rule_{r}" for r in H.H_RULES] + list(S.LEARNERS)]
    assert all(s == "skip (up to date)" for s in status), status
    json.dump(dict(ok=True, code_commit=sha, metas_written=n, all_skip_up_to_date=len(status),
                   identical=sorted(verified), not_regenerated="learner seeds 1-14 of the main grid and the sweep/horizon "
                   "files of SA, IQL-team and VDN: same code and config as the verified files, not re-run (cost)"),
              open(os.path.join(ROOT, "results/provenance_check.json"), "w"), indent=1)
    print(f"ok: {len(verified)} files identical, {n} metas written, {len(status)} outputs up to date")


if __name__ == "__main__":
    main()
