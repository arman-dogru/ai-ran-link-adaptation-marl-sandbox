"""Aggregate results/raw into results/summary.json + CSV tables. Refuses (exit 2, writes nothing) unless the raw
directory holds exactly the expected grid (no missing, no unexpected files). --allow-partial is for debugging only
and must write to a separate --out directory, never over the committed results/.

Statistics (see docs/method.md):
  * learned methods: 15 training seeds x 100 shared evaluation episodes (crossed). 95% CIs: two-way bootstrap
    resampling seeds and episodes independently. Test: exact two-sided sign-flip on the per-seed mean differences.
  * rule vs rule: deterministic per episode -> episode-level bootstrap CI and Monte-Carlo sign-flip over episodes.
  * one Holm family over every pre-registered comparison in COMPARISONS x scenarios."""
import argparse, csv, json, os, sys, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import pandas as pd

from aran import scenarios as S, stats
from run_sweep import expected_sweep_files, SWEEP_RULES
from run_horizon import expected_horizon_files, H_RULES

KPIS = ("se", "edge", "jain", "bler", "power_db", "inr_db")
METHODS = tuple(S.RULES) + tuple(S.LEARNERS)
COMPARISONS = ([(m, "olla-tuned", k) for m in S.LEARNERS for k in ("se", "edge")] +
               [("vdn", "iql", "se"), ("vdn", "iql", "edge"), ("iql_team", "iql", "se"), ("iql_team", "iql", "edge"),
                ("vdn", "iql_team", "se"), ("iql", "sa", "se"),
                ("olla-tuned", "cqi-only", "se"), ("olla-tuned-backoff", "olla-tuned", "se"),
                ("olla-tuned-backoff", "olla-tuned", "edge")])
B = 4000
FLIPS_MC = 100_000


def check_grid(raw):
    """Return (missing, unexpected) relative paths versus the expected main grid + sweep + horizon check."""
    want = set(S.expected_main_files()) | set(expected_sweep_files()) | set(expected_horizon_files())
    have = set()
    for sub in list(S.SCENARIOS) + ["sweep", "horizon"]:
        d = os.path.join(raw, sub)
        for dp, _, fs in os.walk(d):
            for f in fs:
                have.add(os.path.relpath(os.path.join(dp, f), raw))
    return sorted(want - have), sorted(have - want)


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def load_matrix(raw, sc, method, kpi, seeds):
    """(n_seeds, n_episodes) for learners, (1, n_episodes) for rules; columns ordered by episode seed."""
    if method in S.RULES:
        df = pd.read_csv(os.path.join(raw, sc, f"rule_{method}", "eval.csv")).sort_values("episode_seed")
        return df[kpi].to_numpy()[None, :], df["episode_seed"].to_numpy()
    rows, ep = [], None
    for s in seeds:
        df = pd.read_csv(os.path.join(raw, sc, f"{method}_s{s}", "eval.csv")).sort_values("episode_seed")
        if ep is None:
            ep = df["episode_seed"].to_numpy()
        assert np.array_equal(ep, df["episode_seed"].to_numpy()), "evaluation episodes differ between seeds"
        rows.append(df[kpi].to_numpy())
    return np.array(rows), ep


def compare(Ma, Mb, learned_a, learned_b, rng):
    """Difference a - b. Returns dict(diff, ci, p, unit, n, wins)."""
    if learned_a or learned_b:
        D = Ma - Mb  # broadcasting a rule's (1, E) row over seeds
        per_seed = D.mean(1)
        bt = stats.boot_crossed(D, B, rng)
        p = stats.signflip_p(per_seed, 2 ** len(per_seed), rng)
        return dict(diff=float(D.mean()), ci=stats.ci(bt), p=p, unit="training seed", n=len(per_seed),
                    wins=int((per_seed > 0).sum()), losses=int((per_seed < 0).sum()))
    d = (Ma - Mb)[0]
    bt = stats.boot_1d(d, B, rng)
    p = stats.signflip_p(d, FLIPS_MC, rng)
    return dict(diff=float(d.mean()), ci=stats.ci(bt), p=p, unit="evaluation episode", n=len(d),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def holm_family(comps):
    """Holm over every comparison in the list (in place)."""
    adj = stats.holm([c["p"] for c in comps])
    for c, q in zip(comps, adj):
        c["p_holm"] = float(q)
        c["significant"] = bool(q < 0.05 and (c["ci"][0] > 0 or c["ci"][1] < 0))
    return comps


def summarise(raw, seeds):
    out = {"scenarios": {}, "comparisons": []}
    for sc in S.SCENARIOS:
        meth = {}
        for m in METHODS:
            meth[m] = {}
            for k in KPIS:
                M, ep = load_matrix(raw, sc, m, k, seeds)
                rng = rng_for(sc, m, k)
                bt = stats.boot_crossed(M, B, rng) if m in S.LEARNERS else stats.boot_1d(M[0], B, rng)
                meth[m][k] = dict(mean=float(M.mean()), ci=stats.ci(bt))
            meth[m]["n_seeds"] = int(M.shape[0]) if m in S.LEARNERS else 0
            meth[m]["n_episodes"] = int(M.shape[1])
        out["scenarios"][sc] = {"methods": meth}
        for a, b, k in COMPARISONS:
            Ma, _ = load_matrix(raw, sc, a, k, seeds)
            Mb, _ = load_matrix(raw, sc, b, k, seeds)
            c = compare(Ma, Mb, a in S.LEARNERS, b in S.LEARNERS, rng_for("cmp", sc, a, b, k))
            base = float(Mb.mean())
            c.update(scenario=sc, a=a, b=b, metric=k, rel_pct=100 * c["diff"] / base if base else float("nan"))
            out["comparisons"].append(c)
    holm_family(out["comparisons"])
    out["holm_family_size"] = len(out["comparisons"])
    out["tuning"] = {sc: json.load(open(os.path.join(os.path.dirname(os.path.abspath(raw)), f"dev_tuning_{sc}.json")))
                     for sc in S.SCENARIOS}
    out["grid"] = dict(seeds=list(seeds), episodes_train=S.EPISODES, eval_episodes=out["scenarios"]["default"]["methods"]["olla-tuned"]["n_episodes"],
                       bootstrap=B, flips_mc=FLIPS_MC)
    return out


def curves(raw, seeds, win=25):
    rows = []
    for sc in S.SCENARIOS:
        for m in S.LEARNERS:
            C = np.array([pd.read_csv(os.path.join(raw, sc, f"{m}_s{s}", "curve.csv"))["se"].to_numpy() for s in seeds])
            k = np.ones(win) / win
            Sm = np.array([np.convolve(c, k, mode="valid") for c in C])
            mu, sd = Sm.mean(0), Sm.std(0, ddof=1)
            half = 2.145 * sd / np.sqrt(len(seeds))  # t_{0.975, 14}
            for i in range(len(mu)):
                rows.append(dict(scenario=sc, method=m, episode=i + win, se_mean=mu[i], lo=mu[i] - half[i], hi=mu[i] + half[i]))
    return rows


def sweep(raw, seeds):
    ref = pd.read_csv(os.path.join(raw, "sweep", "rule_olla-tuned.csv"))
    rows = []
    for name in [f"rule_{r}" for r in SWEEP_RULES] + list(S.LEARNERS):
        df = pd.read_csv(os.path.join(raw, "sweep", f"{name}.csv"))
        for (v, d), g in df.groupby(["speed_kmh", "cqi_delay"]):
            r = ref[(ref.speed_kmh == v) & (ref.cqi_delay == d)].sort_values("episode_seed")["se"].to_numpy()
            if name.startswith("rule_"):
                x = g.sort_values("episode_seed")["se"].to_numpy()
                bt = stats.boot_1d(x - r, 2000, rng_for("sw", name, v, d))
                D = (x - r)[None]
            else:
                D = np.array([g[g.seed == s].sort_values("episode_seed")["se"].to_numpy() for s in seeds]) - r[None]
                bt = stats.boot_crossed(D, 2000, rng_for("sw", name, v, d))
                x = D + r[None]
            lo, hi = stats.ci(bt)
            rows.append(dict(policy=name.replace("rule_", ""), speed_kmh=v, cqi_delay=d, se=float(np.mean(x)),
                             diff_vs_olla=float(D.mean()), lo=lo, hi=hi))
    return rows


def horizon(raw, seeds):
    """Mean SE per (policy, episode length), with 95% CIs (crossed bootstrap for learners, episodes for rules) and the
    difference to tuned OLLA at the same length."""
    ref = pd.read_csv(os.path.join(raw, "horizon", "rule_olla-tuned.csv"))
    rows = []
    for name in [f"rule_{r}" for r in H_RULES] + list(S.LEARNERS):
        df = pd.read_csv(os.path.join(raw, "horizon", f"{name}.csv"))
        for h, g in df.groupby("episode_tti"):
            r = ref[ref.episode_tti == h].sort_values("episode_seed")["se"].to_numpy()
            if name.startswith("rule_"):
                X = g.sort_values("episode_seed")["se"].to_numpy()[None]
                b1, b2 = stats.boot_1d(X[0], 2000, rng_for("h", name, h)), stats.boot_1d(X[0] - r, 2000, rng_for("hd", name, h))
            else:
                X = np.array([g[g.seed == s].sort_values("episode_seed")["se"].to_numpy() for s in seeds])
                b1, b2 = stats.boot_crossed(X, 2000, rng_for("h", name, h)), stats.boot_crossed(X - r[None], 2000, rng_for("hd", name, h))
            (lo, hi), (dlo, dhi) = stats.ci(b1), stats.ci(b2)
            rows.append(dict(policy=name.replace("rule_", ""), episode_tti=int(h), se=float(X.mean()), lo=lo, hi=hi,
                             diff_vs_olla=float((X - r[None]).mean()), diff_lo=dlo, diff_hi=dhi))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--allow-partial", action="store_true", help="debug only; requires --out other than results/")
    a = ap.parse_args(argv)
    missing, extra = check_grid(a.raw)
    if missing or extra:
        if extra:
            print(f"refusing to aggregate: {len(extra)} unexpected files under {a.raw}, e.g. {extra[:3]}", file=sys.stderr)
        if missing:
            print(f"refusing to aggregate: grid incomplete, {len(missing)} expected files missing, e.g. {missing[:3]}", file=sys.stderr)
        if not a.allow_partial:
            return 2
        if os.path.abspath(a.out) == os.path.abspath(os.path.join(ROOT, "results")):
            print("--allow-partial must write to a separate --out directory", file=sys.stderr)
            return 2
    os.makedirs(a.out, exist_ok=True)
    seeds = list(S.SEEDS)
    summ = summarise(a.raw, seeds)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, sort_keys=True)
    rows = []
    for sc, v in summ["scenarios"].items():
        for m, kp in v["methods"].items():
            r = dict(scenario=sc, method=m, n_seeds=kp["n_seeds"], n_episodes=kp["n_episodes"])
            for k in KPIS:
                r[k], (r[k + "_lo"], r[k + "_hi"]) = kp[k]["mean"], kp[k]["ci"]
            rows.append(r)
    write_csv(os.path.join(a.out, "methods.csv"), rows)
    write_csv(os.path.join(a.out, "comparisons.csv"),
              [{k: (c[k] if k != "ci" else None) for k in ("scenario", "a", "b", "metric", "unit", "n", "diff", "rel_pct", "p", "p_holm", "significant", "wins", "losses")}
               | dict(ci_lo=c["ci"][0], ci_hi=c["ci"][1]) for c in summ["comparisons"]])
    write_csv(os.path.join(a.out, "curves_summary.csv"), curves(a.raw, seeds))
    write_csv(os.path.join(a.out, "sweep_summary.csv"), sweep(a.raw, seeds))
    write_csv(os.path.join(a.out, "horizon_summary.csv"), horizon(a.raw, seeds))
    print(f"wrote {a.out}/summary.json ({len(summ['comparisons'])} comparisons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
