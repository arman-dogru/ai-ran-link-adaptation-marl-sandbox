"""Every number quoted in README alt texts / captions / findings that is not a row of results/methods.csv or
results/comparisons.csv, computed from committed CSVs. Writes results/claim_numbers.json (review r1, m5)."""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd


def r(x, k=3):
    return round(float(x), k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/claim_numbers.json"))
    args = ap.parse_args(argv)
    out = {}
    summ = json.load(open(os.path.join(ROOT, "results/summary.json")))
    comps = summ["comparisons"]
    out["comparisons"] = dict(total=len(comps), significant=sum(bool(c["significant"]) for c in comps),
                              not_significant=sorted(f"{c['scenario']}: {c['a']} - {c['b']} ({c['metric']})"
                                                     for c in comps if not c["significant"]))
    sw = pd.read_csv(os.path.join(ROOT, "results/sweep_summary.csv"))
    out["sweep"] = {}
    for pol, g in sw.groupby("policy"):
        sig = (g.lo > 0) | (g.hi < 0)
        out["sweep"][pol] = dict(cells=len(g), positive=int((g.diff_vs_olla > 0).sum()), negative=int((g.diff_vs_olla < 0).sum()),
                                 sig_positive=int(((g.lo > 0)).sum()), sig_negative=int((g.hi < 0).sum()),
                                 not_sig=int((~sig).sum()), max=r(g.diff_vs_olla.max()), min=r(g.diff_vs_olla.min()),
                                 sig_negative_cells=[f"{a} km/h, D={b}" for a, b in g[g.hi < 0][["speed_kmh", "cqi_delay"]].values])

    pa = pd.read_csv(os.path.join(ROOT, "results/policy_actions.csv"))
    out["actions"] = {}
    for (sc, pol), g in pa.groupby(["scenario", "policy"]):
        by = g.groupby("cqi")
        n = by["count"].sum()
        delta = (g.delta * g["count"]).groupby(g.cqi).sum() / n
        p6 = g[g.power_db <= -6].groupby("cqi")["count"].sum().reindex(n.index, fill_value=0) / n
        p0 = g[g.power_db == 0].groupby("cqi")["count"].sum().reindex(n.index, fill_value=0) / n
        out["actions"][f"{sc}/{pol}"] = dict(mean_delta_by_cqi_index={int(k): r(v, 2) for k, v in delta.items()},
                                             share_minus6dB_by_cqi_index={int(k): r(v, 2) for k, v in p6.items()},
                                             share_0dB_overall=r(g[g.power_db == 0]["count"].sum() / g["count"].sum(), 3))

    tr = pd.read_csv(os.path.join(ROOT, "results/trace_default.csv"))
    c0 = tr[tr.cell == 0]
    out["timeline_cell0"] = {pol: dict(ttis=int(len(g)), ttis_backoff=[int(t) for t in g[g.power_db < 0].t],
                                       share_0dB=r((g.power_db == 0).mean()))
                             for pol, g in c0.groupby("policy")}

    m = pd.read_csv(os.path.join(ROOT, "results/methods.csv"))
    out["proxies"] = {}
    for sc, g in m.groupby("scenario"):
        ref = g[g.method == "olla-tuned"].iloc[0]
        for _, row in g[g.power_db < -1.0].iterrows():
            out["proxies"][f"{sc}/{row.method}"] = dict(power_db=r(row.power_db, 2), inr_drop_vs_olla_db=r(ref.inr_db - row.inr_db, 2),
                                                       bler_minus_olla=r(row.bler - ref.bler, 3))

    ab = pd.read_csv(os.path.join(ROOT, "results/power_ablation.csv"))
    out["power_ablation"] = {}
    for sc, g in ab.groupby("scenario"):
        mm = g.groupby("method").mean(numeric_only=True)
        iql = mm.loc["iql", "se_learned"]
        for meth in mm.index:
            a, b = mm.loc[meth, "se_learned"], mm.loc[meth, "se_full_power"]
            gs = g[g.method == meth]
            gain = gs.se_full_power - gs.se_learned
            d = dict(se_learned=r(a), se_full_power=r(b), change=r(b - a, 4), seeds_improved=int((gain > 0).sum()),
                     per_seed_change_min=r(gain.min(), 4), per_seed_change_max=r(gain.max(), 4))
            if meth != "iql":
                d["gap_to_iql"] = r(iql - a)
                d["fraction_of_gap_explained_by_power"] = r((b - a) / (iql - a), 2)
            out["power_ablation"][f"{sc}/{meth}"] = d

    bo = pd.read_csv(os.path.join(ROOT, "results/backoff_strict_check.csv"))
    out["strict_backoff"] = {f"{x.scenario}/{x.split}": dict(fired=int(x.strict_fired_ttis), se_strict=r(x.se_strict, 4),
                                                             se_olla=r(x.se_olla_tuned, 4), se_relaxed=r(x.se_relaxed, 4))
                             for x in bo.itertuples()}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
