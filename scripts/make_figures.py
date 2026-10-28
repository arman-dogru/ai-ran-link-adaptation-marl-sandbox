"""All README figures, from committed results only (results/*.csv, results/summary.json, results/trace_default.csv,
results/dev_tuning_default.csv, results/power_ablation.csv). Refuses a summary that does not cover the full grid."""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aran import scenarios as S

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "aran", "path.simplify": False})
LABEL = {"fixed": "Fixed MCS", "cqi-only": "CQI only", "olla-10": "OLLA 10%", "olla-tuned": "OLLA tuned",
         "olla-tuned-backoff": "OLLA tuned + back-off", "genie": "Genie (non-causal)", "sa": "Single-agent DQN",
         "iql": "IQL (own reward)", "iql_team": "IQL (team reward)", "vdn": "VDN (cooperative)"}
COLOR = {"fixed": "#bbbbbb", "cqi-only": "#999999", "olla-10": "#7f9fbf", "olla-tuned": "#1f4e79",
         "olla-tuned-backoff": "#4a7fb0", "genie": "#e0c060", "sa": "#9b59b6", "iql": "#2e8b57",
         "iql_team": "#e67e22", "vdn": "#c0392b"}
SC_TITLE = {"default": "default (3 cells + bursty outer ring)", "coupled": "coupled (no outer ring, wider UE spread)"}


def full_summary(res):
    """Exit 2 (like readme_tables.py) unless summary.json covers every seed, scenario and method."""
    s = json.load(open(os.path.join(res, "summary.json")))
    want = set(S.RULES) | set(S.LEARNERS)
    if s.get("grid", {}).get("seeds") != list(S.SEEDS) or set(s.get("scenarios", {})) != set(S.SCENARIOS) or \
            any(set(v["methods"]) != want for v in s["scenarios"].values()):
        print("refusing: summary.json does not cover the full grid (run scripts/aggregate.py on a complete grid)", file=sys.stderr)
        sys.exit(2)
    return s


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def kpi_bars(res, out):
    df = pd.read_csv(os.path.join(res, "methods.csv"))
    meths = list(S.RULES) + list(S.LEARNERS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey="row")
    for j, sc in enumerate(S.SCENARIOS):
        d = df[df.scenario == sc].set_index("method").loc[meths]
        for i, (k, lab) in enumerate((("se", "spectral efficiency (bit/s/Hz per cell)"), ("edge", "edge UE throughput (bit/s/Hz, mean of worst 25%)"))):
            ax = axes[i, j]
            x = np.arange(len(meths))
            ax.bar(x, d[k], color=[COLOR[m] for m in meths], yerr=[d[k] - d[k + "_lo"], d[k + "_hi"] - d[k]], capsize=2)
            ref = d.loc["olla-tuned", k]
            ax.axhline(ref, ls="--", lw=0.8, color=COLOR["olla-tuned"])
            ax.set_xticks(x, [LABEL[m] for m in meths], rotation=40, ha="right")
            ax.set_ylabel(lab if j == 0 else "")
            if i == 0:
                ax.set_title(SC_TITLE[sc])
    fig.suptitle("Evaluation KPIs, 100 shared episodes; learners: mean over 15 training seeds; 95% bootstrap CIs; dashed = OLLA tuned")
    fig.tight_layout()
    save(fig, out, "kpi_bars.png")


def forest(res, out):
    """One panel per (metric, scenario) with its own x-axis, so the small edge differences stay visible (review m8)."""
    c = pd.read_csv(os.path.join(res, "comparisons.csv"))
    metrics = [("se", "SE difference (bit/s/Hz per cell)"), ("edge", "edge-throughput difference (bit/s/Hz)")]
    n = {k: int((c[(c.scenario == "default")].metric == k).sum()) for k, _ in metrics}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharey="row", gridspec_kw={"height_ratios": [n["se"], n["edge"]]})
    for i, (k, xl) in enumerate(metrics):
        for j, sc in enumerate(S.SCENARIOS):
            d = c[(c.scenario == sc) & (c.metric == k)].reset_index(drop=True)
            ax = axes[i, j]
            y = np.arange(len(d))[::-1]
            col = np.where(d.significant, np.where(d["diff"] > 0, "#2e8b57", "#c0392b"), "#999999")
            for yi, (_, r), cc in zip(y, d.iterrows(), col):
                ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="none", ecolor=cc, capsize=2)
                ax.annotate(f"{r.rel_pct:+.1f}%", (r.ci_hi, yi), xytext=(4, -3), textcoords="offset points", fontsize=7, color=cc)
            ax.scatter(d["diff"], y, c=col, zorder=3, s=18)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_yticks(y, [f"{LABEL[a]} - {LABEL[b]}" for a, b in zip(d.a, d.b)])
            ax.set_xlabel(xl)
            lim = 1.25 * max(abs(d.ci_lo).max(), abs(d.ci_hi).max())
            ax.set_xlim(-lim, lim)
            if i == 0:
                ax.set_title(SC_TITLE[sc])
    fig.suptitle("Pre-registered comparisons (one Holm family of 34), separate x-axis per metric; labels = relative difference.\n"
                 "green/red = Holm p < 0.05 and 95% CI excludes 0; grey = not significant")
    fig.tight_layout()
    save(fig, out, "diff_forest.png")


def learning_curves(res, out):
    c = pd.read_csv(os.path.join(res, "curves_summary.csv"))
    m = pd.read_csv(os.path.join(res, "methods.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, sc in enumerate(S.SCENARIOS):
        ax = axes[j]
        for meth in S.LEARNERS:
            d = c[(c.scenario == sc) & (c.method == meth)]
            ax.plot(d.episode, d.se_mean, color=COLOR[meth], label=LABEL[meth], lw=1)
            ax.fill_between(d.episode, d.lo, d.hi, color=COLOR[meth], alpha=0.2, lw=0)
        ref = m[(m.scenario == sc) & (m.method == "olla-tuned")].se.iloc[0]
        ax.axhline(ref, ls="--", color=COLOR["olla-tuned"], lw=0.9, label="OLLA tuned (eval episodes)")
        ax.set_xlabel("training episode (200 TTIs each)")
        ax.set_title(SC_TITLE[sc])
    axes[0].set_ylabel("training-episode SE, 25-episode moving mean")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Learning curves: mean over 15 seeds, 95% t-band; epsilon-greedy exploration decays to 0.05 by episode 600")
    fig.tight_layout()
    save(fig, out, "learning_curves.png")


def timeline(res, out, cell=0, t1=200):
    tr = pd.read_csv(os.path.join(res, "trace_default.csv"))
    tr = tr[(tr.cell == cell) & (tr.t < t1)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 1]})
    for ax, pol in zip(axes[:2], ("olla-tuned", "iql")):
        d = tr[tr.policy == pol]
        ax.plot(d.t, d.sinr_db, color="#555555", lw=0.8, label="true SINR of scheduled UE")
        ax.step(d.t, d.thr10_db, where="mid", color=COLOR[pol], lw=1.0, label="10%-BLER threshold of chosen MCS")
        n = d[d.ack == 0]
        ax.scatter(n.t, n.thr10_db, marker="x", color="#c0392b", s=14, zorder=3, label="NACK")
        ax.set_ylabel("dB")
        ax.set_title(f"{LABEL[pol]}: cell {cell}, eval episode 10000, BLER {100 * (1 - d.ack.mean()):.0f}%", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", ncol=3)
    d = tr[tr.policy == "vdn"]
    axes[2].step(d.t, d.power_db, where="mid", color=COLOR["vdn"], lw=1, label="VDN cell-0 power back-off")
    d = tr[tr.policy == "iql"]
    axes[2].step(d.t, d.power_db + 0.15, where="mid", color=COLOR["iql"], lw=1, label="IQL (offset +0.15 for visibility)")
    axes[2].set_ylabel("power (dB)")
    axes[2].set_xlabel("TTI")
    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "timeline.png")


def sweep_heatmaps(res, out):
    sw = pd.read_csv(os.path.join(res, "sweep_summary.csv"))
    panels = [("olla-tuned", "cqi-only", "OLLA tuned - CQI only"), ("iql", None, "IQL - OLLA tuned"),
              ("vdn", None, "VDN - OLLA tuned"), ("sa", None, "Single-agent DQN - OLLA tuned")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    V, Dl = list(S.SWEEP_SPEEDS), list(S.SWEEP_DELAYS)
    for ax, (a, b, title) in zip(axes, panels):
        M = np.zeros((len(V), len(Dl)))
        sig = np.zeros_like(M, bool)
        for i, v in enumerate(V):
            for j, d in enumerate(Dl):
                ra = sw[(sw.policy == a) & (sw.speed_kmh == v) & (sw.cqi_delay == d)].iloc[0]
                if b is None:
                    M[i, j], sig[i, j] = ra.diff_vs_olla, (ra.lo > 0) or (ra.hi < 0)
                else:  # rule vs rule: olla-tuned minus cqi-only = -(cqi-only diff_vs_olla)
                    rb = sw[(sw.policy == b) & (sw.speed_kmh == v) & (sw.cqi_delay == d)].iloc[0]
                    M[i, j], sig[i, j] = -rb.diff_vs_olla, (rb.lo > 0) or (rb.hi < 0)
        lim = max(0.05, np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(len(V)):
            for j in range(len(Dl)):
                ax.text(j, i, f"{M[i, j]:+.2f}{'*' if sig[i, j] else ''}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(Dl)), Dl)
        ax.set_yticks(range(len(V)), V)
        ax.set_xlabel("CQI delay (TTI)")
        ax.set_ylabel("UE speed (km/h)")
        ax.set_title(title, fontsize=9)
        ax.add_patch(plt.Rectangle((Dl.index(4) - 0.5, V.index(30) - 0.5), 1, 1, fill=False, lw=1.5, ec="k"))
        fig.colorbar(im, ax=ax, shrink=0.8, label="SE difference (bit/s/Hz)")
    fig.suptitle("Speed x CQI-delay sweep (evaluation only, 30 episodes; learners trained at the boxed cell only). * = 95% CI excludes 0 (uncorrected, exploratory)")
    fig.tight_layout()
    save(fig, out, "sweep_heatmap.png")


def tuning_heatmap(res, out):
    t = pd.read_csv(os.path.join(res, "dev_tuning_default.csv"))
    o = t[t.kind == "olla"].copy()
    P = o.groupby(["target", "init"]).se.max().unstack()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    im = ax.imshow(P.values, origin="lower", aspect="auto", cmap="viridis")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(range(P.shape[1]), [f"{x:g}" for x in P.columns])
    ax.set_yticks(range(P.shape[0]), [f"{x:g}" for x in P.index])
    ax.set_xlabel("initial OLLA offset (dB)")
    ax.set_ylabel("BLER target")
    ax.set_title("OLLA dev-set SE (best step size per cell), default scenario", fontsize=9)
    fig.colorbar(im, ax=ax, label="SE (bit/s/Hz)")
    fig.tight_layout()
    save(fig, out, "dev_tuning_heatmap.png")


def proxies(res, out):
    df = pd.read_csv(os.path.join(res, "methods.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (k, lab) in zip(axes, (("bler", "BLER (NACK rate)"), ("inr_db", "interference-to-noise at scheduled UE (dB)"))):
        for sc, mk in zip(S.SCENARIOS, ("o", "s")):
            d = df[df.scenario == sc]
            for _, r in d.iterrows():
                ax.errorbar(r.power_db, r[k], yerr=[[r[k] - r[k + "_lo"]], [r[k + "_hi"] - r[k]]],
                            xerr=[[r.power_db - r.power_db_lo], [r.power_db_hi - r.power_db]], fmt=mk, color=COLOR[r.method], ms=5, capsize=2)
        ax.set_xlabel("mean transmit-power back-off (dB)")
        ax.set_ylabel(lab)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker="o", ls="", label=LABEL[m]) for m in list(S.RULES) + list(S.LEARNERS)]
    h += [Line2D([], [], color="k", marker="o", ls="", label="default"), Line2D([], [], color="k", marker="s", ls="", label="coupled")]
    axes[1].legend(handles=h, fontsize=7, ncol=2, loc="best")
    fig.suptitle("Interference / collision proxies vs power back-off (95% CIs)")
    fig.tight_layout()
    save(fig, out, "proxies.png")


def horizon_plot(res, out):
    h = pd.read_csv(os.path.join(res, "horizon_summary.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for pol, g in h.groupby("policy"):
        g = g.sort_values("episode_tti")
        ax.errorbar(g.episode_tti, g.se, yerr=[g.se - g.lo, g.hi - g.se], marker="o", ms=4, capsize=2, color=COLOR[pol], label=LABEL[pol])
    ax.set_xscale("log")
    ax.set_xticks([200, 1000, 3000], ["200", "1000", "3000"])
    ax.set_xlabel("evaluation episode length (TTIs); learners trained on 200")
    ax.set_ylabel("SE (bit/s/Hz per cell)")
    ax.set_title("Episode-length check, default scenario, 30 episodes (95% CIs)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, out, "horizon.png")


def action_heatmaps(res, out, sc="default"):
    a = pd.read_csv(os.path.join(res, "policy_actions.csv"))
    a = a[a.scenario == sc]
    pols = ["olla-tuned"] + list(S.LEARNERS)
    fig, axes = plt.subplots(2, len(pols), figsize=(15, 5.6), sharex=True)
    cq = np.arange(-1, 15)
    for j, pol in enumerate(pols):
        d = a[a.policy == pol]
        D = d.pivot_table(index="delta", columns="cqi", values="count", aggfunc="sum").reindex(columns=cq).fillna(0)
        D = D.reindex(range(int(min(-6, D.index.min())), int(max(2, D.index.max())) + 1)).fillna(0)
        frac = D / D.sum(axis=0).replace(0, np.nan)
        ax = axes[0, j]
        ax.imshow(frac.values, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                  extent=[cq[0] - 0.5, cq[-1] + 0.5, D.index.min() - 0.5, D.index.max() + 0.5])
        ax.set_title(LABEL[pol], fontsize=9)
        ax.set_ylabel("MCS - CQI index" if j == 0 else "")
        Pw = d.pivot_table(index="power_db", columns="cqi", values="count", aggfunc="sum").reindex(columns=cq).fillna(0)
        Pw = Pw.reindex([-6.0, -3.0, 0.0]).fillna(0)
        pf = Pw / Pw.sum(axis=0).replace(0, np.nan)
        ax = axes[1, j]
        im = ax.imshow(pf.values, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                       extent=[cq[0] - 0.5, cq[-1] + 0.5, -0.5, 2.5])
        ax.set_yticks([0, 1, 2], ["-6", "-3", "0"])
        ax.set_ylabel("power (dB)" if j == 0 else "")
        ax.set_xlabel("CQI index")
    fig.supxlabel("reported CQI index (0-14 = LTE CQI 1-15; -1 = out of range)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="P(action | reported CQI index)")
    fig.suptitle(f"What the policies do ({sc} scenario, 30 eval episodes; learners pooled over 15 seeds): top = MCS offset, bottom = power; CQI -1 = out of range")
    save(fig, out, f"actions_{sc}.png")


def power_ablation_plot(res, out):
    ab = pd.read_csv(os.path.join(res, "power_ablation.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, sc in zip(axes, S.SCENARIOS):
        g = ab[ab.scenario == sc]
        meths = ["iql", "iql_team", "vdn"]
        for i, m in enumerate(meths):
            d = g[g.method == m]
            for k, (col, dx) in enumerate((("se_learned", -0.17), ("se_full_power", 0.17))):
                ax.scatter(np.full(len(d), i + dx), d[col], s=8, color=COLOR[m], alpha=0.35 if k == 0 else 0.9,
                           marker="o" if k == 0 else "s")
                ax.hlines(d[col].mean(), i + dx - 0.12, i + dx + 0.12, color="k", lw=1.5)
            for a_, b_ in zip(d.se_learned, d.se_full_power):
                ax.plot([i - 0.17, i + 0.17], [a_, b_], color=COLOR[m], lw=0.4, alpha=0.5)
        ax.axhline(g[g.method == "iql"].se_learned.mean(), color=COLOR["iql"], ls=":", lw=1, label="IQL as learned (mean)")
        ax.set_xticks(range(3), [f"{LABEL[m]}\nlearned | 0 dB" for m in meths])
        ax.set_ylabel("SE (bit/s/Hz per cell)")
        ax.set_title(f"{sc}: same MCS choices, power forced to 0 dB", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Power ablation: 15 seeds x first 30 eval episodes (circles = as learned, squares = full power, bars = means)", fontsize=9)
    fig.tight_layout()
    save(fig, out, "power_ablation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    full_summary(a.results)
    os.makedirs(a.out, exist_ok=True)
    for f in (kpi_bars, forest, learning_curves, timeline, sweep_heatmaps, tuning_heatmap, proxies, horizon_plot,
              power_ablation_plot, action_heatmaps, lambda r, o: action_heatmaps(r, o, "coupled")):
        f(a.results, a.out)
        print("ok", getattr(f, "__name__", "f"))


if __name__ == "__main__":
    main()
