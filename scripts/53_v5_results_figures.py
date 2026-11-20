"""Result figures and LaTeX tables for the v5 paper, from the result files only.

Inputs (results/metrics/): v5_confirmatory{tag}.json, v5_confirmatory_curves{tag}.npz,
v5_analysis{tag}.json, v5_trase.json, v5_trase_curves.npz, v5_trase_analysis.json;
configs/model_v5/*.json.  Missing inputs skip the corresponding outputs.

Figures (Revision_v4/v5/): Figure_Accuracy, Figure_Tradeoff, Figure_Budget, Figure_Trase.
Tables (Revision_v4/v5/tables/): accuracy_v5.tex, tests_v5.tex, budget_v5.tex,
tuning.tex, trase_v5.tex.  Every number printed is read from a file; nothing is typed in.
Run: python scripts/53_v5_results_figures.py [--tag ""]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from v5_figstyle import (ACCENT, DOUBLE_COL, INK, INK2, LINE, MUTE, OK, OUT, RISK, save)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
CONF = ROOT / "configs" / "model_v5"
TAB = OUT / "tables"
YELLOW = "#eda100"

NAMES = {
    "co_supply": "Co-supply", "popularity": "Popularity", "random": "Random",
    "gnn_sage_bce": "GraphSAGE, binary", "gnn_sage_softmax": "GraphSAGE, softmax",
    "gnn_gat_bce": "GATv2, binary", "gnn_gat_softmax": "GATv2, softmax",
    "kge_RotatE_softmax": "RotatE, softmax", "kge_ComplEx_softmax": "ComplEx, softmax",
    "kge_DistMult_softmax": "DistMult, softmax", "kge_TransE_softmax": "TransE, softmax",
    "kge_RotatE_bce": "RotatE, binary", "kge_ComplEx_bce": "ComplEx, binary", "nbfnet": "NBFNet",
}
GROUPS = [("Rules", ["popularity", "co_supply"]),
          ("Graph neural networks", ["gnn_sage_bce", "gnn_sage_softmax", "gnn_gat_bce", "gnn_gat_softmax"]),
          ("Path-based", ["nbfnet"]),
          ("Knowledge-graph embeddings", ["kge_TransE_softmax", "kge_DistMult_softmax", "kge_ComplEx_bce",
                                          "kge_ComplEx_softmax", "kge_RotatE_bce", "kge_RotatE_softmax"])]


def name(m):
    if m.endswith("_lam1") or m.endswith("_lam5"):
        base, lam = m.rsplit("_lam", 1)
        return f"{NAMES.get(base, base)} + penalty $\\lambda={lam}$"
    return NAMES.get(m, m)


def ci(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ accuracy
def fig_accuracy(d):
    per, best = d["per_seed"], {d["best_gnn"], d["best_kge"]}
    rows = [(g, m) for g, ms in GROUPS for m in ms if m in per]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 0.24 * len(rows) + 1.0), sharey=True)
    ylab, y = [], []
    pos, gap = 0.0, 0.7
    last = None
    for g, m in rows:
        if last is not None and g != last:
            pos += gap
        y.append(pos)
        ylab.append(name(m))
        last = g
        pos += 1
    y = -np.array(y)
    for ax, key, title in ((axes[0], "hit@10", "hit@10"), (axes[1], "rr", "MRR")):
        for (g, m), yy in zip(rows, y):
            mu, h = ci(per[m][key])
            soft = not m.endswith("_bce")
            col = ACCENT if m in best else (MUTE if g == "Rules" else INK2)
            ax.errorbar(mu, yy, xerr=h if h > 0 else None, fmt="D" if m in best else "o", ms=5,
                        mfc=col if soft else "white",
                        mec=col, ecolor=col, elinewidth=1.0, capsize=0, mew=1.2)
            ax.text(mu + max(h, 0) + 0.012, yy, f"{mu:.3f}", va="center", fontsize=7, color=INK2)
        co = np.mean(per["co_supply"][key])
        ax.axvline(co, color=LINE, lw=0.9, ls=(0, (3, 2)), zorder=0)
        ax.set_xlabel(title + " (higher is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ylab, fontsize=7)
    # group labels
    gy = {}
    for (g, _), yy in zip(rows, y):
        gy.setdefault(g, []).append(yy)
    for g, ys in gy.items():
        axes[0].text(0.01, max(ys) + 0.45, g, transform=axes[0].get_yaxis_transform(), ha="left", va="bottom",
                     fontsize=7, color=MUTE, style="italic")
    axes[0].plot([], [], "o", mfc=INK2, mec=INK2, label="softmax loss (or no loss)")
    axes[0].plot([], [], "o", mfc="white", mec=INK2, label="binary cross-entropy")
    axes[0].plot([], [], "D", mfc=ACCENT, mec=ACCENT, label="selected by the pre-registered rule")
    axes[0].plot([], [], color=LINE, ls=(0, (3, 2)), label="co-supply")
    fig.legend(loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "Figure_Accuracy")


def tab_accuracy(d):
    per = d["per_seed"]
    order = [m for _, ms in GROUPS for m in ms if m in per]
    lam = [m for m in per if "_lam" in m]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Ranker & hit@10 & MRR & recall@10 & risk@10 & train, s & re-rank, ms \\", r"\midrule"]
    last = None
    for g, ms in GROUPS:
        for m in ms:
            if m not in per:
                continue
            if last is not None and g != last:
                lines.append(r"\addlinespace")
            last = g
            h, hh = ci(per[m]["hit@10"])
            r = np.mean(per[m]["rr"])
            rc = np.mean(per[m]["recall@10"])
            rk = np.mean(per[m]["risk@10"])
            tr = f"{np.mean(per[m]['train_seconds']):.0f}" if "train_seconds" in per[m] else "n/a"
            rr = f"{np.mean(per[m]['rerank_ms_per_query']):.2f}" if "rerank_ms_per_query" in per[m] else "n/a"
            nm = name(m) + ("$^\\dagger$" if m in (d["best_gnn"], d["best_kge"]) else "")
            lines.append(f"{nm} & {h:.3f} $\\pm$ {hh:.3f} & {r:.3f} & {rc:.3f} & {rk:.3f} & {tr} & {rr} \\\\")
    if lam:
        lines.append(r"\addlinespace")
        base = sorted(lam)[0].rsplit("_lam", 1)[0]
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{NAMES.get(base, base)}, retrained with the learned penalty}}}} \\\\")
        for m in sorted(lam):
            h, hh = ci(per[m]["hit@10"])
            lab = f"\\quad $\\lambda={m.rsplit('_lam', 1)[1]}$"
            lines.append(f"{lab} & {h:.3f} $\\pm$ {hh:.3f} & {np.mean(per[m]['rr']):.3f} & "
                         f"{np.mean(per[m]['recall@10']):.3f} & {np.mean(per[m]['risk@10']):.3f} & "
                         f"{np.mean(per[m]['train_seconds']):.0f} & n/a \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "accuracy_v5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_effects(a, d):
    """Per-seed hit@10 differences (points) behind H1-H4: mean, SD and Cohen's d_z."""
    per = d["per_seed"]
    h = lambda m: np.array(per[m]["hit@10"]) * 100
    diffs = [h(a["best_kge"]) - h("co_supply"), h("nbfnet") - h("co_supply"),
             h(a["best_gnn"]) - h(a["best_gnn"].replace("softmax", "bce")),
             np.array(a["H4_shift_minus_penalty_hit"]["per_seed"]) * 100]
    return [(x.mean(), x.std(ddof=1), int((x > 0).sum())) for x in diffs]


def tab_tests(a, d):
    eff = paired_effects(a, d)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Hypothesis & Diff. & SD & Seeds ahead & $p$ & $p$ (Holm) \\", r"\midrule"]
    short = [f"H1 {NAMES.get(a['best_kge'], a['best_kge']).split(',')[0]} vs. co-supply", "H2 NBFNet vs. co-supply",
             f"H3 {NAMES.get(a['best_gnn'], a['best_gnn']).split(',')[0]}: softmax vs. binary",
             r"H4 Shift vs. penalty ($\lambda=1$), equal risk"]
    for lab, (k, v), (mu, sd, up) in zip(short, a["tests"].items(), eff):
        lines.append(f"{lab} & ${mu:+.1f}$ & {sd:.1f} & {up} of 10 & {fmt_p(v['p'])} & {fmt_p(v['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tests_v5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ trade-off
def mean_curve(cv, seeds, m):
    H = np.mean([cv[f"s{s}/{m}/hit"].mean(1) for s in seeds], 0)
    R = np.mean([cv[f"s{s}/{m}/risk"].mean(1) for s in seeds], 0)
    return H, R


def fig_tradeoff(d, cv):
    seeds = d["seeds_done"]
    bg, bk = d["best_gnn"], d["best_kge"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))
    # (a) shift vs learned penalty on the best GNN
    ax = axes[0]
    H, R = mean_curve(cv, seeds, bg)
    ax.plot(R, H, "-", color=ACCENT, lw=1.6, marker=".", ms=4, label="shift on the plain model")
    ax.plot(R[0], H[0], "o", color=ACCENT, ms=5)
    for lam, mk in (("1", "s"), ("5", "D")):
        k = f"{bg}_lam{lam}"
        if f"s{seeds[0]}/{k}/hit" in cv.files:
            H2, R2 = mean_curve(cv, seeds, k)
            ax.plot(R2, H2, ":", color=RISK, lw=1.1)
            ax.plot(R2[0], H2[0], mk, color=RISK, ms=5.5, mfc="white", mew=1.4,
                    label=f"learned penalty $\\lambda={lam}$ (then shifted, dotted)")
    ax.set_xlabel("risk@10 (lower is better)")
    ax.set_ylabel("hit@10")
    ax.set_title(f"(a) {name(bg)}: retrain or re-rank (high-risk end)", fontsize=8, loc="left", color=INK)
    ax.set_xlim(R[0] - 0.035, R[0] + 0.003)
    ax.set_ylim(H[0] - 0.09, H[0] + 0.02)
    ax.legend(fontsize=7, loc="lower right")
    # (b) shift curves of the main rankers
    ax = axes[1]
    series = [(bk, ACCENT, "-", "o"), ("nbfnet", OK, (0, (4, 2)), "s"), (bg, YELLOW, "-.", "^"),
              ("co_supply", MUTE, ":", "v")]
    for m, col, ls, mk in series:
        if f"s{seeds[0]}/{m}/hit" not in cv.files:
            continue
        H, R = mean_curve(cv, seeds, m)
        ax.plot(R, H, ls=ls, color=col, lw=1.5, marker=mk, ms=3, markevery=3, label=name(m))
        ax.plot(R[0], H[0], mk, color=col, ms=5)
    ax.set_xlabel("risk@10 (lower is better)")
    ax.set_ylabel("hit@10")
    ax.set_title("(b) Shift applied to different rankers", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=7, loc="lower right")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "Figure_Tradeoff")


# ------------------------------------------------------------------ budget
def fig_budget(a, d, target="target20"):
    bud = a["risk_budget"]
    rankers = [m for m in (d["best_kge"], "co_supply", d["best_gnn"], "nbfnet") if m in bud]
    sizes = ["25", "50", "100", "half"]
    n_cust = int(np.mean(d["per_seed"]["_n_queries"]["customers"]))
    xs = [25, 50, 100, n_cust // 2]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.75))
    metrics = [("mean_test_risk_minus_alpha", "(a) evaluation risk minus target α"),
               ("share_within", "(b) share of splits with risk ≤ α"),
               ("hit_lost", "(c) hit@10 lost")]
    styles = {"crc": ("-", "guaranteed rule"), "grid": ("--", "grid rule")}
    marks = dict(zip(rankers, ["o", "v", "^", "s"]))
    cols = {rankers[0]: ACCENT}
    if len(rankers) > 1:
        cols[rankers[1]] = MUTE
    if len(rankers) > 2:
        cols[rankers[2]] = YELLOW
    if len(rankers) > 3:
        cols[rankers[3]] = OK
    for ax, (k, title) in zip(axes, metrics):
        for m in rankers:
            for rule, (ls, _) in styles.items():
                v = [bud[m][target][s][rule][k] for s in sizes]
                ax.plot(xs, v, ls=ls, marker=marks[m], ms=3.8, color=cols[m], lw=1.2,
                        mfc=cols[m] if rule == "crc" else "white")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs[:-1]] + [f"~{int(round(xs[-1], -1))}"], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("calibration customers (log scale)", fontsize=7)
        ax.set_title(title, fontsize=7.6, loc="left", color=INK)
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[1].set_ylim(0, 1.02)
    axes[0].set_ylabel("risk@10 minus α")
    axes[1].set_ylabel("share of splits")
    axes[2].set_ylabel("hit@10 lost")
    for m in rankers:
        axes[2].plot([], [], color=cols[m], lw=1.4, marker=marks[m], ms=4, label=name(m))
    for rule, (ls, lab) in styles.items():
        axes[2].plot([], [], ls=ls, color=INK2, lw=1.2, label=lab)
    fig.legend(*axes[2].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, "Figure_Budget")


def fig_modes(cm, d):
    """Hit@10 lost to meet a risk target: one calibrated weight vs a weight chosen per customer or per list."""
    an = cm["anonnet"]
    m = d["best_kge"]
    n_half = int(np.mean(d["per_seed"]["_n_queries"]["customers"])) // 2
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6), sharey=True)
    for ax, tk, title in zip(axes, ("target10", "target20", "target30"), ("(a) 10% target", "(b) 20% target",
                                                                           "(c) 30% target")):
        e = an[m][tk]
        xs = [25, 100, n_half]
        ax.plot(xs, [e[f"crc_{s}"]["lost"] for s in ("25", "100", "half")], "-o", color=ACCENT, lw=1.5, ms=4,
                label="one weight, guaranteed rule")
        ax.plot(xs, [e[f"grid_{s}"]["lost"] for s in ("25", "100", "half")], "--o", color=ACCENT, lw=1.2, ms=4,
                mfc="white", label="one weight, grid rule (no guarantee)")
        ax.axhline(e["per_customer"]["lost"], color=OK, lw=1.4, ls=(0, (4, 2)), label="weight per customer")
        ax.axhline(e["per_list"]["lost"], color=RISK, lw=1.4, ls=":", label="weight per shortlist")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["25", "100", f"~{int(round(n_half, -1))}"])
        ax.minorticks_off()
        ax.set_xlabel("calibration customers (log scale)")
        ax.set_title(title, fontsize=8, loc="left", color=INK)
    axes[0].set_ylabel(f"hit@10 lost, {name(m)}")
    axes[0].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "Figure_Modes")


def tab_budget(a, d):
    bud = a["risk_budget"]
    rankers = [m for m in (d["best_kge"], "nbfnet", d["best_gnn"], "co_supply") if m in bud]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Grid rule} & \multicolumn{3}{c}{Guaranteed rule} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Ranker & Target & risk $-\,\alpha$ & within & hit lost & risk $-\,\alpha$ & within & hit lost \\",
             r"\midrule"]
    for i, m in enumerate(rankers):
        if i:
            lines.append(r"\addlinespace")
        for t in ("target10", "target20", "target30"):
            g, c = bud[m][t]["half"]["grid"], bud[m][t]["half"]["crc"]
            lab = name(m) if t == "target10" else ""
            lines.append(f"{lab} & {t[6:]}\\% & {g['mean_test_risk_minus_alpha']:+.4f} & {g['share_within']:.2f} & "
                         f"{g['hit_lost']:.3f} & {c['mean_test_risk_minus_alpha']:+.4f} & {c['share_within']:.2f} & "
                         f"{c['hit_lost']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "budget_v5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ tuning table
def hp_text(b):
    if b["family"] == "nbfnet":
        return f"dim {b['dim']}, {b['layers']} layers; tuned at 240 s, evaluated at {b['seconds']:.0f} s"
    h = b["hp"]
    if b["family"] == "gnn":
        return (f"{h['num_layers']} layers, emb. {h['embedding_dim']}, hidden {h['hidden_channels']}, "
                f"out {h['out_channels']}, lr {h['lr']:.1e}, batch {h['batch_size']}, {h['epochs']} ep., {h['num_neg']} neg.")
    return (f"dim {h['dim']}, lr {h['lr']:.1e}, batch {h['batch_size']}, {h['epochs']} ep., "
            f"{h['num_neg']} neg." + (f", $\\gamma$ {h['gamma']:g}" if "gamma" in h else ""))


def tab_tuning():
    rows = []
    for p in sorted(CONF.glob("substitution_*.json")):
        c = json.loads(p.read_text())
        rows.append((c["key"], c))
    lines = [r"\begin{tabularx}{\textwidth}{lXrrr}", r"\toprule",
             r"Ranker & Chosen setting & Trials & Minutes & Tuning hit@10 \\", r"\midrule"]
    order = [m for _, ms in GROUPS for m in ms]
    for k, c in sorted(rows, key=lambda r: order.index(r[0]) if r[0] in order else 99):
        lines.append(f"{name(k)} & {hp_text(c['best'])} & {len(c['trials'])} & {c['total_seconds'] / 60:.0f} & "
                     f"{c['best_val_hit10']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    body = "\n".join(lines)
    out = (r"\begin{table}[htbp]" "\n"
           r"\caption{Settings chosen on the tuning scenario (seed 100) by mean hit@10 of its substitution "
           r"queries. emb.: input embedding size; hidden and out: layer widths; dim: embedding size; lr: learning "
           r"rate; ep.: epochs; neg.: sampled negatives per positive; $\gamma$: margin of the distance models. "
           r"Minutes are wall-clock time on one shared 5\,GB GPU.}\label{tab:tuning}" "\n"
           r"\centering\footnotesize" "\n" + body + "\n" r"\end{table}" "\n")
    (TAB / "tuning.tex").write_text(out, encoding="utf-8")


# ------------------------------------------------------------------ Trase
def fig_trase(t, ta, tc):
    years = [int(y) for y in t["years"]]
    per = t["per_year"]
    best = t["best"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))
    ax = axes[0]
    series = [(best, ACCENT, "-"), ("nbfnet", OK, "-"), ("co_supply", MUTE, "-"), ("random", LINE, ":")]
    seen = set()
    for m, col, ls in series:
        if m in seen or m not in per:
            continue
        seen.add(m)
        full = list(range(min(years), max(years) + 1))
        v = [per[m][str(y)]["hit"] if y in years else np.nan for y in full]
        ax.plot(full, v, ls=ls, marker="o", ms=3.5, color=col, lw=1.4, label=name(m))
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=7)
    ax.set_ylabel("hit@10")
    ax.set_ylim(0, 1)
    ax.set_title("(a) Observed replacements, per evaluation year", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax = axes[1]
    for m, col, _ in series[:3]:
        if f"{years[0]}/{m}/hit" not in tc.files:
            continue
        H = np.mean([tc[f"{y}/{m}/hit"].mean(1) for y in years], 0)
        R = np.mean([tc[f"{y}/{m}/risk"].mean(1) for y in years], 0)
        ax.plot(R, H, "-", color=col, lw=1.6, label=name(m))
        ax.plot(R[0], H[0], "o", color=col, ms=4.5)
    ax.set_xlabel("deforestation exposure of the top 10 (normalised)")
    ax.set_ylabel("hit@10")
    ax.set_ylim(bottom=0)
    ax.set_title("(b) Shift on Trase: accuracy against risk", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "Figure_Trase")


def tab_trase(t, ta):
    years = t["years"]
    per = t["per_year"]
    ms = [m for m in ("random", "popularity", "co_supply", "gnn_sage_bce", "gnn_sage_softmax",
                      "kge_RotatE_softmax", "kge_ComplEx_softmax", "nbfnet") if m in per]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Ranker & hit@10 & MRR & risk@10 & years above co-supply \\", r"\midrule"]
    co = [per["co_supply"][str(y)]["hit"] for y in years]
    for m in ms:
        h = [per[m][str(y)]["hit"] for y in years]
        r = np.mean([per[m][str(y)]["rr"] for y in years])
        rk = np.mean([per[m][str(y)]["risk"] for y in years])
        above = "n/a" if m == "co_supply" else f"{sum(a > b for a, b in zip(h, co))} of {len(years)}"
        nm = name(m) + ("$^\\dagger$" if m == t["best"] else "")
        lines.append(f"{nm} & {np.mean(h):.3f} & {r:.3f} & {rk:.3f} & {above} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "trase_v5.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a_ = ap.parse_args()
    TAB.mkdir(exist_ok=True)
    tab_tuning()
    d = load(MET / f"v5_confirmatory{a_.tag}.json")
    if d:
        cv = np.load(MET / f"v5_confirmatory_curves{a_.tag}.npz")
        fig_accuracy(d)
        tab_accuracy(d)
        fig_tradeoff(d, cv)
        an = load(MET / f"v5_analysis{a_.tag}.json")
        if an:
            tab_tests(an, d)
            fig_budget(an, d)
            tab_budget(an, d)
        cm = load(MET / "v5_control_modes.json")
        if cm:
            fig_modes(cm, d)
    t = load(MET / "v5_trase.json")
    ta = load(MET / "v5_trase_analysis.json")
    if t and ta:
        tc = np.load(MET / "v5_trase_curves.npz")
        fig_trase(t, ta, tc)
        tab_trase(t, ta)
    print("done")


if __name__ == "__main__":
    main()
