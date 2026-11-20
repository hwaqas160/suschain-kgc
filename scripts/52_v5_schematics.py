"""Schematic figures for the v5 paper (no data; every element names a step of the paper).

  Figure_Framework.pdf   decision framework: ranker trained once, per-query shift,
                         calibration of the shift weight from a buyer's risk target
  Figure_Task.pdf        one substitution query on a toy network, and what the shift
                         does to its ranked list
  Figure_Design.pdf      study design: protocol timeline and research questions -> evidence
  Graphical_Abstract.pdf/.png
Output folder: Revision_v4/v5/.  Run: python scripts/52_v5_schematics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402

from v5_figstyle import (ACCENT, DOUBLE_COL, FILL, FILL_ACC, FILL_OK, FILL_RISK, INK, INK2,  # noqa: E402
                         LINE, MUTE, OK, OUT, RISK, RISK_RAMP, arrow, box, canvas, save, sphere)

BH = r"$\hat{\beta}$"
T, L = 7.8, 6.7  # title and line font sizes inside boxes


# ---------------------------------------------------------------- framework
def framework():
    fig, ax = canvas(DOUBLE_COL, 3.6, 120, 58)
    lane = dict(fontsize=6.6, color=MUTE, fontweight="bold", rotation=90, ha="center", va="center",
                linespacing=1.2)
    for y0, y1, name in ((41, 57, "OFFLINE\n(once)"), (21, 39, "PER\nQUERY"), (1, 19, "WHEN THE\nTARGET CHANGES")):
        ax.text(2.6, (y0 + y1) / 2, name, **lane)
        ax.plot([5.6, 5.6], [y0 + 0.5, y1 - 0.5], color=LINE, lw=0.8)

    kw = dict(tsize=T, lsize=L)
    yo, ho = 43, 12.5
    box(ax, 8, yo, 23, ho, "Visible network", ["supply links known", "before the disruption"], **kw)
    box(ax, 36, yo, 25, ho, "Train a ranker", ["any family: co-supply,", "embedding, GNN, NBFNet"], **kw)
    box(ax, 66, yo, 21, ho, "Scores s(c, v)", ["every customer", "and firm"], **kw)
    arrow(ax, 31.4, yo + ho / 2, 35.6, yo + ho / 2)
    arrow(ax, 61.4, yo + ho / 2, 65.6, yo + ho / 2)
    box(ax, 92, yo, 27, ho, "Alternative tested", ["learned penalty: retrain", "for every risk target"],
        fill="white", edge=MUTE, ls=(0, (3, 2)), tcolor=INK2, lcolor=MUTE, **kw)

    yq, hq = 23, 12.5
    box(ax, 8, yq, 19, hq, "Query (c, s)", ["c loses", "supplier s"], fill=FILL_RISK, edge=RISK, **kw)
    box(ax, 31, yq, 20, hq, "Candidates C(s)", ["products", "similar to s"], **kw)
    box(ax, 55, yq, 18, hq, "Standardise", ["z-scores", "over C(s)"], **kw)
    box(ax, 77, yq, 20, hq, "Shift", ["rank by", r"$z(c,v)-\hat{\beta}\,r(v)$"], fill=FILL_ACC, edge=ACCENT,
        lw=1.3, **kw)
    box(ax, 101, yq, 18, hq, "Shortlist", ["top 10", "substitutes"], fill=FILL_OK, edge=OK, **kw)
    for x0, x1 in ((27.4, 30.6), (51.4, 54.6), (73.4, 76.6), (97.4, 100.6)):
        arrow(ax, x0, yq + hq / 2, x1, yq + hq / 2)
    arrow(ax, 76.5, yo - 0.4, 66, yq + hq + 0.4)

    yc, hc = 3, 13.5
    box(ax, 8, yc, 24, hc, "Buyer's target", ["cut risk by t:", r"$\alpha=(1-t)\,R_0$"], fill="white",
        edge=ACCENT, **kw)
    box(ax, 37, yc, 25, hc, "Calibration", ["n customers: mean", r"risk@10 $\bar{L}_n(\beta)$ per $\beta$"],
        fill="white", edge=ACCENT, **kw)
    box(ax, 67, yc, 30, hc, "Guaranteed rule", [r"smallest $\beta$ with",
                                                 r"$\frac{n}{n+1}\bar{L}_n(\beta)+\frac{1}{n+1}\leq\alpha$"],
        fill="white", edge=ACCENT, **kw)
    arrow(ax, 32.4, yc + hc / 2, 36.6, yc + hc / 2, color=ACCENT)
    arrow(ax, 62.4, yc + hc / 2, 66.6, yc + hc / 2, color=ACCENT)
    arrow(ax, 87, yc + hc + 0.4, 87, yq - 0.4, color=ACCENT, lw=1.3)
    ax.text(88.5, (yc + hc + yq) / 2, BH, color=ACCENT, fontsize=9, va="center")
    ax.text(109, yc + hc / 2, "expected risk of\na new customer\n" + r"$\leq\alpha$" + " (Prop. 3)",
            ha="center", va="center", fontsize=6.6, color=ACCENT, linespacing=1.35)
    ax.text(44, 19.6, "risk r(v): emission or deforestation exposure per supplier", fontsize=6.4, color=RISK,
            ha="center", va="center", style="italic")
    save(fig, "Figure_Framework")


# ---------------------------------------------------------------- task
def node(ax, x, y, label, fc=FILL, ec=INK2, lw=0.8, r=2.6, ring=None, fs=7.2, tc=INK):
    if ring:
        ax.add_patch(Circle((x, y), r + 1.0, fc="none", ec=ring, lw=1.6, zorder=3.5))
    sphere(ax, x, y, r, fc, ec, lw=lw, zorder=3)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def task():
    fig, ax = canvas(DOUBLE_COL, 3.85, 120, 62)
    ax.set_aspect("equal")
    ax.text(1, 61, "(a) One substitution query", fontsize=8.2, fontweight="bold", color=INK, va="top")
    ax.text(83, 61, "(b) Its ranked candidates", fontsize=8.2, fontweight="bold", color=INK, va="top")

    C = (9, 34)
    node(ax, *C, "c", fc=FILL_ACC, ec=ACCENT, lw=1.1)
    ax.text(C[0], C[1] - 4.6, "customer", ha="center", va="top", fontsize=6.4, color=INK2)
    S, V1, H2 = (30, 51), (30, 41), (30, 19)
    node(ax, *S, "s", fc=FILL_RISK, ec=RISK, lw=1.2)
    ax.text(S[0], S[1] + 4.0, "failed supplier", ha="center", fontsize=6.4, color=RISK)
    node(ax, *V1, "v₁")
    ax.text(V1[0] + 3.6, V1[1] - 0.6, "visible supplier", fontsize=6.3, color=INK2)
    node(ax, *H2, "h₂")

    def link(p, q, **kw):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", shrinkA=8, shrinkB=8,
                                                        mutation_scale=8, **kw), zorder=2)
    link(S, C, color=RISK, lw=1.0)
    ax.text((S[0] + C[0]) / 2 + 0.6, (S[1] + C[1]) / 2 + 0.3, "×", fontsize=12, color=RISK, ha="center",
            va="center", fontweight="bold")
    link(V1, C, color=INK2, lw=1.0)
    link(H2, C, color=MUTE, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.text(H2[0], H2[1] - 4.2, "hidden supplier,\nproducts not like s:\nnot an answer", ha="center", va="top",
            fontsize=6.3, color=MUTE, linespacing=1.2)

    P = (61, 34)
    ax.add_patch(Ellipse(P, 30, 42, fc="none", ec=ACCENT, lw=1.0, ls=(0, (4, 2))))
    ax.text(P[0], 56.6, r"candidates $C(s)$: sim$(v,s)\geq\tau$", ha="center", fontsize=6.6, color=ACCENT)
    cands = {"a": ((55, 46), 2), "b": ((67, 47), 1), "h₁": ((53, 30), 1), "d": ((67, 35), 3),
             "e": ((56, 20), 0), "f": ((68, 22), 2)}
    for lab, ((x, y), rk) in cands.items():
        node(ax, x, y, lab, fc=RISK_RAMP[rk], ring=OK if lab == "h₁" else None, tc="white" if rk == 3 else INK)
    link(cands["h₁"][0], C, color=OK, lw=1.2, ls=(0, (1.5, 1.5)))
    ax.text(39.0, 26.4, "hidden link:\ncorrect answer", ha="center", va="top", fontsize=6.3, color=OK,
            linespacing=1.2)
    ax.annotate("", xy=(49, 49), xytext=(33.4, 50.5),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, ls=(0, (2, 2))))

    # legend, panel a
    ly = 4.0
    ax.text(1, ly, "risk r(v)", fontsize=6.2, color=INK2, va="center")
    for i, c in enumerate(RISK_RAMP):
        ax.add_patch(Rectangle((10.5 + 3.0 * i, ly - 1.1), 2.6, 2.2, fc=c, ec=INK2, lw=0.4))
    ax.text(10.5, ly - 2.0, "low", fontsize=6.2, color=MUTE, va="top")
    ax.text(22.3, ly - 2.0, "high", fontsize=6.2, color=MUTE, va="top", ha="right")
    ax.plot([27, 31], [ly, ly], color=INK2, lw=1.0)
    ax.text(32, ly, "visible link", fontsize=6.2, color=INK2, va="center")
    ax.plot([45, 49], [ly, ly], color=MUTE, lw=1.0, ls=(0, (1.5, 1.5)))
    ax.text(50, ly, "hidden link", fontsize=6.2, color=INK2, va="center")
    ax.add_patch(Circle((65, ly), 1.5, fc="none", ec=OK, lw=1.4))
    ax.text(67.5, ly, "correct answer", fontsize=6.2, color=INK2, va="center")

    # panel b
    plain = ["d", "h₁", "a", "f", "b", "e"]
    shifted = ["h₁", "b", "e", "a", "f", "d"]
    risk = {k: v[1] for k, v in cands.items()}
    cols = ((92, "plain", r"by $z(c,v)$"), (110, "after the shift", r"by $z(c,v)-\beta\,r(v)$"))
    k, top, step = 3, 47, 5.6
    for (x, title, sub), lst in zip(cols, (plain, shifted)):
        ax.text(x, 55.4, title, ha="center", fontsize=6.9, fontweight="bold", color=ACCENT if x > 100 else INK)
        ax.text(x, 52.4, sub, ha="center", fontsize=6.3, color=INK2)
        for i, lab in enumerate(lst):
            y = top - i * step
            node(ax, x, y, lab, fc=RISK_RAMP[risk[lab]], r=2.0, fs=6.4, ring=OK if lab == "h₁" else None,
                 tc="white" if risk[lab] == 3 else INK)
        ax.add_patch(Rectangle((x - 4.2, top - (k - 1) * step - 3.3), 8.4, (k - 1) * step + 6.6, fc="none",
                               ec=INK2, lw=0.8, ls=(0, (3, 2))))
    for i in range(6):
        ax.text(84.5, top - i * step, f"{i + 1}", ha="center", va="center", fontsize=6.2, color=MUTE)
    ax.text(101, top - (k - 1) * step - 5.4, f"top {k}", ha="center", va="center", fontsize=6.3, color=INK2)
    arrow(ax, 96.6, top - step, 105.2, top - step, color=ACCENT, lw=1.1)
    ax.text(100, 6.0, "The correct answer stays in the top 3;\nthe top 3 now holds lower-risk firms.\n"
            "No retraining; β comes from the buyer's\ntarget (Figure 1).", ha="center", va="center",
            fontsize=6.3, color=INK2, linespacing=1.3)
    save(fig, "Figure_Task")


# ---------------------------------------------------------------- design
def design():
    """Protocol only; the research questions and their evidence are a table in the manuscript."""
    fig, ax = canvas(DOUBLE_COL, 1.05, 120, 17)
    steps = [("Round 1", ["exploratory", "seeds 0 to 9"], FILL, INK2),
             ("Registration", ["hypotheses, rules", "and analysis"], FILL_ACC, ACCENT),
             ("Tuning", ["on seed 100;", "settings frozen"], FILL, INK2),
             ("Round 2", ["confirmatory", "seeds 10 to 19"], FILL_ACC, ACCENT),
             ("Tests", ["Wilcoxon, Holm;", "t-intervals"], FILL, INK2)]
    w, gap = 21.6, 2.0
    for i, (t_, ls_, fc, ec) in enumerate(steps):
        x = 1 + i * (w + gap)
        box(ax, x, 1.5, w, 13.0, t_, ls_, fill=fc, edge=ec, tsize=7.4, lsize=7.0, line_gap=3.4)
        if i:
            arrow(ax, x - gap + 0.15, 8.0, x - 0.15, 8.0)
    save(fig, "Figure_Design")


# ---------------------------------------------------------------- graphical abstract
def graphical_abstract():
    f = OUT / "graphical_abstract_findings.json"   # written once results exist
    findings = json.loads(f.read_text()) if f.exists() else [
        "Tested on AnonNet (7,245 firms) and on observed replacements in Trase Brazil soy (deforestation exposure)."]
    top = 35 + 5.6 * len(findings)
    fig, ax = canvas(6.6, 0.05 * top, 132, top)
    ax.text(66, top - 0.5, "Retrain or re-rank? Sustainability-aware supplier substitution", ha="center",
            va="top", fontsize=9.4, fontweight="bold", color=INK)
    y, h = top - 30.2, 16
    kw = dict(tsize=8.0, lsize=6.9, line_gap=3.7)
    box(ax, 1, y, 26, h, "A supplier fails", ["customer c", "loses supplier s"], fill=FILL_RISK, edge=RISK, **kw)
    box(ax, 32, y, 28, h, "Rank substitutes", ["any ranker, trained", "once on the network"], **kw)
    box(ax, 65, y, 30, h, "Shift by risk", [r"$z(c,v)-\hat{\beta}\,r(v)$", "no retraining"],
        fill=FILL_ACC, edge=ACCENT, lw=1.3, **kw)
    box(ax, 100, y, 31, h, "Low-risk shortlist", ["expected risk ≤ the", "buyer's target α"], fill=FILL_OK,
        edge=OK, **kw)
    for x0, x1 in ((27.4, 31.6), (60.4, 64.6), (95.4, 99.6)):
        arrow(ax, x0, y + h / 2, x1, y + h / 2, lw=1.2)
    box(ax, 63, y + h + 2.6, 34, 5.6, None, [BH + " calibrated on customers"], fill="white", edge=ACCENT,
        lsize=6.6, lcolor=ACCENT, title_gap=1.6)
    arrow(ax, 80, y + h + 2.4, 80, y + h + 0.3, color=ACCENT, lw=1.1)
    for i, s in enumerate(findings):
        ax.text(66, y - 3.5 - 5.6 * i, s, ha="center", va="top", fontsize=6.9, color=INK2)
    save(fig, "Graphical_Abstract", png=True, dpi=450)


if __name__ == "__main__":
    framework()
    task()
    design()
    graphical_abstract()
