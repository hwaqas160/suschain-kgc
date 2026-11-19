"""Pipeline diagram: from a failed supplier to a risk-budgeted shortlist.

A schematic, not a data plot; every box names a step of the method in the paper.
Output: Revision_v4/v5/Figure_Pipeline.pdf (vector, fonts embedded as TrueType).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUT = Path(__file__).resolve().parents[2] / "Revision_v4" / "v5" / "Figure_Pipeline.pdf"
INK, MUTE, ACCENT, FILL, FILL2 = "#222222", "#666666", "#0072B2", "#F2F2F2", "#E3EEF7"

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 110)
ax.set_ylim(0, 44)
ax.axis("off")


def box(x, y, w, h, title, lines, fill=FILL, edge=INK, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.4 - 3.3 * i, s, ha="center", va="top", fontsize=8.2, color=MUTE)


def arrow(x0, x1, y, label=None):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    if label:
        ax.text((x0 + x1) / 2, y + 1.6, label, ha="center", va="bottom", fontsize=7.8, color=MUTE)


W, H, Y = 19.5, 24, 19
box(1, Y, W, H, "1  Query", ["customer c loses", "supplier s", "candidates C(s):", "product-similar", "to s"])
box(24.5, Y, W, H, "2  Ranker", ["any model, trained", "once, on the", "visible links:", "co-supply, KGE,", "GNN, NBFNet"])
box(48, Y, W, H, "3  Standardise", ["scores of the", "candidates of one", "query: z-scores,", "so one weight", "suits every ranker"])
box(71.5, Y, W, H, "4  Shift", ["rank by", "z(c,v) − β·risk(v)", "no retraining;", "β = 0 is the", "plain ranking"], fill=FILL2, edge=ACCENT, lw=1.6)
box(95, Y, 14, H, "5  Top 10", ["shortlist", "with lower", "mean risk"])
for x0, x1 in ((20.8, 24.2), (44.3, 47.7), (67.8, 71.2), (91.3, 94.7)):
    arrow(x0, x1, Y + H / 2)

# calibration strip under step 4
ax.add_patch(FancyBboxPatch((29.5, 1.0), 68, 11.5, boxstyle="round,pad=0.25,rounding_size=1.0",
                            fc="white", ec=ACCENT, lw=1.2, ls=(0, (4, 2))))
ax.text(63.5, 11.0, "Risk budget: choosing β from the buyer's target", ha="center", va="top", fontsize=9.2,
        fontweight="bold", color=ACCENT)
ax.text(63.5, 7.2, "target risk level α → calibration customers → smallest β with\n"
        "(n/(n+1))·mean risk + 1/(n+1) ≤ α → expected risk on a new customer ≤ α",
        ha="center", va="top", fontsize=8.0, color=MUTE, linespacing=1.5)
ax.annotate("", xy=(81.2, Y - 0.3), xytext=(81.2, 12.8), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
ax.text(82.4, 16.0, "β", fontsize=10, color=ACCENT, va="center", fontweight="bold")
ax.text(1, 9, "Trained once.\nNo change to the\nranker's weights.", fontsize=8.2, color=MUTE, va="center",
        linespacing=1.5)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.1)
print("saved", OUT)
