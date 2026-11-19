"""Shared style for the v5 paper figures (print, vector PDF, fonts embedded).

Colour roles (validated categorical slots 1-3 of the reference palette; at most
three hues per figure, identity never by colour alone):
  ACCENT  blue    the proposed procedure (shift / risk budget)
  RISK    orange  risk, the failed supplier, the learned penalty
  OK      aqua    correct answers, guarantees met
Risk magnitude uses one hue, light to dark (RISK_RAMP).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False,
    "savefig.dpi": 300,
})

OUT = Path(__file__).resolve().parents[2] / "Revision_v4" / "v5"
INK, INK2, MUTE, LINE = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2"
ACCENT, RISK, OK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
FILL, FILL_ACC, FILL_RISK, FILL_OK = "#f4f4f2", "#e6eefa", "#fcebe3", "#e3f5ee"
RISK_RAMP = ["#fde7dc", "#f8b896", "#eb6834", "#a8431a"]  # low -> high risk
SINGLE_COL, DOUBLE_COL = 3.54, 7.48  # inches (90 mm, 190 mm)


def box(ax, x, y, w, h, title=None, lines=(), fill=FILL, edge=INK2, lw=0.8, ls="-",
        tsize=8.5, lsize=7.4, tcolor=INK, lcolor=INK2, rounding=1.0, title_gap=2.2, line_gap=2.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rounding}",
                                fc=fill, ec=edge, lw=lw, ls=ls))
    top = y + h - title_gap
    if title:
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=tcolor)
        top -= line_gap + 0.9
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - line_gap * i, s, ha="center", va="top", fontsize=lsize, color=lcolor)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0, ls="-", label=None, lpos=0.5, loff=(0, 1.2),
          lsize=6.8, lcolor=None, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=9, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(x0 + (x1 - x0) * lpos + loff[0], y0 + (y1 - y0) * lpos + loff[1], label, ha="center",
                va="bottom", fontsize=lsize, color=lcolor or color)


def canvas(w_in, h_in, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.set_aspect("auto")
    return fig, ax


def save(fig, name, png=False):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    if png:
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")
