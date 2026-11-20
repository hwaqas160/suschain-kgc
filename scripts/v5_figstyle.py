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
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch  # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
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
SINGLE_COL, DOUBLE_COL = 3.54, 6.5  # inches; 6.5 in is the text width of the manuscript, so fonts print at size


SOFT3D = False  # soft drop shadow + gentle vertical gradient on diagram boxes and nodes


def _mix(c, other, t):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return tuple(a + (b - a) * t)


def _shadow(ax, patch_factory, dx, dy, layers=((1.1, 0.05), (0.7, 0.07), (0.35, 0.09))):
    """Soft shadow: a few offset, slightly grown copies with low alpha (blur imitation)."""
    for grow, alpha in layers:
        p = patch_factory(dx, dy, grow)
        p.set_facecolor((0.10, 0.12, 0.16, alpha))
        p.set_edgecolor("none")
        p.set_zorder(0.5)
        ax.add_patch(p)


def _gradient(ax, patch, x0, x1, y0, y1, top, bottom, zorder):
    g = np.linspace(0, 1, 64)[:, None, None]
    img = np.array(top)[None, None, :] * (1 - g) + np.array(bottom)[None, None, :] * g
    im = ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", aspect=ax.get_aspect(), zorder=zorder,
                   interpolation="bicubic")
    im.set_clip_path(patch)


def box(ax, x, y, w, h, title=None, lines=(), fill=FILL, edge=INK2, lw=0.8, ls="-",
        tsize=8.5, lsize=7.4, tcolor=INK, lcolor=INK2, rounding=1.0, title_gap=2.2, line_gap=2.9, flat=False):
    style = f"round,pad=0.2,rounding_size={rounding}"
    solid = ls == "-" and fill != "white"
    if SOFT3D and not flat:
        _shadow(ax, lambda dx, dy, g: FancyBboxPatch((x + dx - g / 2, y + dy - g / 2), w + g, h + g,
                                                     boxstyle=style), 0.55, -0.75)
        face = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="none", zorder=1)
        ax.add_patch(face)
        top_c = _mix(fill, "#ffffff", 0.75)
        bot_c = _mix(fill, edge if solid else "#9a9a96", 0.16)
        _gradient(ax, face, x - 0.3, x + w + 0.3, y - 0.3, y + h + 0.3, top_c, bot_c, 1.1)
        # thin highlight along the top edge
        ax.plot([x + rounding * 0.8, x + w - rounding * 0.8], [y + h + 0.05, y + h + 0.05], color="white",
                lw=0.9, alpha=0.9, zorder=1.3, solid_capstyle="round")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc="none", ec=edge, lw=lw, ls=ls, zorder=1.4))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec=edge, lw=lw, ls=ls))
    top = y + h - title_gap
    if title:
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=tcolor)
        top -= line_gap + 0.9
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - line_gap * i, s, ha="center", va="top", fontsize=lsize, color=lcolor)


def sphere(ax, x, y, r, fill, edge, lw=0.8, zorder=3):
    """A circle shaded like a soft sphere: light from the upper left, soft shadow below right."""
    if not SOFT3D:
        ax.add_patch(Circle((x, y), r, fc=fill, ec=edge, lw=lw, zorder=zorder))
        return
    for grow, alpha in ((0.9, 0.05), (0.5, 0.08), (0.2, 0.10)):
        ax.add_patch(Ellipse((x + 0.45, y - 0.6), 2 * r + grow, 2 * r + grow, fc=(0.1, 0.12, 0.16, alpha),
                             ec="none", zorder=zorder - 0.3))
    face = Circle((x, y), r, fc=fill, ec="none", zorder=zorder)
    ax.add_patch(face)
    n = 96
    yy, xx = np.mgrid[1:-1:n * 1j, -1:1:n * 1j]
    d = np.clip(np.hypot(xx + 0.38, yy - 0.42) / 1.55, 0, 1)[..., None]   # 0 at the highlight
    light, base, dark = np.array(_mix(fill, "#ffffff", 0.7)), np.array(to_rgb(fill)), np.array(_mix(fill, "#000000", 0.18))
    img = np.where(d < 0.55, light + (base - light) * (d / 0.55), base + (dark - base) * ((d - 0.55) / 0.45))
    im = ax.imshow(img, extent=(x - r, x + r, y - r, y + r), origin="upper", aspect=ax.get_aspect(), zorder=zorder + 0.05,
                   interpolation="bicubic")
    im.set_clip_path(face)
    ax.add_patch(Circle((x, y), r, fc="none", ec=edge, lw=lw, zorder=zorder + 0.1))


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


def save(fig, name, png=False, dpi=300):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    if png:
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")
