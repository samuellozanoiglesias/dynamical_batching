"""
fig2_style.py
------------------------------------------------------------------------
Small drawing helpers shared by fig2_subpanels.py, fig2_decision_boundaries.py
and fig2_compound.py.

* style_axes(...)        -> spines / ticks / label pads with a global scale
                            factor, so a panel drawn at scale s looks exactly
                            like the standalone panel shrunk by s.
* add_framed_square(...) -> the black|colour|black square markers of panel (a)
* add_framed_box(...)    -> the black|colour|black frames around panel (b)
* bold_text(...)         -> LaTeX-like bold titles placed in inches from the top-left
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle

PT = 1.0 / 72.0  # one typographic point in inches

# matplotlib default tick / label geometry (points). Multiplying all of them by
# the same factor as the fonts reproduces a uniformly shrunk standalone panel.
TICK_MAJOR_SIZE, TICK_MAJOR_WIDTH, TICK_MAJOR_PAD = 3.5, 0.8, 3.5
TICK_MINOR_SIZE, TICK_MINOR_WIDTH, TICK_MINOR_PAD = 2.0, 0.6, 3.4
LABELPAD = 4.0

# Frame colours used in the figure (PowerPoint "Gold" and "Light green")
UNIFORM_COLOR = "#E69F00"
DYNAMICAL_COLOR = "#4DAF4A"


def style_axes(ax, scale: float = 1.0, spine_lw: float = 0.8) -> None:
    """Scale spines, ticks and label pads of an axes. scale=1 == matplotlib defaults."""
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw * scale)
    ax.tick_params(which="major", length=TICK_MAJOR_SIZE * scale,
                   width=TICK_MAJOR_WIDTH * scale, pad=TICK_MAJOR_PAD * scale)
    ax.tick_params(which="minor", length=TICK_MINOR_SIZE * scale,
                   width=TICK_MINOR_WIDTH * scale, pad=TICK_MINOR_PAD * scale)
    ax.xaxis.labelpad = LABELPAD * scale
    ax.yaxis.labelpad = LABELPAD * scale


def _frame_layers(x0, y0, w, h, lines_pt, color):
    """Rectangles (x, y, w, h, lw_pt, colour) for an outer-black / colour / inner-black frame.

    (x0, y0, w, h) is the OUTER edge of the frame, in inches.
    lines_pt = (outer black width, colour band width, inner black width) in points.
    """
    outer, band, inner = lines_pt
    layers = []
    for inset_pt, lw, c in (
        (outer / 2.0, outer, "black"),
        (outer + band / 2.0, band, color),
        (outer + band + inner / 2.0, inner, "black"),
    ):
        d = inset_pt * PT
        layers.append((x0 + d, y0 + d, w - 2 * d, h - 2 * d, lw, c))
    return layers


def add_framed_square(ax, x_data, y_data, color, side_pt=22.0,
                      lines_pt=(1.5, 5.0, 1.5), zorder=6) -> None:
    """Hollow framed square of fixed physical size centred on a data point."""
    fig = ax.figure
    trans = fig.dpi_scale_trans + mtransforms.ScaledTranslation(x_data, y_data, ax.transData)
    s = side_pt * PT
    for x, y, w, h, lw, c in _frame_layers(-s / 2, -s / 2, s, s, lines_pt, color):
        # add_artist (not add_patch) so the square never changes the data limits
        ax.add_artist(Rectangle((x, y), w, h, transform=trans, fill=False,
                                edgecolor=c, linewidth=lw, joinstyle="miter",
                                clip_on=False, zorder=zorder))


def add_framed_box(fig, left_in, top_in, width_in, height_in, color,
                   lines_pt=(1.1, 3.7, 1.0), zorder=0) -> None:
    """Framed box in figure inches measured from the TOP-LEFT corner of the figure."""
    y0 = fig.get_figheight() - top_in - height_in
    for x, y, w, h, lw, c in _frame_layers(left_in, y0, width_in, height_in, lines_pt, color):
        fig.add_artist(Rectangle((x, y), w, h, transform=fig.dpi_scale_trans,
                                 fill=False, edgecolor=c, linewidth=lw,
                                 joinstyle="miter", zorder=zorder))


# ---- bold title font ----------------------------------------------------
# The reference titles are LaTeX \textbf (Computer Modern Bold Extended).
# The script looks for Latin Modern Roman 10 Bold (ships with every TeX
# distribution, same metrics as cmbx10). Set TITLE_FONT_FILE to a path to
# force a font; if nothing is found it falls back to matplotlib's bold
# Computer Modern (mathtext, slightly narrower).
TITLE_FONT_FILE = None
TITLE_FONT_GLOBS = [
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-bold.otf",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-bold.otf",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-bold.otf",
    "~/Library/Fonts/lmroman10-bold.otf",
    "~/.fonts/lmroman10-bold.otf",
    "~/.local/share/fonts/lmroman10-bold.otf",
    "C:/Windows/Fonts/lmroman10-bold.otf",
]


def _find_title_font():
    if TITLE_FONT_FILE:
        return str(Path(TITLE_FONT_FILE).expanduser())
    for pattern in TITLE_FONT_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[0]
    try:  # installed system-wide under its family name?
        path = findfont(FontProperties(family="Latin Modern Roman", weight="bold"),
                        fallback_to_default=False)
        return path
    except Exception:
        return None


TITLE_FONT = _find_title_font()


def bold_text(fig, x_in, y_top_in, text, fontsize, **kw):
    """Bold label placed in inches measured from the top-left corner."""
    W, H = fig.get_size_inches()
    if TITLE_FONT:
        s, kw["fontproperties"] = text, FontProperties(fname=TITLE_FONT, size=fontsize)
    else:
        s = r"$\mathbf{" + text.replace(" ", r"\ ") + "}$"
        kw.update(math_fontfamily="cm", fontsize=fontsize)
    return fig.text(x_in / W, 1.0 - y_top_in / H, s, **kw)