"""
fig2_subpanels.py
------------------------------------------------------------------------
Reads two grid-search result files:

    results_new.csv          -- "small" network
    results_large_nn_new.csv  -- "large" network

Each file has columns: T, w_max, train_accuracy, test_accuracy.
For every (T, w_max) configuration this script computes the accuracy gain
relative to the w_max = 1 baseline evaluated at the *same* T, separately
for train and test accuracy, in percentage points.

Output (written to figures/)
------------------------------------------------------------------------
    panel_small_train.png   panel_small_test.png
    panel_large_train.png   panel_large_test.png
    colorbar_small.png      colorbar_large.png
    fig2_subpanels_preview.png

The drawing functions (prepare_network, draw_panel, draw_heat_colorbar) are
also used by fig2_compound.py to build the full Figure 2.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from fig2_style import (style_axes, add_framed_square,
                        UNIFORM_COLOR, DYNAMICAL_COLOR)

# ========================================================================
# CONFIG -- tweak freely
# ========================================================================
USE_TEX = False              # True on a machine with a full LaTeX install

LABEL_FONTSIZE = 28          # x/y axis label font size
TICK_FONTSIZE = 22           # x/y tick label font size
TITLE_FONTSIZE = 24          # per-panel title font size
CBAR_LABEL_FONTSIZE = 36
CBAR_TICK_FONTSIZE = 28
CBAR_LABELPAD = 50
AXES_LINEWIDTH = 0.9

# Fixed physical dimensions of the heatmap box itself across ALL panels (width, height in inches)
HEATMAP_BOX_SIZE = (3.5, 3.2)

# Outer canvas size (large enough to fit full labels before tight-cropping)
PANEL_FIGSIZE = (5.9, 5.9)
CBAR_FIGSIZE = (2, 4.7)                  # size of each standalone colorbar file (inches)
CBAR_AX_RECT = [0.35, 0.06, 0.30, 0.88]  # colorbar axes inside CBAR_FIGSIZE (fractions)

Y_TICKS = [100, 300, 1000, 3000, 10000]
X_TICKS = [1, 50, 100, 150, 200]

BASELINE_W = 1.0             # w_max value used as the reference column
MISSING_COLOR = "#ffffff"
# Large-network colorbar range and ticks. Cells outside [LARGE_VMIN, LARGE_VMAX]
# saturate to the end colours (anything below -30 is drawn as the darkest blue).
LARGE_VMAX = 5.0
LARGE_VMIN = -30.0                     # None -> use the data minimum
LARGE_CBAR_TICKS = [5, 0, -10, -20, -30]    # add -30 here to also label the bottom end

CBAR_LABEL = "Gain with respect\nto A=1 (%)"

# --- framed square markers (small network only) --------------------------
# Sizes are given at STANDALONE scale; the compound figure shrinks them with
# the rest of the panel (22 pt -> ~14.8 pt, exactly as in the reference figure).
DRAW_MARKERS_STANDALONE = False   # squares in the standalone panels / preview too?
MARKER_SIDE_PT = 22.0
MARKER_LINES_PT = (1.5, 5.0, 1.5)  # outer black, colour band, inner black
# Force the marker position instead of using the best test-gain cell:
MARKER_T = None                    # e.g. 350.0
MARKER_A = None                    # e.g. 20.0  (uniform marker always sits at A = BASELINE_W)

DATASETS = {
    "small": "results_new.csv",
    "large": "results_large_nn_new.csv",
}

PANEL_TITLES = {"train": "Training gain", "test": "Validation gain"}
OUT_DIR = Path("figures")

RC = {
    "text.usetex": USE_TEX,
    "font.family": "STIXGeneral" if not USE_TEX else "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": AXES_LINEWIDTH,
}


# ========================================================================
# helpers
# ========================================================================
def edges_from_centers(centers, log=False):
    """Bin edges for pcolormesh from (possibly log-spaced) cell centers."""
    c = np.log10(centers) if log else np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    edges = np.concatenate([[first], mid, [last]])
    return 10 ** edges if log else edges


def load_gain_tables(csv_path, baseline_w=BASELINE_W):
    """Return (T_values, w_values, gain_train, gain_test) as 1D/2D arrays."""
    df = pd.read_csv(csv_path)
    pivot_train = df.pivot(index="T", columns="w_max", values="train_accuracy")
    pivot_test = df.pivot(index="T", columns="w_max", values="test_accuracy")
    pivot_train = pivot_train.sort_index().sort_index(axis=1)
    pivot_test = pivot_test.sort_index().sort_index(axis=1)

    baseline_train = pivot_train[baseline_w]
    baseline_test = pivot_test[baseline_w]

    gain_train = pivot_train.sub(baseline_train, axis=0) * 100.0
    gain_test = pivot_test.sub(baseline_test, axis=0) * 100.0

    T_values = pivot_train.index.to_numpy(dtype=float)
    w_values = pivot_train.columns.to_numpy(dtype=float)
    return T_values, w_values, gain_train.to_numpy(), gain_test.to_numpy()


def best_nonbaseline(gain, T_values, w_values, baseline_w=BASELINE_W):
    """(T, w_max) of the maximum gain in this panel, excluding baseline."""
    masked = gain.copy()
    baseline_col = np.isclose(w_values, baseline_w)
    masked[:, baseline_col] = np.nan
    if np.all(np.isnan(masked)):
        return None
    i, j = np.unravel_index(np.nanargmax(masked), masked.shape)
    return T_values[i], w_values[j]


def _asymmetric_cmap(base_cmap, vmin, vmax):
    """RdBu_r whose white point sits exactly at 0 for an asymmetric [vmin, vmax]."""
    zero_pos = (0.0 - vmin) / (vmax - vmin)
    n_neg, n_pos = 128, 128
    x_neg = np.linspace(0.0, zero_pos, n_neg)
    color_neg = base_cmap(np.linspace(0.0, 0.5, n_neg))
    x_pos = np.linspace(zero_pos, 1.0, n_pos)
    color_pos = base_cmap(np.linspace(0.5, 1.0, n_pos))
    x_all = np.concatenate([x_neg, x_pos[1:]])
    colors_all = np.vstack([color_neg, color_pos[1:]])
    cmap = mcolors.LinearSegmentedColormap.from_list("asym_RdBu_r", list(zip(x_all, colors_all)))
    cmap.set_bad(MISSING_COLOR)
    return cmap


def prepare_network(net_name, csv_path):
    """Load one network's grid and return everything needed to draw its row.

    Returns dict with keys: T, w, gains{'train','test'}, norm, cmap, markers,
    cbar_ticks (None -> matplotlib's automatic ticks).
    markers is a list of (T, A, colour) framed squares (small network only):
        green  -> best non-baseline test gain  (Dynamical Batching run)
        yellow -> same T, A = 1                (Uniform Batching run)
    """
    base_cmap = plt.get_cmap("RdBu_r").copy()
    base_cmap.set_bad(MISSING_COLOR)
    T_values, w_values, gain_train, gain_test = load_gain_tables(csv_path)
    all_gains = np.concatenate([gain_train.ravel(), gain_test.ravel()])

    markers = []
    cbar_ticks = None
    if net_name == "small":
        vmax = np.nanmax(np.abs(all_gains))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = base_cmap
        best = best_nonbaseline(gain_test, T_values, w_values)
        if best is not None:
            T_star, w_star = best
            T_star = MARKER_T if MARKER_T is not None else T_star
            w_star = MARKER_A if MARKER_A is not None else w_star
            markers = [(T_star, BASELINE_W, UNIFORM_COLOR),
                       (T_star, w_star, DYNAMICAL_COLOR)]
    else:
        vmin = np.nanmin(all_gains) if LARGE_VMIN is None else LARGE_VMIN
        vmax = LARGE_VMAX
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = _asymmetric_cmap(base_cmap, vmin, vmax)
        cbar_ticks = LARGE_CBAR_TICKS

    return {"T": T_values, "w": w_values,
            "gains": {"train": gain_train, "test": gain_test},
            "norm": norm, "cmap": cmap, "markers": markers, "cbar_ticks": cbar_ticks}


def draw_panel(ax, T_values, w_values, gain, norm, cmap, markers=None,
               show_x=True, show_y=True, scale=1.0, rasterized=False):
    """Draw one gain heatmap. `scale` shrinks fonts/ticks/lines/markers uniformly.
    rasterized=True avoids hairline seams between cells in PDF viewers."""
    w_edges = edges_from_centers(w_values, log=False)
    T_edges = edges_from_centers(T_values, log=True)
    masked_gain = np.ma.masked_invalid(gain)

    mesh = ax.pcolormesh(w_edges, T_edges, masked_gain, cmap=cmap, norm=norm,
                         shading="flat", rasterized=rasterized)

    style_axes(ax, scale=scale, spine_lw=AXES_LINEWIDTH)

    ax.set_yscale("log")
    ax.set_ylim(T_edges[0], T_edges[-1])
    ax.set_xlim(w_edges[0], w_edges[-1])
    y_ticks = [t for t in Y_TICKS if T_edges[0] <= t <= T_edges[-1]]
    ax.set_yticks(y_ticks)
    if show_y:
        ax.set_yticklabels([str(t) for t in y_ticks], fontsize=TICK_FONTSIZE * scale)
        ax.set_ylabel("Period (T)", fontsize=LABEL_FONTSIZE * scale)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    x_ticks = [t for t in X_TICKS if w_edges[0] <= t <= w_edges[-1]]
    ax.set_xticks(x_ticks)
    if show_x:
        ax.set_xticklabels([str(t) for t in x_ticks], fontsize=TICK_FONTSIZE * scale)
        ax.set_xlabel("Amplitude (A)", fontsize=LABEL_FONTSIZE * scale)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    ax.tick_params(axis="both", which="both", direction="out")

    for T_m, A_m, color in (markers or []):
        add_framed_square(ax, A_m, T_m, color,
                          side_pt=MARKER_SIDE_PT * scale,
                          lines_pt=tuple(l * scale for l in MARKER_LINES_PT))
    return mesh


def draw_heat_colorbar(fig, cax, mesh, label=CBAR_LABEL, scale=1.0, ticks=None):
    """Vertical gain colorbar into `cax`, sized relative to the standalone one."""
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks)
    style_axes(cbar.ax, scale=scale, spine_lw=AXES_LINEWIDTH)  # before set_label (labelpad)
    cbar.set_label(label, fontsize=CBAR_LABEL_FONTSIZE * scale,
                   labelpad=CBAR_LABELPAD * scale, va="center")
    cbar.outline.set_linewidth(AXES_LINEWIDTH * scale)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE * scale)
    return cbar


def create_fixed_panel_axes(figsize=PANEL_FIGSIZE, box_size=HEATMAP_BOX_SIZE):
    """Create a figure and axes where the axes (heatmap box) has a fixed physical size."""
    fig = plt.figure(figsize=figsize)
    left, bottom = 0.28, 0.25
    ax = fig.add_axes([left, bottom, box_size[0] / figsize[0], box_size[1] / figsize[1]])
    return fig, ax


def save_colorbar(mesh, path, label=CBAR_LABEL, figsize=CBAR_FIGSIZE, ticks=None):
    """Save a single, standalone colorbar."""
    fig_cb = plt.figure(figsize=figsize)
    ax_cb = fig_cb.add_axes(CBAR_AX_RECT)
    draw_heat_colorbar(fig_cb, ax_cb, mesh, label, ticks=ticks)
    fig_cb.savefig(path, bbox_inches="tight")
    plt.close(fig_cb)


def axis_visibility(net_name, split):
    """x labels only on the bottom (large) row, y labels only on the train column."""
    return net_name != "small", split != "test"


# ========================================================================
# main
# ========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig_prev, axes_prev = plt.subplots(2, 2, figsize=(11.5, 9.0))
        saved_panels, saved_colorbars = [], []

        for row, (net_name, csv_path) in enumerate(DATASETS.items()):
            data = prepare_network(net_name, csv_path)
            markers = data["markers"] if DRAW_MARKERS_STANDALONE else None
            meshes = {}

            for col, split in enumerate(["train", "test"]):
                gain = data["gains"][split]
                show_x, show_y = axis_visibility(net_name, split)

                fig_p, ax_p = create_fixed_panel_axes(PANEL_FIGSIZE, HEATMAP_BOX_SIZE)
                mesh = draw_panel(ax_p, data["T"], data["w"], gain, data["norm"], data["cmap"],
                                  markers, show_x=show_x, show_y=show_y)
                out_path = OUT_DIR / f"panel_{net_name}_{split}.png"
                fig_p.savefig(out_path, bbox_inches="tight")
                plt.close(fig_p)
                saved_panels.append(str(out_path))
                meshes[split] = mesh

                draw_panel(axes_prev[row, col], data["T"], data["w"], gain, data["norm"],
                           data["cmap"], markers, show_x=show_x, show_y=show_y)

            cbar_path = OUT_DIR / f"colorbar_{net_name}.png"
            save_colorbar(meshes["train"], cbar_path, ticks=data["cbar_ticks"])
            saved_colorbars.append(str(cbar_path))

            fig_prev.colorbar(meshes["train"], ax=list(axes_prev[row, :]), shrink=0.9,
                              ticks=data["cbar_ticks"], label=r"Gain with respect to $A=1$ (%)")

        fig_prev.savefig(OUT_DIR / "fig2_subpanels_preview.png", dpi=170, bbox_inches="tight")
        plt.close(fig_prev)

    print("Saved standalone panels:", saved_panels)
    print("Saved standalone colorbars:", saved_colorbars)
    print(f"Saved combined preview: {OUT_DIR / 'fig2_subpanels_preview.png'}")


if __name__ == "__main__":
    main()