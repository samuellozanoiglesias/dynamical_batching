"""fig4_subpanels.py

------------------------------------------------------------------------
Reads grid-search result files from:

    grid_analysis/{study_name}/grid_analysis_summary.csv

Computes accuracy gains relative to the baseline amplitude (A = 1.0)
evaluated at the same period T for training and testing accuracy (in % points).

Outputs:
    - Standalone panel heatmaps (PNG/PDF)
    - Standalone colorbars (PNG/PDF)
    - Preview composite figure
"""

import os
import re
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ========================================================================
# CONFIG -- tweak freely
# ========================================================================
STUDY_NAME = "cifar10_resnet_narrow"  # Target study folder name

CSV_PATH = f"grid_analysis/{STUDY_NAME}/grid_analysis_summary.csv"
OUTPUT_DIR = f"figures/grid_analysis/{STUDY_NAME}"

MAX_X = 250  # Maximum x value (Amplitude A / w_max) to plot; set to None for no limit
MAX_Y = 800  # Maximum y value (Period T) to plot; set to None for no limit

USE_TEX = False

LABEL_FONTSIZE = 24
TICK_FONTSIZE = 22
TITLE_FONTSIZE = 24
CBAR_LABEL_FONTSIZE = 24
CBAR_TICK_FONTSIZE = 22
STAR_MARKERSIZE = 20
CIRCLE_MARKERSIZE = 12

HEATMAP_BOX_SIZE = (6, 3)
PANEL_FIGSIZE = (8, 4)
CBAR_FIGSIZE = (2.5, 3.0)

CBAR_VMIN = -30.0   # bottom of colorbar
CBAR_VMAX = 5.0     # top of colorbar
CBAR_VCENTER = 0.0  # white point
CBAR_TICKS = [-30, -20, -10, 0, 5]

Y_TICKS = [800, 200, 50, 10]
X_TICKS = [
    1,
    10,
    50,
    100,
    250,
]

BASELINE_W = 1.0
MISSING_COLOR = "#ffffff"

PANEL_TITLES = {"train": "Training set", "test": "Test set"}

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "STIXGeneral" if not USE_TEX else "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
})


# ========================================================================
# Helpers
# ========================================================================
def build_offcenter_cmap(base_name="RdBu_r", vmin=CBAR_VMIN, vmax=CBAR_VMAX,
                         vcenter=CBAR_VCENTER, n=256):
    """Diverging colormap for a *linear* norm whose white point sits at vcenter.

    The negative (blue) half of the base colormap is spread over [vmin, vcenter],
    the positive (red) half is compressed into [vcenter, vmax].
    """
    base = plt.get_cmap(base_name)
    frac = (vcenter - vmin) / (vmax - vmin)  # position of 0 on the bar (6/7 here)

    neg_pos = np.linspace(0.0, frac, n)
    neg_col = base(np.linspace(0.0, 0.5, n))
    pos_pos = np.linspace(frac, 1.0, n)[1:]
    pos_col = base(np.linspace(0.5, 1.0, n))[1:]

    neg_col[-1] = (1.0, 1.0, 1.0, 1.0)  # force pure white at 0 (RdBu center is #f7f7f7)

    nodes = list(zip(neg_pos, neg_col)) + list(zip(pos_pos, pos_col))
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{base_name}_offcenter", nodes)
    cmap.set_bad(MISSING_COLOR)

    # Resample so that vcenter falls exactly on a bin: then 0 renders pure white.
    from fractions import Fraction
    m = Fraction(frac).limit_denominator(1000).denominator
    cmap = cmap.resampled(m * n + 1)
    cmap.set_bad(MISSING_COLOR)
    return cmap

def parse_w_max(val):
    """Converts w_max string values to floats."""
    val_str = str(val).strip()
    if val_str in ("1_without_bumps", "1 (without_bumps)"):
        return 1.0
    match = re.match(r"^([0-9.]+)", val_str)
    if match:
        return float(match.group(1))
    return float(val_str)



def load_gain_tables(csv_path, baseline_w=BASELINE_W, max_x=MAX_X, max_y=MAX_Y):
    """Parses CSV and returns sorted T_values, w_values, and gain matrices."""
    df = pd.read_csv(csv_path)

    if "metric" in df.columns and "value" in df.columns:
        df_sub = df[df["metric"].isin(["train_accuracy", "test_accuracy"])].copy()
        df_sub["w_max_num"] = df_sub["w_max"].apply(parse_w_max)
        df_sub["T"] = df_sub["period_length"].astype(float)

        pivot_train = df_sub[df_sub["metric"] == "train_accuracy"].pivot(
            index="T", columns="w_max_num", values="value"
        )
        pivot_test = df_sub[df_sub["metric"] == "test_accuracy"].pivot(
            index="T", columns="w_max_num", values="value"
        )
    else:
        col_map = {
            "period_length": "T",
            "amplitude": "w_max",
            "train_Accuracy": "train_accuracy",
            "test_Accuracy": "test_accuracy",
        }
        df = df.rename(columns=col_map)
        df["w_max"] = df["w_max"].apply(parse_w_max)

        pivot_train = df.pivot(index="T", columns="w_max", values="train_accuracy")
        pivot_test = df.pivot(index="T", columns="w_max", values="test_accuracy")

    pivot_train = pivot_train.sort_index().sort_index(axis=1)
    pivot_test = pivot_test.sort_index().sort_index(axis=1)

    baseline_train = pivot_train[baseline_w]
    baseline_test = pivot_test[baseline_w]

    scale_train = 100.0 if pivot_train.max().max() <= 1.0 else 1.0
    scale_test = 100.0 if pivot_test.max().max() <= 1.0 else 1.0

    gain_train = pivot_train.sub(baseline_train, axis=0) * scale_train
    gain_test = pivot_test.sub(baseline_test, axis=0) * scale_test

    T_values = pivot_train.index.to_numpy(dtype=float)
    w_values = pivot_train.columns.to_numpy(dtype=float)

    # Filter columns to only include x values up to max_x
    if max_x is not None:
        valid_cols = w_values <= max_x
        w_values = w_values[valid_cols]
        gain_train = gain_train.iloc[:, valid_cols]
        gain_test = gain_test.iloc[:, valid_cols]

    # Filter rows to only include y values up to max_y
    if max_y is not None:
        valid_rows = T_values <= max_y
        T_values = T_values[valid_rows]
        gain_train = gain_train.iloc[valid_rows, :]
        gain_test = gain_test.iloc[valid_rows, :]

    return T_values, w_values, gain_train.to_numpy(), gain_test.to_numpy()


def best_nonbaseline(gain, T_values, w_values, baseline_w=BASELINE_W):
    """Locates (T, w_max) of maximum gain excluding baseline."""
    masked = gain.copy()
    baseline_col = np.isclose(w_values, baseline_w)
    masked[:, baseline_col] = np.nan
    if np.all(np.isnan(masked)):
        return None
    i, j = np.unravel_index(np.nanargmax(masked), masked.shape)
    return T_values[i], w_values[j]


def draw_panel(
    ax,
    T_values,
    w_values,
    gain,
    norm,
    cmap,
    stars=None,
    show_x=True,
    show_y=True,
):
    """Draws heatmap with equal-sized cell squares and Y-axis from 10 to 2000."""
    x_edges = np.arange(len(w_values) + 1)
    y_edges = np.arange(len(T_values) + 1)
    masked_gain = np.ma.masked_invalid(gain)

    mesh = ax.pcolormesh(
        x_edges, y_edges, masked_gain, cmap=cmap, norm=norm, shading="flat"
    )
    ax.set_aspect("equal")

    # Enforce Y-axis limits explicitly from T=10 to T=2000
    y_10_match = np.where(np.isclose(T_values, 10))[0]
    y_2000_match = np.where(np.isclose(T_values, 2000))[0]

    y_min = y_10_match[0] if len(y_10_match) > 0 else 0
    y_max = (y_2000_match[0] + 1) if len(y_2000_match) > 0 else len(T_values)
    ax.set_ylim(y_min, y_max)

    # Map Y-ticks (Period T) onto cell-centre positions. A value that is not a
    # grid period is placed by interpolation between its neighbouring rows.
    centres = np.arange(len(T_values)) + 0.5
    y_tick_positions, y_tick_labels = [], []
    for t in Y_TICKS:
        if not (T_values.min() <= t <= T_values.max()):
            print(f"[y-ticks] T={t} outside the grid range; skipped.")
            continue
        exact = np.where(np.isclose(T_values, t))[0]
        pos = centres[exact[0]] if len(exact) else float(np.interp(t, T_values, centres))
        if not len(exact):
            print(f"[y-ticks] T={t} is not a grid period; placed between rows.")
        if y_min <= pos <= y_max:
            y_tick_positions.append(pos)
            y_tick_labels.append(str(int(t)) if t == int(t) else str(t))

    ax.set_yticks(y_tick_positions)
    if show_y:
        ax.set_yticklabels(y_tick_labels, fontsize=TICK_FONTSIZE)
        ax.set_ylabel("Period (T)", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    # Map X-ticks (w_max / bump width) onto integer cell center positions
    x_tick_positions = []
    x_tick_labels = []
    for x in X_TICKS:
        idx = np.where(np.isclose(w_values, x))[0]
        if len(idx) > 0:
            x_tick_positions.append(idx[0] + 0.5)
            x_label = (
                "1"
                if x == 1
                else (str(int(x)) if x == int(x) else str(x))
            )
            x_tick_labels.append(x_label)

    ax.set_xticks(x_tick_positions)
    if show_x:
        ax.set_xticklabels(x_tick_labels, fontsize=TICK_FONTSIZE)
        ax.set_xlabel("Amplitude (A)", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    ax.tick_params(axis="both", which="both", direction="out")
    return mesh


def create_fixed_panel_axes(figsize=PANEL_FIGSIZE, box_size=HEATMAP_BOX_SIZE):
    """Creates a figure and axes with fixed heatmap box size."""
    fig = plt.figure(figsize=figsize)
    left, bottom = 0.28, 0.25
    width = box_size[0] / figsize[0]
    height = box_size[1] / figsize[1]
    ax = fig.add_axes([left, bottom, width, height])
    return fig, ax


def save_colorbar(
    mesh,
    path,
    label="Gain with respect\nto A=1 (%)",
    figsize=CBAR_FIGSIZE,
):
    """Saves a standalone colorbar PDF/PNG formatted to match example layout."""
    fig_cb = plt.figure(figsize=figsize)
    ax_cb = fig_cb.add_axes([0.30, 0.08, 0.25, 0.84])
    cbar = fig_cb.colorbar(mesh, cax=ax_cb, ticks=CBAR_TICKS)
    cbar.set_label(label, fontsize=CBAR_LABEL_FONTSIZE, labelpad=20, va="center")
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    fig_cb.savefig(path, bbox_inches="tight")
    plt.close(fig_cb)


# ========================================================================
# Main
# ========================================================================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find summary CSV at: '{CSV_PATH}'")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    T_values, w_values, gain_train, gain_test = load_gain_tables(CSV_PATH, max_x=MAX_X, max_y=MAX_Y)

    # 1. Enforce strict 0.0 for baseline columns to eliminate float precision noise (-1e-16)
    gain_train[:, np.isclose(w_values, BASELINE_W)] = 0.0
    gain_test[:, np.isclose(w_values, BASELINE_W)] = 0.0

    # 2./3. Fixed linear scale from -30 to +5; white exactly at 0,
    #       red half of the colormap compressed into [0, 5]
    norm = mcolors.Normalize(vmin=CBAR_VMIN, vmax=CBAR_VMAX)
    cmap_curr = build_offcenter_cmap("RdBu_r", CBAR_VMIN, CBAR_VMAX, CBAR_VCENTER)

    best_pt = best_nonbaseline(gain_test, T_values, w_values)
    stars = []
    if best_pt:
        T_star, w_star = best_pt
        stars = [
            (T_star, w_star, "#00FF00", "*"),
            (T_star, BASELINE_W, "#FFFF00", "o"),
        ]

    fig_prev, axes_prev = plt.subplots(1, 2, figsize=(11.5, 5.0))
    saved_panels, meshes = [], {}

    gains = {"train": gain_train, "test": gain_test}

    for col, split in enumerate(["train", "test"]):
        gain = gains[split]

        fig_p, ax_p = create_fixed_panel_axes(PANEL_FIGSIZE, HEATMAP_BOX_SIZE)
        mesh = draw_panel(
            ax_p,
            T_values,
            w_values,
            gain,
            norm,
            cmap_curr,
            stars=(stars if split == "test" else None),
            show_x=True,
            show_y=(split == "train"),
        )

        out_path = os.path.join(OUTPUT_DIR, f"panel_{STUDY_NAME}_{split}.png")
        fig_p.savefig(out_path, bbox_inches="tight")
        plt.close(fig_p)
        saved_panels.append(out_path)
        meshes[split] = mesh

        # Preview plot
        draw_panel(
            axes_prev[col],
            T_values,
            w_values,
            gain,
            norm,
            cmap_curr,
            stars=(stars if split == "test" else None),
            show_x=True,
            show_y=(col == 0),
        )
        axes_prev[col].set_title(PANEL_TITLES[split], fontsize=TITLE_FONTSIZE)

    # Save standalone colorbar matching target style
    cbar_path = os.path.join(OUTPUT_DIR, f"colorbar_{STUDY_NAME}.png")
    save_colorbar(
        meshes["train"], cbar_path, label="Gain with respect\nto A=1 (%)"
    )

    # Place colorbar outside right panel without stealing axes space
    cax = inset_axes(
        axes_prev[1],
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1.0, 1.0),
        bbox_transform=axes_prev[1].transAxes,
        borderpad=0,
    )
    cbar_prev = fig_prev.colorbar(meshes["test"], cax=cax, ticks=CBAR_TICKS)
    cbar_prev.set_label(
        "Gain with respect\nto A=1 (%)", fontsize=CBAR_LABEL_FONTSIZE, labelpad=15
    )
    cbar_prev.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    
    preview_base = os.path.join(OUTPUT_DIR, STUDY_NAME)
    for ext in ("png", "pdf"):
        path = f"{preview_base}.{ext}"
        try:
            fig_prev.savefig(path, dpi=170, bbox_inches="tight")
        except PermissionError:
            raise SystemExit(f"Cannot write {path}: close it in your PDF viewer and rerun.")
    plt.close(fig_prev)
    preview_path = f"{preview_base}.pdf"

    print(f"Successfully generated subpanels for study '{STUDY_NAME}':")
    print("  Panels:", saved_panels)
    print("  Colorbar:", cbar_path)
    print("  Preview:", preview_path)

if __name__ == "__main__":
    main()