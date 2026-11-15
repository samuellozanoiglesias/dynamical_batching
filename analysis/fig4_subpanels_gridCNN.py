"""fig4_subpanels_gridCNN.py

Heatmaps of accuracy gain vs. the A = 1 baseline (same period T), built
directly from the folders written by gridsearch_gridCNN.sbatch:

    {BASE_DIR}/{model}/w{W}_p{P}/seed{S}/summary.json
    {BASE_DIR}/{model}/w{W}_p{P}/seed{S}/training_metrics.csv

For every (A = w_max, T = period) cell the accuracy is averaged over all
finished seeds, then the gain  mean_acc(A, T) - mean_acc(A=1, T)  is plotted
in % points (same colormap / layout as fig4_subpanels.py).

Usage
-----
    python fig4_subpanels_gridCNN.py cifar10_ICLR_myrtle5_gridCNN cifar10_ICLR_resnet10_gridCNN
    python fig4_subpanels_gridCNN.py --all                       # every model folder in BASE_DIR
    python fig4_subpanels_gridCNN.py myrtle5 --metric last_k --last_k 10
    python fig4_subpanels_gridCNN.py resnet10 --vmin -10 --vmax 10 --stars

A model argument can be the exact folder name, a unique substring of one
(e.g. "myrtle5"), or a path to a model folder.

Outputs, per model, in {OUT_DIR}/{model}/:
    grid_analysis_summary.csv        long format (metric, w_max, period_length,
                                     value, std, n_seeds) -- same columns the
                                     old script read, plus seed statistics
    panel_{model}_train.png / _test.png
    colorbar_{model}.png
    {model}.png / .pdf               train + test preview
and, when more than one model is given, {OUT_DIR}/all_models_overview.png/.pdf
(one row per model, shared colorbar).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ========================================================================
# DEFAULTS -- all overridable from the command line
# ========================================================================
BASE_DIR = "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods"
OUT_DIR = "figures/gridCNN_analysis"


#MODEL = "cifar10_ICLR_plaincnn_bottleneck50_gridCNN"
#MODEL = "cifar10_ICLR_myrtle5_gridCNN"
MODEL = "cifar10_ICLR_resnet10_gridCNN"
#MODEL = "cifar10_ICLR_resnet10narrow_gridCNN"

MAX_X = 250      # max amplitude A to plot (None = no limit)
MAX_Y = 1500      # max period T to plot (None = no limit)

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

CBAR_VMIN = -30.0
CBAR_VMAX = 5.0
CBAR_VCENTER = 0.0
CBAR_TICKS = [-30, -20, -10, 0, 5]

Y_TICKS = [1500, 800, 150, 50, 20, 10]
X_TICKS = [1, 10, 50, 150, 250]

BASELINE_W = 1.0
MISSING_COLOR = "#ffffff"
PANEL_TITLES = {"train": "Training set", "test": "Test set"}
CBAR_LABEL = "Gain with respect\nto A=1 (%)"

RUN_DIR_RE = re.compile(r"^w([0-9.]+)_p([0-9.]+)$")
SEED_DIR_RE = re.compile(r"^seed(-?\d+)$")

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
})


# ========================================================================
# Loading the new folder layout
# ========================================================================


def _run_accuracy(seed_dir: Path, metric: str, last_k: int) -> tuple[float, float] | None:
    """Returns (train_acc, test_acc) in [0, 1] for one finished run, or None."""
    summary_path = seed_dir / "summary.json"
    if not summary_path.exists():
        return None  # unfinished / crashed run
    if metric == "final":
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        return float(s["final_train_accuracy"]), float(s["final_test_accuracy"])

    csv_path = seed_dir / "training_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, usecols=["train_accuracy", "test_accuracy"])
    if df.empty:
        return None
    if metric == "last_k":
        tail = df.tail(max(1, last_k))
        return float(tail["train_accuracy"].mean()), float(tail["test_accuracy"].mean())
    if metric == "best":
        return float(df["train_accuracy"].max()), float(df["test_accuracy"].max())
    raise ValueError(f"Unknown metric '{metric}'")


def collect_runs(model_dir: Path, metric: str, last_k: int, seeds: list[int] | None) -> pd.DataFrame:
    """One row per finished run: w_max, period_length, seed, train_accuracy, test_accuracy."""
    rows = []
    for run_dir in sorted(model_dir.iterdir()):
        m = RUN_DIR_RE.match(run_dir.name)
        if not (m and run_dir.is_dir()):
            continue
        w, T = float(m.group(1)), float(m.group(2))
        for seed_dir in sorted(run_dir.iterdir()):
            sm = SEED_DIR_RE.match(seed_dir.name)
            if not (sm and seed_dir.is_dir()):
                continue
            seed = int(sm.group(1))
            if seeds is not None and seed not in seeds:
                continue
            acc = _run_accuracy(seed_dir, metric, last_k)
            if acc is None:
                continue
            rows.append({"w_max": w, "period_length": T, "seed": seed,
                         "train_accuracy": acc[0], "test_accuracy": acc[1]})
    return pd.DataFrame(rows)


def aggregate_and_save_summary(runs: pd.DataFrame, csv_out: Path, min_seeds: int) -> pd.DataFrame:
    """Seed-mean per cell, written in the long format fig4_subpanels.py used."""
    grouped = runs.groupby(["w_max", "period_length"])
    long_rows = []
    for metric in ("train_accuracy", "test_accuracy"):
        stats = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        stats = stats[stats["count"] >= min_seeds]
        for _, r in stats.iterrows():
            long_rows.append({
                "metric": metric, "w_max": r["w_max"], "period_length": r["period_length"],
                "value": r["mean"], "std": r["std"], "n_seeds": int(r["count"]),
            })
    long_df = pd.DataFrame(long_rows).sort_values(["metric", "period_length", "w_max"])
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(csv_out, index=False)
    return long_df


def gain_tables(long_df: pd.DataFrame, max_x, max_y, baseline_w=BASELINE_W):
    """Pivot to (T x A) and subtract the A = baseline_w column (in % points)."""
    pivots = {}
    for split in ("train", "test"):
        sub = long_df[long_df["metric"] == f"{split}_accuracy"]
        pivots[split] = sub.pivot(index="period_length", columns="w_max", values="value")

    # common, sorted grid for both splits
    T_values = np.array(sorted(set(pivots["train"].index) | set(pivots["test"].index)), dtype=float)
    w_values = np.array(sorted(set(pivots["train"].columns) | set(pivots["test"].columns)), dtype=float)
    if max_x is not None:
        w_values = w_values[w_values <= max_x]
    if max_y is not None:
        T_values = T_values[T_values <= max_y]

    gains = {}
    for split, pv in pivots.items():
        pv = pv.reindex(index=T_values, columns=w_values)
        if baseline_w not in pv.columns or pv[baseline_w].isna().all():
            raise SystemExit(f"No baseline runs (A = {baseline_w}) found -- cannot compute gains.")
        scale = 100.0 if np.nanmax(pv.to_numpy()) <= 1.0 else 1.0
        g = pv.sub(pv[baseline_w], axis=0).to_numpy() * scale
        g[:, np.isclose(w_values, baseline_w)] = np.where(
            np.isnan(g[:, np.isclose(w_values, baseline_w)]), np.nan, 0.0
        )  # exact 0 on the baseline column (removes -1e-16 noise)
        gains[split] = g
    return T_values, w_values, gains["train"], gains["test"]


# ========================================================================
# Plotting (same look as fig4_subpanels.py)
# ========================================================================
def build_offcenter_cmap(base_name, vmin, vmax, vcenter, n=256):
    base = plt.get_cmap(base_name)
    frac = (vcenter - vmin) / (vmax - vmin)
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    neg_pos = np.linspace(0.0, frac, n)
    neg_col = base(np.linspace(0.0, 0.5, n))
    pos_pos = np.linspace(frac, 1.0, n)[1:]
    pos_col = base(np.linspace(0.5, 1.0, n))[1:]
    neg_col[-1] = (1.0, 1.0, 1.0, 1.0)
    nodes = list(zip(neg_pos, neg_col)) + list(zip(pos_pos, pos_col))
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{base_name}_offcenter", nodes)
    m = Fraction(frac).limit_denominator(1000).denominator
    cmap = cmap.resampled(m * n + 1)
    cmap.set_bad(MISSING_COLOR)
    return cmap


def best_nonbaseline(gain, T_values, w_values, baseline_w=BASELINE_W):
    masked = gain.copy()
    masked[:, np.isclose(w_values, baseline_w)] = np.nan
    if np.all(np.isnan(masked)):
        return None
    i, j = np.unravel_index(np.nanargmax(masked), masked.shape)
    return T_values[i], w_values[j]


def draw_panel(ax, T_values, w_values, gain, norm, cmap, stars=None, show_x=True, show_y=True):
    x_edges = np.arange(len(w_values) + 1)
    y_edges = np.arange(len(T_values) + 1)
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(gain),
                     cmap=cmap, norm=norm, shading="flat",
                     edgecolors="face", linewidth=0, antialiased=False,
                     rasterized=True)
    ax.set_aspect("auto")
    ax.set_ylim(0, len(T_values))

    centres = np.arange(len(T_values)) + 0.5
    y_pos, y_lab = [], []
    for t in Y_TICKS:
        if len(T_values) == 0 or not (T_values.min() <= t <= T_values.max()):
            continue
        exact = np.where(np.isclose(T_values, t))[0]
        y_pos.append(centres[exact[0]] if len(exact) else float(np.interp(t, T_values, centres)))
        y_lab.append(str(int(t)) if t == int(t) else str(t))
    ax.set_yticks(y_pos)
    if show_y:
        ax.set_yticklabels(y_lab, fontsize=TICK_FONTSIZE)
        ax.set_ylabel("Period (T)", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_yticklabels([])

    x_pos, x_lab = [], []
    for x in X_TICKS:
        idx = np.where(np.isclose(w_values, x))[0]
        if len(idx):
            x_pos.append(idx[0] + 0.5)
            x_lab.append(str(int(x)) if x == int(x) else str(x))
    ax.set_xticks(x_pos)
    if show_x:
        ax.set_xticklabels(x_lab, fontsize=TICK_FONTSIZE)
        ax.set_xlabel("Amplitude (A)", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_xticklabels([])

    for T_s, w_s, color, marker in (stars or []):
        i = np.where(np.isclose(T_values, T_s))[0]
        j = np.where(np.isclose(w_values, w_s))[0]
        if len(i) and len(j):
            ax.plot(j[0] + 0.5, i[0] + 0.5, marker=marker, color=color, markeredgecolor="black",
                    markersize=STAR_MARKERSIZE if marker == "*" else CIRCLE_MARKERSIZE, zorder=5)

    ax.tick_params(axis="both", which="both", direction="out")
    return mesh


def create_fixed_panel_axes(figsize=PANEL_FIGSIZE, box_size=HEATMAP_BOX_SIZE):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.28, 0.25, box_size[0] / figsize[0], box_size[1] / figsize[1]])
    return fig, ax


def save_colorbar(mesh, path, ticks):
    fig_cb = plt.figure(figsize=CBAR_FIGSIZE)
    ax_cb = fig_cb.add_axes([0.30, 0.08, 0.25, 0.84])
    cbar = fig_cb.colorbar(mesh, cax=ax_cb, ticks=ticks)
    cbar.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE, labelpad=20, va="center")
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    fig_cb.savefig(path, bbox_inches="tight")
    plt.close(fig_cb)


def add_side_colorbar(fig, ax, mesh, ticks):
    # Use a dedicated figure-level axes so the test heatmap is not shrunk
    # relative to the training heatmap when Matplotlib allocates colorbar space.
    # The caller reserves room on the right with subplots_adjust(right=...).
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cbar.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE, labelpad=15)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)


def save_fig(fig, base: Path, dpi=170):
    for ext in ("png", "pdf"):
        path = f"{base}.{ext}"
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except PermissionError:
            raise SystemExit(f"Cannot write {path}: close it in your viewer and rerun.")


# ========================================================================
# Main
# ========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=[MODEL],
                    help=f"model folder names inside --base_dir, or paths (default: {MODEL})")
    ap.add_argument("--base_dir", default=BASE_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--metric", choices=["final", "last_k", "best"], default="final",
                    help="final = accuracy at the last step (summary.json); "
                         "last_k = mean of the last K logged evaluations; best = max over training")
    ap.add_argument("--last_k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="only use these seeds")
    ap.add_argument("--min_seeds", type=int, default=1, help="cells with fewer finished seeds are left blank")
    ap.add_argument("--max_x", type=float, default=MAX_X)
    ap.add_argument("--max_y", type=float, default=MAX_Y)
    ap.add_argument("--vmin", type=float, default=CBAR_VMIN)
    ap.add_argument("--vmax", type=float, default=CBAR_VMAX)
    ap.add_argument("--ticks", type=float, nargs="*", default=None, help="colorbar ticks")
    ap.add_argument("--stars", action="store_true",
                    help="mark best test cell (green star) and its A=1 baseline (yellow circle)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_dir)
    model_dirs = [Path(m) if Path(m).is_dir() else base_dir / m for m in args.models]
    missing = [str(p) for p in model_dirs if not p.is_dir()]
    if missing:
        raise SystemExit(f"Model folder(s) not found: {missing}")

    ticks = args.ticks if args.ticks else (
        CBAR_TICKS if (args.vmin, args.vmax) == (CBAR_VMIN, CBAR_VMAX)
        else list(np.round(np.linspace(args.vmin, args.vmax, 5), 1))
    )
    norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)
    cmap = build_offcenter_cmap("RdBu_r", args.vmin, args.vmax, CBAR_VCENTER)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # figure-independent, for colorbars

    results = []  # for the overview figure
    for model_dir in model_dirs:
        name = model_dir.name
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        runs = collect_runs(model_dir, args.metric, args.last_k, args.seeds)
        if runs.empty:
            print(f"[{name}] no finished runs yet -- skipped.")
            continue
        long_df = aggregate_and_save_summary(runs, out_dir / "grid_analysis_summary.csv", args.min_seeds)
        if long_df.empty:
            print(f"[{name}] no cell has >= {args.min_seeds} seeds -- skipped.")
            continue
        T_values, w_values, gain_train, gain_test = gain_tables(long_df, args.max_x, args.max_y)

        n_cells = runs.groupby(["w_max", "period_length"]).ngroups
        seeds_per_cell = runs.groupby(["w_max", "period_length"]).size()
        print(f"[{name}] {len(runs)} finished runs, {n_cells} cells, "
              f"seeds/cell min={seeds_per_cell.min()} max={seeds_per_cell.max()} | metric={args.metric}")

        stars = []
        best = best_nonbaseline(gain_test, T_values, w_values)
        if best:
            T_s, w_s = best
            j = np.where(np.isclose(w_values, w_s))[0][0]
            i = np.where(np.isclose(T_values, T_s))[0][0]
            print(f"[{name}] best test gain: {gain_test[i, j]:+.2f} pp at A={w_s:g}, T={T_s:g}")
            if args.stars:
                stars = [(T_s, w_s, "#00FF00", "*"), (T_s, BASELINE_W, "#FFFF00", "o")]

        gains = {"train": gain_train, "test": gain_test}
        fig_prev, axes_prev = plt.subplots(1, 2, figsize=(16, 5.5))
        # Reserve a colorbar column without changing either heatmap's size.
        fig_prev.subplots_adjust(left=0.10, right=0.84, wspace=0.28, bottom=0.20, top=0.84)
        mesh = None
        for col, split in enumerate(["train", "test"]):
            fig_p, ax_p = create_fixed_panel_axes()
            mesh = draw_panel(ax_p, T_values, w_values, gains[split], norm, cmap,
                              stars=stars if split == "test" else None,
                              show_y=(split == "train"))
            fig_p.savefig(out_dir / f"panel_{name}_{split}.png", bbox_inches="tight", dpi=300)
            fig_p.savefig(out_dir / f"panel_{name}_{split}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig_p)

            draw_panel(axes_prev[col], T_values, w_values, gains[split], norm, cmap,
                       stars=stars if split == "test" else None, show_y=(col == 0))
            axes_prev[col].set_title(PANEL_TITLES[split], fontsize=TITLE_FONTSIZE)

        save_colorbar(scalar_map, out_dir / f"colorbar_{name}.png", ticks)
        save_colorbar(scalar_map, out_dir / f"colorbar_{name}.pdf", ticks)
        add_side_colorbar(fig_prev, axes_prev[1], scalar_map, ticks)
        save_fig(fig_prev, out_dir / name)
        plt.close(fig_prev)
        print(f"[{name}] figures -> {out_dir}")

        results.append((name, T_values, w_values, gains, stars))

    # ---------- overview: one row per model, shared colorbar ----------
    if len(results) > 1:
        n = len(results)
        fig, axes = plt.subplots(n, 2, figsize=(16, 4.2 * n), squeeze=False)
        mesh = None
        for r, (name, T_values, w_values, gains, stars) in enumerate(results):
            for c, split in enumerate(["train", "test"]):
                mesh = draw_panel(axes[r, c], T_values, w_values, gains[split], norm, cmap,
                                  stars=stars if split == "test" else None,
                                  show_x=(r == n - 1), show_y=(c == 0))
                if r == 0:
                    axes[r, c].set_title(PANEL_TITLES[split], fontsize=TITLE_FONTSIZE)
            axes[r, 0].annotate(name.replace("_gridCNN", "").replace("cifar10_ICLR_", ""),
                                xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                                ha="center", va="center", fontsize=LABEL_FONTSIZE)
        # Reserve space for the colorbar inside the canvas. Placing cax at x=1.0
        # puts it outside the figure and can break its position in vector PDFs.
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        cax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
        cbar = fig.colorbar(mesh, cax=cax, ticks=ticks)
        cbar.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE, labelpad=15)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
        save_fig(fig, out_root / "all_models_overview")
        plt.close(fig)
        print(f"Overview -> {out_root / 'all_models_overview'}.png/.pdf")


if __name__ == "__main__":
    main()
