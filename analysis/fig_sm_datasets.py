"""SM figure sm_datasets: the five synthetic datasets used in the paper.

One row of five panels (Spiral, Blobs, Checkerboard, Dartboard, Rings), each
showing the training samples (circles, black edge) and the test samples
(crosses), coloured by class with the same Red - Grey - Blue palette and STIX
fonts as the decision-boundary figure (Figure 2b).

The datasets are regenerated from the exact configurations used in the paper
(CONFIGS below) with the project's own generator, build_dataset_bundle, so the
figure shows precisely the data the networks were trained and tested on.

Usage:
    python fig_sm_datasets.py -o figures
Output: <outdir>/fig_sm_datasets.png and .pdf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parent.parent))

# The full generator (all five datasets); the "easy" copy only contains the spiral.
try:
    from generate_dataset import build_dataset_bundle
except ImportError:                                   # pragma: no cover
    from generate_dataset import build_dataset_bundle

try:                                                  # same axis styling as Figure 2b
    from fig2_style import style_axes
except ImportError:                                   # pragma: no cover
    style_axes = None

OUT_NAME = "fig_sm_datasets"
LIM = (-1.1, 1.1)                    # all datasets live in [-1, 1] x [-1, 1]
TICKS = (-1.0, 0.0, 1.0)
TICK_LABELS = ("\u22121", "0", "1")    # proper minus sign, as matplotlib draws it

# ========================================================================
# Exact data configurations used in the paper (the `data:` block of each run)
# ========================================================================
CONFIGS = {
    "Spiral": dict(dataset_name="spiral", data_dir="./data", num_classes=3, points_per_class=100,
                   revolutions=4.0, noise_std=0.2, min_radius=0.01, random_seed=0),
    "Blobs": dict(dataset_name="blobs", data_dir="./data", num_classes=3, points_per_class=100,
                  noise_std=0.05, class_separation=120, center_radius=0.6, num_modes=5, random_seed=0),
    "Checkerboard": dict(dataset_name="checkerboard", data_dir="./data", num_classes=3,
                         points_per_class=100, noise_std=0.10, grid_size=7, random_seed=0),
    "Dartboard": dict(dataset_name="dartboard", num_classes=3, num_rings=3, num_sectors=12,
                  points_per_class=100, noise_std=0.03, inner_radius=0.1, outer_radius=1.0,
                  cell_gap=0.15, random_seed=0),
    #"Dartboard": dict(dataset_name="dartboard", num_classes=3, num_rings=2, num_sectors=12,
    #                  points_per_class=100, noise_std=0.05, inner_radius=0.1, outer_radius=1.0,
    #                  random_seed=0),
    "Rings": dict(dataset_name="rings", data_dir="./data", num_classes=3, points_per_class=100,
                  noise_std=0.05, class_separation=0.15, center_radius=0.1, num_rings=2, random_seed=0),
}

# ========================================================================
# Style (fonts and class colours as in the decision-boundary figure)
# ========================================================================
FIG_W, FIG_H = 18.0, 4.6
TITLE_FONTSIZE = 24
TICK_FONTSIZE = 18
LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 24
AXES_LINEWIDTH = 1.1

TRAIN_S, TRAIN_EDGE_LW = 16, 0.25
TEST_S, TEST_LW = 20, 0.9
GRID_LW = 0.8

RC = {
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": AXES_LINEWIDTH,
    "pdf.fonttype": 42,              # embed TrueType fonts (camera-ready friendly)
    "ps.fonttype": 42,
}


def class_colors(num_classes: int) -> list:
    """Point colours of the Red - Grey - Blue palette used in Figure 2b."""
    red_grey_blue = mcolors.LinearSegmentedColormap.from_list(
        "red_grey_blue", ["#d7191c", "#808080", "#2b83ba"])
    idx = [0.5] if num_classes == 1 else np.linspace(0.0, 1.0, num_classes)
    return [red_grey_blue(i) for i in idx]


# ========================================================================
# Data
# ========================================================================
def _points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4 and x.shape[1:] == (2, 1, 1):
        x = x[:, :, 0, 0]
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected 2D inputs of shape (N, 2), got {x.shape}.")
    return x


def load_dataset(data_cfg: dict):
    """Regenerate one dataset with the project generator. The generator also saves
    its own preview PNGs; those go to a temporary directory and are discarded."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = build_dataset_bundle({"data": dict(data_cfg)}, Path(tmp))
    return (_points(bundle.train_inputs), np.asarray(bundle.train_targets, dtype=np.int64),
            _points(bundle.test_inputs), np.asarray(bundle.test_targets, dtype=np.int64),
            int(bundle.num_classes))


# ========================================================================
# Drawing
# ========================================================================
def draw_dataset(ax, name: str, tr_x, tr_y, te_x, te_y, num_classes: int, first: bool,
                 letter: str = "a") -> None:
    colors = class_colors(num_classes)
    cmap = ListedColormap(colors)
    vmax = max(0, num_classes - 1)
    # clip_on=False: points on the boundary of [-1, 1]^2 are drawn whole, not cut in half
    ax.scatter(tr_x[:, 0], tr_x[:, 1], c=tr_y, cmap=cmap, vmin=0, vmax=vmax, s=TRAIN_S,
               alpha=0.98, edgecolors="black", linewidths=TRAIN_EDGE_LW, zorder=3, clip_on=False)
    ax.scatter(te_x[:, 0], te_x[:, 1], c=te_y, cmap=cmap, vmin=0, vmax=vmax, marker="x",
               s=TEST_S, alpha=1.0, linewidths=TEST_LW, zorder=2, clip_on=False)

    ax.set_xlim(*LIM)                                    # same square window for every dataset
    ax.set_ylim(*LIM)
    ax.set_aspect("equal", adjustable="box")

    if style_axes is not None:
        style_axes(ax, scale=1.0, spine_lw=AXES_LINEWIDTH)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_xticklabels(TICK_LABELS)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("x", fontsize=LABEL_FONTSIZE)
    if first:                                            # y labels only on the left-most panel
        ax.set_yticklabels(TICK_LABELS)
        ax.set_ylabel("y", fontsize=LABEL_FONTSIZE, rotation=0, labelpad=10)
    else:
        ax.set_yticklabels([])
    ax.set_title(name, fontsize=TITLE_FONTSIZE, pad=8)
    ax.text(-0.14, 1.07, f"({letter})", transform=ax.transAxes, fontsize=TITLE_FONTSIZE,
            fontweight="bold", va="bottom", ha="left")
    ax.grid(True, alpha=0.2, linewidth=GRID_LW, zorder=0)


def legend_handles(num_classes: int) -> list:
    colors = class_colors(num_classes)
    h = [Line2D([], [], ls="none", marker="o", ms=9, mfc=c, mec="black", mew=0.4, label=f"Class {i}")
         for i, c in enumerate(colors)]
    h += [Line2D([], [], ls="none", marker="o", ms=9, mfc="white", mec="black", mew=0.8, label="Training"),
          Line2D([], [], ls="none", marker="x", ms=9, mec="black", mew=1.4, label="Test")]
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, cfg in CONFIGS.items():
        data[name] = load_dataset(cfg)
        tr_x, tr_y, te_x, te_y, c = data[name]
        print(f"[data] {name:<12s} train {tr_x.shape[0]:4d}  test {te_x.shape[0]:4d}  classes {c}")
        allx = np.concatenate([tr_x, te_x])
        if allx.min() < LIM[0] - 1e-6 or allx.max() > LIM[1] + 1e-6:
            print(f"[warn] {name}: points outside [-1, 1]^2 (min {allx.min():.3f}, max {allx.max():.3f})")

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(data), figsize=(FIG_W, FIG_H))
        fig.subplots_adjust(left=0.04, right=0.995, bottom=0.14, top=0.80, wspace=0.12)
        for i, (ax, (name, d)) in enumerate(zip(axes, data.items())):
            draw_dataset(ax, name, *d, first=(i == 0), letter="abcde"[i])
        n_cls = max(d[4] for d in data.values())
        fig.legend(handles=legend_handles(n_cls), loc="upper center", ncol=n_cls + 2,
                   frameon=False, fontsize=LEGEND_FONTSIZE, handletextpad=0.3,
                   columnspacing=1.8, bbox_to_anchor=(0.52, 1.05))
        for ext in ("png", "pdf"):
            path = args.outdir / f"{OUT_NAME}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[save] {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()