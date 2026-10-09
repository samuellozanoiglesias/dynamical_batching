"""
fig_sm_gridmap.py
------------------------------------------------------------------------
Supplementary figure: the 2x2 accuracy-gain grid maps for the LARGE-BATCH
(B = 200) spiral sweep, laid out like Fig_1_sm_large_bs.pdf:

                 Training set      Test set
    Small NN     [heatmap]         [heatmap]       [colorbar]
    Large NN     [heatmap]         [heatmap]       [colorbar]

Data
------------------------------------------------------------------------
Unlike Figure 2a (B = 50, read from fig2_subpanels.DATASETS), this figure
reads the large-batch sweep:

    Small NN (H = 50)   <- results_large_bs_new.csv
    Large NN (H = 500)  <- results_large_bs_large_nn_new.csv

By default both files are looked up in the same folder as the CSVs of
fig2_subpanels.DATASETS; use --small-csv / --large-csv to point elsewhere.
The script refuses to run if either path coincides with a Figure-2 CSV,
so the SM figure can no longer silently duplicate Figure 2a.

Colour maps, colorbar ranges (incl. the large-NN range 5 .. -30 with ticks
5, 0, -10, -20) and the drawing still come from fig2_subpanels.py, so both
figures share the same style and scale.

Geometry
------------------------------------------------------------------------
Page size and every box were measured from the reference PDF
(609.48 x 453.48 pt). Positions are INCHES from the TOP-LEFT corner.
In that PDF the heatmap images were stretched more than their text, so the
axes boxes and the font scale are set independently (HEAT_FONT_SCALE).

Usage
------------------------------------------------------------------------
    python fig_sm_gridmap.py                  # -> figures/fig_sm_gridmap.pdf / .png
    python fig_sm_gridmap.py --out some/path  # custom output (no extension)
    python fig_sm_gridmap.py --small-csv path/results_large_bs_new.csv \
                             --large-csv path/results_large_bs_large_nn_new.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import fig2_style as style
import fig2_subpanels as sp
from fig2_style import bold_text

# ========================================================================
# CONFIG
# ========================================================================
# ---- data: large-batch (B = 200) sweep ---------------------------------
SMALL_CSV_NAME = "results_large_bs_new.csv"            # H = 50
LARGE_CSV_NAME = "results_large_bs_large_nn_new.csv"   # H = 500
# Default folder = folder of the Figure-2 CSVs (change if yours live elsewhere)
DATA_DIR = Path(next(iter(sp.DATASETS.values()))).parent

FIGSIZE = (8.465, 6.298)           # 609.48 x 453.48 pt, same as the reference PDF

# (left, top, width, height) in inches from the top-left corner
HEAT_W, HEAT_H = 2.425, 2.360
COL_X = {"train": 1.452, "test": 4.362}
ROW_Y = {"small": 0.473, "large": 3.305}
CBAR_X, CBAR_W, CBAR_H = 6.913, 0.360, 2.387
CBAR_Y = {"small": 0.460, "large": 3.290}

# Fonts/ticks of the heatmaps relative to the standalone panels
# (tick labels ~16.6 pt, axis labels ~18 pt, as measured in the reference).
HEAT_FONT_SCALE = 0.753
# Colorbar scale is derived from its height, exactly like in fig2_compound.py
CBAR_SCALE = CBAR_H / (sp.CBAR_FIGSIZE[1] * sp.CBAR_AX_RECT[3])

# ---- text (bold title font is configured in fig2_style.TITLE_FONT_FILE) --
COL_TITLE_FONTSIZE = 22            # "Training set", "Test set"
ROW_TITLE_FONTSIZE = 22            # "Small NN", "Large NN"
TITLE_BASELINE = 0.303             # inches from top
COL_TITLES = {"Training set": COL_X["train"] + HEAT_W / 2,
              "Test set": COL_X["test"] + HEAT_W / 2}
ROW_TITLE_BASELINE_X = 0.347       # inches from left (rotated text, baseline on the right)
ROW_TITLES = {"Small NN": ROW_Y["small"] + HEAT_H / 2,
              "Large NN": ROW_Y["large"] + HEAT_H / 2}

DRAW_MARKERS = False               # the reference has no framed squares
RASTERIZE = True                   # rasterize heatmap cells inside the PDF (no seams)
OUT_BASE = Path("figures/fig_sm_gridmap")
PNG_DPI = 300

RC = {**sp.RC, "axes.linewidth": 1.0}


# ========================================================================
def add_axes_in(fig, left, top, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1.0 - (top + h) / H, w / W, h / H])


def resolve_datasets(small_csv: Path | None = None,
                     large_csv: Path | None = None) -> dict[str, Path]:
    """Map network name -> large-batch CSV, and guard against reusing Fig. 2 data."""
    datasets = {
        "small": Path(small_csv) if small_csv else DATA_DIR / SMALL_CSV_NAME,
        "large": Path(large_csv) if large_csv else DATA_DIR / LARGE_CSV_NAME,
    }
    fig2_csvs = {Path(p).resolve() for p in sp.DATASETS.values()}
    for net_name, path in datasets.items():
        if not path.is_file():
            raise FileNotFoundError(f"[fig_sm_gridmap] {net_name} NN CSV not found: {path}")
        if path.resolve() in fig2_csvs:
            raise ValueError(f"[fig_sm_gridmap] {path} is a Figure-2 (B=50) CSV; "
                             "this figure must use the large-batch sweep.")
    return datasets


def build_gridmap(out_base: Path = OUT_BASE, datasets: dict[str, Path] | None = None):
    datasets = datasets or resolve_datasets()

    with plt.rc_context(RC):
        fig = plt.figure(figsize=FIGSIZE)

        for net_name, csv_path in datasets.items():
            print(f"[fig_sm_gridmap] {net_name} NN <- {csv_path}")
            data = sp.prepare_network(net_name, csv_path)
            markers = data["markers"] if DRAW_MARKERS else None
            mesh = None
            for split in ("train", "test"):
                ax = add_axes_in(fig, COL_X[split], ROW_Y[net_name], HEAT_W, HEAT_H)
                show_x, show_y = sp.axis_visibility(net_name, split)
                m = sp.draw_panel(ax, data["T"], data["w"], data["gains"][split],
                                  data["norm"], data["cmap"], markers=markers,
                                  show_x=show_x, show_y=show_y,
                                  scale=HEAT_FONT_SCALE, rasterized=RASTERIZE)
                mesh = mesh or m
            cax = add_axes_in(fig, CBAR_X, CBAR_Y[net_name], CBAR_W, CBAR_H)
            sp.draw_heat_colorbar(fig, cax, mesh, scale=CBAR_SCALE, ticks=data["cbar_ticks"])

        for text, x in COL_TITLES.items():
            bold_text(fig, x, TITLE_BASELINE, text, COL_TITLE_FONTSIZE, ha="center", va="baseline")
        for text, y in ROW_TITLES.items():
            bold_text(fig, ROW_TITLE_BASELINE_X, y, text, ROW_TITLE_FONTSIZE, rotation=90,
                      rotation_mode="anchor", ha="center", va="baseline")

        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight": the page keeps the exact reference size.
        fig.savefig(out_base.with_suffix(".pdf"), dpi=PNG_DPI)
        fig.savefig(out_base.with_suffix(".png"), dpi=PNG_DPI)
        plt.close(fig)

    if not style.TITLE_FONT:
        print("[fig_sm_gridmap] Latin Modern Bold not found -> titles use matplotlib's bold CM "
              "(slightly narrower). Set fig2_style.TITLE_FONT_FILE to match the reference exactly.")
    print(f"Saved grid-map figure: {out_base.with_suffix('.pdf')} / .png")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the supplementary large-batch grid-map figure.")
    p.add_argument("--out", default=str(OUT_BASE), help="Output path without extension.")
    p.add_argument("--small-csv", default=None,
                   help=f"Small-NN CSV (default: {DATA_DIR / SMALL_CSV_NAME}).")
    p.add_argument("--large-csv", default=None,
                   help=f"Large-NN CSV (default: {DATA_DIR / LARGE_CSV_NAME}).")
    args = p.parse_args(argv)
    build_gridmap(Path(args.out), resolve_datasets(args.small_csv, args.large_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())