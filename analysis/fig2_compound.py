"""
fig2_compound.py
------------------------------------------------------------------------
Builds the full Figure 2 in one matplotlib figure:

  (a) 2x2 accuracy-gain heatmaps (Small NN / Large NN x Training / Test set)
      with one colorbar per row and the framed squares marking the
      Uniform (A = 1, yellow) and Dynamical (best A, green) configurations.
  (b) decision boundaries of the Uniform Batching and Dynamical Batching
      runs, each inside a matching coloured frame, plus the shared
      "Predicted Class" colorbar.

It also (re)creates every standalone piece exactly as before:
    figures/panel_*.png, figures/colorbar_small|large.png,
    figures/fig2_subpanels_preview.png,
    figures/fig2_decision_boundaries_uniform|dynamical.png,
    figures/colorbar_decision_boundaries.png

Geometry
------------------------------------------------------------------------
The page size and every box below were measured from the reference PDF
(1190.52 x 425.28 pt). All positions are in INCHES from the TOP-LEFT corner.
Fonts, ticks, line widths and markers of each block are the standalone values
multiplied by (size of the box here / size of the box in the standalone file),
which is exactly the shrink factor applied when the standalone images were
pasted into the slide.

Usage
------------------------------------------------------------------------
    python fig2_compound.py                       # everything
    python fig2_compound.py --no-standalone       # compound only
    python fig2_compound.py --uniform RUN --dynamical RUN --out figures/fig2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import fig2_subpanels as sp
import fig2_decision_boundaries as dbm
import fig2_style as style
from fig2_style import add_framed_box, bold_text, UNIFORM_COLOR, DYNAMICAL_COLOR

# ========================================================================
# CONFIG
# ========================================================================
FIGSIZE = (16.535, 5.907)          # 1190.52 x 425.28 pt, same as the reference PDF

# (left, top, width, height) in inches from the top-left corner
HEAT_W, HEAT_H = 2.360, 2.162
LAYOUT = {
    # ---- (a) heatmaps ----------------------------------------------------
    "small_train": (1.510, 0.780, HEAT_W, HEAT_H),
    "small_test":  (4.140, 0.780, HEAT_W, HEAT_H),
    "large_train": (1.510, 3.178, HEAT_W, HEAT_H),
    "large_test":  (4.140, 3.178, HEAT_W, HEAT_H),
    "cbar_small":  (6.660, 0.787, 0.330, 2.183),
    "cbar_large":  (6.660, 3.177, 0.330, 2.183),
    # ---- (b) decision boundaries ------------------------------------------
    "dec_uniform":   (8.930, 0.983, 3.235, 3.215),
    "dec_dynamical": (13.100, 0.983, 3.235, 3.215),
    "frame_uniform":   (8.277, 0.830, 4.027, 3.950),   # outer edge of the frame
    "frame_dynamical": (12.450, 0.830, 4.027, 3.950),
    "cbar_class":    (9.883, 4.877, 4.983, 0.313),
}

FRAME_LINES_PT = (1.1, 3.7, 1.0)   # outer black, colour band, inner black (b frames)

# ---- text (bold title font is configured in fig2_style.TITLE_FONT_FILE) --
COL_TITLE_FONTSIZE = 23.5          # "Training set", "Test set", "Uniform/Dynamical Batching"
ROW_TITLE_FONTSIZE = 22            # "Small NN", "Large NN"
PANEL_LABEL_FONTSIZE = 28          # "(a)", "(b)"
TITLE_BASELINE = 0.580             # inches from top, all column titles
COL_TITLES = {                     # text: x centre (inches)
    "Training set": 1.510 + HEAT_W / 2,
    "Test set": 4.140 + HEAT_W / 2,
    "Uniform Batching": 8.277 + 4.027 / 2,
    "Dynamical Batching": 12.450 + 4.027 / 2,
}
ROW_TITLE_BASELINE_X = 0.397       # inches from left (rotated text, baseline on the right)
ROW_TITLES = {                     # text: y centre (inches from top)
    "Small NN": 0.780 + HEAT_H / 2,
    "Large NN": 3.178 + HEAT_H / 2 + 0.06,
}

PANEL_LABELS = {"(a)": (0.097, 0.143), "(b)": (7.803, 0.143)}  # anchor of ha="left", va="top" (inches)

UNIFORM_RUN = dbm.PATH_STATIC
DYNAMICAL_RUN = dbm.PATH_DYN
OUT_BASE = Path("figures/fig2_compound")   # -> .pdf and .png
PNG_DPI = 300
RASTERIZE = True                   # rasterize heatmap cells / filled regions inside the PDF

RC = {**sp.RC, "axes.linewidth": 1.0}


# ========================================================================
# helpers
# ========================================================================
def add_axes_in(fig, key):
    left, top, w, h = LAYOUT[key]
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1.0 - (top + h) / H, w / W, h / H])


def scale_factors():
    """Shrink factor of each block relative to its standalone file."""
    sub_w = plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"]
    return {
        "heat": HEAT_W / sp.HEATMAP_BOX_SIZE[0],
        "heat_cbar": LAYOUT["cbar_small"][3] / (sp.CBAR_FIGSIZE[1] * sp.CBAR_AX_RECT[3]),
        "dec": LAYOUT["dec_uniform"][2] / (dbm.PLOT_FIGSIZE[0] * sub_w),
        "class_cbar": LAYOUT["cbar_class"][2] / (dbm.CBAR_FIGSIZE[0] * dbm.CBAR_AX_RECT[2]),
    }


# ========================================================================
# compound figure
# ========================================================================
def build_compound(uniform_db: dict, dynamical_db: dict, out_base: Path = OUT_BASE):
    s = scale_factors()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=FIGSIZE)

        # ---------------- (a) heatmaps ----------------
        for net_name, csv_path in sp.DATASETS.items():
            data = sp.prepare_network(net_name, csv_path)
            mesh = None
            for split in ("train", "test"):
                ax = add_axes_in(fig, f"{net_name}_{split}")
                show_x, show_y = sp.axis_visibility(net_name, split)
                m = sp.draw_panel(ax, data["T"], data["w"], data["gains"][split],
                                  data["norm"], data["cmap"], markers=data["markers"],
                                  show_x=show_x, show_y=show_y, scale=s["heat"],
                                  rasterized=RASTERIZE)
                mesh = mesh or m
            sp.draw_heat_colorbar(fig, add_axes_in(fig, f"cbar_{net_name}"), mesh,
                                  scale=s["heat_cbar"], ticks=data["cbar_ticks"])

        # ---------------- (b) decision boundaries ----------------
        num_classes = max(uniform_db["num_classes"], dynamical_db["num_classes"])
        contour = None
        for key, db in (("dec_uniform", uniform_db), ("dec_dynamical", dynamical_db)):
            cs = dbm.draw_decision_boundary(add_axes_in(fig, key), db,
                                            num_classes=num_classes, scale=s["dec"],
                                            rasterized=RASTERIZE)
            contour = contour or cs
        dbm.draw_class_colorbar(fig, add_axes_in(fig, "cbar_class"), contour,
                                num_classes, scale=s["class_cbar"])

        add_framed_box(fig, *LAYOUT["frame_uniform"], UNIFORM_COLOR, lines_pt=FRAME_LINES_PT)
        add_framed_box(fig, *LAYOUT["frame_dynamical"], DYNAMICAL_COLOR, lines_pt=FRAME_LINES_PT)

        # ---------------- titles & labels ----------------
        for text, x in COL_TITLES.items():
            bold_text(fig, x, TITLE_BASELINE, text, COL_TITLE_FONTSIZE, ha="center", va="baseline")
        for text, y in ROW_TITLES.items():
            bold_text(fig, ROW_TITLE_BASELINE_X, y, text, ROW_TITLE_FONTSIZE, rotation=90,
                      rotation_mode="anchor", ha="center", va="baseline")
        for text, (x, y) in PANEL_LABELS.items():
            bold_text(fig, x, y, text, PANEL_LABEL_FONTSIZE, ha="left", va="top")

        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight": the page must keep the exact reference size.
        fig.savefig(out_base.with_suffix(".pdf"), dpi=PNG_DPI)
        fig.savefig(out_base.with_suffix(".png"), dpi=PNG_DPI)
        plt.close(fig)
    if not style.TITLE_FONT:
        print("[fig2_compound] Latin Modern Bold not found -> titles use matplotlib's bold CM "
              "(slightly narrower). Set TITLE_FONT_FILE to match the reference exactly.")
    print(f"Saved compound figure: {out_base.with_suffix('.pdf')} / .png")


# ========================================================================
# standalone pieces (same outputs as the individual scripts)
# ========================================================================
def build_standalone(uniform_db: dict, dynamical_db: dict, out_dir: Path = sp.OUT_DIR):
    sp.main()
    num_classes = max(uniform_db["num_classes"], dynamical_db["num_classes"])
    with plt.rc_context(dbm.RC):
        contour = None
        for tag, db in (("uniform", uniform_db), ("dynamical", dynamical_db)):
            path = Path(out_dir) / f"fig2_decision_boundaries_{tag}.png"
            cs, _ = dbm.save_decision_boundary_plot(db, path, num_classes)
            contour = contour or cs
            print(f"Saved decision boundaries: {path}")
        cbar_path = Path(out_dir) / "colorbar_decision_boundaries.png"
        dbm.save_standalone_colorbar(contour, num_classes, cbar_path)
        print(f"Saved class colorbar: {cbar_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build Figure 2 (standalone panels + compound).")
    p.add_argument("--uniform", default=UNIFORM_RUN, help="Run dir of the Uniform Batching model.")
    p.add_argument("--dynamical", default=DYNAMICAL_RUN, help="Run dir of the Dynamical Batching model.")
    p.add_argument("--grid-size", type=int, default=400)
    p.add_argument("--out", default=str(OUT_BASE), help="Output path without extension.")
    p.add_argument("--no-standalone", action="store_true", help="Only build the compound figure.")
    args = p.parse_args(argv)

    uniform_db = dbm.load_decision_boundary_data(Path(args.uniform).expanduser(), args.grid_size)
    dynamical_db = dbm.load_decision_boundary_data(Path(args.dynamical).expanduser(), args.grid_size)

    if not args.no_standalone:
        build_standalone(uniform_db, dynamical_db)
    build_compound(uniform_db, dynamical_db, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())