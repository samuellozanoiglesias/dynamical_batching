"""Fig. 3, variant 3.

Top row    : (a) switching-set size |S|/n, (b) persistence of the switching set
             (Jaccard between consecutive cycles, with the random-set chance level
             of each protocol dotted), (c) switching distance d_ij (pair-averaged);
             one legend (Uniform / Dynamical Batching) above the three panels.
Bottom row : (d) early-vs-late in-cycle train-accuracy inset (unchanged from v2);
             (e) weight-level schematic, early training left -> late training right:
             many switching weights redrawn every cycle -> few switching weights,
             the same every cycle, on a scaffold that barely moves.

Data: see fig_common.py (switching_metrics.csv + switch_decomposition.csv per run;
UB keeps only uniform cycles, DB only bump cycles; matched by cycle index).
Output: <outdir>/fig3_v3_<dataset>.png and .pdf, plus fig3_v3_<dataset>.txt with
mean +/- SD across runs of |S|/n, persistence (and its random-set chance level)
and the switching distance d_ij (see fig_stats.py for the final/late convention).

Switching-set definition (--detrend, panels a-b only):
  drift   (default, paper) u = theta - m - D (p - (C-1)/2) / C, D = m(c) - m(c-1)
  linfit  u = residual of a least-squares line fitted to the C class states of the cycle
  mean    u = theta - m (no trend removal)
Non-default modes are recomputed from each run's switching_snapshots/ (no retraining)
and cached as switch_detrend_comparison.csv in the run dir. --compare-detrend also
writes <outdir>/fig3_detrend_compare_<dataset>.png: |S|/n, persistence and the
overlap J(S_mode, S_drift) for every mode, UB vs DB.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fig_common as fc
import fig_stats as st
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import switching_metrics as sm

DATASET_TYPE = "spiral"
SGD_FOLDER = Path(f"/data/samuel_lozano/dynamical-sgd/with_bumps/{DATASET_TYPE}-1_320_bumps/")
DYNAMICAL_FOLDER = Path(f"/data/samuel_lozano/dynamical-sgd/with_bumps/{DATASET_TYPE}-320_bumps/")

# Panel (d): one example run's per-step training_metrics.csv (independent of (a)-(c)).
INSET_CSV = Path("data_training_accuracy_figure_4.csv")
INSET_FOCUS_CLASS = 0
LOSS_COLOR = "#F65F52"      # soft purple, right axis of panel (d)
ACC_COLOR = "0.25"          # left axis of panel (d)

# None = show cycles up to the last dynamical (bump) cycle of the DB runs.
MAX_CYCLES: int | None = None

# Switching-set definition for panels (a)-(b): "drift" (paper), "linfit" or "mean".
DETREND = sm.DEFAULT_DETREND

OUT_NAME = f"fig3_v3_{DATASET_TYPE}"
FIG_W, FIG_H = 11.6, 6.8          # inches

LABELS = {
    "size": "Switching-set size\n$|\\mathcal{S}|/n$",
    "persist": "Switching-set persistence\n$J(\\mathcal{S}_c,\\mathcal{S}_{c-1})$",
    "dist": "Switching distance $d$",
}

CLASS_COLORS = ["#d7191c", "#808080", "#2b83ba"]   # class 1, 2, 3 (schematic)

# --------------------------------------------------------------------------- #
# Alternative switching-set definitions, recomputed from the snapshots
# --------------------------------------------------------------------------- #
@dataclass
class LocalCurve:
    cycle: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def _run_dirs(folder: Path) -> list[Path]:
    return sorted({p.parent for p in Path(folder).rglob("switching_snapshots") if p.is_dir()})


def ensure_detrend_tables(folder: Path, refresh: bool = False) -> list[Path]:
    """Every run under `folder` with snapshots gets switch_detrend_comparison.csv
    (all modes). Recomputed only if missing or incomplete."""
    need = [f"switch_frac_{m}" for m in sm.DETREND_MODES]
    tables = []
    for run in _run_dirs(folder):
        path = run / sm.DETREND_COMPARISON_CSV
        ok = False
        if path.exists() and not refresh:
            with open(path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            ok = all(c in header for c in need)
        if not ok:
            print(f"[detrend] recomputing decompositions from snapshots: {run}")
            sm.detrend_comparison_from_snapshots(run, sm.DETREND_MODES)
        tables.append(path)
    if not tables:
        print(f"[warn] no switching_snapshots/ under {folder}: alternative definitions unavailable "
              "(these runs would need a rerun with analysis.switching_save_snapshots = true).")
    return tables


def local_curve(tables: list[Path], regime: str, column: str, max_cycle: int) -> LocalCurve | None:
    """Across-run mean and band (fc.BAND: sd or sem) of one column, per cycle index."""
    per_cycle: dict[int, list[float]] = {}
    for path in tables:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("regime") != regime or column not in r:
                    continue
                c, v = int(float(r["cycle"])), fc._num(r[column])
                if c <= max_cycle and np.isfinite(v):
                    per_cycle.setdefault(c, []).append(v)
    if not per_cycle:
        return None
    cyc = np.array(sorted(per_cycle))
    vals = [np.asarray(per_cycle[c]) for c in cyc]
    mean = np.array([v.mean() for v in vals])
    spread = np.array([v.std(ddof=1) if v.size > 1 else 0.0 for v in vals])
    if fc.BAND == "sem":
        spread = spread / np.sqrt([max(1, v.size) for v in vals])
    return LocalCurve(cyc, mean, mean - spread, mean + spread)


def local_draw(ax, cv: LocalCurve | None, dynamical: bool) -> None:
    if cv is None:
        return
    color = fc.DYN_COLOR if dynamical else fc.SGD_COLOR
    ax.fill_between(cv.cycle, cv.lo, cv.hi, color=color, alpha=0.2, lw=0)
    ax.plot(cv.cycle, cv.mean, color=color, ls="-" if dynamical else "--",
            lw=fc.LW_DYN if dynamical else fc.LW_STD)


def _local_top(*curves: LocalCurve | None, pad: float = 1.1) -> float:
    tops = [float(np.nanmax(c.hi)) for c in curves if c is not None and np.isfinite(c.hi).any()]
    return pad * max(tops) if tops else 1.0


# --------------------------------------------------------------------------- #
# Panels a-c
# --------------------------------------------------------------------------- #
def draw_top_row(axes, ub: list[fc.Run], db: list[fc.Run], max_cycle: int,
                 detrend: str = sm.DEFAULT_DETREND, tables: dict | None = None) -> None:
    kw = dict(max_cycle=max_cycle)

    if detrend == sm.DEFAULT_DETREND:
        # (a) |S|/n
        ax = axes[0]
        cu, cd = fc.curve(ub, "dec/switch_frac", **kw), fc.curve(db, "dec/switch_frac", **kw)
        fc.draw(ax, cu, dynamical=False)
        fc.draw(ax, cd, dynamical=True, markers=False)
        ax.set_ylim(0, fc.top_of(cu, cd, pad=1.15))
        ax.set_ylabel(LABELS["size"])

        # (b) persistence + random-set chance (per protocol: |S| differs between them)
        ax = axes[1]
        fc.draw(ax, fc.curve(ub, "dec/switch_persist", **kw), dynamical=False)
        fc.draw(ax, fc.curve(db, "dec/switch_persist", **kw), dynamical=True, markers=False)
        #fc.draw_chance(ax, fc.curve(ub, "dec/switch_persist_chance", **kw), fc.SGD_COLOR)
        #fc.draw_chance(ax, fc.curve(db, "dec/switch_persist_chance", **kw), fc.DYN_COLOR)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(LABELS["persist"])
    else:
        # Same panels, switching set recomputed from the snapshots with another detrending.
        tu, td = tables["ub"], tables["db"]
        ax = axes[0]
        cu = local_curve(tu, fc.UB_REGIME, f"switch_frac_{detrend}", max_cycle)
        cd = local_curve(td, fc.DB_REGIME, f"switch_frac_{detrend}", max_cycle)
        local_draw(ax, cu, dynamical=False)
        local_draw(ax, cd, dynamical=True)
        ax.set_ylim(0, _local_top(cu, cd, pad=1.15))
        ax.set_ylabel(LABELS["size"])
        ax = axes[1]
        local_draw(ax, local_curve(tu, fc.UB_REGIME, f"switch_persist_{detrend}", max_cycle), dynamical=False)
        local_draw(ax, local_curve(td, fc.DB_REGIME, f"switch_persist_{detrend}", max_cycle), dynamical=True)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(LABELS["persist"])

    # (c) d_ij, pair-averaged within each run
    ax = axes[2]
    cu, cd = fc.curve(ub, "avg/d", **kw), fc.curve(db, "avg/d", **kw)
    fc.draw(ax, cu, dynamical=False)
    fc.draw(ax, cd, dynamical=True, markers=False)
    ax.set_ylim(0, fc.top_of(cu, cd))
    ax.set_ylabel(LABELS["dist"])

    for ax, letter in zip(axes, "abc"):
        fc.cycle_axis(ax, max_cycle)
        fc.panel_label(ax, letter)


# --------------------------------------------------------------------------- #
# Panel d: early-vs-late in-cycle accuracy inset (unchanged from v2)
# --------------------------------------------------------------------------- #
@dataclass
class CycleTrace:
    step: np.ndarray            # global_step column (steps elapsed since the cycle started)
    train_accuracy: np.ndarray
    train_loss: np.ndarray


def load_inset_cycles(csv_path: Path, focus_class: int = 0) -> tuple[CycleTrace, CycleTrace] | None:
    """First and last contiguous stretch where bump_focus_class == focus_class,
    as (early, late) traces of train_accuracy and train_loss vs. global_step."""
    if not csv_path.exists():
        print(f"[warn] inset csv not found: {csv_path}; panel (d) will be skipped.")
        return None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("epoch") not in (None, "")]
    if not rows:
        return None
    epoch = np.array([int(float(r["epoch"])) for r in rows])
    focus = np.array([int(float(r["bump_focus_class"])) for r in rows])
    acc = np.array([fc._num(r.get("train_accuracy")) for r in rows])
    loss = np.array([fc._num(r.get("train_loss")) for r in rows])
    gstep = np.array([fc._num(r.get("global_step")) for r in rows])
    idx = np.where(focus == focus_class)[0]
    if idx.size == 0:
        return None
    breaks = np.where(np.diff(epoch[idx]) > 1)[0]
    bounds = list(zip([idx[0]] + list(idx[breaks + 1]), list(idx[breaks]) + [idx[-1]]))
    if len(bounds) < 2:
        return None

    def trace(b: tuple[int, int]) -> CycleTrace:
        s, e = b
        return CycleTrace(step=gstep[s:e + 1] - gstep[s], train_accuracy=acc[s:e + 1],
                          train_loss=loss[s:e + 1])

    return trace(bounds[0]), trace(bounds[-1])


def draw_inset_panel(fig, cell, inset: tuple[CycleTrace, CycleTrace] | None) -> None:
    if inset is None:
        ax = fig.add_subplot(cell)
        ax.axis("off")
        ax.text(0.5, 0.5, "(inset data unavailable)", ha="center", va="center", fontsize=14,
                color="0.5", transform=ax.transAxes)
        fc.panel_label(ax, "d", dx=-52, dy=26)
        return
    early, late = inset
    outer_sub = cell.subgridspec(3, 1, height_ratios=[0.1, 1.0, 0.15])
    sub = outer_sub[1, 0].subgridspec(1, 2, wspace=0.95)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    for i, (ax, tr, title) in enumerate(zip(axes, (early, late), ("Early training", "Late training"))):
        x_end = float(tr.step.max())
        mid = x_end / 2.0                      # phase boundary: middle of the focus-class period
        ax.axvline(mid, color="0.6", lw=1.6, zorder=0, ls="--")
        ax.plot(tr.step, tr.train_accuracy, color=ACC_COLOR, lw=2.2, zorder=3)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
        ax.set_xlim(0, x_end)
        ax.set_ylim(0.3, 1.01)
        ax.set_xticks([0, mid, x_end])
        ax.set_xticklabels([f"{v:g}" for v in (0, round(mid), round(x_end))])
        ax.tick_params(axis="y", colors=ACC_COLOR)
        ax.grid(False)

        # right axis: training loss (own scale per panel; early losses are ~10x larger)
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(LOSS_COLOR)
        ax2.plot(tr.step, tr.train_loss, color=LOSS_COLOR, lw=1.8, zorder=2)
        top = float(np.nanmax(tr.train_loss)) if np.isfinite(tr.train_loss).any() else 1.0
        ax2.set_ylim(0, top * 1.08)
        ax2.tick_params(axis="y", colors=LOSS_COLOR, labelsize=13)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax2.grid(False)
        if i == 1:
            ax2.set_ylabel("Training loss", color=LOSS_COLOR, alpha=1)
    axes[0].set_ylabel("Training accuracy", color=ACC_COLOR)
    fc.panel_label(axes[0], "d", dx=-52, dy=26)
    ax_shared = fig.add_subplot(outer_sub[2, 0])
    ax_shared.axis("off")
    ax_shared.text(0.5, 0.2, "Steps elapsed since the cycle started", ha="center", va="top", fontsize=15)


# --------------------------------------------------------------------------- #
# Panel e: horizontal schematic (early left -> late right), key below
# --------------------------------------------------------------------------- #
def draw_schematic(ax) -> None:
    box_w, box_h, gap = 8, 5.5, 4.5               # frame size; gap holds the arrow
    yf = 1.55                                     # bottom edge of the frames
    W = 2 * box_w + gap
    ax.set_xlim(-0.5, W + 0.5)
    # Expanded bottom limit to accommodate the 2-line legend
    ax.set_ylim(-1.5, yf + box_h + 0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.axis("off")

    # --- DYNAMIC GRID SIZING ---
    # Define an internal margin (e.g., 8% horizontal, 12% vertical padding)
    margin_x = box_w * 0.08
    margin_y = box_h * 0.12
    avail_w = box_w - 2 * margin_x
    avail_h = box_h - 2 * margin_y

    sp_ratio = 0.35  # Spacing between cells is 35% of the cell size
    
    # Calculate required spatial units: max 6 columns wide, 4 rows high
    cols_width = 6 + 5 * sp_ratio
    rows_height = 4 + 3 * sp_ratio

    # Calculate optimal cell size to fit cleanly into the available space
    cell = min(avail_w / cols_width, avail_h / rows_height)
    sp = cell * sp_ratio
    r = cell * 0.32  # Inner cell rounding size scales with cell

    # Compute coordinates to perfectly center the entire lattice within the frame
    lattice_w = cell * cols_width
    lattice_h = cell * rows_height
    base_x0 = (box_w - lattice_w) / 2
    base_y0 = (box_h - lattice_h) / 2

    # Stagger multipliers: 0.5 means shift right by half a unit for a brick/honeycomb lay
    rows = [(0.5, 4), (0.0, 6), (0.0, 6), (0.5, 4)]   
    # ---------------------------

    scaf_fc, scaf_ec = "0.95", "0.72"             # class-agnostic scaffold cells

    def rbox(x, y, w, h, fc, ec, lw, z, rs):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

    def lattice(x_off, contested, title):
        # Outer frame bounding box
        rbox(x_off, yf, box_w, box_h, "white", "0.55", 0.8, 0, box_w * 0.05)
        
        x0 = x_off + base_x0
        y0 = yf + base_y0
        
        for ri, (stagger, n) in enumerate(rows):
            for c in range(n):
                # Apply the stagger relative to the computed cell and spacing size
                x = x0 + stagger * (cell + sp) + c * (cell + sp)
                y = y0 + (len(rows) - 1 - ri) * (cell + sp)
                
                if (ri, c) in contested:
                    for k, col in enumerate(CLASS_COLORS[::-1]):        # class 1 on top
                        # Calculate the 3D-stacking offset proportionally
                        depth = cell * 0.20 
                        rbox(x + depth * (2 - k), y - depth * (2 - k), cell, cell,
                             col, "0.30", 0.4, 2 + k, r)
                else:
                    rbox(x, y, cell, cell, scaf_fc, scaf_ec, 0.4, 1, r)
                    
        # Reduced y padding above boxes
        cx, y_top = x_off + box_w / 2, yf + box_h + 0.3
        ax.text(cx, y_top, title, ha="center", va="bottom", fontsize=16, fontweight="bold")

    lattice(0.0, {(0, 1), (1, 0), (1, 3), (2, 2), (2, 5), (3, 2)},
            "Early training")
    lattice(box_w + gap, {(1, 3), (2, 5)},
            "Late training")

    # arrow early -> late
    ym = yf + box_h / 2
    ax.add_patch(FancyArrowPatch((box_w + 0.3, ym), (box_w + gap - 0.3, ym), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color="0.35", zorder=3))
    ax.text(box_w + gap / 2, ym + 0.4, "Training", ha="center", va="bottom", fontsize=15,
            style="italic", color="0.30")

    # KEY (Bigger boxes, 2 lines)
    ks_classes = 1  # Increased from 0.5 for bigger boxes
    ks_boxes = 0.8     # Increased from 0.4 for bigger boxes
    yk1 = 0.4 # First row Y position
    yk2 = -0.7 # Second row Y position
    txt = dict(fontsize=14, va="center", ha="left", color="black")

    # Modified ksq to accept y_pos dynamically
    def ksq(x, y_pos, fc, ec, ks, off=0.0, z=1):
        rbox(x + off, y_pos - ks / 2 - off, ks, ks, fc, ec, 0.4, z, 0.14)

    # --- Row 1 ---
    ax.text(0.5, yk1, "Classes:", **txt)
    for i, col in enumerate(CLASS_COLORS):
        x = 4 + i * 1.2  # Spaced slightly more to accommodate bigger boxes
        ksq(x, yk1, col, "0.30", ks_classes)
        ax.text(x + ks_classes / 2, yk1 - 0.07, str(i + 1), fontsize=14, color="white", fontweight="bold",
                ha="center", va="center", zorder=5)
                
    # --- Row 1, Column 2 ---
    for k, col in enumerate(CLASS_COLORS[::-1]):
        ksq(10.4, yk1+0.1, col, "0.30", ks_boxes, off=0.1 * (2 - k), z=2 + k)
    ax.text(11.8, yk1, "Switching weight", **txt)
    
    # --- Row 2 ---
    # Aligned vertically under the shared weights
    ksq(10.5, yk2, scaf_fc, scaf_ec, ks_boxes)
    ax.text(11.9, yk2, "Scaffold weight", **txt)

CLASS_COLORS_REV = fc.CLASS_COLORS[::-1]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_detrend_comparison(tables: dict, outdir: Path, max_cycle: int) -> None:
    """Rows: |S|/n, persistence, overlap with the paper set J(S_mode, S_drift).
    Columns: detrending modes. UB dashed, DB solid (same colours as Fig. 3)."""
    modes = sm.DETREND_MODES
    rows = [("switch_frac", LABELS["size"]), ("switch_persist", LABELS["persist"]),
            (f"jaccard_vs_{sm.DEFAULT_DETREND}", "Overlap with paper set\n$J(\\mathcal{S}_{mode},\\mathcal{S}_{drift})$")]
    titles = {"drift": "drift (paper)", "linfit": "within-cycle linear fit", "mean": "cycle mean only"}
    with plt.rc_context(fc.RC):
        fig, axes = plt.subplots(len(rows), len(modes), figsize=(4.2 * len(modes), 3.4 * len(rows)),
                                 sharex=True, squeeze=False)
        for j, m in enumerate(modes):
            for i, (col, lab) in enumerate(rows):
                ax = axes[i, j]
                cu = local_curve(tables["ub"], fc.UB_REGIME, f"{col}_{m}", max_cycle)
                cd = local_curve(tables["db"], fc.DB_REGIME, f"{col}_{m}", max_cycle)
                local_draw(ax, cu, dynamical=False)
                local_draw(ax, cd, dynamical=True)
                if cu is None and cd is None:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.5")
                ax.set_ylim(0, 1.05 if i else _local_top(cu, cd, pad=1.15))
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 0:
                    ax.set_title(titles.get(m, m))
                if i == len(rows) - 1:
                    fc.cycle_axis(ax, max_cycle)
        # share the |S|/n scale across modes so the sizes are comparable
        top = max(ax.get_ylim()[1] for ax in axes[0])
        for ax in axes[0]:
            ax.set_ylim(0, top)
        handles = [Line2D([], [], color=fc.SGD_COLOR, ls="--", lw=fc.LW_STD, label=fc.SGD_LABEL),
                   Line2D([], [], color=fc.DYN_COLOR, ls="-", lw=fc.LW_DYN, label=fc.DYN_LABEL)]
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fc.save(fig, outdir, f"fig3_detrend_compare_{DATASET_TYPE}")


def print_detrend_summary(tables: dict, max_cycle: int) -> None:
    """Late-training means (last 20% of plotted cycles) per protocol and mode."""
    lo = int(0.8 * max_cycle)
    print(f"\n[detrend] means over cycles {lo}..{max_cycle}")
    for key, regime in (("ub", fc.UB_REGIME), ("db", fc.DB_REGIME)):
        for m in sm.DETREND_MODES:
            vals = []
            for col in ("switch_frac", "switch_persist", f"jaccard_vs_{sm.DEFAULT_DETREND}", "energy_kept"):
                cv = local_curve(tables[key], regime, f"{col}_{m}", max_cycle)
                v = float(np.nanmean(cv.mean[cv.cycle >= lo])) if cv is not None and (cv.cycle >= lo).any() else np.nan
                vals.append(v)
            print(f"  {key.upper()} {m:6s} |S|/n={vals[0]:.4f}  persist={vals[1]:.3f}  "
                  f"J(vs drift)={vals[2]:.3f}  energy kept={vals[3]:.3f}")


def write_fig3_stats(ub, db, outdir: Path, name: str, max_cycle: int,
                     detrend: str = sm.DEFAULT_DETREND) -> None:
    """Numbers behind panels (a)-(c), written to <outdir>/<name>.txt."""
    w = st.StatsWriter(f"{name}: panels (a)-(c), mean +/- SD across runs, cycles <= {max_cycle}, "
                       f"switching-set detrending = {detrend}")
    if detrend == sm.DEFAULT_DETREND:
        src = sm.DECOMPOSITION_CSV
        cols = {"size": "switch_frac", "persist": "switch_persist", "chance": "switch_persist_chance"}
    else:
        src = sm.DETREND_COMPARISON_CSV
        cols = {"size": f"switch_frac_{detrend}", "persist": f"switch_persist_{detrend}",
                "chance": f"switch_persist_chance_{detrend}"}
    items = [("(a) switching-set size |S|/n", src, st.column(cols["size"])),
             ("(b) persistence J(S_c, S_c-1)", src, st.column(cols["persist"])),
             ("(b) random-set chance J_rand", src, st.column(cols["chance"])),
             ("(c) switching distance d_ij (pair mean)", "switching_metrics.csv", st.pair_mean_distance)]
    for label, csv_name, fn in items:
        w.section(label)
        for proto, runs, regime in (("UB", ub, fc.UB_REGIME), ("DB", db, fc.DB_REGIME)):
            w.add(proto, st.summarize(st.cycle_series(r.path, csv_name, regime, fn, max_cycle) for r in runs))
    w.write(outdir, name)


def make_figure(ub, db, inset, outdir: Path, max_cycle: int,
                detrend: str = sm.DEFAULT_DETREND, tables: dict | None = None) -> None:
    with plt.rc_context(fc.RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], left=0.085, right=0.99,
                                 bottom=0.07, top=0.88, hspace=0.45)
        top = outer[0].subgridspec(1, 3, wspace=0.55)
        bottom = outer[1].subgridspec(1, 2, width_ratios=[1.45, 1.55], wspace=0.24)

        draw_top_row([fig.add_subplot(top[0, i]) for i in range(3)], ub, db, max_cycle, detrend, tables)
        draw_inset_panel(fig, bottom[0, 0], inset)
        ax_sch = fig.add_subplot(bottom[0, 1])
        draw_schematic(ax_sch)
        ax_sch.apply_aspect()
        fc.panel_label(ax_sch, "e", dx=-26, dy=8)

        # Plain line handles (no markers), independent of fig_common's defaults.
        handles = []
        if ub:
            handles.append(Line2D([], [], color=fc.SGD_COLOR, ls="--", lw=fc.LW_STD, label=fc.SGD_LABEL))
        if db:
            handles.append(Line2D([], [], color=fc.DYN_COLOR, ls="-", lw=fc.LW_DYN, label=fc.DYN_LABEL))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), handlelength=2.4,
                   columnspacing=3.8, handletextpad=0.7, borderaxespad=0.0,
                   bbox_to_anchor=(0.52, 0.99), fontsize=18)
        fc.save(fig, outdir, OUT_NAME if detrend == sm.DEFAULT_DETREND else f"{OUT_NAME}_{detrend}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--sgd-folder", type=Path, default=SGD_FOLDER)
    parser.add_argument("--dyn-folder", type=Path, default=DYNAMICAL_FOLDER)
    parser.add_argument("--inset-csv", type=Path, default=INSET_CSV)
    parser.add_argument("--inset-focus-class", type=int, default=INSET_FOCUS_CLASS)
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    parser.add_argument("--band", choices=("sd", "sem"), default=fc.BAND)
    parser.add_argument("--detrend", choices=sm.DETREND_MODES, default=DETREND,
                        help="switching-set definition for panels (a)-(b)")
    parser.add_argument("--compare-detrend", action="store_true",
                        help="also plot all definitions side by side and print a summary")
    parser.add_argument("--refresh-detrend", action="store_true",
                        help="recompute switch_detrend_comparison.csv even if it exists")
    args = parser.parse_args()
    fc.BAND = args.band

    ub = fc.load_folder(args.sgd_folder, fc.UB_REGIME, "UB")
    db = fc.load_folder(args.dyn_folder, fc.DB_REGIME, "DB")
    if not ub and not db:
        raise SystemExit("No switching data found in either folder.")
    max_cycle = args.max_cycles or fc.last_cycle(db or ub, "dec/switch_frac")
    if max_cycle is None:
        raise SystemExit("No switch_decomposition.csv data: run switching_plots_full.py once "
                         "(it computes the decomposition from the snapshots).")
    tables = None
    if args.detrend != sm.DEFAULT_DETREND or args.compare_detrend:
        tables = {"ub": ensure_detrend_tables(args.sgd_folder, args.refresh_detrend) if ub else [],
                  "db": ensure_detrend_tables(args.dyn_folder, args.refresh_detrend) if db else []}
        for key, runs in (("ub", ub), ("db", db)):
            if len(tables[key]) != len(runs):
                print(f"[warn] {key.upper()}: {len(runs)} runs loaded but {len(tables[key])} have snapshots.")
    make_figure(ub, db, load_inset_cycles(args.inset_csv, args.inset_focus_class), args.outdir, max_cycle,
                args.detrend, tables)
    write_fig3_stats(ub, db, args.outdir,
                     OUT_NAME if args.detrend == sm.DEFAULT_DETREND else f"{OUT_NAME}_{args.detrend}",
                     max_cycle, args.detrend)
    if args.compare_detrend:
        make_detrend_comparison(tables, args.outdir, max_cycle)
        print_detrend_summary(tables, max_cycle)


if __name__ == "__main__":
    main()