"""SM figure sm_spiral_accuracy: training and test accuracy of the spiral runs.

(a) small network (H=50): train and test accuracy, UB vs DB
(b) large network (H=500): the same
Four solid lines per panel, one colour per (protocol, split): see COLORS below.
Lines are the across-run mean, shaded bands the SD across runs
(or SEM with --band sem), aligned by global_step.

Data: training_metrics.csv of every run found (recursively) under each folder,
one row per global_step (columns global_step, train_accuracy, test_accuracy).
Output: <outdir>/fig_sm_spiral_accuracy.png/.pdf and fig_sm_spiral_accuracy.txt
(final and late accuracy, mean +/- SD across runs, see fig_stats.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fig_common as fc
import fig_stats as st

SMALL_SGD = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_320_bumps/")     # H=50, A=1
SMALL_DYN = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-320_bumps/")    # H=50, A=20
LARGE_SGD = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_150_bumps_large/")    # EDIT: H=500, A=1
LARGE_DYN = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-150_bumps_large/")          # EDIT: H=500, A>1
METRICS_CSV = "training_metrics.csv"
OUT_NAME = "fig_sm_spiral_accuracy"
FIG_W, FIG_H = 12.0, 4.6

# One colour per (protocol, split); edit freely. All lines are solid.
COLORS = {
    ("UB", "train_accuracy"): "#E69F00",   # UB, training  (dark orange)
    ("UB", "test_accuracy"):  "#F7C873",   # UB, test      (light orange)
    ("DB", "train_accuracy"): "#2E8B2E",   # DB, training  (dark green)
    ("DB", "test_accuracy"):  "#9AD68A",   # DB, test      (light green)
}
LINE_WIDTH = 2.0
BAND_ALPHA = 0.18

PROTOCOLS = (("UB", "UB"), ("DB", "DB"))                                  # (key, legend label)
SPLITS = (("train_accuracy", "Training"), ("test_accuracy", "Test"))     # (column, label)


def run_dirs(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        print(f"[warn] folder not found: {folder}")
        return []
    return sorted({p.parent for p in folder.rglob(METRICS_CSV)})


def global_step_series(run_dir: Path, col: str) -> dict[int, float]:
    """global_step -> value (last row of each global_step), rows without a value skipped."""
    out: dict[int, float] = {}
    for r in st.read_rows(run_dir / METRICS_CSV):
        e, v = st.num(r.get("global_step")), st.num(r.get(col))
        if np.isfinite(e) and np.isfinite(v):
            out[int(e)] = v
    return out


# Drawing order: higher zorder is drawn on top. UB in front, DB behind.
ZORDER = {"DB": 2, "UB": 3}


def draw(ax, runs: list[Path], col: str, color: str, zorder: float = 2) -> None:
    mb = st.mean_band([global_step_series(r, col) for r in runs], fc.BAND)
    if mb is None:
        return
    x, m, lo, hi = mb
    # Band just below its own line, so each protocol's band+line stay together.
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=zorder - 0.5)
    ax.plot(x, m, color=color, ls="-", lw=LINE_WIDTH, zorder=zorder)


def panel(ax, ub: list[Path], db: list[Path], title: str) -> None:
    if not (ub or db):
        ax.text(0.5, 0.5, "no training_metrics.csv found", ha="center", va="center",
                transform=ax.transAxes, fontsize=20, color="0.5")
    # DB first (back), UB last (front).
    for proto, runs in (("DB", db), ("UB", ub)):
        for col, _ in SPLITS:
            draw(ax, runs, col, COLORS[(proto, col)], zorder=ZORDER[proto])
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Training steps", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.set_title(title, fontsize=24, pad=6)
    ax.set_xlim(left=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--small-sgd", type=Path, default=SMALL_SGD)
    parser.add_argument("--small-dyn", type=Path, default=SMALL_DYN)
    parser.add_argument("--large-sgd", type=Path, default=LARGE_SGD)
    parser.add_argument("--large-dyn", type=Path, default=LARGE_DYN)
    parser.add_argument("--band", choices=("sd", "sem"), default=fc.BAND)
    args = parser.parse_args()
    fc.BAND = args.band

    nets = {
        "H=50": (run_dirs(args.small_sgd), run_dirs(args.small_dyn)),
        "H=500": (run_dirs(args.large_sgd), run_dirs(args.large_dyn)),
    }
    for net, (ub, db) in nets.items():
        print(f"[runs] {net}: UB {len(ub)}, DB {len(db)}")

    with plt.rc_context(fc.RC):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.76, wspace=0.12)
        panel(axes[0], *nets["H=50"], "Small network, $H=50$")
        panel(axes[1], *nets["H=500"], "Large network, $H=500$")
        axes[1].set_ylabel("")
        for ax, letter in zip(axes, "ab"):
            fc.panel_label(ax, letter, dx=-50, dy=6, fontsize=20)
        # One-line legend: UB, training | UB, test | DB, training | DB, test
        handles = [Line2D([], [], color=COLORS[(proto, col)], ls="-", lw=LINE_WIDTH + 1.0,
                          label=f"{label}, {split.lower()}")
                   for proto, label in PROTOCOLS for col, split in SPLITS]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), handlelength=1.8,
                   columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0,
                   bbox_to_anchor=(0.53, 0.99), fontsize=20, frameon=False)
        fc.save(fig, args.outdir, OUT_NAME)

    w = st.StatsWriter(f"{OUT_NAME}: accuracy, mean +/- SD across runs (global_step series)")
    for net, (ub, db) in nets.items():
        for col, split in SPLITS:
            w.section(f"{split} accuracy (%), {net}")
            summ = {}
            for proto, runs in (("UB", ub), ("DB", db)):
                summ[proto] = st.summarize(global_step_series(r, col) for r in runs)
                w.add(proto, summ[proto], scale=100.0)
            if summ["UB"]["n"] and summ["DB"]["n"]:
                w.note(f"{'DB - UB (final mean)':<44s} {100 * (summ['DB']['final_mean'] - summ['UB']['final_mean']):+.2f} points")
    w.write(args.outdir, OUT_NAME)


if __name__ == "__main__":
    main()