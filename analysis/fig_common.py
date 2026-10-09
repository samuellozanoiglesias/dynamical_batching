"""Shared helpers for fig3_subpanels_v3.py, fig_sm_large.py and fig_sm_spiral_metrics.py.

Every ``training_*`` sub-folder of an input folder is scanned for
``switching_metrics.csv``; its directory is one run, and the sibling files
written by training_runner_easy.py / switching_metrics_easy.py are read from
the same place:

  switching_metrics.csv               per-cycle pair metrics (d, pr, rho_full, rho_probe, ...)
  switch_decomposition.csv            |S|/n, persistence, oscillation, drift, ...
  switch_decomposition_layerwise.csv  oscillation-energy share per tensor
  switching_vs_null.csv               DB runs only: top-k ranked relative to UB
  switching_snapshots/                class states per cycle (threshold robustness)

Only cycles of the expected regime are kept (UB: "uniform", DB: "bump");
"mixed" cycles (e.g. the uniform tail at the end of a DB run) are dropped.
Cycle indices are absolute in both protocols, so DB and UB are matched by
cycle index: no stretching or truncation.

Statistics: within each run, pair metrics are averaged over class pairs; then
mean and a band over runs are computed. The band is the same for every
measure and both protocols (BAND = "sd" or "sem"). Positive, log-scaled
quantities use geometric statistics (mean and band of log10), so the band never
crosses zero on a log axis.
"""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (clusters)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# --------------------------------------------------------------------------- #
# Style (same as fig3_subpanels_v2.py)
# --------------------------------------------------------------------------- #
SGD_COLOR, DYN_COLOR = "#E69F00", "#4DAF4A"
SGD_LABEL, DYN_LABEL = "Uniform Batching", "Dynamical Batching"
SWITCH_COLOR, SCAFFOLD_COLOR = "#d7191c", "#2b83ba"   # switching set / scaffold
CLASS_COLORS = ["#d7191c", "#808080", "#2b83ba"]       # class 1, 2, 3 (schematic)
REF_COLOR = "0.15"
LW_DYN, LW_STD, MS, BAND_ALPHA = 2.2, 1.7, 5.4, 0.25
PNG_DPI = 600

# "sd": +/- 1 standard deviation across runs; "sem": +/- 1 standard error of the mean.
# Applied identically to every measure and to both protocols.
BAND = "sd"

# Sparse markers on the Dynamical Batching curves. They are decoration only (placed
# every ~1/10 of the x-range, not at special cycles); solid vs. dashed already
# distinguishes the protocols. Set True only if the figure must survive grayscale.
DB_MARKERS = False

RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "text.usetex": False,
    "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14.5, "legend.frameon": False,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.color": "0.2", "ytick.color": "0.2",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (no Type 3)
}

LAYER_NAMES = {   # parameter leaf -> readable name
    "classifier/bias": "output biases",
    "classifier/kernel": "hidden-to-output weights",
    "hidden_layers/0/bias": "hidden biases",
    "hidden_layers/0/kernel": "input-to-hidden weights",
}

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
UB_REGIME, DB_REGIME = "uniform", "bump"
_PAIR_RE = re.compile(r"^([a-z_]+?)_(\d+)_(\d+)$")
_TEXT_COLS = {"cycle", "regime", "snapshot_mode", "proxy_name", "pair", "layer", "class"}

Series = dict[int, float]


@dataclass
class Run:
    path: Path
    series: dict[str, Series] = field(default_factory=dict)


def _num(text) -> float:
    try:
        return float(text) if text not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _keep(rows: list[dict[str, str]], regime: str) -> list[dict[str, str]]:
    if rows and "regime" in rows[0]:
        return [r for r in rows if r["regime"] == regime]
    return rows


def _add(series: dict[str, Series], rows: list[dict[str, str]], prefix: str) -> None:
    for row in rows:
        cyc = int(float(row["cycle"]))
        for col, val in row.items():
            if col is None or col in _TEXT_COLS:
                continue
            series.setdefault(prefix + col, {})[cyc] = _num(val)


def load_run(run_dir: Path, regime: str) -> Run | None:
    """Keys: sw/<col> and avg/<family> (pair-averaged) from switching_metrics.csv;
    dec/<col> from switch_decomposition.csv; lw/<col>/<layer> from its layer-wise
    file; null/<col> from switching_vs_null.csv."""
    rows = _keep(_rows(run_dir / "switching_metrics.csv"), regime)
    if not rows:
        return None
    run = Run(run_dir)
    _add(run.series, rows, "sw/")
    cycles = sorted({int(float(r["cycle"])) for r in rows})

    families: dict[str, list[str]] = {}
    for col in rows[0]:
        m = _PAIR_RE.match(col or "")
        if m:
            families.setdefault(m.group(1), []).append("sw/" + col)
    for fam, cols in families.items():
        avg = {}
        for c in cycles:
            vals = [run.series[k].get(c, np.nan) for k in cols]
            vals = [v for v in vals if np.isfinite(v)]
            avg[c] = float(np.mean(vals)) if vals else float("nan")
        run.series["avg/" + fam] = avg

    _add(run.series, _keep(_rows(run_dir / "switch_decomposition.csv"), regime), "dec/")
    for r in _keep(_rows(run_dir / "switch_decomposition_layerwise.csv"), regime):
        cyc = int(float(r["cycle"]))
        for col in ("weight_share", "osc_energy_share", "drift_energy_share", "frac_in_switch_set"):
            run.series.setdefault(f"lw/{col}/{r['layer']}", {})[cyc] = _num(r.get(col))
    _add(run.series, _rows(run_dir / "switching_vs_null.csv"), "null/")
    return run


def find_run_dirs(root: Path) -> list[Path]:
    training_dirs = sorted(p for p in Path(root).glob("training_*") if p.is_dir())
    if not training_dirs:
        raise FileNotFoundError(f"No 'training_*' folders found in {root}")
    return [p.parent for d in training_dirs for p in sorted(d.rglob("switching_metrics.csv"))]


def load_folder(root: Path, regime: str, label: str) -> list[Run]:
    runs = []
    for run_dir in find_run_dirs(root):
        run = load_run(run_dir, regime)
        if run is None:
            print(f"[warn] {run_dir}: no '{regime}' cycles; skipped.")
        else:
            runs.append(run)
    print(f"[info] {label}: {len(runs)} run(s) from {root}")
    if not runs and regime == UB_REGIME:
        print(f"[WARN] {label}: no uniform cycles found. Is this folder really an A=1 run?")
    return runs


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Curve:
    x: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: int


def curve(runs: list[Run], key: str, log: bool = False, max_cycle: int | None = None,
          transform=None) -> Curve | None:
    """Mean and band across runs of one series. `transform(run) -> Series`
    replaces the key lookup (used for derived quantities such as ratios)."""
    per_run = [transform(r) if transform else r.series.get(key) for r in runs]
    per_run = [s for s in per_run if s]
    if not per_run:
        return None
    x = np.asarray(sorted({c for s in per_run for c in s}), dtype=np.int64)
    if max_cycle is not None:
        x = x[x <= max_cycle]
    if not x.size:
        return None
    arr = np.full((len(per_run), x.size), np.nan)
    pos = {int(c): i for i, c in enumerate(x)}
    for i, s in enumerate(per_run):
        for c, v in s.items():
            if c in pos:
                arr[i, pos[c]] = v
    if log:
        arr = np.where(arr > 0, arr, np.nan)
        arr = np.log10(arr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        n_eff = np.sum(np.isfinite(arr), axis=0)
    spread = sd / np.sqrt(np.maximum(n_eff, 1)) if BAND == "sem" else sd
    lo, hi = m - spread, m + spread
    if log:
        m, lo, hi = 10 ** m, 10 ** lo, 10 ** hi
    keep = np.isfinite(m)
    return Curve(x[keep], m[keep], lo[keep], hi[keep], len(per_run))


def last_cycle(runs: list[Run], key: str) -> int | None:
    cyc = [c for r in runs for c, v in r.series.get(key, {}).items() if np.isfinite(v)]
    return max(cyc) if cyc else None


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw(ax, c: Curve | None, dynamical: bool, color: str | None = None, band: bool = True,
         lw: float | None = None, markers: bool | None = None, zorder: float = 2) -> None:
    """Dynamical: solid with sparse markers. Uniform: dashed. Band = BAND."""
    if c is None or not c.x.size:
        return
    col = color or (DYN_COLOR if dynamical else SGD_COLOR)
    if markers is None:
        markers = DB_MARKERS
    if dynamical:
        kw = {}
        if markers:
            every = 1 if len(c.x) <= 12 else int(np.ceil(len(c.x) / 10))
            kw = dict(markevery=every, marker="o", ms=MS, mfc=col, mec="0.25", mew=0.6)
        ax.plot(c.x, c.mean, "-", color=col, lw=lw or LW_DYN, zorder=zorder + 1, **kw)
    else:
        ax.plot(c.x, c.mean, "--", color=col, lw=lw or LW_STD, zorder=zorder)
    if band:
        ax.fill_between(c.x, c.lo, c.hi, color=col, alpha=BAND_ALPHA, linewidth=0, zorder=1)


def draw_chance(ax, c: Curve | None, color: str) -> None:
    if c is not None and c.x.size:
        ax.plot(c.x, c.mean, ":", color=color, lw=1.6, zorder=2)


def ref_line(ax, y: float, text: str | None = None, below: bool = True, fontsize: float = 14) -> None:
    ax.axhline(y, color=REF_COLOR, ls=":", lw=1.3, zorder=1)
    if text:
        ax.text(0.03, y - 0.03 if below else y + 0.03, text, color=REF_COLOR, fontsize=fontsize,
                transform=ax.get_yaxis_transform(), va="top" if below else "bottom", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))


def cycle_axis(ax, x_max: float, x_min: float = 0.0, label: bool = True) -> None:
    pad = max(0.35, 0.03 * (x_max - x_min))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if label:
        ax.set_xlabel("Cycle")


def panel_label(ax, letter: str, dx: float = -56, dy: float = 8, fontsize: float = 18) -> None:
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=fontsize, fontweight="bold",
                va="bottom", ha="left")


def top_of(*curves: Curve | None, pad: float = 1.12) -> float:
    vals = [np.nanmax(c.hi) for c in curves if c is not None and c.x.size]
    return (max(vals) if vals else 1.0) * pad


def condition_handles(ub: bool = True, db: bool = True) -> list[Line2D]:
    h = []
    if ub:
        h.append(Line2D([], [], color=SGD_COLOR, ls="--", lw=LW_STD, label=SGD_LABEL))
    if db:
        mk = dict(marker="o", ms=MS, mfc=DYN_COLOR, mec="0.25", mew=0.6) if DB_MARKERS else {}
        h.append(Line2D([], [], color=DYN_COLOR, ls="-", lw=LW_DYN, label=DYN_LABEL, **mk))
    return h


def chance_handle(label: str = "Random sets of equal size") -> Line2D:
    return Line2D([], [], color="0.35", ls=":", lw=1.6, label=label)


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)
    print(f"[info] saved {outdir / name}.png / .pdf")


# --------------------------------------------------------------------------- #
# Switch/scaffold decomposition from snapshots (numpy only; threshold robustness)
# --------------------------------------------------------------------------- #
def switch_fraction_from_snapshots(run_dir: Path, regime: str, energy_q: float) -> Series:
    """|S|/n per cycle at another energy threshold, recomputed from
    switching_snapshots/ exactly as in switching_metrics_easy.switch_decomposition
    (drift-corrected oscillation; previous cycle used only if consecutive and of
    the same regime)."""
    snap_dir = Path(run_dir) / "switching_snapshots"
    out: Series = {}
    prev_cycle, prev_mean = None, None
    for p in sorted(snap_dir.glob("cycle_*.npz")):
        cyc = int(p.stem.split("_")[1])
        z = np.load(p)
        if str(z["regime"]) != regime:
            prev_cycle, prev_mean = None, None
            continue
        order = [int(c) for c in z["order"]]
        classes = list(dict.fromkeys(order))
        theta = np.stack([z[f"class_{c}"].astype(np.float64) for c in classes])
        C = len(classes)
        mean = theta.mean(axis=0)
        drift = mean - prev_mean if prev_cycle == cyc - 1 and prev_mean is not None else None
        energy = np.zeros_like(mean)
        for pos in range(C):
            u = theta[pos] - mean
            if drift is not None:
                u = u - drift * (pos - (C - 1) / 2.0) / C
            energy += u * u
        total = energy.sum()
        if total > 0:
            cum = np.cumsum(np.sort(energy)[::-1])
            out[cyc] = (int(np.searchsorted(cum, energy_q * total)) + 1) / energy.size
        prev_cycle, prev_mean = cyc, mean
    return out