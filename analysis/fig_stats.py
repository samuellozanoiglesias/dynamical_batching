"""Small helpers shared by the figure scripts: read per-run CSVs, summarise a
per-cycle (or per-epoch) quantity across runs, and write the numbers next to
the figure as <name>.txt.

Summary convention (same for every metric):
  final : value at the last point of each run (last cycle <= max_cycle, or last
          epoch), then mean and SD across runs
  late  : mean of each run over its last LATE_FRAC of points, then mean and SD
          across runs (less noisy than a single cycle)
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

LATE_FRAC = 0.10
PAIR_D_COL = re.compile(r"^d_\d+_\d+$")


def num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def read_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cycle_series(run_dir: Path, csv_name: str, regime: str | None,
                 value: Callable[[dict[str, str]], float],
                 max_cycle: int | None = None) -> dict[int, float]:
    """cycle -> value for one run, keeping only rows of `regime` (if the CSV has
    a regime column) and cycles <= max_cycle."""
    out: dict[int, float] = {}
    for r in read_rows(Path(run_dir) / csv_name):
        if regime is not None and "regime" in r and r["regime"] != regime:
            continue
        c = num(r.get("cycle"))
        if not math.isfinite(c) or (max_cycle is not None and c > max_cycle):
            continue
        v = value(r)
        if math.isfinite(v):
            out[int(c)] = v
    return out


def column(name: str) -> Callable[[dict[str, str]], float]:
    return lambda r: num(r.get(name))


def pair_mean_distance(r: dict[str, str]) -> float:
    """avg/d: mean of d_i_j over class pairs (switching_metrics.csv)."""
    vals = [num(v) for k, v in r.items() if PAIR_D_COL.match(k or "")]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(series: Iterable[dict[int, float]], late_frac: float = LATE_FRAC) -> dict[str, float]:
    finals, lates = [], []
    for s in series:
        if not s:
            continue
        keys = sorted(s)
        finals.append(s[keys[-1]])
        k = max(1, int(round(late_frac * len(keys))))
        lates.append(float(np.mean([s[c] for c in keys[-k:]])))

    def ms(v):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0

    fm, fs = ms(finals)
    lm, ls = ms(lates)
    return {"n": len(finals), "final_mean": fm, "final_std": fs, "late_mean": lm, "late_std": ls}


def mean_band(series: list[dict[int, float]], band: str = "sd"):
    """x, mean, lo, hi across runs aligned by key (cycle or epoch)."""
    per: dict[int, list[float]] = {}
    for s in series:
        for c, v in s.items():
            per.setdefault(c, []).append(v)
    if not per:
        return None
    x = np.array(sorted(per))
    vals = [np.asarray(per[c]) for c in x]
    mean = np.array([v.mean() for v in vals])
    sd = np.array([v.std(ddof=1) if v.size > 1 else 0.0 for v in vals])
    if band == "sem":
        sd = sd / np.sqrt([v.size for v in vals])
    return x, mean, mean - sd, mean + sd


class StatsWriter:
    """Collects summary lines and writes them to <outdir>/<name>.txt."""

    def __init__(self, title: str, late_frac: float = LATE_FRAC):
        self.lines = [title, "",
                      "final = last point of each run; late = per-run mean over its last "
                      f"{100 * late_frac:.0f}% of points. mean +/- SD across runs (n = runs).", ""]

    def section(self, name: str) -> None:
        self.lines += ["", f"== {name} =="]

    def add(self, label: str, stats: dict[str, float], scale: float = 1.0, unit: str = "") -> None:
        if stats["n"] == 0:
            self.lines.append(f"{label:<44s} no data")
            return
        f = lambda v: f"{scale * v:.4g}{unit}"
        self.lines.append(
            f"{label:<44s} final {f(stats['final_mean'])} +/- {f(stats['final_std'])}   "
            f"late {f(stats['late_mean'])} +/- {f(stats['late_std'])}   (n={stats['n']})")

    def note(self, text: str) -> None:
        self.lines.append(text)

    def write(self, outdir: Path, name: str) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{name}.txt"
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"[stats] {path}")
        return path
