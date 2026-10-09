"""Switching-cost and weight-space modularity metrics.

All per-cycle quantities are computed from *flat* float vectors (one per class
per cycle), produced by the training loop with `jax.flatten_util.ravel_pytree`,
whose ordering is identical to `jax.tree_util.tree_leaves` and therefore to
`param_leaf_names` below.

Per class i and cycle c the training loop provides:
  theta_i(c)     class-specialised parameters (snapshot mode: mean over the
                 central window [P/4, 3P/4), the value at 3P/4, or end of slot)
  g_i(c)         mean *data-loss* gradient (weight decay excluded) of the actual
                 training minibatches over the central window, all parameters
  proxy_i(c)     same, restricted to one tensor (first-layer bias by default)
  u_i(c)         mean optimiser update (theta_{t+1} - theta_t) over the window
  p_i(c)         class-restricted gradient grad L_i on a fixed class-i probe set,
                 at the bump centre (control; see note below)

Interpretation notes
--------------------
* rho / rho_full / rho_update use the gradients / updates the optimiser actually
  followed. Under a closed limit cycle, sum_i u_i ~ 0, hence rho -> -1/(C-1)
  when norms are equal. Under uniform training the "class windows" are just
  consecutive time windows and there is no reason for that value.
* rho_probe (class-restricted gradients) converges to -1/(C-1) at ANY
  stationary point of the balanced loss, with or without oscillations, because
  sum_i grad L_i = 0 is the first-order optimality condition. It is a
  convergence diagnostic, not an oscillation diagnostic.
* Distances and PRs must be read against a uniform (A=1) null run at the same
  cycle: use `compare_to_null`.
"""
from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np

from model import ParamTree


EPS = 1e-12
DEFAULT_TAU = 0.1
DEFAULT_TOPK = 20

REGIME_BUMP = "bump"
REGIME_UNIFORM = "uniform"
REGIME_MIXED = "mixed"


# =============================================================================
# Flattening / layer helpers
# =============================================================================


def _flatten_param_tree(params: ParamTree) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        raise ValueError("Parameter tree has no leaves.")
    return np.concatenate([np.asarray(leaf, dtype=np.float64).reshape(-1) for leaf in leaves])


def _flatten_param_tree_with_names(params: ParamTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    if not path_leaves:
        raise ValueError("Parameter tree has no leaves.")
    vectors, names, indices = [], [], []
    for path, leaf in path_leaves:
        vec = np.asarray(leaf, dtype=np.float64).reshape(-1)
        name = "/".join(
            str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))) for k in path
        )
        vectors.append(vec)
        names.append(np.full(vec.shape[0], name, dtype=object))
        indices.append(np.arange(vec.shape[0], dtype=np.int64))
    return np.concatenate(vectors), np.concatenate(names), np.concatenate(indices)


def param_leaf_names(params: ParamTree) -> tuple[np.ndarray, np.ndarray]:
    """Leaf name and within-leaf index for every entry of the flat vector."""
    _, names, index = _flatten_param_tree_with_names(params)
    return names, index


def _layer_groups(leaf_names: np.ndarray) -> dict[str, np.ndarray]:
    return {
        layer: np.where(leaf_names == layer)[0]
        for layer in dict.fromkeys(leaf_names.tolist())
    }


def _layer_rms(vec: np.ndarray, groups: dict[str, np.ndarray]) -> np.ndarray:
    """Per-entry vector holding the RMS of `vec` over that entry's layer."""
    out = np.empty_like(vec, dtype=np.float64)
    for idx in groups.values():
        out[idx] = math.sqrt(float(np.mean(vec[idx] * vec[idx]))) if idx.size else 0.0
    return out


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _participation_ratio(delta: np.ndarray) -> float:
    """Effective number of coordinates carrying ||delta||^2 (1 .. n)."""
    sq = delta * delta
    total = float(np.sum(sq))
    if total <= EPS:
        return float("nan")
    p = sq / total
    return 1.0 / float(np.sum(p * p))


def _unit_mean_residual(vectors: list[np.ndarray]) -> float:
    """|| mean_c v_c/||v_c|| ||: 0 iff the unit vectors sum to zero."""
    units = []
    for v in vectors:
        n = float(np.linalg.norm(v))
        if n > EPS:
            units.append(v / n)
    if not units:
        return float("nan")
    m = np.mean(np.stack(units, axis=0), axis=0)
    return float(np.linalg.norm(m))


# =============================================================================
# Top-k overlap
# =============================================================================


def topk_indices(delta: np.ndarray, k: int, scale: np.ndarray | None = None) -> np.ndarray:
    score = np.abs(delta) if scale is None else np.abs(delta) / np.maximum(np.abs(scale), EPS)
    k = int(min(max(1, k), score.size))
    return np.argpartition(score, -k)[-k:]


def _overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.intersect1d(a, b).size)


def transition_displacements(
    theta_by_class: dict[int, np.ndarray], order: list[int]
) -> dict[tuple[int, int], np.ndarray]:
    """delta_{a->b} = theta_b - theta_a for consecutive classes in bump order."""
    out = {}
    n = len(order)
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        if a != b:
            out[(a, b)] = theta_by_class[b] - theta_by_class[a]
    return out


def class_displacements(
    theta_by_class: dict[int, np.ndarray], order: list[int]
) -> dict[int, np.ndarray]:
    """delta_i = theta_i - mean_k theta_k (class-centred)."""
    classes = list(dict.fromkeys(order))
    mean = np.mean([theta_by_class[c] for c in classes], axis=0)
    return {c: theta_by_class[c] - mean for c in classes}


def _keys_share_endpoint(a: Any, b: Any) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return bool(set(a) & set(b))
    return False


def topk_overlap_metrics(
    theta_now: dict[int, np.ndarray],
    theta_prev: dict[int, np.ndarray] | None,
    order: list[int],
    k: int,
    groups: dict[str, np.ndarray],
    scale_fn=None,
    prefix: str = "topk",
) -> dict[str, float]:
    """Top-k overlap (counts out of k) for one cycle.

    For each mode ("transition", "class") and normalisation ("raw",
    "layernorm"):
      <prefix>_<mode>_<norm>_across_cycles : mean |S_x(c) ∩ S_x(c-1)|
      <prefix>_<mode>_<norm>_across_keys   : mean |S_x(c) ∩ S_y(c)|, x != y

    Across-keys excludes transitions that share an endpoint (0->1 vs 1->2 both
    contain theta_1, which correlates them by construction), so it needs
    C >= 4 for "transition"; the class-centred mode needs C >= 3 (for C = 2,
    delta_0 = -delta_1 and the sets are identical by construction).

    `scale_fn(mode, key, delta) -> scale` optionally overrides normalisation
    (used by `compare_to_null` to divide by the null run's |delta|); when
    given, the "raw"/"layernorm" split is replaced by a single "scaled" norm.
    """
    out: dict[str, float] = {}
    n = next(iter(theta_now.values())).size
    out[f"{prefix}_k"] = float(min(k, n))
    out[f"{prefix}_chance"] = float(min(k, n) ** 2 / n)

    builders = {"transition": transition_displacements, "class": class_displacements}
    norms = ["scaled"] if scale_fn is not None else ["raw", "layernorm"]

    for mode, builder in builders.items():
        d_now = builder(theta_now, order)
        d_prev = builder(theta_prev, order) if theta_prev is not None else None
        for norm in norms:
            def _scale(key, delta, norm=norm, mode=mode):
                if norm == "raw":
                    return None
                if norm == "layernorm":
                    return _layer_rms(delta, groups)
                return scale_fn(mode, key, delta)

            s_now = {x: topk_indices(d, k, _scale(x, d)) for x, d in d_now.items()}
            keys = list(s_now)
            across_keys = []
            if not (mode == "class" and len(keys) < 3):
                for i, a in enumerate(keys):
                    for b in keys[i + 1:]:
                        if _keys_share_endpoint(a, b):
                            continue
                        across_keys.append(_overlap(s_now[a], s_now[b]))
            across_cycles = []
            if d_prev is not None:
                for x in keys:
                    if x in d_prev:
                        across_cycles.append(
                            _overlap(s_now[x], topk_indices(d_prev[x], k, _scale(x, d_prev[x])))
                        )
            base = f"{prefix}_{mode}_{norm}"
            out[f"{base}_across_cycles"] = float(np.mean(across_cycles)) if across_cycles else float("nan")
            out[f"{base}_across_keys"] = float(np.mean(across_keys)) if across_keys else float("nan")
    return out


def topk_column_names(prefix: str = "topk", scaled: bool = False) -> list[str]:
    cols = [f"{prefix}_k", f"{prefix}_chance"]
    norms = ["scaled"] if scaled else ["raw", "layernorm"]
    for mode in ("transition", "class"):
        for norm in norms:
            cols += [f"{prefix}_{mode}_{norm}_across_cycles", f"{prefix}_{mode}_{norm}_across_keys"]
    return cols


# =============================================================================
# Per-cycle switching metrics
# =============================================================================


PAIR_FAMILIES = ["s", "d", "dnorm", "pr", "prfrac", "amp", "overlap", "rho", "rho_full", "rho_update", "rho_probe"]
SCALAR_COLUMNS = [
    "grad_balance_residual",
    "grad_balance_residual_full",
    "update_balance_residual",
    "probe_balance_residual",
]
META_COLUMNS = ["cycle", "regime", "frac_active", "focus_frac_central_min", "snapshot_mode", "proxy_name"]
LAYERWISE_HEADER = ["cycle", "regime", "pair", "layer", "n_weights", "d", "pr", "prfrac", "s"]


@dataclass
class SwitchingCycleRaw:
    """One cycle of switching metrics.

    row: column -> value for the main CSV (see `switching_csv_columns`).
    layerwise_rows: per (pair, layer) distance / PR / s for the layer-wise CSV.
    active_masks_by_pair: boolean "switching" masks behind s(c).
    """

    row: dict[str, float]
    layerwise_rows: list[dict[str, Any]] = field(default_factory=list)
    active_masks_by_pair: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def switching_csv_columns(class_pairs: list[tuple[int, int]]) -> list[str]:
    cols = list(META_COLUMNS)
    for fam in PAIR_FAMILIES:
        cols += [f"{fam}_{l}_{r}" for l, r in class_pairs]
    cols += SCALAR_COLUMNS
    cols += topk_column_names("topk")
    return cols


def collect_switching_raw_cycle(
    theta_by_class: dict[int, np.ndarray],
    proxy_grad_by_class: dict[int, np.ndarray],
    class_pairs: list[tuple[int, int]],
    leaf_names: np.ndarray,
    tau: float = DEFAULT_TAU,
    num_classes: int | None = None,
    full_grad_by_class: dict[int, np.ndarray] | None = None,
    update_by_class: dict[int, np.ndarray] | None = None,
    probe_grad_by_class: dict[int, np.ndarray] | None = None,
    transition_order: list[int] | None = None,
    prev_theta_by_class: dict[int, np.ndarray] | None = None,
    topk: int = DEFAULT_TOPK,
) -> SwitchingCycleRaw:
    """Compute all switching metrics for one cycle from flat vectors.

    s_ij  fraction of weights with |delta_k|/2 / max(|mean_k|, RMS_layer(mean))
          > tau. The per-layer RMS floor stops near-zero weights from counting
          as "switching" for arbitrarily small displacements.
    d_ij  ||theta_i - theta_j||_2;  dnorm_ij = d_ij / ||(theta_i+theta_j)/2||.
          NB: with CE and no weight decay ||theta|| grows, so dnorm falls even
          when d is constant; compare against a null run.
    pr_ij participation ratio of delta^2 (effective # of weights), prfrac = pr/n.
          Also reported per layer in the layer-wise CSV.
    overlap_ij mean Jaccard of the s-masks between (i,j) and every pair with
          disjoint endpoints (pairs sharing a class overlap by construction).
    rho, rho_full, rho_update, rho_probe: cosines, see module docstring.
    """
    if not class_pairs:
        raise ValueError("class_pairs must contain at least one (i, j) pair.")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    needed = {c for pair in class_pairs for c in pair}
    if num_classes is not None and len(needed) < num_classes:
        warnings.warn(
            f"class_pairs covers {len(needed)}/{num_classes} classes; balance residuals "
            "are computed over the covered subset only.",
            stacklevel=2,
        )
    for name, dct in (("theta_by_class", theta_by_class), ("proxy_grad_by_class", proxy_grad_by_class)):
        missing = sorted(needed - set(dct))
        if missing:
            raise ValueError(f"Missing {name} entries for classes: {missing}")

    groups = _layer_groups(leaf_names)
    n_weights = next(iter(theta_by_class.values())).size
    row: dict[str, float] = {}
    layer_rows: list[dict[str, Any]] = []
    masks: dict[tuple[int, int], np.ndarray] = {}

    def _get(dct, c):
        return None if dct is None or c not in dct else dct[c]

    for left, right in class_pairs:
        tl, tr = theta_by_class[left], theta_by_class[right]
        if tl.shape != tr.shape:
            raise ValueError(f"Parameters for classes {left} and {right} have mismatched shapes.")
        key = f"{left}_{right}"
        delta = tl - tr
        mean_vec = 0.5 * (tl + tr)

        floor = _layer_rms(mean_vec, groups)
        rel = (np.abs(delta) / 2.0) / (np.maximum(np.abs(mean_vec), floor) + EPS)
        active = rel > tau
        masks[(left, right)] = active
        row[f"s_{key}"] = float(np.mean(active))

        d = float(np.linalg.norm(delta))
        row[f"d_{key}"] = d
        row[f"dnorm_{key}"] = d / (float(np.linalg.norm(mean_vec)) + EPS)
        pr = _participation_ratio(delta)
        row[f"pr_{key}"] = pr
        row[f"prfrac_{key}"] = pr / n_weights
        # a_ij = d_ij / sqrt(PR_ij): typical displacement of a participating weight.
        row[f"amp_{key}"] = d / math.sqrt(pr) if np.isfinite(pr) and pr > EPS else float("nan")

        row[f"rho_{key}"] = _cos(proxy_grad_by_class[left], proxy_grad_by_class[right])
        row[f"rho_full_{key}"] = _cos(_get(full_grad_by_class, left), _get(full_grad_by_class, right))
        row[f"rho_update_{key}"] = _cos(_get(update_by_class, left), _get(update_by_class, right))
        row[f"rho_probe_{key}"] = _cos(_get(probe_grad_by_class, left), _get(probe_grad_by_class, right))

        for layer, idx in groups.items():
            dl = delta[idx]
            pr_l = _participation_ratio(dl)
            layer_rows.append(
                {
                    "pair": key,
                    "layer": layer,
                    "n_weights": int(idx.size),
                    "d": float(np.linalg.norm(dl)),
                    "pr": pr_l,
                    "prfrac": pr_l / idx.size if idx.size else float("nan"),
                    "s": float(np.mean(active[idx])) if idx.size else float("nan"),
                }
            )

    for left, right in class_pairs:
        a = masks[(left, right)]
        vals = []
        for (l2, r2), b in masks.items():
            if {l2, r2} & {left, right}:
                continue
            union = np.count_nonzero(a | b)
            if union:
                vals.append(np.count_nonzero(a & b) / union)
        row[f"overlap_{left}_{right}"] = float(np.mean(vals)) if vals else float("nan")

    classes = sorted(needed)
    row["grad_balance_residual"] = _unit_mean_residual([proxy_grad_by_class[c] for c in classes])
    for col, dct in (
        ("grad_balance_residual_full", full_grad_by_class),
        ("update_balance_residual", update_by_class),
        ("probe_balance_residual", probe_grad_by_class),
    ):
        row[col] = (
            _unit_mean_residual([dct[c] for c in classes if c in dct]) if dct else float("nan")
        )

    if transition_order is not None and len(set(transition_order)) >= 2:
        row.update(
            topk_overlap_metrics(
                theta_by_class, prev_theta_by_class, list(transition_order), topk, groups, prefix="topk"
            )
        )

    return SwitchingCycleRaw(row=row, layerwise_rows=layer_rows, active_masks_by_pair=masks)


# =============================================================================
# CSV I/O
# =============================================================================


def switching_layerwise_csv_path(switching_csv_path: Path) -> Path:
    p = Path(switching_csv_path)
    return p.with_name(p.stem + "_layerwise" + p.suffix)


def initialize_switching_csv(switching_csv_path: Path, class_pairs: list[tuple[int, int]]) -> None:
    with open(switching_csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(switching_csv_columns(class_pairs))
    with open(switching_layerwise_csv_path(switching_csv_path), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(LAYERWISE_HEADER)


def append_switching_csv_row(
    switching_csv_path: Path,
    cycle: int,
    raw: SwitchingCycleRaw,
    meta: dict[str, Any] | None = None,
) -> None:
    with open(switching_csv_path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    row = {"cycle": cycle, **(meta or {}), **raw.row}
    with open(switching_csv_path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=header, restval="nan", extrasaction="ignore").writerow(row)
    regime = (meta or {}).get("regime", "")
    with open(switching_layerwise_csv_path(switching_csv_path), "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LAYERWISE_HEADER, restval="nan")
        for lr in raw.layerwise_rows:
            w.writerow({"cycle": cycle, "regime": regime, **lr})


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            casted: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    casted[key] = float("nan")
                    continue
                try:
                    casted[key] = float(value)
                except ValueError:
                    casted[key] = value
            rows.append(casted)
    return rows


_load_switching_columns = _load_csv_rows  # backwards-compatible name


def filter_rows_by_regime(rows: list[dict[str, Any]], regime_filter: str = "auto") -> tuple[list[dict[str, Any]], str]:
    """'auto': drop mixed cycles; keep bump cycles if any exist, else uniform.
    'bump' / 'uniform': keep only that regime. 'all': keep everything."""
    if regime_filter == "all" or not rows or "regime" not in rows[0]:
        return rows, "all"
    if regime_filter == "auto":
        has_bump = any(r.get("regime") == REGIME_BUMP for r in rows)
        regime_filter = REGIME_BUMP if has_bump else REGIME_UNIFORM
    return [r for r in rows if r.get("regime") == regime_filter], regime_filter


def _col(rows, name) -> np.ndarray:
    return np.asarray([r.get(name, float("nan")) for r in rows], dtype=np.float64)


def _pair_matrix(rows, family, class_pairs) -> np.ndarray:
    return np.stack([_col(rows, f"{family}_{l}_{r}") for l, r in class_pairs], axis=1)


# =============================================================================
# Plots
# =============================================================================


def _plot_pairs(ax, cycles, values, class_pairs, title, ylabel, hline=None, ylim=None):
    many = len(class_pairs) > 10
    for idx, (l, r) in enumerate(class_pairs):
        ax.plot(cycles, values[:, idx], linewidth=0.8 if many else 1.6, alpha=0.35 if many else 0.9,
                label=None if many else f"pair {l}-{r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ax.plot(cycles, np.nanmean(values, axis=1), color="black", linewidth=2.4, label="mean over pairs")
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1.4, label=r"$-1/(C-1)$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def _save(fig, output_path: Path, title: str):
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sibling(path: Path, suffix: str) -> Path:
    path = Path(path)
    return path.with_name(path.stem + suffix + path.suffix)


def finalize_switching_metrics(
    switching_csv_path: Path,
    output_path: Path,
    class_pairs: list[tuple[int, int]],
    num_classes: int,
    regime_filter: str = "auto",
) -> None:
    rows = _load_csv_rows(switching_csv_path)
    if not rows:
        print(f"Warning: No switching-cost rows found in {switching_csv_path}. Skipping switching plots.")
        return
    n_all = len(rows)
    rows, used = filter_rows_by_regime(rows, regime_filter)
    print(f"[switching] plotting {len(rows)}/{n_all} cycles (regime filter: {used}).")
    if not rows:
        print("Warning: no cycles left after regime filtering. Skipping switching plots.")
        return

    cycles = _col(rows, "cycle").astype(np.int64)
    stat = -1.0 / float(num_classes - 1) if num_classes > 1 else None
    tag = f" [{used} cycles]"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    _plot_pairs(axes[0], cycles, _pair_matrix(rows, "s", class_pairs), class_pairs,
                "Switching weight fraction $s$", "$s(c)$", ylim=(0, 1))
    _plot_pairs(axes[1], cycles, _pair_matrix(rows, "d", class_pairs), class_pairs,
                "Switching distance $d_{ij}$", "$d_{ij}(c)$")
    _plot_pairs(axes[2], cycles, _pair_matrix(rows, "rho", class_pairs), class_pairs,
                r"Window-gradient angle $\rho_{ij}$ (proxy tensor)", r"$\rho_{ij}$", hline=stat, ylim=(-1, 1))
    _save(fig, output_path, "Switching Cost Measures" + tag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    for ax, fam, title in zip(
        axes,
        ["rho", "rho_full", "rho_update", "rho_probe"],
        [
            "Window gradient, proxy tensor",
            "Window gradient, full network",
            "Window optimiser update, full network",
            "Class-restricted probe gradient (control:\n→ -1/(C-1) at any balanced stationary point)",
        ],
    ):
        _plot_pairs(ax, cycles, _pair_matrix(rows, fam, class_pairs), class_pairs, title, "cosine",
                    hline=stat, ylim=(-1, 1))
    _save(fig, _sibling(output_path, "_gradient_angles"), "Gradient / update alignment between class windows" + tag)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    _plot_pairs(axes[0], cycles, _pair_matrix(rows, "dnorm", class_pairs), class_pairs,
                "Normalised distance", r"$d_{ij}/\|\bar\theta\|$")
    _plot_pairs(axes[1], cycles, _pair_matrix(rows, "pr", class_pairs), class_pairs,
                "Participation ratio", "effective # weights")
    _plot_pairs(axes[2], cycles, _pair_matrix(rows, "prfrac", class_pairs), class_pairs,
                "Participation fraction", "PR / n", ylim=(0, 1))
    _plot_pairs(axes[3], cycles, _pair_matrix(rows, "overlap", class_pairs), class_pairs,
                "s-mask Jaccard vs disjoint pairs", "Jaccard", ylim=(0, 1))
    ax = axes[4]
    for col, lab in zip(SCALAR_COLUMNS, ["window grad (proxy)", "window grad (full)", "window update", "probe grad"]):
        ax.plot(cycles, _col(rows, col), linewidth=1.8, label=lab)
    ax.set_title("Unit-mean residual  $\\|\\mathrm{mean}_c\\, v_c/\\|v_c\\|\\|$")
    ax.set_xlabel("Cycle")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[5]
    kcol = _col(rows, "topk_k")
    k = np.nanmax(kcol) if np.any(np.isfinite(kcol)) else float("nan")
    for mode, ls in (("transition", "-"), ("class", "--")):
        for norm, color in (("raw", "tab:blue"), ("layernorm", "tab:red")):
            ax.plot(cycles, _col(rows, f"topk_{mode}_{norm}_across_cycles"), ls, color=color,
                    label=f"{mode}/{norm} across cycles")
            ax.plot(cycles, _col(rows, f"topk_{mode}_{norm}_across_keys"), ls, color=color, alpha=0.4,
                    label=f"{mode}/{norm} across keys")
    ax.plot(cycles, _col(rows, "topk_chance"), ":", color="gray", label="chance $k^2/N$")
    ax.set_title(f"Top-k overlap (k={k:g})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("shared weights (out of k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    _save(fig, _sibling(output_path, "_extra_metrics"), "Complementary switching measures" + tag)

    if (Path(switching_csv_path).parent / DECOMPOSITION_CSV).exists():
        plot_decomposition_dashboard(Path(switching_csv_path).parent, _sibling(output_path, "_decomposition"),
                                     regime_filter)


# =============================================================================
# Snapshots (needed for null-normalised comparisons)
# =============================================================================


def save_cycle_snapshot(
    snapshot_dir: Path,
    cycle: int,
    regime: str,
    theta_by_class: dict[int, np.ndarray],
    order: list[int],
    leaf_names: np.ndarray,
) -> None:
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    names_path = snapshot_dir / "leaf_names.npy"
    if not names_path.exists():
        np.save(names_path, np.asarray(leaf_names, dtype=str))
    np.savez_compressed(
        snapshot_dir / f"cycle_{cycle:06d}.npz",
        regime=np.asarray(regime),
        order=np.asarray(order, dtype=np.int64),
        **{f"class_{c}": v.astype(np.float32) for c, v in theta_by_class.items()},
    )


def load_snapshots(snapshot_dir: Path) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    snapshot_dir = Path(snapshot_dir)
    names = np.load(snapshot_dir / "leaf_names.npy").astype(object)
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(snapshot_dir.glob("cycle_*.npz")):
        cyc = int(p.stem.split("_")[1])
        z = np.load(p)
        theta = {int(k.split("_")[1]): z[k].astype(np.float64) for k in z.files if k.startswith("class_")}
        out[cyc] = {"regime": str(z["regime"]), "order": z["order"].tolist(), "theta": theta}
    return out, names


# =============================================================================
# Comparison against a uniform (A=1) null run
# =============================================================================


def _null_scale(ref_delta: np.ndarray, groups: dict[str, np.ndarray]) -> np.ndarray:
    ref = np.abs(ref_delta)
    return np.maximum(ref, 0.1 * _layer_rms(ref, groups))


def compare_to_null(
    bump_run_dir: Path,
    null_run_dir: Path,
    output_path: Path | None = None,
    class_pairs: list[tuple[int, int]] | None = None,
    topk: int = DEFAULT_TOPK,
) -> Path:
    """Ratios bump / null at matched cycle index for d, dnorm, PR, s, plus
    top-k overlap with ranking by |delta_bump| / |delta_null| (same key, same
    cycle; floored at 0.1 x the null's per-layer RMS).

    Both runs must share model, period_length, bump order and cycle alignment
    (true if only w_max differs). Writes `switching_vs_null.csv` and a figure
    in the bump run dir.
    """
    bump_run_dir, null_run_dir = Path(bump_run_dir), Path(null_run_dir)
    b_rows, _ = filter_rows_by_regime(_load_csv_rows(bump_run_dir / "switching_metrics.csv"), REGIME_BUMP)
    n_rows, _ = filter_rows_by_regime(_load_csv_rows(null_run_dir / "switching_metrics.csv"), REGIME_UNIFORM)
    if not b_rows or not n_rows:
        raise ValueError("Need bump cycles in the bump run and uniform cycles in the null run.")
    if class_pairs is None:
        class_pairs = []
        for c in b_rows[0]:
            parts = c.split("_")
            if len(parts) == 3 and parts[0] == "d":
                class_pairs.append((int(parts[1]), int(parts[2])))
    b_by, n_by = {int(r["cycle"]): r for r in b_rows}, {int(r["cycle"]): r for r in n_rows}
    common = sorted(set(b_by) & set(n_by))
    if not common:
        raise ValueError("No common cycle indices between bump and null runs.")

    def _pmean(r, fam):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean([r.get(f"{fam}_{l}_{rr}", np.nan) for l, rr in class_pairs]))

    fams = ("d", "dnorm", "pr", "prfrac", "amp", "s")
    out_rows = []
    for c in common:
        rec: dict[str, float] = {"cycle": c}
        for fam in fams:
            b, n = _pmean(b_by[c], fam), _pmean(n_by[c], fam)
            rec[f"{fam}_bump"], rec[f"{fam}_null"] = b, n
            rec[f"{fam}_diff"] = b - n
            # Ratios are unstable when the null is ~0 (typically s): NaN below 1e-3 x |bump|.
            ok = np.isfinite(n) and abs(n) > max(EPS, 1e-3 * abs(b))
            rec[f"{fam}_ratio"] = b / n if ok else float("nan")
            # Average log-ratios (not ratios) across seeds: a mean of ratios is
            # dominated by the few seeds where the null value dips toward 0.
            rec[f"{fam}_logratio"] = math.log(b / n) if ok and b > 0 and n > 0 else float("nan")
        out_rows.append(rec)

    b_snap_dir, n_snap_dir = bump_run_dir / "switching_snapshots", null_run_dir / "switching_snapshots"
    topk_cols: list[str] = []
    if b_snap_dir.exists() and n_snap_dir.exists():
        b_snaps, names = load_snapshots(b_snap_dir)
        n_snaps, _ = load_snapshots(n_snap_dir)
        groups = _layer_groups(names)
        topk_cols = topk_column_names("topk_nullnorm", scaled=True)
        builders = {"transition": transition_displacements, "class": class_displacements}
        by_cycle = {r["cycle"]: r for r in out_rows}
        for c in common:
            if c not in b_snaps or c not in n_snaps:
                continue
            order = b_snaps[c]["order"]
            null_now = {m: bld(n_snaps[c]["theta"], order) for m, bld in builders.items()}

            def scale_fn(mode, key, _delta, nd=null_now):
                return _null_scale(nd[mode][key], groups)

            metrics = topk_overlap_metrics(
                b_snaps[c]["theta"], None, order, topk, groups, scale_fn=scale_fn, prefix="topk_nullnorm"
            )
            # Across cycles: each cycle ranked with its own matched null scale.
            if (c - 1) in b_snaps and (c - 1) in n_snaps:
                null_prev = {m: bld(n_snaps[c - 1]["theta"], order) for m, bld in builders.items()}
                for mode, bld in builders.items():
                    d_now = bld(b_snaps[c]["theta"], order)
                    d_prev = bld(b_snaps[c - 1]["theta"], order)
                    vals = [
                        _overlap(
                            topk_indices(d_now[x], topk, _null_scale(null_now[mode][x], groups)),
                            topk_indices(d_prev[x], topk, _null_scale(null_prev[mode][x], groups)),
                        )
                        for x in d_now if x in d_prev
                    ]
                    metrics[f"topk_nullnorm_{mode}_scaled_across_cycles"] = (
                        float(np.mean(vals)) if vals else float("nan")
                    )
            by_cycle[c].update(metrics)
    else:
        print("[compare_to_null] snapshots missing in one of the runs; skipping null-normalised top-k.")

    header = ["cycle"] + [f"{f}_{s}" for f in fams for s in ("bump", "null", "ratio", "logratio", "diff")] + topk_cols
    csv_out = bump_run_dir / "switching_vs_null.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="nan", extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    output_path = Path(output_path) if output_path else bump_run_dir / "switching_vs_null.png"
    cyc = np.asarray([r["cycle"] for r in out_rows])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for ax, fam, title in zip(axes[:3], ("d", "pr", "s"), ("Distance", "Participation ratio", "Switching fraction s")):
        ax.plot(cyc, [r[f"{fam}_bump"] for r in out_rows], label="bump")
        ax.plot(cyc, [r[f"{fam}_null"] for r in out_rows], label="null (A=1)")
        ax.set_title(title + " (mean over pairs)")
        ax.set_xlabel("Cycle")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax = axes[3]
    for fam in ("d", "dnorm", "pr"):
        ax.plot(cyc, [r[f"{fam}_ratio"] for r in out_rows], label=f"{fam} bump/null")
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_title("Ratios bump / null (s: see difference panel)")
    ax.set_xlabel("Cycle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = axes[4]
    if topk_cols:
        for mode in ("transition", "class"):
            for which in ("across_cycles", "across_keys"):
                col = f"topk_nullnorm_{mode}_scaled_{which}"
                ax.plot(cyc, [r.get(col, np.nan) for r in out_rows], label=f"{mode} {which}")
        ax.plot(cyc, [r.get("topk_nullnorm_chance", np.nan) for r in out_rows], ":", color="gray", label="chance")
        ax.set_title(f"Top-{topk} overlap, ranked by |Δ_bump|/|Δ_null|")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no snapshots", ha="center", va="center")
    ax.set_xlabel("Cycle")
    ax.grid(True, alpha=0.3)
    ax = axes[5]
    ax.plot(cyc, [r["s_diff"] for r in out_rows], label="s bump - null")
    ax.plot(cyc, [r["prfrac_diff"] for r in out_rows], label="PR/n bump - null")
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_title("Differences bump - null")
    ax.set_xlabel("Cycle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_path, "Bump run relative to uniform null")
    print(f"[compare_to_null] wrote {csv_out} and {output_path}")
    return csv_out


# =============================================================================
# Switch / scaffold decomposition
# =============================================================================
#
# Per weight k and cycle c, with class states theta_i(c) (i in bump order) and
# cycle mean  m_k(c) = mean_i theta_i,k(c):
#
#   drift       D_k(c) = m_k(c) - m_k(c-1)            irreversible, between cycles
#   oscillation u_i,k(c) = theta_i,k(c) - m_k(c) - D_k(c) * (pos_i - (C-1)/2) / C
#               a_k(c) = sqrt(mean_i u_i,k(c)^2)       reversible, within the cycle
#
# The drift term inside u removes the part of the class deviations that is just
# steady drift across the C sequential class windows (pos_i = position of class
# i in the bump order). Without it, a uniformly drifting UB run shows a
# spurious, perfectly repeatable "switching" pattern (-1, 0, +1) x drift.
#
# Equivalently, u removes a straight line alpha_k + beta_k * pos_i from the C class
# states, with the slope FIXED by the between-cycle drift, beta_k = D_k(c) / C,
# and the intercept chosen so the residuals have zero mean. This is the default
# `detrend="drift"`. Two alternatives are available for comparison:
#
#   detrend="mean"    u_i,k = theta_i,k - m_k            (no trend removal: steady
#                     drift leaks into a_k as (pos_i - (C-1)/2) * D_k / C)
#   detrend="linfit"  u_i,k = residual of a least-squares line fitted to the C
#                     class states of THIS cycle (slope estimated within the cycle)
#
# "linfit" is NOT equivalent to "drift": the fitted slope absorbs every genuine
# class pattern that is linear in cycle position. It removes 2 of the C degrees
# of freedom (1 more than the mean), so only C-2 residual directions survive:
#   C = 2  u = 0 identically (undefined)
#   C = 3  u is always proportional to (1, -2, 1): a_k^2 = (theta_0 - 2 theta_1 +
#          theta_2)^2 / 18; a weight separating the first and last class in the
#          order, (+1, 0, -1), gets a_k = 0, and sel_k is the same constant for
#          every weight (ownership becomes meaningless). The result also depends
#          on which class the bump order happens to start with.
#   large C  an isotropic oscillation loses ~1/(C-1) of its energy.
# The "drift" slope uses information from OUTSIDE the cycle (the previous
# cycle mean), which is why it can separate drift from oscillation at all:
# under a closed limit cycle D_k = 0 whatever the oscillation amplitude.
#
# Switching set S(c): the fewest weights that carry `energy_q` (default 80%) of
# the total oscillation energy sum_k a_k^2. Scaffold = every other weight.
#
# Metrics (one row per cycle):
#   switch_frac          |S| / n                      how many weights carry the switch
#   switch_persist       Jaccard(S(c), S(c-1))        is it the same set each cycle?
#   switch_persist_chance  expected Jaccard of random sets of the same sizes
#   osc_rms_switch       RMS_{k in S} a_k             how far a switching weight moves
#   osc_rms_scaffold     RMS_{k not in S} a_k         does the scaffold move during a switch?
#   drift_rms_switch     RMS_{k in S} |D_k|           do switching weights stay put on average?
#   drift_rms_scaffold   RMS_{k not in S} |D_k|       is the scaffold still learning?
#   osc_rel_switch       osc_rms_switch / RMS_{k in S} |m_k|   scale-free switch amplitude
#   osc_rel_scaffold     same for the scaffold
#   osc_energy_total, drift_energy_total   sum_k a_k^2, sum_k D_k^2
# Layer-wise rows: fraction of each tensor in S and its share of the
# oscillation / drift energy (compare with its share of the weights).
#
# Class ownership ("how many weights does each class have?")
# ----------------------------------------------------------
# Gini(sel) in the modularity suite says that DB produces a subset of strongly
# class-selective weights, but not how many per class; and the modularity
# "owned" sets are the top (1 - quantile) of weights per class by
# construction, so their size cannot change. This measure counts instead:
#
#   sel_k  = max_i |u_i,k| / ||u_.,k||      selectivity of weight k (same drift-
#                                           corrected deviations u as above)
#   dom_k  = argmax_i |u_i,k|               its dominant class
#   owned by class i  <=>  sel_k > t_C  and  dom_k = i
#
# where t_C is the `own_q` (default 95th) percentile of sel_k for class-less
# noise (i.i.d. Gaussian class states, centred; depends only on C). Pure noise
# therefore marks (1 - own_q) of all weights as owned, split evenly over the
# classes: that is the chance line of every ownership metric.
#
#   owned_frac               owned weights / n, all classes together
#   owned_frac_of_switch     fraction of the switching set S that is class-owned
#   owned_in_switch_frac     fraction of the owned weights that lie inside S
#   (+ chance values), and per class in class_ownership.csv:
#   owned_frac (of n), owned_count, owned_in_switch_count, owned_share_of_switch
#
# owned_frac_of_switch is what reconciles "fewer weights change" with "more
# weights per class": if the few switching weights are the class-owned ones,
# it is high (>> 1 - own_q) while |S|/n stays small.
#
# The opposite tail is counted too: a weight is "shared" if sel_k is BELOW the
# (1 - own_q) noise percentile, i.e. several classes move it with comparable
# strength (for C = 3 this is the pairwise-contrast pattern (+1, -1, 0): a
# weight separating two classes). Gini(sel) rises when EITHER tail grows, so
# owned_* and shared_* together say which one it is. Chance is (1 - own_q) for both.

DECOMPOSITION_CSV = "switch_decomposition.csv"
DECOMPOSITION_LAYERWISE_CSV = "switch_decomposition_layerwise.csv"
DEFAULT_ENERGY_Q = 0.8
DEFAULT_OWNERSHIP_Q = 0.8
CLASS_OWNERSHIP_CSV = "class_ownership.csv"
CLASS_OWNERSHIP_HEADER = [
    "cycle", "regime", "class", "n_weights", "switch_size", "owned_count", "owned_frac",
    "owned_in_switch_count", "owned_share_of_switch", "owned_frac_chance",
]
DECOMPOSITION_COLUMNS = [
    "cycle", "regime", "detrended", "energy_q", "n_weights",
    "switch_frac", "switch_persist", "switch_persist_chance",
    "osc_rms_switch", "osc_rms_scaffold", "drift_rms_switch", "drift_rms_scaffold",
    "osc_rel_switch", "osc_rel_scaffold", "osc_energy_total", "drift_energy_total",
    "own_q", "sel_threshold", "owned_frac", "owned_frac_chance",
    "owned_frac_of_switch", "owned_frac_of_switch_chance",
    "owned_in_switch_frac", "owned_in_switch_chance",
    "shared_threshold", "shared_frac", "shared_frac_of_switch", "shared_frac_chance",
]
DECOMPOSITION_LAYER_HEADER = [
    "cycle", "regime", "layer", "n_weights", "weight_share",
    "frac_in_switch_set", "osc_energy_share", "drift_energy_share",
]

DETREND_DRIFT = "drift"     # paper definition: slope D_k(c)/C from the previous cycle
DETREND_LINFIT = "linfit"   # least-squares line through the C class states of the cycle
DETREND_MEAN = "mean"       # cycle mean only, no trend removal
DETREND_MODES = (DETREND_DRIFT, DETREND_LINFIT, DETREND_MEAN)
DEFAULT_DETREND = DETREND_DRIFT
DETREND_COMPARISON_CSV = "switch_detrend_comparison.csv"


def _check_detrend(detrend: str) -> str:
    detrend = str(detrend).strip().lower()
    if detrend not in DETREND_MODES:
        raise ValueError(f"detrend must be one of {DETREND_MODES}, got {detrend!r}.")
    return detrend


def decomposition_file_names(detrend: str = DEFAULT_DETREND) -> tuple[str, str, str]:
    """(main, layerwise, class-ownership) CSV names for one detrending mode.
    The default mode keeps the historical names, so existing files stay valid."""
    detrend = _check_detrend(detrend)
    names = (DECOMPOSITION_CSV, DECOMPOSITION_LAYERWISE_CSV, CLASS_OWNERSHIP_CSV)
    if detrend == DEFAULT_DETREND:
        return names
    return tuple(str(Path(n).with_name(Path(n).stem + f"_{detrend}" + Path(n).suffix)) for n in names)


_LINFIT_WARNED: set[int] = set()


def oscillation_residuals(
    theta_now: dict[int, np.ndarray],
    classes: list[int],
    drift: np.ndarray | None,
    detrend: str = DEFAULT_DETREND,
) -> np.ndarray:
    """u_i,k (rows in cycle-position order) for the chosen detrending mode."""
    detrend = _check_detrend(detrend)
    C = len(classes)
    theta = np.stack([np.asarray(theta_now[c], dtype=np.float64) for c in classes], axis=0)
    U = theta - theta.mean(axis=0, keepdims=True)
    pc = np.arange(C, dtype=np.float64) - (C - 1) / 2.0          # centred positions
    if detrend == DETREND_DRIFT:
        if drift is not None:
            U -= np.outer(pc / C, drift)
    elif detrend == DETREND_LINFIT:
        if C <= 2:
            raise ValueError("detrend='linfit' needs C >= 3 (for C = 2 the residuals are identically 0).")
        if C == 3 and C not in _LINFIT_WARNED:
            _LINFIT_WARNED.add(C)
            warnings.warn("detrend='linfit' with C = 3 leaves one residual direction, (1, -2, 1): "
                          "class ownership is degenerate and first-vs-last class switching is invisible.",
                          stacklevel=3)
        beta = (pc @ U) / float(pc @ pc)                            # OLS slope per weight
        U -= np.outer(pc, beta)
    return U


def _energy_set(energy: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of the fewest entries whose energy sums to >= q * total."""
    total = float(energy.sum())
    mask = np.zeros(energy.size, dtype=bool)
    if total <= EPS:
        return mask
    order = np.argsort(energy)[::-1]
    m = int(np.searchsorted(np.cumsum(energy[order]), q * total)) + 1
    mask[order[:min(m, energy.size)]] = True
    return mask


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def _random_jaccard(s1: int, s2: int, n: int) -> float:
    inter = s1 * s2 / n if n else 0.0
    union = s1 + s2 - inter
    return inter / union if union > 0 else float("nan")


_SEL_NULL_CACHE: dict[tuple[int, float, bool], float] = {}


def selectivity_null_threshold(num_classes: int, q: float = DEFAULT_OWNERSHIP_Q,
                               n_samples: int = 400_000, seed: int = 0, linfit: bool = False) -> float:
    """q-quantile of sel_k = max_i|u_i|/||u|| when the C class states are
    i.i.d. Gaussian (then centred, and linearly detrended if `linfit`).
    Scale-free, so it depends only on C."""
    key = (int(num_classes), round(float(q), 6), bool(linfit))
    if key not in _SEL_NULL_CACHE:
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((key[0], n_samples))
        u -= u.mean(axis=0, keepdims=True)
        if linfit:
            pc = np.arange(key[0], dtype=np.float64) - (key[0] - 1) / 2.0
            u -= np.outer(pc, (pc @ u) / float(pc @ pc))
        sel = np.abs(u).max(axis=0) / (np.sqrt((u * u).sum(axis=0)) + EPS)
        _SEL_NULL_CACHE[key] = float(np.quantile(sel, key[1]))
    return _SEL_NULL_CACHE[key]


def switch_decomposition(
    theta_now: dict[int, np.ndarray],
    theta_prev: dict[int, np.ndarray] | None,
    order: list[int],
    leaf_names: np.ndarray,
    energy_q: float = DEFAULT_ENERGY_Q,
    prev_switch_mask: np.ndarray | None = None,
    own_q: float = DEFAULT_OWNERSHIP_Q,
    detrend: str = DEFAULT_DETREND,
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """One cycle of the switch/scaffold decomposition and class ownership (see
    block comment). theta_prev must be the immediately preceding cycle of the
    same regime (else pass None: drift is then NaN and no detrending is applied).
    `detrend` selects how u_i,k is built ("drift", "linfit", "mean"; see block
    comment). Drift metrics are reported for every mode when theta_prev exists.
    Returns (row, layer_rows, switch_mask, class_rows)."""
    classes = list(dict.fromkeys(order))
    C = len(classes)
    mean_now = np.mean([theta_now[c] for c in classes], axis=0)
    drift = None
    if theta_prev is not None:
        drift = mean_now - np.mean([theta_prev[c] for c in classes], axis=0)

    detrend = _check_detrend(detrend)
    U = oscillation_residuals(theta_now, classes, drift, detrend)   # u_i,k, rows = cycle position
    osc_energy = np.mean(U * U, axis=0)                  # a_k^2
    S = _energy_set(osc_energy, energy_q)
    scaffold = ~S
    n = mean_now.size

    row: dict[str, float] = {
        "detrended": float(drift is not None) if detrend == DETREND_DRIFT else float(detrend == DETREND_LINFIT),
        "energy_q": float(energy_q),
        "n_weights": float(n),
        "switch_frac": float(S.mean()),
        "osc_rms_switch": math.sqrt(float(osc_energy[S].mean())) if S.any() else float("nan"),
        "osc_rms_scaffold": math.sqrt(float(osc_energy[scaffold].mean())) if scaffold.any() else float("nan"),
        "osc_energy_total": float(osc_energy.sum()),
    }
    mag_s, mag_f = _rms(mean_now[S]), _rms(mean_now[scaffold])
    row["osc_rel_switch"] = row["osc_rms_switch"] / mag_s if mag_s > EPS else float("nan")
    row["osc_rel_scaffold"] = row["osc_rms_scaffold"] / mag_f if mag_f > EPS else float("nan")
    if drift is not None:
        row["drift_rms_switch"] = _rms(drift[S])
        row["drift_rms_scaffold"] = _rms(drift[scaffold])
        row["drift_energy_total"] = float(np.sum(drift * drift))
    else:
        row["drift_rms_switch"] = row["drift_rms_scaffold"] = row["drift_energy_total"] = float("nan")
    if prev_switch_mask is not None and prev_switch_mask.shape == S.shape:
        union = np.count_nonzero(S | prev_switch_mask)
        row["switch_persist"] = np.count_nonzero(S & prev_switch_mask) / union if union else float("nan")
        row["switch_persist_chance"] = _random_jaccard(int(S.sum()), int(prev_switch_mask.sum()), n)
    else:
        row["switch_persist"] = row["switch_persist_chance"] = float("nan")

    layer_rows = []
    tot_osc = float(osc_energy.sum())
    tot_drift = float(np.sum(drift * drift)) if drift is not None else 0.0
    for layer, idx in _layer_groups(leaf_names).items():
        layer_rows.append({
            "layer": layer,
            "n_weights": int(idx.size),
            "weight_share": idx.size / n,
            "frac_in_switch_set": float(S[idx].mean()) if idx.size else float("nan"),
            "osc_energy_share": float(osc_energy[idx].sum()) / tot_osc if tot_osc > EPS else float("nan"),
            "drift_energy_share": (float(np.sum(drift[idx] ** 2)) / tot_drift
                                   if drift is not None and tot_drift > EPS else float("nan")),
        })

    # ---- class ownership
    norm = np.sqrt(np.sum(U * U, axis=0))
    moving = norm > EPS
    sel = np.zeros_like(norm)
    sel[moving] = np.abs(U[:, moving]).max(axis=0) / norm[moving]
    dom = np.argmax(np.abs(U), axis=0)
    lin = detrend == DETREND_LINFIT
    thr = selectivity_null_threshold(C, own_q, linfit=lin) if C >= 2 else float("nan")
    owned = moving & (sel > thr)
    n_s, n_owned = int(S.sum()), int(owned.sum())
    chance = 1.0 - own_q
    row.update({
        "own_q": float(own_q),
        "sel_threshold": thr,
        "owned_frac": n_owned / n,
        "owned_frac_chance": chance,
        "owned_frac_of_switch": np.count_nonzero(owned & S) / n_s if n_s else float("nan"),
        "owned_frac_of_switch_chance": chance,
        "owned_in_switch_frac": np.count_nonzero(owned & S) / n_owned if n_owned else float("nan"),
        "owned_in_switch_chance": n_s / n,
    })
    thr_low = selectivity_null_threshold(C, 1.0 - own_q, linfit=lin) if C >= 2 else float("nan")
    shared = moving & (sel < thr_low)
    row.update({
        "shared_threshold": thr_low,
        "shared_frac": float(shared.mean()),
        "shared_frac_of_switch": np.count_nonzero(shared & S) / n_s if n_s else float("nan"),
        "shared_frac_chance": chance,
    })
    class_rows = []
    for pos, c in enumerate(classes):
        mine = owned & (dom == pos)
        in_s = int(np.count_nonzero(mine & S))
        class_rows.append({
            "class": c,
            "n_weights": n,
            "switch_size": n_s,
            "owned_count": int(mine.sum()),
            "owned_frac": float(mine.sum()) / n,
            "owned_in_switch_count": in_s,
            "owned_share_of_switch": in_s / n_s if n_s else float("nan"),
            "owned_frac_chance": chance / C,
        })
    return row, layer_rows, S, class_rows


def initialize_decomposition_csv(run_dir: Path, detrend: str = DEFAULT_DETREND) -> None:
    run_dir = Path(run_dir)
    main, layer, own = decomposition_file_names(detrend)
    with open(run_dir / main, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(DECOMPOSITION_COLUMNS)
    with open(run_dir / layer, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(DECOMPOSITION_LAYER_HEADER)
    with open(run_dir / own, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(CLASS_OWNERSHIP_HEADER)


def append_decomposition_rows(run_dir: Path, cycle: int, regime: str, row: dict, layer_rows: list[dict],
                              class_rows: list[dict] | None = None, detrend: str = DEFAULT_DETREND) -> None:
    run_dir = Path(run_dir)
    main, layer, own = decomposition_file_names(detrend)
    with open(run_dir / main, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=DECOMPOSITION_COLUMNS, restval="nan", extrasaction="ignore").writerow(
            {"cycle": cycle, "regime": regime, **row})
    with open(run_dir / layer, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECOMPOSITION_LAYER_HEADER, restval="nan")
        for lr in layer_rows:
            w.writerow({"cycle": cycle, "regime": regime, **lr})
    if class_rows:
        with open(run_dir / own, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CLASS_OWNERSHIP_HEADER, restval="nan")
            for cr in class_rows:
                w.writerow({"cycle": cycle, "regime": regime, **cr})


def decomposition_is_current(run_dir: Path, detrend: str = DEFAULT_DETREND) -> bool:
    """True if the decomposition CSV of this detrending mode exists and already
    has every current column (older files lack the class-ownership columns)."""
    main, _, own = decomposition_file_names(detrend)
    path = Path(run_dir) / main
    if not path.exists() or not (Path(run_dir) / own).exists():
        return False
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    return all(c in header for c in DECOMPOSITION_COLUMNS)


def switch_decomposition_from_snapshots(run_dir: Path, energy_q: float = DEFAULT_ENERGY_Q,
                                        own_q: float = DEFAULT_OWNERSHIP_Q,
                                        detrend: str = DEFAULT_DETREND) -> Path:
    """Post-hoc version for runs that already have switching_snapshots/:
    writes switch_decomposition*.csv (or switch_decomposition_<detrend>*.csv
    for a non-default mode) next to switching_metrics.csv, identical to what
    the training loop writes online. No retraining needed."""
    run_dir = Path(run_dir)
    snaps, names = load_snapshots(run_dir / "switching_snapshots")
    initialize_decomposition_csv(run_dir, detrend)
    prev_cycle, prev_regime, prev_theta, prev_mask = None, None, None, None
    for cyc in sorted(snaps):
        s = snaps[cyc]
        linked = prev_cycle == cyc - 1 and prev_regime == s["regime"]
        row, lrows, mask, crows = switch_decomposition(
            s["theta"], prev_theta if linked else None, s["order"], names, energy_q,
            prev_mask if linked else None, own_q, detrend,
        )
        append_decomposition_rows(run_dir, cyc, s["regime"], row, lrows, crows, detrend)
        prev_cycle, prev_regime, prev_theta, prev_mask = cyc, s["regime"], s["theta"], mask
    return run_dir / decomposition_file_names(detrend)[0]


def detrend_comparison_from_snapshots(
    run_dir: Path,
    modes: tuple[str, ...] = DETREND_MODES,
    energy_q: float = DEFAULT_ENERGY_Q,
    own_q: float = DEFAULT_OWNERSHIP_Q,
    reference: str = DEFAULT_DETREND,
) -> Path:
    """One pass over switching_snapshots/ that writes the decomposition CSVs of
    every mode in `modes` plus switch_detrend_comparison.csv, one row per cycle:
      switch_frac_<m>, switch_persist_<m>, switch_persist_chance_<m>
      jaccard_vs_<reference>_<m>   Jaccard(S_m(c), S_reference(c)), same cycle
      osc_energy_<m>               sum_k a_k^2 under mode m
      energy_kept_<m>              osc_energy_<m> / osc_energy_mean (share of the
                                   raw class variance the detrending leaves)
    Modes that need a previous cycle ("drift") fall back to "mean" on the first
    cycle of a regime, exactly as online."""
    run_dir = Path(run_dir)
    modes = tuple(dict.fromkeys(_check_detrend(m) for m in (reference, *modes)))
    snaps, names = load_snapshots(run_dir / "switching_snapshots")
    for m in modes:
        initialize_decomposition_csv(run_dir, m)
    header = ["cycle", "regime", "num_classes"]
    for m in modes:
        header += [f"switch_frac_{m}", f"switch_persist_{m}", f"switch_persist_chance_{m}",
                   f"jaccard_vs_{reference}_{m}", f"osc_energy_{m}", f"energy_kept_{m}"]
    out_path = run_dir / DETREND_COMPARISON_CSV
    prev_cycle = prev_regime = prev_theta = None
    prev_mask: dict[str, np.ndarray | None] = {m: None for m in modes}
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="nan")
        w.writeheader()
        for cyc in sorted(snaps):
            s = snaps[cyc]
            linked = prev_cycle == cyc - 1 and prev_regime == s["regime"]
            classes = list(dict.fromkeys(s["order"]))
            raw_energy = float(np.sum(np.var(np.stack([s["theta"][c] for c in classes]), axis=0)))
            rec: dict[str, Any] = {"cycle": cyc, "regime": s["regime"], "num_classes": len(classes)}
            masks = {}
            for m in modes:
                if m == DETREND_LINFIT and len(classes) <= 2:
                    continue
                row, lrows, mask, crows = switch_decomposition(
                    s["theta"], prev_theta if linked else None, s["order"], names, energy_q,
                    prev_mask[m] if linked else None, own_q, m,
                )
                append_decomposition_rows(run_dir, cyc, s["regime"], row, lrows, crows, m)
                masks[m] = mask
                prev_mask[m] = mask
                rec[f"switch_frac_{m}"] = row["switch_frac"]
                rec[f"switch_persist_{m}"] = row["switch_persist"]
                rec[f"switch_persist_chance_{m}"] = row["switch_persist_chance"]
                rec[f"osc_energy_{m}"] = row["osc_energy_total"]
                rec[f"energy_kept_{m}"] = row["osc_energy_total"] / raw_energy if raw_energy > EPS else float("nan")
            ref = masks.get(reference)
            for m, mask in masks.items():
                union = np.count_nonzero(mask | ref) if ref is not None else 0
                rec[f"jaccard_vs_{reference}_{m}"] = np.count_nonzero(mask & ref) / union if union else float("nan")
            w.writerow(rec)
            prev_cycle, prev_regime, prev_theta = cyc, s["regime"], s["theta"]
    return out_path


def plot_decomposition_dashboard(run_dir: Path, output_path: Path | None = None, regime_filter: str = "auto",
                                 detrend: str = DEFAULT_DETREND) -> None:
    """Single-run view of switch_decomposition.csv (the multi-run DB vs UB
    comparison lives in switching_plots_full.py)."""
    run_dir = Path(run_dir)
    rows, used = filter_rows_by_regime(
        _load_csv_rows(run_dir / decomposition_file_names(detrend)[0]), regime_filter)
    if not rows:
        return
    cyc = _col(rows, "cycle")
    fig, axes = plt.subplots(1, 5, figsize=(27, 5))
    axes[0].plot(cyc, _col(rows, "switch_frac"))
    axes[0].set_title(f"switching-set size |S|/n ({int(100 * rows[0]['energy_q'])}% of osc. energy)")
    axes[0].set_ylim(0, 1)
    axes[1].plot(cyc, _col(rows, "switch_persist"), label="S(c) vs S(c-1)")
    axes[1].plot(cyc, _col(rows, "switch_persist_chance"), ":", color="gray", label="random sets")
    axes[1].set_title("switching-set persistence (Jaccard)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    for ax, what, title in ((axes[2], "osc", "oscillation per weight (RMS a_k)"),
                            (axes[3], "drift", "drift per weight per cycle (RMS |D_k|)")):
        ax.plot(cyc, _col(rows, f"{what}_rms_switch"), color="tab:red", label="switching set")
        ax.plot(cyc, _col(rows, f"{what}_rms_scaffold"), color="tab:blue", label="scaffold")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[4].plot(cyc, _col(rows, "owned_frac_of_switch"), color="tab:purple", label="class-owned share of S")
    axes[4].plot(cyc, _col(rows, "owned_frac"), color="tab:gray", label="class-owned share of all weights")
    axes[4].plot(cyc, _col(rows, "shared_frac_of_switch"), color="tab:brown", label="shared share of S")
    axes[4].plot(cyc, _col(rows, "owned_frac_chance"), ":", color="gray", label="noise (chance)")
    axes[4].set_ylim(0, 1)
    axes[4].set_title(f"class-owned weights (sel > noise q{int(100 * rows[0].get('own_q', 0.95))})")
    axes[4].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Cycle")
        ax.grid(True, alpha=0.3)
    default_out = run_dir / Path(decomposition_file_names(detrend)[0]).with_suffix(".png")
    _save(fig, output_path or default_out, f"Switch / scaffold decomposition, detrend={detrend} [{used} cycles]")


# =============================================================================
# Weight-space modularity (all classes at once, not pairwise)
# =============================================================================


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = float(x.sum())
    if n == 0 or total <= EPS:
        return float("nan")
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


MODULARITY_CSV_NAME = "modularity_metrics.csv"
MODULARITY_HEADER = [
    "cycle",
    "regime",
    "gini_sel",
    "top1pct_variance_share",
    "modularity_Q",
    "effective_rank_classes",
    "cross_class_owned_jaccard_mean",
    "num_owned_total",
    "num_classes",
    "n_weights",
]
MODULARITY_LAYERWISE_CSV_NAME = "modularity_layerwise.csv"
MODULARITY_LAYERWISE_HEADER = ["cycle", "regime", "layer", "class", "n_weights", "mean_sel", "owned_frac"]


def modularity_csv_path(base_dir: Path) -> Path:
    return Path(base_dir) / MODULARITY_CSV_NAME


def modularity_layerwise_csv_path(base_dir: Path) -> Path:
    return Path(base_dir) / MODULARITY_LAYERWISE_CSV_NAME


@dataclass
class ModularityCycleRaw:
    """sel_k = max_i|theta_i,k - mean_k| / ||theta_.,k - mean_k||_2 in [1/sqrt(C), 1];
    gini_sel, top1pct_variance_share, bipartite modularity Q of the
    weight-class ownership graph, effective rank of the C class states,
    cross-class Jaccard of owned sets, and a layer-wise breakdown."""

    gini_sel: float
    top1pct_variance_share: float
    modularity_Q: float
    effective_rank_classes: float
    cross_class_owned_jaccard_mean: float
    num_owned_total: int
    num_classes: int
    n_weights: int
    classes: list[int]
    owned_mask_by_class: dict[int, np.ndarray]
    sel_by_layer: dict[str, float]
    owned_frac_by_layer_class: dict[tuple[str, int], float]
    n_weights_by_layer: dict[str, int]
    param_leaf_names: np.ndarray


def collect_modularity_raw_cycle(
    theta_by_class: dict[int, np.ndarray],
    leaf_names: np.ndarray,
    quantile: float = 0.9,
) -> ModularityCycleRaw:
    classes = sorted(theta_by_class)
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes to compute modularity metrics.")
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")

    theta = np.stack([theta_by_class[c] for c in classes], axis=0)
    n_weights = theta.shape[1]
    mean_vec = theta.mean(axis=0)
    centered = theta - mean_vec[None, :]
    abs_centered = np.abs(centered)
    l2norm = np.sqrt(np.sum(centered * centered, axis=0)) + EPS
    sel = abs_centered.max(axis=0) / l2norm
    gini_sel = _gini(sel)

    per_weight_variance = np.sum(centered * centered, axis=0)
    total_variance = float(per_weight_variance.sum()) + EPS
    k1pct = max(1, int(round(0.01 * n_weights)))
    top_idx = np.argpartition(per_weight_variance, -k1pct)[-k1pct:]
    top1pct_variance_share = float(per_weight_variance[top_idx].sum()) / total_variance

    owned = {}
    for ci, c in enumerate(classes):
        thr = float(np.quantile(abs_centered[ci], quantile))
        owned[c] = abs_centered[ci] > thr
    num_owned_total = int(sum(m.sum() for m in owned.values()))

    jac = []
    for a in range(num_classes):
        for b in range(a + 1, num_classes):
            ma, mb = owned[classes[a]], owned[classes[b]]
            union = np.count_nonzero(ma | mb)
            if union:
                jac.append(np.count_nonzero(ma & mb) / union)
    cross_jac = float(np.mean(jac)) if jac else float("nan")

    owned_stack = np.stack([owned[c] for c in classes], axis=0)
    d_i = owned_stack.sum(axis=1).astype(np.float64)
    d_k = owned_stack.sum(axis=0).astype(np.float64)
    m = float(d_i.sum())
    if m <= EPS:
        modularity_Q = float("nan")
    else:
        g_k = np.argmax(np.where(owned_stack, abs_centered, -np.inf), axis=0)
        g_k = np.where(d_k > 0, g_k, -1)
        q = 0.0
        for ci in range(num_classes):
            comm = g_k == ci
            q += float(owned_stack[ci, comm].sum()) - d_i[ci] * float(d_k[comm].sum()) / m
        modularity_Q = q / m

    cov = (centered @ centered.T) / float(n_weights)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s_sq = float(np.sum(eig * eig))
    eff_rank = float(eig.sum()) ** 2 / s_sq if s_sq > EPS else float("nan")

    sel_by_layer, n_by_layer, owned_frac = {}, {}, {}
    for layer, idx in _layer_groups(leaf_names).items():
        n_by_layer[layer] = int(idx.size)
        sel_by_layer[layer] = float(sel[idx].mean()) if idx.size else float("nan")
        for c in classes:
            owned_frac[(layer, c)] = float(owned[c][idx].mean()) if idx.size else float("nan")

    return ModularityCycleRaw(
        gini_sel=gini_sel,
        top1pct_variance_share=top1pct_variance_share,
        modularity_Q=modularity_Q,
        effective_rank_classes=eff_rank,
        cross_class_owned_jaccard_mean=cross_jac,
        num_owned_total=num_owned_total,
        num_classes=num_classes,
        n_weights=n_weights,
        classes=classes,
        owned_mask_by_class=owned,
        sel_by_layer=sel_by_layer,
        owned_frac_by_layer_class=owned_frac,
        n_weights_by_layer=n_by_layer,
        param_leaf_names=leaf_names,
    )


def initialize_modularity_csv(base_dir: Path) -> None:
    with open(modularity_csv_path(base_dir), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(MODULARITY_HEADER)
    with open(modularity_layerwise_csv_path(base_dir), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(MODULARITY_LAYERWISE_HEADER)


def append_modularity_csv_row(base_dir: Path, cycle: int, raw: ModularityCycleRaw, regime: str = "") -> None:
    row = [
        cycle, regime, raw.gini_sel, raw.top1pct_variance_share, raw.modularity_Q,
        raw.effective_rank_classes, raw.cross_class_owned_jaccard_mean,
        raw.num_owned_total, raw.num_classes, raw.n_weights,
    ]
    with open(modularity_csv_path(base_dir), "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)
    with open(modularity_layerwise_csv_path(base_dir), "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for layer, mean_sel in raw.sel_by_layer.items():
            for c in raw.classes:
                w.writerow([cycle, regime, layer, c, raw.n_weights_by_layer[layer], mean_sel,
                            raw.owned_frac_by_layer_class[(layer, c)]])


def finalize_modularity_metrics(
    base_dir: Path,
    output_path: Path,
    classes: list[int],
    regime_filter: str = "auto",
) -> None:
    base_dir = Path(base_dir)
    rows = _load_csv_rows(modularity_csv_path(base_dir))
    if not rows:
        print(f"Warning: No modularity rows found in {modularity_csv_path(base_dir)}. Skipping modularity plots.")
        return
    rows, used = filter_rows_by_regime(rows, regime_filter)
    if not rows:
        print("Warning: no modularity cycles left after regime filtering.")
        return
    cycles = _col(rows, "cycle").astype(np.int64)
    num_classes = int(rows[0]["num_classes"])

    layer_sel_final: dict[str, float] = {}
    lp = modularity_layerwise_csv_path(base_dir)
    if lp.exists():
        lrows = [r for r in _load_csv_rows(lp) if used == "all" or r.get("regime") == used]
        if lrows:
            last = max(int(r["cycle"]) for r in lrows)
            for r in lrows:
                if int(r["cycle"]) == last:
                    layer_sel_final.setdefault(str(r["layer"]), float(r["mean_sel"]))

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    axes = axes.ravel()
    specs = [
        ("gini_sel", "Class-selectivity distribution: Gini($sel_k$)", "Gini", "tab:blue"),
        ("top1pct_variance_share", "Variance concentration", "Top-1% share", "tab:orange"),
        ("modularity_Q", "Bipartite modularity $Q$", "$Q$", "tab:green"),
        ("effective_rank_classes", "Effective rank of class states", "PR", "tab:red"),
        ("cross_class_owned_jaccard_mean", "Cross-class owned-weight overlap", "Jaccard", "tab:purple"),
    ]
    for ax, (col, title, ylabel, color) in zip(axes, specs):
        ax.plot(cycles, _col(rows, col), linewidth=1.8, color=color)
        ax.set_title(title)
        ax.set_xlabel("Cycle")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[2].axhline(0.0, color="gray", linestyle="--")
    axes[3].axhline(num_classes, color="gray", linestyle="--", label=f"C = {num_classes}")
    axes[3].axhline(1.0, color="gray", linestyle=":", label="fully collapsed")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(fontsize=8)
    axes[4].set_ylim(0, 1)
    ax = axes[5]
    if layer_sel_final:
        layers = list(layer_sel_final)
        ax.barh(range(len(layers)), [layer_sel_final[l] for l in layers], color="tab:cyan")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        # Reference: selectivity of a weight that all C classes push equally.
        ax.axvline(1.0 / np.sqrt(max(2, num_classes)), color="gray", linestyle=":", label=r"$1/\sqrt{C}$")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No layer-wise data", ha="center", va="center")
    ax.set_title("Layer-wise mean $sel_k$ (final cycle)")
    ax.grid(True, alpha=0.3, axis="x")
    _save(fig, output_path, f"Weight-Space Modularity Measures [{used} cycles]")

# =============================================================================
# CLI: backfill existing runs from their snapshots (no retraining)
# =============================================================================


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Recompute the switch/scaffold decomposition of existing runs "
                                             "from switching_snapshots/ for one or more detrending modes.")
    ap.add_argument("folders", nargs="+", type=Path, help="run dirs, or parents searched recursively")
    ap.add_argument("--detrend", nargs="+", default=list(DETREND_MODES), choices=DETREND_MODES)
    ap.add_argument("--energy-q", type=float, default=DEFAULT_ENERGY_Q)
    ap.add_argument("--own-q", type=float, default=DEFAULT_OWNERSHIP_Q)
    args = ap.parse_args()
    runs = sorted({p.parent for f in args.folders for p in Path(f).rglob("switching_snapshots") if p.is_dir()})
    if not runs:
        raise SystemExit("No switching_snapshots/ directories found: these runs need a rerun "
                         "with analysis.switching_save_snapshots = true.")
    for r in runs:
        out = detrend_comparison_from_snapshots(r, tuple(args.detrend), args.energy_q, args.own_q)
        print(f"[detrend] {r} -> {out.name}")


if __name__ == "__main__":
    _cli()