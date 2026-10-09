"""SM figure sm_spiral_metrics: additional weight-level measures for the spiral.

One row of four panels
(a) class-restricted probe-gradient cosine, small network (H=50), with all C
    per-class gradients evaluated at the SAME parameters: at each class state
    theta_i(c) of the cycle (the stored snapshot), the C probe gradients
    grad L_k(theta_i(c)), k = 1..C, their mean pairwise cosine, then the mean
    over the C states of the cycle. Recomputed offline from switching_snapshots/
    and cached as probe_same.csv in each run dir (no rerun needed).
(b) the same, large network (H=500)
(c) participation ratio PR_ij of theta_i - theta_j, small network
(d) switching-set size |S|/n at energy thresholds of 50 / 80 / 90 %, small network
With --diagnostic, a separate figure fig_sm_probe_diagnostic.png/.pdf compares,
for both networks: the online rho_probe of switching_metrics.csv (each class
gradient at its own window centre, i.e. at different parameters), the
same-point cosine over all parameters, and the same-point cosine restricted to
the hidden layer(s) and to the readout layer.
The .txt also reports, for both networks, the share and coverage of every layer
and the hidden-unit concentration of S (N_eff and unit coverage vs. random).

Units. Every weight is attached to the hidden unit(s) it connects: incoming
kernel entries and the bias of hidden unit j belong to j, outgoing kernel entries
of j belong to j (a hidden-to-hidden weight belongs to both of its units). With
q_u the share of S-incidences on unit u, N_eff = 1 / sum_u q_u^2. The random
reference shuffles S over all weights (same size), N_SHUFFLES times per cycle.
Layer panels read switch_decomposition_layerwise.csv; the threshold and unit
panels recompute S from switching_snapshots/ (same definition as
switching_metrics_easy.switch_decomposition, default detrending) and cache the
result as switch_compactness.csv in each run dir.

Output: <outdir>/fig_sm_spiral_metrics.png/.pdf and fig_sm_spiral_metrics.txt
(mean +/- SD across runs of every plotted quantity, see fig_stats.py).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fig_common as fc
import fig_stats as st
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import switching_metrics as sm

SMALL_SGD = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_320_bumps/")     # H=50, A=1
SMALL_DYN = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-320_bumps/")    # H=50, A=20
LARGE_SGD = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_150_bumps_large/")    # EDIT: H=500, A=1
LARGE_DYN = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-150_bumps_large/")          # EDIT: H=500, A>1
NUM_CLASSES = 3
INPUT_DIM = 2                        # spiral inputs; used to infer the kernel shapes
KERNEL_LAYOUT = "in_out"             # flattened kernel is (n_in, n_out) row-major; "out_in" otherwise
THRESHOLDS = (0.5, 0.8, 0.9)         # 0.8 = the value used in the main text
N_SHUFFLES = 5
COMPACTNESS_CSV = "switch_compactness.csv"
PROBE_SAME_CSV = "probe_same.csv"
PROBE_COLS = ("rho_same", "resid_same", "rho_same_hidden", "rho_same_readout")
PROBE_SIZE_DEFAULT = 256             # analysis.switching_probe_size_per_class default in the runner
PROBE_SEED_OFFSET = 303              # the runner seeds the probe sets with random_seed + 303
DIAG_NAME = "fig_sm_probe_diagnostic"
OUT_NAME = "fig_sm_spiral_metrics"
FIG_W, FIG_H = 18.0, 4.9

LAYER_MARKERS = ["o", "s", "^", "D", "v", "P"]     # layers are told apart by marker, not colour
N_LAYER_MARKERS = 9                                # markers per line


def _empty(ax, text="no data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=18, color="0.5")


def _title(ax, text):
    ax.set_title(text, fontsize=20, pad=6)


# --------------------------------------------------------------------------- #
# Layers and hidden units
# --------------------------------------------------------------------------- #
def _layer_of(leaf: str) -> str:
    """'hidden_layers/0/kernel' -> 'hidden_layers/0'."""
    return leaf.rsplit("/", 1)[0] if "/" in leaf else leaf


def _layer_sort_key(layer: str):
    m = re.search(r"hidden_layers/(\d+)", layer)
    return (0, int(m.group(1))) if m else (1, layer)


def layer_name(layer: str, n_hidden: int = 1) -> str:
    """Legend name: 'Hidden layer' (numbered if several) or 'Classifier'."""
    m = re.search(r"hidden_layers/(\d+)", layer)
    if m:
        return "Hidden layer" if n_hidden <= 1 else f"Hidden layer {int(m.group(1)) + 1}"
    return "Classifier"


def layer_label(layer: str) -> str:
    m = re.search(r"hidden_layers/(\d+)", layer)
    if m:
        i = int(m.group(1))
        return "input → H1" if i == 0 else f"H{i} → H{i + 1}"
    return "hidden → output"


def unit_incidence(names: np.ndarray, input_dim: int = INPUT_DIM, layout: str = KERNEL_LAYOUT):
    """(weight index, hidden-unit index) pairs and the number of hidden units.
    Layer order: hidden_layers/0, 1, ..., then the output layer. Kernel shapes are
    inferred from the chain of widths starting at input_dim."""
    names = np.asarray(names).astype(str)
    leaves = list(dict.fromkeys(names.tolist()))
    start = {leaf: int(np.argmax(names == leaf)) for leaf in leaves}
    size = {leaf: int(np.sum(names == leaf)) for leaf in leaves}
    layers = sorted(dict.fromkeys(_layer_of(l) for l in leaves), key=_layer_sort_key)
    kernels = {_layer_of(l): l for l in leaves if l.endswith("kernel")}
    biases = {_layer_of(l): l for l in leaves if l.endswith("bias")}
    if set(kernels) != set(layers):
        raise ValueError(f"Could not find one kernel per layer in {leaves}.")

    w_idx, u_idx, unit_offset, n_in = [], [], 0, int(input_dim)
    prev_offset = None                           # unit offset of the previous hidden layer
    for li, layer in enumerate(layers):
        k = kernels[layer]
        if size[k] % n_in:
            raise ValueError(f"{k}: {size[k]} entries not divisible by n_in={n_in}; check INPUT_DIM.")
        n_out = size[k] // n_in
        hidden = li < len(layers) - 1
        flat = np.arange(size[k])
        i, j = (flat // n_out, flat % n_out) if layout == "in_out" else (flat % n_in, flat // n_in)
        glob = start[k] + flat
        if hidden:                               # incoming weight of unit j
            w_idx.append(glob); u_idx.append(unit_offset + j)
        if prev_offset is not None:              # outgoing weight of previous-layer unit i
            w_idx.append(glob); u_idx.append(prev_offset + i)
        if hidden and layer in biases:
            b = biases[layer]
            w_idx.append(start[b] + np.arange(size[b])); u_idx.append(unit_offset + np.arange(size[b]))
        if hidden:
            prev_offset, unit_offset = unit_offset, unit_offset + n_out
        n_in = n_out
    if n_in != NUM_CLASSES:
        print(f"[units] warning: inferred output width {n_in} != NUM_CLASSES={NUM_CLASSES}; "
              "check INPUT_DIM / KERNEL_LAYOUT.")
    return np.concatenate(w_idx), np.concatenate(u_idx), unit_offset


def _unit_stats(mask, w_idx, u_idx, n_units):
    counts = np.bincount(u_idx[mask[w_idx]], minlength=n_units).astype(float)
    tot = counts.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    q = counts / tot
    return 1.0 / float(np.sum(q * q)), float(np.mean(counts > 0))


def compactness_from_snapshots(run_dir: Path, refresh: bool = False) -> Path:
    """Per cycle: |S|/n at every threshold, and N_eff / coverage of the 80 % set
    over hidden units, with the random-set reference. Cached in the run dir."""
    run_dir = Path(run_dir)
    out = run_dir / COMPACTNESS_CSV
    cols = ["cycle", "regime", "n_hidden_units"] + [f"switch_frac_q{q:g}" for q in THRESHOLDS] + \
           ["neff", "neff_rand", "neff_ratio", "cover", "cover_rand", "cover_ratio"]
    if out.exists() and not refresh and st.read_rows(out) and set(cols) <= set(st.read_rows(out)[0]):
        return out
    snap_dir = run_dir / "switching_snapshots"
    if not snap_dir.is_dir():
        return out
    print(f"[compactness] {run_dir}")
    snaps, names = sm.load_snapshots(snap_dir)
    w_idx, u_idx, n_units = unit_incidence(names)
    rng = np.random.default_rng(0)
    rows, prev = [], None
    for cyc in sorted(snaps):
        s = snaps[cyc]
        classes = list(dict.fromkeys(s["order"]))
        linked = prev is not None and prev[0] == cyc - 1 and prev[1] == s["regime"]
        mean_now = np.mean([s["theta"][c] for c in classes], axis=0)
        drift = mean_now - np.mean([prev[2][c] for c in classes], axis=0) if linked else None
        U = sm.oscillation_residuals(s["theta"], classes, drift, sm.DEFAULT_DETREND)
        a2 = np.mean(U * U, axis=0)
        r = {"cycle": cyc, "regime": s["regime"], "n_hidden_units": n_units}
        for q in THRESHOLDS:
            r[f"switch_frac_q{q:g}"] = float(sm._energy_set(a2, q).mean())
        S = sm._energy_set(a2, 0.8)
        neff, cover = _unit_stats(S, w_idx, u_idx, n_units)
        rand = [_unit_stats(rng.permutation(S), w_idx, u_idx, n_units) for _ in range(N_SHUFFLES)]
        neff_r, cover_r = (float(np.nanmean(v)) for v in zip(*rand))
        r.update(neff=neff, neff_rand=neff_r, neff_ratio=neff / neff_r if neff_r > 0 else float("nan"),
                 cover=cover, cover_rand=cover_r, cover_ratio=cover / cover_r if cover_r > 0 else float("nan"))
        rows.append(r)
        prev = (cyc, s["regime"], s["theta"])
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out


def layer_series(runs, regime, xmax):
    """{layer: (share of S, coverage by S, share of all weights), each a list of
    per-run series}; layers merged over their kernel and bias leaves."""
    out: dict[str, tuple[list, list, list]] = {}
    for r in runs:
        per_cycle: dict[int, dict[str, list[float]]] = {}
        for row in st.read_rows(Path(r.path) / sm.DECOMPOSITION_LAYERWISE_CSV):
            if row.get("regime") != regime:
                continue
            c = int(st.num(row["cycle"]))
            if xmax is not None and c > xmax:
                continue
            nw, frac = st.num(row["n_weights"]), st.num(row["frac_in_switch_set"])
            d = per_cycle.setdefault(c, {}).setdefault(_layer_of(row["layer"]), [0.0, 0.0])
            d[0] += nw * frac                          # weights of this layer in S
            d[1] += nw                                 # weights of this layer
        share, cover, wshare = {}, {}, {}
        for c, layers in per_cycle.items():
            tot_s = sum(v[0] for v in layers.values())
            tot_w = sum(v[1] for v in layers.values())
            for layer, (in_s, nw) in layers.items():
                share.setdefault(layer, {})[c] = in_s / tot_s if tot_s > 0 else float("nan")
                cover.setdefault(layer, {})[c] = in_s / nw if nw > 0 else float("nan")
                wshare.setdefault(layer, {})[c] = nw / tot_w if tot_w > 0 else float("nan")
        for layer in share:
            out.setdefault(layer, ([], [], []))
            out[layer][0].append(share[layer])
            out[layer][1].append(cover[layer])
            out[layer][2].append(wshare[layer])
    return dict(sorted(out.items(), key=lambda kv: _layer_sort_key(kv[0])))


# --------------------------------------------------------------------------- #
# Probe gradients at the same parameters (offline, from the snapshots)
# --------------------------------------------------------------------------- #
def load_run_config(run_dir: Path) -> dict:
    """Config the run was launched with. EDIT if your runner stores it elsewhere."""
    import json
    run_dir = Path(run_dir)
    for name in ("config.json", "resolved_config.json", "run_config.json"):
        if (run_dir / name).exists():
            return json.loads((run_dir / name).read_text())
    for name in ("config.yaml", "config.yml", "run_config.yaml"):
        if (run_dir / name).exists():
            import yaml
            return yaml.safe_load((run_dir / name).read_text())
    summ = run_dir / "summary.json"
    if summ.exists():
        d = json.loads(summ.read_text())
        if isinstance(d, dict) and isinstance(d.get("config"), dict):
            return d["config"]
    raise FileNotFoundError(f"No run config found in {run_dir}; edit load_run_config().")


def _probe_sets(train_x, train_y, config):
    """Exact replica of training_runner_easy._build_probe_sets (same seed, same size)."""
    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("random_seed", config.get("random_seed", 42)))
    size = int(config.get("analysis", {}).get("switching_probe_size_per_class", PROBE_SIZE_DEFAULT))
    rng = np.random.default_rng(seed + PROBE_SEED_OFFSET)
    out = {}
    for c in range(NUM_CLASSES):
        idx = np.where(train_y == c)[0]
        if idx.size == 0:
            continue
        chosen = rng.choice(idx, size=min(size, idx.size), replace=False)
        out[c] = (train_x[chosen], train_y[chosen])
    return out


def _mean_pair_cos(G: np.ndarray) -> float:
    v = [sm._cos(G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G))]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def probe_same_from_snapshots(run_dir: Path, refresh: bool = False) -> Path:
    """Per cycle, at each stored class state theta_i(c): the C class-restricted
    probe gradients at that same point. Columns (means over the C states):
      rho_same          mean pairwise cosine, all parameters
      resid_same        || mean_k g_k/||g_k|| ||  (0 when the unit gradients cancel)
      rho_same_hidden   mean pairwise cosine, hidden-layer parameters only
      rho_same_readout  mean pairwise cosine, readout (last) layer only"""
    run_dir = Path(run_dir)
    out = run_dir / PROBE_SAME_CSV
    cols = ["cycle", "regime", *PROBE_COLS]
    if out.exists() and not refresh:
        rows = st.read_rows(out)
        if rows and set(cols) <= set(rows[0]):
            return out
    snap_dir = run_dir / "switching_snapshots"
    if not snap_dir.is_dir():
        return out
    print(f"[probe-same] {run_dir}")

    import jax
    import jax.numpy as jnp
    import optax
    from jax.flatten_util import ravel_pytree
    from generate_dataset import build_dataset_bundle
    from model import build_model

    config = load_run_config(run_dir)
    data = build_dataset_bundle(config, run_dir)
    built = build_model(model_cfg=config.get("model", {}), input_shape=data.input_shape,
                        num_classes=NUM_CLASSES, random_seed=0)   # values are overwritten below
    model = built.model
    template, unravel = ravel_pytree(built.params)

    tx, ty = np.asarray(data.train_inputs), np.asarray(data.train_targets)
    probes = {c: (jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.int32))
              for c, (x, y) in _probe_sets(tx, ty, config).items()}

    def loss(p, x, y):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model.apply(p, x), y))

    grad_flat = jax.jit(lambda flat, x, y: ravel_pytree(jax.grad(loss)(unravel(flat), x, y))[0])

    snaps, names = sm.load_snapshots(snap_dir)
    if names.size != template.size:
        raise ValueError(f"{run_dir}: snapshot has {names.size} weights, rebuilt model {template.size}; "
                         "check the model config.")
    layers = np.array([_layer_of(str(n)) for n in names])
    is_out = layers == _readout_layer(list(dict.fromkeys(layers.tolist())))

    rows = []
    for cyc in sorted(snaps):
        snap = snaps[cyc]
        acc = {k: [] for k in PROBE_COLS}
        for c in dict.fromkeys(snap["order"]):
            flat = jnp.asarray(snap["theta"][c], dtype=template.dtype)
            G = np.stack([np.asarray(grad_flat(flat, *probes[k]), dtype=np.float64) for k in sorted(probes)])
            acc["rho_same"].append(_mean_pair_cos(G))
            acc["resid_same"].append(sm._unit_mean_residual(list(G)))
            acc["rho_same_hidden"].append(_mean_pair_cos(G[:, ~is_out]))
            acc["rho_same_readout"].append(_mean_pair_cos(G[:, is_out]))
        r = {"cycle": cyc, "regime": snap["regime"]}
        r.update({k: float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan") for k, v in acc.items()})
        rows.append(r)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out


def probe_series(runs, regime, xmax, refresh=False):
    """{column: list of per-run cycle series} from probe_same.csv."""
    paths = [probe_same_from_snapshots(r.path, refresh) for r in runs]
    paths = [p for p in paths if p.exists()]
    return {c: [s for s in (st.cycle_series(p.parent, p.name, regime, st.column(c), xmax) for p in paths) if s]
            for c in PROBE_COLS}


def _draw_band(ax, series, color, dynamical, scale=1.0, lw=None, marker=None):
    mb = st.mean_band(series, fc.BAND)
    if mb is None:
        return float("nan")
    x, m, lo, hi = mb
    ax.fill_between(x, scale * lo, scale * hi, color=color, alpha=0.15, lw=0)
    mk = {} if marker is None else dict(marker=marker, ms=6, mfc="white" if not dynamical else color,
                                        mec=color, mew=1.3, markevery=max(1, len(x) // N_LAYER_MARKERS))
    ax.plot(x, scale * m, color=color, ls="-" if dynamical else "--",
            lw=lw or (fc.LW_DYN if dynamical else fc.LW_STD), **mk)
    return float(np.nanmax(scale * hi))


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _probe_axes(ax, xmax, title, ylabel="Mean pairwise cosine"):
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    fc.ref_line(ax, -1.0 / (NUM_CLASSES - 1), r"$-1/(C-1)$")
    ax.set_ylim(-0.65, -0.30)
    ax.set_ylabel(ylabel)
    _title(ax, title)
    fc.cycle_axis(ax, xmax or 1)


def panel_probe(ax, p_ub, p_db, xmax, title, col="rho_same"):
    """Same-point probe-gradient cosine. p_ub / p_db: probe_series() dicts."""
    if not (p_ub.get(col) or p_db.get(col)):
        _empty(ax, "no switching_snapshots/ or run config")
    for series, color, dyn in ((p_ub.get(col) or [], fc.SGD_COLOR, False), (p_db.get(col) or [], fc.DYN_COLOR, True)):
        if series:
            _draw_band(ax, series, color, dyn)
    _probe_axes(ax, xmax, title)


def panel_probe_online(ax, ub, db, xmax, title):
    """Old metric: each class gradient at its own window centre (different parameters)."""
    if not (ub or db):
        _empty(ax, "folders not found")
    fc.draw(ax, fc.curve(ub, "avg/rho_probe", max_cycle=xmax), dynamical=False)
    fc.draw(ax, fc.curve(db, "avg/rho_probe", max_cycle=xmax), dynamical=True, markers=False)
    _probe_axes(ax, xmax, title)


def panel_pr(ax, ub, db, xmax):
    cu, cd = fc.curve(ub, "avg/pr", max_cycle=xmax), fc.curve(db, "avg/pr", max_cycle=xmax)
    fc.draw(ax, cu, dynamical=False)
    fc.draw(ax, cd, dynamical=True, markers=False)
    ax.set_ylim(0, fc.top_of(cu, cd))
    ax.set_ylabel(r"$\mathrm{PR}_{ij}$")
    _title(ax, "Participation ratio, $H=50$")
    fc.cycle_axis(ax, xmax)


def panel_thresholds(ax, ub, db, xmax):
    for q in THRESHOLDS:
        main = abs(q - 0.8) < 1e-9
        for runs, dyn, regime in ((ub, False, fc.UB_REGIME), (db, True, fc.DB_REGIME)):
            if main:
                c = fc.curve(runs, "dec/switch_frac", max_cycle=xmax)
            else:
                c = fc.curve(runs, f"q{q}", max_cycle=xmax,
                             transform=lambda r, q=q, g=regime: fc.switch_fraction_from_snapshots(r.path, g, q))
            fc.draw(ax, c, dynamical=dyn, band=main, markers=False, lw=None if main else 1.1)
    ax.text(0.97, 0.97, "thick: 80% (main text)\nthin: 50% (below), 90% (above)", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color="0.1",
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$|\mathcal{S}|/n$")
    _title(ax, "Switching-set size vs. threshold\n$H=50$")
    fc.cycle_axis(ax, xmax)


def _readout_layer(layers: list[str]) -> str:
    """The output layer (last in network order)."""
    return sorted(layers, key=_layer_sort_key)[-1]


def panel_layers(ax, lay_ub, lay_db, xmax, net: str):
    """Share of S per layer, protocol colours (UB dashed, DB solid). Two weight
    layers: only the readout share (the other is 100 % minus it), with its share
    of all weights dotted. More layers: one line per layer, told apart by marker."""
    layers = sorted(dict.fromkeys([*lay_ub, *lay_db]), key=_layer_sort_key)
    if not layers:
        _empty(ax, "no switch_decomposition_layerwise.csv")
        _title(ax, rf"Composition of $\mathcal{{S}}$, ${net}$")
        return
    if len(layers) == 2:
        out = _readout_layer(layers)
        for lay, color, dyn in ((lay_ub, fc.SGD_COLOR, False), (lay_db, fc.DYN_COLOR, True)):
            if out in lay:
                _draw_band(ax, lay[out][0], color, dyn, scale=100.0)
        ref = next((lay[out][2] for lay in (lay_db, lay_ub) if out in lay), None)
        mb = st.mean_band(ref, fc.BAND) if ref else None
        if mb is not None:
            w_share = 100.0 * float(np.nanmean(mb[1]))
            ax.axhline(w_share, color="0.45", lw=1.0, ls=":", zorder=0)
            ax.text(0.02, w_share / 100.0 - 0.02, f"readout = {w_share:.0f}% of all weights",
                    transform=ax.transAxes, fontsize=14, color="0.4", va="top")
        ax.set_ylabel(r"Share of $\mathcal{S}$ in readout layer (%)")
    else:
        handles = []
        for li, layer in enumerate(layers):
            mk = LAYER_MARKERS[li % len(LAYER_MARKERS)]
            for lay, color, dyn in ((lay_ub, fc.SGD_COLOR, False), (lay_db, fc.DYN_COLOR, True)):
                if layer in lay:
                    _draw_band(ax, lay[layer][0], color, dyn, scale=100.0, marker=mk)
            handles.append(Line2D([], [], color="0.35", ls="none", marker=mk, ms=7, label=layer_label(layer)))
        ax.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
        ax.set_ylabel(r"Share of $\mathcal{S}$ in layer (%)")
    ax.set_ylim(0, 100)
    _title(ax, rf"Composition of $\mathcal{{S}}$, ${net}$")
    fc.cycle_axis(ax, xmax or 1)


def panel_coverage(ax, lay_ub, lay_db, xmax, net: str):
    """Percentage of each layer's weights that are in S: one line per
    (protocol, layer). Colour / line style = protocol, marker shape = layer."""
    layers = sorted(dict.fromkeys([*lay_ub, *lay_db]), key=_layer_sort_key)
    if not layers:
        _empty(ax, "no switch_decomposition_layerwise.csv")
    n_hidden = sum(1 for l in layers if _layer_sort_key(l)[0] == 0)
    handles, tops = [], []
    for li, layer in enumerate(layers):
        mk = LAYER_MARKERS[li % len(LAYER_MARKERS)]
        for lay, color, dyn in ((lay_ub, fc.SGD_COLOR, False), (lay_db, fc.DYN_COLOR, True)):
            if layer in lay:
                tops.append(_draw_band(ax, lay[layer][1], color, dyn, scale=100.0, marker=mk))
        handles.append(Line2D([], [], color="0.3", ls="none", marker=mk, ms=7, mfc="0.3",
                              label=layer_name(layer, n_hidden)))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=16, frameon=False)
    top = max([t for t in tops if np.isfinite(t)], default=100.0)
    ax.set_ylim(0, min(100.0, 1.25 * top))
    ax.set_ylabel(r"Layer weights in $\mathcal{S}$ (%)")
    _title(ax, rf"Layer coverage by $\mathcal{{S}}$, ${net}$")
    fc.cycle_axis(ax, xmax or 1)


def panel_units(ax, series_ub, series_db, xmax, ylabel, title):
    if not (series_ub or series_db):
        _empty(ax, "no switching_snapshots/")
    tops = [_draw_band(ax, series_ub, fc.SGD_COLOR, False), _draw_band(ax, series_db, fc.DYN_COLOR, True)]
    ax.axhline(1.0, color="0.45", lw=1.0, ls=":", zorder=0)
    ax.text(0.02, 0.03, "1 = random set of the same size", transform=ax.transAxes, fontsize=14,
            color="0.4", va="bottom")
    ax.set_ylim(0, max([1.25] + [1.1 * t for t in tops if np.isfinite(t)]))
    ax.set_ylabel(ylabel)
    _title(ax, title)
    fc.cycle_axis(ax, xmax)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def _pair_mean(prefix: str):
    pat = re.compile(rf"^{prefix}_\d+_\d+$")

    def f(r):
        v = [st.num(x) for k, x in r.items() if pat.match(k or "")]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")
    return f


PROBE_LABELS = {
    "rho_same": "SAME-POINT probe-gradient cosine, all parameters (panels a, b)",
    "resid_same": "SAME-POINT unit-gradient balance residual ||mean_k g_k/||g_k|| || (0 = cancel)",
    "rho_same_hidden": "SAME-POINT probe-gradient cosine, hidden-layer parameters only",
    "rho_same_readout": "SAME-POINT probe-gradient cosine, readout-layer parameters only",
}


def write_stats(outdir, groups, xmax, xmax_l, lay, comp, probe):
    w = st.StatsWriter(f"{OUT_NAME}: mean +/- SD across runs")
    for net in ("H=50", "H=500"):
        for col in PROBE_COLS:
            w.section(f"{PROBE_LABELS[col]}, {net}")
            for proto in ("UB", "DB"):
                w.add(proto, st.summarize(probe[net][proto].get(col, [])))
    for (key, label, csv_name, fn, big) in (
        ("rho", "ONLINE probe-gradient cosine rho_probe, each class at its own window centre (mean over pairs)", "switching_metrics.csv", _pair_mean("rho_probe"), False),
        ("rho", "ONLINE probe-gradient cosine rho_probe, each class at its own window centre (mean over pairs)", "switching_metrics.csv", _pair_mean("rho_probe"), True),
        ("pr", "participation ratio PR_ij (mean over pairs)", "switching_metrics.csv", _pair_mean("pr"), False),
    ):
        net = "H=500" if big else "H=50"
        w.section(f"{label}, {net}")
        for proto in ("UB", "DB"):
            runs, regime = groups[(proto, big)]
            w.add(proto, st.summarize(st.cycle_series(r.path, csv_name, regime, fn, xmax_l if big else xmax)
                                       for r in runs))
    w.section("switching-set size |S|/n vs. energy threshold, H=50")
    for q in THRESHOLDS:
        for proto in ("UB", "DB"):
            w.add(f"{proto}  q={q:g}", st.summarize(comp["H=50"][proto][f"switch_frac_q{q:g}"]))
    for net in ("H=50", "H=500"):
        w.section(f"composition of S: share of |S| in each layer (%), {net}")
        for proto in ("UB", "DB"):
            for layer, (share, _, wsh) in lay[net][proto].items():
                w.add(f"{proto}  {layer_label(layer)}", st.summarize(share), scale=100.0)
        for layer, (_, _, wsh) in (lay[net]["DB"] or lay[net]["UB"]).items():
            w.add(f"     {layer_label(layer)}: share of all weights", st.summarize(wsh), scale=100.0)
        w.section(f"layer coverage: fraction of the layer's weights in S (%), {net}")
        for proto in ("UB", "DB"):
            for layer, (_, cover, _) in lay[net][proto].items():
                w.add(f"{proto}  {layer_label(layer)}", st.summarize(cover), scale=100.0)
        w.section(f"hidden-unit concentration of S (80 % set), {net}")
        for col, label in (("neff", "N_eff (effective hidden units)"), ("neff_rand", "N_eff, random set"),
                           ("neff_ratio", "N_eff / N_eff random"), ("cover", "unit coverage"),
                           ("cover_rand", "unit coverage, random set"), ("cover_ratio", "coverage / random")):
            for proto in ("UB", "DB"):
                w.add(f"{proto}  {label}", st.summarize(comp[net][proto][col]))
    w.write(outdir, OUT_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--small-sgd", type=Path, default=SMALL_SGD)
    parser.add_argument("--small-dyn", type=Path, default=SMALL_DYN)
    parser.add_argument("--large-sgd", type=Path, default=LARGE_SGD)
    parser.add_argument("--large-dyn", type=Path, default=LARGE_DYN)
    parser.add_argument("--band", choices=("sd", "sem"), default=fc.BAND)
    parser.add_argument("--refresh", action="store_true", help=f"recompute {COMPACTNESS_CSV}")
    parser.add_argument("--refresh-probe", action="store_true", help=f"recompute {PROBE_SAME_CSV}")
    parser.add_argument("--diagnostic", action="store_true", help=f"also write {DIAG_NAME} (not for the paper)")
    args = parser.parse_args()
    fc.BAND = args.band

    ub = fc.load_folder(args.small_sgd, fc.UB_REGIME, "UB (H=50)")
    db = fc.load_folder(args.small_dyn, fc.DB_REGIME, "DB (H=50)")
    try:
        ub_l = fc.load_folder(args.large_sgd, fc.UB_REGIME, "UB (H=500)")
        db_l = fc.load_folder(args.large_dyn, fc.DB_REGIME, "DB (H=500)")
    except FileNotFoundError as exc:
        print(f"[warn] {exc}; panel (b) left empty.")
        ub_l, db_l = [], []
    xmax = fc.last_cycle(db or ub, "sw/d_0_1") or 1
    xmax_l = fc.last_cycle(db_l or ub_l, "sw/d_0_1") if (db_l or ub_l) else None

    # layer and hidden-unit measures, both networks
    nets = {"H=50": (ub, db, xmax), "H=500": (ub_l, db_l, xmax_l)}
    cols = [f"switch_frac_q{q:g}" for q in THRESHOLDS] + ["neff", "neff_rand", "neff_ratio",
                                                          "cover", "cover_rand", "cover_ratio"]
    lay, comp, probe = {}, {}, {}
    for net, (u, d, xm) in nets.items():
        probe[net] = {"UB": probe_series(u, fc.UB_REGIME, xm, args.refresh_probe),
                      "DB": probe_series(d, fc.DB_REGIME, xm, args.refresh_probe)}
        lay[net] = {"UB": layer_series(u, fc.UB_REGIME, xm), "DB": layer_series(d, fc.DB_REGIME, xm)}
        comp[net] = {}
        for proto, runs, regime in (("UB", u, fc.UB_REGIME), ("DB", d, fc.DB_REGIME)):
            paths = [compactness_from_snapshots(r.path, args.refresh) for r in runs]
            comp[net][proto] = {c: [s for s in (st.cycle_series(p.parent, p.name, regime, st.column(c), xm)
                                                for p in paths) if s] for c in cols}

    with plt.rc_context(fc.RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(1, 4, left=0.055, right=0.99, bottom=0.14, top=0.78, wspace=0.34)
        axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
        panel_probe(axes[0], probe["H=50"]["UB"], probe["H=50"]["DB"], xmax, "Probe gradients, $H=50$")
        panel_probe(axes[1], probe["H=500"]["UB"], probe["H=500"]["DB"], xmax_l, "Probe gradients, $H=500$")
        panel_pr(axes[2], ub, db, xmax)
        panel_thresholds(axes[3], ub, db, xmax)
        for ax, letter in zip(axes, "abcd"):
            fc.panel_label(ax, letter, dx=-50, dy=6, fontsize=20)

        handles = [Line2D([], [], color=fc.SGD_COLOR, ls="--", lw=fc.LW_STD, label=fc.SGD_LABEL),
                   Line2D([], [], color=fc.DYN_COLOR, ls="-", lw=fc.LW_DYN, label=fc.DYN_LABEL)]
        fig.legend(handles=handles, loc="upper center", ncol=2, handlelength=2.4, columnspacing=3.8,
                   borderaxespad=0.0, bbox_to_anchor=(0.52, 0.99), fontsize=20)
        fc.save(fig, args.outdir, OUT_NAME)

    groups = {("UB", False): (ub, fc.UB_REGIME), ("DB", False): (db, fc.DB_REGIME),
              ("UB", True): (ub_l, fc.UB_REGIME), ("DB", True): (db_l, fc.DB_REGIME)}
    write_stats(args.outdir, groups, xmax, xmax_l, lay, comp, probe)

    # Diagnostic (not for the paper): online vs same-point, and the layer split.
    if not args.diagnostic:
        return
    with plt.rc_context(fc.RC):
        fig, axd = plt.subplots(2, 4, figsize=(18.0, 8.6))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.86, wspace=0.32, hspace=0.45)
        for row, (net, u, d, xm) in enumerate((("H=50", ub, db, xmax), ("H=500", ub_l, db_l, xmax_l))):
            n = net
            panel_probe_online(axd[row, 0], u, d, xm, f"Online (own window centre), ${n}$")
            panel_probe(axd[row, 1], probe[net]["UB"], probe[net]["DB"], xm, f"Same point, all params, ${n}$")
            panel_probe(axd[row, 2], probe[net]["UB"], probe[net]["DB"], xm, f"Same point, hidden layer, ${n}$",
                        col="rho_same_hidden")
            panel_probe(axd[row, 3], probe[net]["UB"], probe[net]["DB"], xm, f"Same point, readout, ${n}$",
                        col="rho_same_readout")
        fig.legend(handles=handles, loc="upper center", ncol=2, handlelength=2.4, columnspacing=3.8,
                   borderaxespad=0.0, bbox_to_anchor=(0.52, 0.99), fontsize=20)
        fc.save(fig, args.outdir, DIAG_NAME)


if __name__ == "__main__":
    main()