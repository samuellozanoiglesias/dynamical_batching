from __future__ import annotations

import csv
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
import pickle

import torch

from generate_dataset import DatasetBundle, build_dataset_bundle
from metrics import (
    plot_example_distribution_dynamics,
    plot_2d_decision_boundaries,
    plot_training_report,
)

from switching_metrics import (
    DEFAULT_TOPK,
    append_modularity_csv_row,
    append_switching_csv_row,
    collect_modularity_raw_cycle,
    collect_switching_raw_cycle,
    compare_to_null,
    finalize_modularity_metrics,
    finalize_switching_metrics,
    initialize_modularity_csv,
    initialize_switching_csv,
    param_leaf_names,
    save_cycle_snapshot,
    DEFAULT_ENERGY_Q,
    DEFAULT_OWNERSHIP_Q,
    DEFAULT_DETREND,
    DETREND_LINFIT,
    DETREND_MODES,
    append_decomposition_rows,
    initialize_decomposition_csv,
    switch_decomposition,
)

from model import JAXModel, ParamTree, build_model

@dataclass
class RunningStats:
    loss_sum: float
    correct_sum: int
    count_sum: int
    class_loss_sum: np.ndarray
    class_correct_sum: np.ndarray
    class_count_sum: np.ndarray


def build_class_pairs(num_classes: int) -> list[tuple[int, int]]:
    # Keep adjacency-first ordering: (0,1), (1,2), ... then remaining pairs.
    pairs: list[tuple[int, int]] = []
    for left in range(num_classes - 1):
        pairs.append((left, left + 1))
    for left in range(num_classes):
        for right in range(left + 2, num_classes):
            pairs.append((left, right))
    return pairs

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _analysis_output_enabled(
    analysis_cfg: Dict[str, Any],
    keys: tuple[str, ...],
    default: bool = True,
) -> bool:
    outputs_cfg = analysis_cfg.get("outputs", analysis_cfg.get("metrics", {}))
    if isinstance(outputs_cfg, dict):
        for key in keys:
            if key in outputs_cfg:
                return _as_bool(outputs_cfg[key])
    for key in keys:
        if key in analysis_cfg:
            return _as_bool(analysis_cfg[key])
    return default


def _normalize_plot_version(raw: Any) -> str:
    if raw is None:
        return "full"
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"", "full"}:
            return "full"
        if text == "simplified":
            return "simplified"
    raise ValueError("analysis.plot_version must be 'full' or 'simplified'.")


def _build_grad_mask(params: ParamTree, frozen_part: str | None) -> ParamTree:
    mask = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    if frozen_part == "classifier":
        mask["classifier"] = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x),
            params["classifier"],
        )
    elif frozen_part == "preclassifier":
        mask["hidden_layers"] = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x),
            params["hidden_layers"],
        )
    return mask


def _apply_grad_mask(grads: ParamTree, mask: ParamTree) -> ParamTree:
    return jax.tree_util.tree_map(lambda g, m: g * m, grads, mask)


def _parse_bump_order(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.strip() == "":
            return []
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [int(part) for part in raw]
    raise ValueError("dynamics.bump_order must be null, a list of ints, or a comma-separated string.")


def _validate_bump_order(bump_order: list[int] | None, num_classes: int) -> list[int] | None:
    if bump_order is None:
        return None
    if len(bump_order) != num_classes:
        raise ValueError(
            f"dynamics.bump_order must have length equal to num_classes ({num_classes}). "
            f"Got length {len(bump_order)}."
        )
    for idx, class_id in enumerate(bump_order):
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"dynamics.bump_order has invalid class id {class_id} at index {idx}. "
                f"Valid range is [0, {num_classes - 1}]."
            )
    return bump_order



_W_MAX_ALIASES = ("w_max", "A", "a", "amplitude", "bump_amplitude")
_W_MAX_DEFAULT = 50.0


def _resolve_w_max(dynamics_cfg: Dict[str, Any]) -> tuple[float, str]:
    """Read the bump amplitude. The old code only looked at `dynamics.w_max`
    and silently fell back to 50.0 -- so a config that spelled it `A` or put it
    elsewhere produced a bump run labelled as a uniform one."""
    found = {k: dynamics_cfg[k] for k in _W_MAX_ALIASES if k in dynamics_cfg and dynamics_cfg[k] is not None}
    if not found:
        msg = (
            f"dynamics.w_max not set (also looked for {_W_MAX_ALIASES[1:]}); "
            f"falling back to the default w_max={_W_MAX_DEFAULT}. If this was meant to be a "
            "uniform (A=1) run, set dynamics.w_max: 1."
        )
        warnings.warn(msg, stacklevel=2)
        print("WARNING:", msg)
        return _W_MAX_DEFAULT, "default"
    values = {float(v) for v in found.values()}
    if len(values) > 1:
        raise ValueError(f"Conflicting bump amplitudes in dynamics config: {found}")
    key = next(iter(found))
    return float(found[key]), f"dynamics.{key}"


def _batch_composition(w: float, batch_size: int, num_classes: int) -> tuple[int, int]:
    """(focus count, per-other-class count) for focus weight w, with the same
    int truncation as the training loop."""
    probs = np.ones(num_classes, dtype=np.float64)
    probs[0] = w
    probs /= probs.sum()
    counts = (probs * float(batch_size)).astype(np.int64)
    return int(counts[0]), int(counts[1]) if num_classes > 1 else 0


def _build_probe_sets(
    inputs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    size_per_class: int,
    seed: int,
    device: jax.Device,
) -> Dict[int, tuple[jax.Array, jax.Array]]:
    """Fixed per-class probe sets for the class-restricted gradient control."""
    if size_per_class <= 0:
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if idx.size == 0:
            continue
        chosen = rng.choice(idx, size=min(size_per_class, idx.size), replace=False)
        out[c] = (
            jax.device_put(jnp.asarray(inputs[chosen], dtype=jnp.float32), device=device),
            jax.device_put(jnp.asarray(targets[chosen], dtype=jnp.int32), device=device),
        )
    return out


def _build_class_to_indices(targets: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    return {
        class_id: np.where(targets == class_id)[0].astype(np.int64, copy=False)
        for class_id in range(num_classes)
    }


def _build_metric_subset(
    inputs: np.ndarray,
    targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    num_classes: int,
    subset_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    total_samples = int(targets.shape[0])
    subset_size = min(int(subset_size), total_samples)
    if subset_size <= 0:
        raise ValueError("analysis.metric_subset_size must be > 0")

    if subset_size < num_classes:
        chosen = rng.choice(total_samples, size=subset_size, replace=False)
        subset_inputs = inputs[chosen]
        subset_targets = targets[chosen]
        return subset_inputs, subset_targets, _build_class_to_indices(subset_targets, num_classes)

    counts = np.array(
        [len(class_to_indices[class_id]) for class_id in range(num_classes)],
        dtype=np.int64,
    )
    base = subset_size // num_classes
    per_class = np.minimum(base, counts)
    remaining = subset_size - int(per_class.sum())
    if remaining > 0:
        available = counts - per_class
        if int(available.sum()) > 0:
            extra = rng.multinomial(remaining, available / available.sum())
            per_class += extra

    chosen_indices: list[np.ndarray] = []
    for class_id, count in enumerate(per_class):
        if count <= 0:
            continue
        class_indices = class_to_indices[class_id]
        replace = int(count) > class_indices.shape[0]
        chosen = rng.choice(class_indices, size=int(count), replace=replace)
        chosen_indices.append(chosen.astype(np.int64, copy=False))

    if not chosen_indices:
        raise ValueError("Metric subset selection produced an empty sample.")

    chosen_all = np.concatenate(chosen_indices, axis=0)
    chosen_all = rng.permutation(chosen_all)
    subset_inputs = inputs[chosen_all]
    subset_targets = targets[chosen_all]
    return subset_inputs, subset_targets, _build_class_to_indices(subset_targets, num_classes)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _devices_for(platform: str) -> list[jax.Device]:
    try:
        return list(jax.devices(platform))
    except Exception:
        return []


def _pin_jax_platform(platform: str) -> None:
    normalized = str(platform).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized not in {"cpu", "cuda"}:
        return

    os.environ["JAX_PLATFORMS"] = normalized
    os.environ["JAX_PLATFORM_NAME"] = normalized
    try:
        jax.config.update("jax_platforms", normalized)
    except Exception:
        pass
    try:
        jax.config.update("jax_platform_name", normalized)
    except Exception:
        pass


def _resolve_device(device_raw: Any) -> jax.Device:
    requested = str(device_raw if device_raw is not None else "auto").strip().lower()

    if requested in {"auto", ""}:
        gpu_devices = _devices_for("gpu")
        if gpu_devices:
            return gpu_devices[0]
        _pin_jax_platform("cpu")
        return jax.devices("cpu")[0]

    if requested in {"cuda", "gpu"}:
        _pin_jax_platform("cuda")
        gpu_devices = _devices_for("gpu")
        if not gpu_devices:
            raise RuntimeError("Config requested CUDA/GPU but no GPU device is available to JAX.")
        return gpu_devices[0]

    if requested == "cpu":
        _pin_jax_platform("cpu")
        return jax.devices("cpu")[0]

    raise ValueError(f"Unsupported device '{requested}'. Use auto, cuda/gpu, or cpu.")


def _device_label(device: jax.Device) -> str:
    return f"{device.platform}:{device.id}"


def _init_stats(num_classes: int) -> RunningStats:
    return RunningStats(
        loss_sum=0.0,
        correct_sum=0,
        count_sum=0,
        class_loss_sum=np.zeros(num_classes, dtype=np.float64),
        class_correct_sum=np.zeros(num_classes, dtype=np.float64),
        class_count_sum=np.zeros(num_classes, dtype=np.float64),
    )


def _update_stats(
    stats: RunningStats,
    per_sample_loss: jax.Array,
    preds: jax.Array,
    target: jax.Array,
    num_classes: int,
    compute_per_class: bool = True,
) -> None:
    loss_np = np.asarray(per_sample_loss, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    target_np = np.asarray(target, dtype=np.int64)
    correct_np = (preds_np == target_np).astype(np.float64)

    stats.loss_sum += float(np.sum(loss_np))
    stats.correct_sum += int(np.sum(correct_np))
    stats.count_sum += int(target_np.shape[0])

    if compute_per_class:
        class_count = np.bincount(target_np, minlength=num_classes).astype(np.float64)
        class_loss = np.bincount(target_np, weights=loss_np, minlength=num_classes).astype(np.float64)
        class_correct = np.bincount(target_np, weights=correct_np, minlength=num_classes).astype(np.float64)

        stats.class_count_sum += class_count
        stats.class_loss_sum += class_loss
        stats.class_correct_sum += class_correct


def _finalize_stats(stats: RunningStats) -> Dict[str, Any]:
    denom = max(1, stats.count_sum)
    accuracy = stats.correct_sum / denom
    training_error = 1.0 - accuracy
    per_class_loss = np.full_like(stats.class_loss_sum, np.nan)
    per_class_acc = np.full_like(stats.class_correct_sum, np.nan)
    valid = stats.class_count_sum > 0
    per_class_loss[valid] = stats.class_loss_sum[valid] / stats.class_count_sum[valid]
    per_class_acc[valid] = stats.class_correct_sum[valid] / stats.class_count_sum[valid]

    return {
        "loss": stats.loss_sum / denom,
        "accuracy": accuracy,
        "training_error": training_error,
        "zero_training_error": bool(stats.count_sum > 0 and stats.correct_sum == stats.count_sum),
        "per_class_loss": per_class_loss,
        "per_class_accuracy": per_class_acc,
    }


def _compute_focus_weight(period_step: int, period_length: int, w_max: float) -> tuple[float, float]:
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / period
    return focus_weight, phase


def _compute_class_probabilities(
    step: int,
    num_classes: int,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None = None,
) -> tuple[np.ndarray, int, float, float]:
    if bump_order is None:
        focus_class = (step // period_length) % num_classes
    else:
        cycle_index = (step // period_length) % len(bump_order)
        focus_class = int(bump_order[cycle_index])
    period_step = step % period_length
    focus_weight, bump_phase = _compute_focus_weight(period_step, period_length, w_max)

    weights = np.ones(num_classes, dtype=np.float64)
    weights[focus_class] = focus_weight
    probabilities = weights / weights.sum()
    return probabilities, int(focus_class), float(focus_weight), float(bump_phase)


def _sample_batch_by_class_counts(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    class_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_counts_int = np.asarray(class_counts, dtype=np.int64)
    sampled_index_parts: list[np.ndarray] = []
    for class_id, count in enumerate(class_counts_int):
        if count <= 0:
            continue
        class_indices = class_to_indices[class_id]
        chosen = rng.choice(class_indices, size=int(count), replace=True)
        sampled_index_parts.append(chosen.astype(np.int64, copy=False))

    if not sampled_index_parts:
        raise ValueError(
            "Deterministic class_counts produced an empty batch. "
            "Increase training.batch_size or adjust dynamics settings."
        )

    sampled_indices = np.concatenate(sampled_index_parts, axis=0)
    return train_inputs[sampled_indices], train_targets[sampled_indices], class_counts_int


def _iterate_minibatches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = int(inputs.shape[0])
    for start in range(0, num_samples, batch_size):
        end = min(num_samples, start + batch_size)
        yield inputs[start:end], targets[start:end]


def _collect_pre_classifier_outputs(
    model: JAXModel,
    params: ParamTree,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        logits, intermediates = model.apply(params, x, return_intermediates=True)
        pre_classifier = intermediates.get("pre_classifier")
        if pre_classifier is None:
            raise RuntimeError("Failed to capture activations for layer 'pre_classifier'.")
        features.append(np.asarray(pre_classifier, dtype=np.float64))
        logits_list.append(np.asarray(logits, dtype=np.float64))
        labels.append(np.asarray(batch_targets, dtype=np.int64))

    if not features:
        raise ValueError("No samples provided for pre-classifier collection.")

    features_arr = np.concatenate(features, axis=0)
    logits_arr = np.concatenate(logits_list, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    if features_arr.shape[0] != inputs.shape[0]:
        raise ValueError("Collected activations do not match expected sample count.")

    return features_arr, logits_arr, labels_arr


def evaluate_arrays(
    params: ParamTree,
    predict_step: Callable[[ParamTree, jax.Array], jax.Array],
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
    num_classes: int,
    compute_per_class: bool = True,
) -> Dict[str, Any]:
    stats = _init_stats(num_classes)

    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        y = jax.device_put(jnp.asarray(batch_targets, dtype=jnp.int32), device=device)
        logits = predict_step(params, x)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        preds = jnp.argmax(logits, axis=1)
        
        _update_stats(stats, per_sample_loss, preds, y, num_classes, compute_per_class)

    return _finalize_stats(stats)


def _evaluate_per_example_correctness(
    params: ParamTree,
    predict_step: Callable[[ParamTree, jax.Array], jax.Array],
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
) -> np.ndarray:
    """Forward-only eval pass returning a 1D bool array of per-example
    correctness, in the same order as `inputs`/`targets`. `inputs`/`targets`
    should be a fixed array across calls (e.g. the held-out metric subset)
    so that repeated calls produce a consistent example ordering -- this is
    what the behavioral (switching_metrics) own/other-window, persistent-
    failure-overlap, and oscillation-flip-rate measures all need."""
    chunks: list[np.ndarray] = []
    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        logits = predict_step(params, x)
        preds = np.asarray(jnp.argmax(logits, axis=1))
        chunks.append(preds == np.asarray(batch_targets))
    if not chunks:
        return np.zeros((0,), dtype=bool)
    return np.concatenate(chunks, axis=0)


def _l2_penalty(model: JAXModel, params: ParamTree) -> jax.Array:
    total = jnp.array(0.0, dtype=jnp.float32)
    for _name, param in model.iter_named_parameters(params):
        total = total + jnp.sum(jnp.square(param))
    return total


def _classifier_weight_grads(
    pre_classifier: jax.Array,
    logits: jax.Array,
    targets: jax.Array,
    num_classes: int,
) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=1)
    one_hot = jax.nn.one_hot(targets, num_classes, dtype=logits.dtype)
    diff = probs - one_hot
    grad_kernel = jnp.einsum("bi,bj->bij", pre_classifier, diff)
    return jnp.transpose(grad_kernel, (0, 2, 1))



# =============================================================================
# Train step (built and jitted ONCE per run, not once per chunk)
# =============================================================================


def _make_train_step(
    model: JAXModel,
    optimizer: optax.GradientTransformation,
    weight_decay: float,
    num_classes: int,
) -> Callable:
    """Returns a jitted step. The gradient returned for analysis is the
    *data-loss* gradient (mean CE on the actual minibatch), before masking, before
    the optimiser, and WITHOUT the weight-decay term. The optimiser still sees
    data gradient + weight-decay gradient, exactly as before."""

    def _data_loss(p: ParamTree, x: jax.Array, y: jax.Array):
        logits, intermediates = model.apply(p, x, return_intermediates=True)
        per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(per_sample), (logits, per_sample, intermediates["pre_classifier"])

    def _penalty(p: ParamTree) -> jax.Array:
        return 0.5 * weight_decay * _l2_penalty(model, p)

    @jax.jit
    def train_step(step_params, step_opt_state, batch_x, batch_y, mask):
        (_, (logits, per_sample_loss, pre_classifier)), data_grads = jax.value_and_grad(
            _data_loss, has_aux=True
        )(step_params, batch_x, batch_y)
        if weight_decay > 0.0:
            penalty_grads = jax.grad(_penalty)(step_params)
            grads = jax.tree_util.tree_map(lambda a, b: a + b, data_grads, penalty_grads)
        else:
            grads = data_grads
        masked_grads = _apply_grad_mask(grads, mask)
        updates, next_opt_state = optimizer.update(masked_grads, step_opt_state, step_params)
        next_params = optax.apply_updates(step_params, updates)
        preds = jnp.argmax(logits, axis=1)
        classifier_grads = _classifier_weight_grads(pre_classifier, logits, batch_y, num_classes)
        return next_params, next_opt_state, per_sample_loss, preds, classifier_grads, data_grads

    return train_step


# =============================================================================
# Switching-metric bookkeeping
# =============================================================================


def _select_proxy(model: JAXModel) -> tuple[str, Callable[[ParamTree], jax.Array]]:
    """Single-tensor gradient proxy for rho. First-layer bias if it exists and
    is used; otherwise the first-layer kernel; with no hidden layer, the
    classifier. (The old code always read hidden_layers[0]['bias'], which is
    identically zero when use_bias=False and crashes with 0 hidden layers.)"""
    if model.num_hidden_layers >= 1:
        if model.use_bias:
            return "hidden_layers/0/bias", lambda g: g["hidden_layers"][0]["bias"].reshape(-1)
        return "hidden_layers/0/kernel", lambda g: g["hidden_layers"][0]["kernel"].reshape(-1)
    if model.use_bias:
        return "classifier/bias", lambda g: g["classifier"]["bias"].reshape(-1)
    return "classifier/kernel", lambda g: g["classifier"]["kernel"].reshape(-1)


_SNAPSHOT_MODES = {"central_mean", "three_quarter", "end"}


def _as_detrend_modes(value: Any) -> tuple[str, ...]:
    """analysis.switching_detrend: "drift", "linfit", "mean", a comma-separated
    string, a list, or "all". The default mode is always included."""
    if value is None:
        return (DEFAULT_DETREND,)
    if isinstance(value, str):
        value = DETREND_MODES if value.strip().lower() == "all" else value.split(",")
    return tuple(str(v).strip().lower() for v in value if str(v).strip())


class SwitchingTracker:
    """Accumulates, per class slot and per cycle, everything the switching
    metrics need, on device, and writes one CSV row per completed cycle.

    Per class window (steps with period_step in [P/4, 3P/4)):
      proxy / full  mean data-loss gradient of the actual training minibatch
      update        mean optimiser update theta_{t+1} - theta_t
      theta_sum     running sum of theta_{t+1} (for snapshot_mode=central_mean)
    Plus theta at 3P/4 and at the end of the slot, and the class-restricted
    probe gradient at the bump centre (P/2).

    Every cycle is tagged with its regime from the fraction of steps whose
    bumps were active: 'bump' (1.0), 'uniform' (0.0) or 'mixed'. Pre-TPT
    cycles (bumps_before_tpt=False), the uniform tail (last 5% of steps) and
    cycles straddling a switch are therefore identifiable and filtered at
    plot time instead of silently mixed into the curves.
    """

    def __init__(
        self,
        *,
        model: JAXModel,
        params: ParamTree,
        csv_path: Path,
        run_dir: Path,
        class_pairs: list[tuple[int, int]],
        num_classes: int,
        period_length: int,
        cycle_order: list[int],
        tau: float,
        topk: int,
        snapshot_mode: str,
        probe_sets: Dict[int, tuple[jax.Array, jax.Array]] | None,
        modularity_enabled: bool,
        modularity_quantile: float,
        save_snapshots: bool,
        energy_q: float = DEFAULT_ENERGY_Q,
        own_q: float = DEFAULT_OWNERSHIP_Q,
        detrend_modes: tuple[str, ...] = (DEFAULT_DETREND,),
    ) -> None:
        if period_length < 4:
            raise ValueError("Switching metrics need dynamics.period_length >= 4 (central window [P/4, 3P/4)).")
        if snapshot_mode not in _SNAPSHOT_MODES:
            raise ValueError(f"analysis.switching_snapshot must be one of {sorted(_SNAPSHOT_MODES)}.")
        self.model = model
        self.csv_path = Path(csv_path)
        self.run_dir = Path(run_dir)
        self.class_pairs = class_pairs
        self.num_classes = num_classes
        self.P = int(period_length)
        self.order = list(cycle_order)
        self.cycle_length = self.P * len(self.order)
        self.tau = float(tau)
        self.topk = int(topk)
        self.snapshot_mode = snapshot_mode
        self.probe_sets = probe_sets or {}
        self.modularity_enabled = modularity_enabled
        self.modularity_quantile = float(modularity_quantile)
        self.snapshot_dir = self.run_dir / "switching_snapshots" if save_snapshots else None
        if not (0.0 < float(energy_q) < 1.0):
            raise ValueError("analysis.switching_energy_quantile must be in (0, 1).")
        self.energy_q = float(energy_q)
        if not (0.0 < float(own_q) < 1.0):
            raise ValueError("analysis.switching_ownership_quantile must be in (0, 1).")
        self.own_q = float(own_q)
        # Detrending modes of the switch decomposition written online. The
        # default ("drift") always comes first and keeps switch_decomposition.csv;
        # every other mode writes switch_decomposition_<mode>.csv.
        modes = [DEFAULT_DETREND] + [str(m).strip().lower() for m in detrend_modes]
        modes = list(dict.fromkeys(modes))
        bad = [m for m in modes if m not in DETREND_MODES]
        if bad:
            raise ValueError(f"analysis.switching_detrend: unknown mode(s) {bad}; valid: {DETREND_MODES}.")
        if DETREND_LINFIT in modes and len(set(cycle_order)) <= 2:
            print("[switching] detrend 'linfit' needs C >= 3; dropping it.")
            modes.remove(DETREND_LINFIT)
        self.detrend_modes = tuple(modes)
        self.prev_switch_mask: Dict[str, np.ndarray | None] = {m: None for m in self.detrend_modes}

        self.leaf_names, _ = param_leaf_names(params)
        self.proxy_name, proxy_fn = _select_proxy(model)
        self._ravel = jax.jit(lambda t: ravel_pytree(t)[0])
        self._proxy = jax.jit(proxy_fn)

        def _probe_loss(p, x, y):
            logits = model.apply(p, x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

        self._probe_grad = jax.jit(lambda p, x, y: ravel_pytree(jax.grad(_probe_loss)(p, x, y))[0])

        self.prev_theta: Dict[int, np.ndarray] | None = None
        self.prev_regime: str | None = None
        self.prev_cycle: int | None = None
        self.logged = {"bump": 0, "uniform": 0, "mixed": 0, "skipped": 0}
        self._reset()

        initialize_switching_csv(self.csv_path, class_pairs)
        if self.modularity_enabled:
            initialize_modularity_csv(self.run_dir)
        for mode in self.detrend_modes:
            initialize_decomposition_csv(self.run_dir, mode)

    def _reset(self) -> None:
        self.acc: Dict[int, Dict[str, Any]] = {}
        self.active_steps = 0
        self.focus_frac_min = float("inf")

    def observe(
        self,
        *,
        step: int,
        bump_state: Dict[str, Any],
        params_before: ParamTree,
        params_after: ParamTree,
        data_grads: ParamTree,
        sampled_counts: np.ndarray,
    ) -> None:
        """Call once per optimiser step with the PRE-increment global step."""
        focus = int(bump_state["focus_class"])
        ps = step % self.P
        self.active_steps += int(bool(bump_state["active"]))
        a = self.acc.setdefault(focus, {"n": 0})

        if (self.P // 4) <= ps < (3 * self.P // 4):
            flat_after = self._ravel(params_after)
            update = flat_after - self._ravel(params_before)
            proxy = self._proxy(data_grads)
            full = self._ravel(data_grads)
            if a["n"] == 0:
                a["proxy"], a["full"], a["update"], a["theta_sum"] = proxy, full, update, flat_after
            else:
                a["proxy"] = a["proxy"] + proxy
                a["full"] = a["full"] + full
                a["update"] = a["update"] + update
                a["theta_sum"] = a["theta_sum"] + flat_after
            a["n"] += 1
            total = int(np.sum(sampled_counts))
            if total > 0:
                self.focus_frac_min = min(self.focus_frac_min, float(sampled_counts[focus]) / total)

        if ps == self.P // 2 and focus in self.probe_sets:
            x_c, y_c = self.probe_sets[focus]
            a["probe"] = self._probe_grad(params_before, x_c, y_c)
        if ps + 1 == (3 * self.P) // 4:
            a["theta_3q"] = self._ravel(params_after)
        if ps == self.P - 1:
            a["theta_end"] = self._ravel(params_after)

        if (step + 1) % self.cycle_length == 0:
            self._close_cycle((step + 1) // self.cycle_length - 1)

    def _theta(self, a: Dict[str, Any]) -> np.ndarray | None:
        if self.snapshot_mode == "central_mean":
            return np.asarray(a["theta_sum"], dtype=np.float64) / a["n"] if a.get("n", 0) else None
        key = "theta_3q" if self.snapshot_mode == "three_quarter" else "theta_end"
        return np.asarray(a[key], dtype=np.float64) if key in a else None

    def _close_cycle(self, cycle: int) -> None:
        frac_active = self.active_steps / float(self.cycle_length)
        regime = "bump" if frac_active >= 1.0 else ("uniform" if frac_active <= 0.0 else "mixed")
        classes = list(dict.fromkeys(self.order))
        incomplete = [c for c in classes if self.acc.get(c, {}).get("n", 0) == 0 or self._theta(self.acc[c]) is None]
        if incomplete:
            print(f"Warning: cycle {cycle} is missing class windows {incomplete}; skipping.")
            self.logged["skipped"] += 1
            self._reset()
            return

        def _mean(key):
            return {c: np.asarray(self.acc[c][key], dtype=np.float64) / self.acc[c]["n"] for c in classes}

        theta = {c: self._theta(self.acc[c]) for c in classes}
        probe = {c: np.asarray(self.acc[c]["probe"], dtype=np.float64) for c in classes if "probe" in self.acc[c]}
        # Across-cycle top-k is only meaningful between consecutive cycles of the same regime.
        prev = (
            self.prev_theta
            if (self.prev_regime == regime and self.prev_cycle == cycle - 1)
            else None
        )
        raw = collect_switching_raw_cycle(
            theta_by_class=theta,
            proxy_grad_by_class=_mean("proxy"),
            class_pairs=self.class_pairs,
            leaf_names=self.leaf_names,
            tau=self.tau,
            num_classes=self.num_classes,
            full_grad_by_class=_mean("full"),
            update_by_class=_mean("update"),
            probe_grad_by_class=probe or None,
            transition_order=self.order,
            prev_theta_by_class=prev,
            topk=self.topk,
        )
        meta = {
            "regime": regime,
            "frac_active": frac_active,
            "focus_frac_central_min": self.focus_frac_min if np.isfinite(self.focus_frac_min) else float("nan"),
            "snapshot_mode": self.snapshot_mode,
            "proxy_name": self.proxy_name,
        }
        append_switching_csv_row(self.csv_path, cycle, raw, meta)

        # Switch / scaffold decomposition (same linking rule as the top-k persistence).
        for mode in self.detrend_modes:
            dec_row, dec_layers, switch_mask, class_rows = switch_decomposition(
                theta, prev, self.order, self.leaf_names, self.energy_q,
                self.prev_switch_mask[mode] if prev is not None else None, self.own_q, mode,
            )
            append_decomposition_rows(self.run_dir, cycle, regime, dec_row, dec_layers, class_rows, mode)
            self.prev_switch_mask[mode] = switch_mask

        if self.modularity_enabled and len(classes) >= 2:
            mod = collect_modularity_raw_cycle(theta, self.leaf_names, quantile=self.modularity_quantile)
            append_modularity_csv_row(self.run_dir, cycle, mod, regime=regime)
        if self.snapshot_dir is not None:
            save_cycle_snapshot(self.snapshot_dir, cycle, regime, theta, self.order, self.leaf_names)

        self.prev_theta, self.prev_regime, self.prev_cycle = theta, regime, cycle
        self.logged[regime] += 1
        self._reset()


def train_epoch(
    params: ParamTree,
    opt_state: optax.OptState,
    train_step: Callable,
    model: JAXModel,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    batch_size: int,
    device: jax.Device,
    num_classes: int,
    global_step: int,
    steps_this_epoch: int,
    total_steps: int,
    bumps_enabled: bool,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None,
    cumulative_weight_distance: float,
    prev_params: ParamTree,
    rng: np.random.Generator,
    grad_mask: ParamTree,
    reinit_key: jax.Array,
    initial_params: ParamTree,
    compute_per_class: bool = True,
    switching_tracker: SwitchingTracker | None = None,
) -> tuple[
    ParamTree,
    optax.OptState,
    Dict[str, Any],
    int,
    Dict[str, Any],
    np.ndarray,
    float,
    ParamTree,
    jax.Array | None,
    ParamTree,
    ParamTree,
    jax.Array,
]:
    stats = _init_stats(num_classes)
    bump_state: Dict[str, Any] = {"active": False, "focus_class": -1, "focus_weight": 1.0, "phase": 0.0}
    sampled_distributions: list[np.ndarray] = []
    last_classifier_grads: jax.Array | None = None

    for _ in range(steps_this_epoch):
        in_uniform_tail = global_step >= (0.95 * float(total_steps))

        # Pseudo-focus class: keeps class windows defined when bumps are off,
        # so uniform runs produce a time-window null. Such cycles are tagged
        # 'uniform' by the tracker.
        _, pseudo_focus, _, _ = _compute_class_probabilities(
            step=global_step, num_classes=num_classes, period_length=period_length,
            w_max=w_max, bump_order=bump_order,
        )

        # A bump is only "active" if it actually changes the batch distribution:
        # w_max == 1 with bump flags on is a uniform run and must be tagged so.
        if bumps_enabled and (not in_uniform_tail) and w_max > 1.0:
            class_probs, focus_class, focus_weight, bump_phase = _compute_class_probabilities(
                step=global_step, num_classes=num_classes, period_length=period_length,
                w_max=w_max, bump_order=bump_order,
            )
            bump_state = {"active": True, "focus_class": focus_class, "focus_weight": focus_weight, "phase": bump_phase}
        else:
            class_probs = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
            bump_state = {"active": False, "focus_class": pseudo_focus, "focus_weight": 1.0, "phase": 0.0}

        class_counts = (class_probs * float(batch_size)).astype(np.int64)
        batch_data, batch_target, sampled_class_counts = _sample_batch_by_class_counts(
            train_inputs=train_inputs, train_targets=train_targets,
            class_to_indices=class_to_indices, class_counts=class_counts, rng=rng,
        )
        sampled_count_total = int(sampled_class_counts.sum())
        sampled_distributions.append(sampled_class_counts.astype(np.float64) / float(max(1, sampled_count_total)))

        x = jax.device_put(jnp.asarray(batch_data, dtype=jnp.float32), device=device)
        y = jax.device_put(jnp.asarray(batch_target, dtype=jnp.int32), device=device)

        params_before = params
        prev_w = model.classifier_weight_matrix(prev_params)
        params, opt_state, per_sample_loss, preds, classifier_grads, data_grads = train_step(
            params, opt_state, x, y, grad_mask,
        )
        current_w = model.classifier_weight_matrix(params)
        cumulative_weight_distance += float(jnp.linalg.norm(current_w - prev_w))
        prev_params = params
        last_classifier_grads = classifier_grads

        _update_stats(stats, per_sample_loss, preds, y, num_classes, compute_per_class)

        # Per-step switching bookkeeping, with the PRE-increment step (the old
        # code used global_step after `+= 1`, shifting the window by one step).
        if switching_tracker is not None:
            switching_tracker.observe(
                step=global_step,
                bump_state=bump_state,
                params_before=params_before,
                params_after=params,
                data_grads=data_grads,
                sampled_counts=sampled_class_counts,
            )
        global_step += 1

    if sampled_distributions:
        sampled_distribution_arr = np.stack(sampled_distributions, axis=0)
    else:
        sampled_distribution_arr = np.zeros((0, num_classes), dtype=np.float64)

    return (
        params, opt_state, _finalize_stats(stats), global_step, bump_state, sampled_distribution_arr,
        cumulative_weight_distance, prev_params, last_classifier_grads, initial_params, grad_mask, reinit_key,
    )




def _csv_header(num_classes: int, training_results: str = "all") -> list[str]:
    base = [
        "epoch",
        "global_step",
        "tpt_reached",
        "tpt_step",
        "bumps_active",
        "bump_focus_class",
        "bump_focus_weight",
        "bump_phase",
        "train_loss",
        "train_accuracy",
        "train_error",
        "test_loss",
        "test_accuracy",
    ]
    if training_results == "all":
        for class_id in range(num_classes):
            base.append(f"train_loss_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"train_accuracy_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"test_loss_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"test_accuracy_class_{class_id}")
    return base


def _metric_row(
    epoch: int,
    global_step: int,
    tpt_reached: bool,
    tpt_step: int,
    bump_state: Dict[str, Any],
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    num_classes: int,
    training_results: str = "all",
) -> list[Any]:
    row: list[Any] = [
        epoch,
        global_step,
        int(tpt_reached),
        tpt_step,
        int(bool(bump_state["active"])),
        int(bump_state["focus_class"]),
        float(bump_state["focus_weight"]),
        float(bump_state["phase"]),
        float(train_metrics["loss"]),
        float(train_metrics["accuracy"]),
        float(train_metrics["training_error"]),
        float(test_metrics["loss"]),
        float(test_metrics["accuracy"]),
    ]
    if training_results == "all":
        for class_id in range(num_classes):
            row.append(float(train_metrics["per_class_loss"][class_id]))
        for class_id in range(num_classes):
            row.append(float(train_metrics["per_class_accuracy"][class_id]))
        for class_id in range(num_classes):
            row.append(float(test_metrics["per_class_loss"][class_id]))
        for class_id in range(num_classes):
            row.append(float(test_metrics["per_class_accuracy"][class_id]))
    return row


def run_training(config: dict, run_dir: Path, run_label: str) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    dynamics_cfg = config.get("dynamics", {})
    optimizer_cfg = config.get("optimizer", {})
    analysis_cfg = config.get("analysis", {})

    training_results = str(analysis_cfg.get("compute_results", "all")).strip().lower()
    if training_results not in {"all", "average"}:
        training_results = "all"

    compute_per_class_flag = (training_results == "all")

    dataset_name = str(data_cfg.get("dataset_name", "spiral"))
    dataset_key = dataset_name.strip().lower()

    random_seed = int(training_cfg.get("random_seed", config.get("random_seed", 42)))
    _set_global_seed(random_seed)

    device = _resolve_device(config.get("device", "auto"))

    total_steps = int(training_cfg.get("training_steps", training_cfg.get("total_steps", 25000)))
    save_metrics_every_n_steps = int(training_cfg.get("save_metrics_every_n_steps", 50))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))

    if total_steps <= 0:
        raise ValueError("training.training_steps must be > 0")
    if save_metrics_every_n_steps <= 0:
        raise ValueError("training.save_metrics_every_n_steps must be > 0")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if eval_batch_size <= 0:
        raise ValueError("training.eval_batch_size must be > 0")

    optimizer_name = str(optimizer_cfg.get("optimizer_type", training_cfg.get("optimizer", "adam"))).strip().lower()
    loss_name = str(training_cfg.get("loss", "cross_entropy")).strip().lower()
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: adam, sgd.")
    if loss_name not in {"cross_entropy", "cross-entropy", "ce"}:
        raise ValueError(f"Only cross-entropy loss is supported in this runner. Got '{loss_name}'.")

    learning_rate = float(optimizer_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.002)))
    beta1 = float(optimizer_cfg.get("beta1", training_cfg.get("beta1", 0.9)))
    beta2 = float(optimizer_cfg.get("beta2", training_cfg.get("beta2", 0.999)))
    eps = float(optimizer_cfg.get("eps", training_cfg.get("eps", 1e-8)))
    momentum = float(optimizer_cfg.get("momentum", training_cfg.get("momentum", 0.0)))
    nesterov = _as_bool(optimizer_cfg.get("nesterov", training_cfg.get("nesterov", False)))
    weight_decay = float(
        optimizer_cfg.get(
            "weight_decay",
            optimizer_cfg.get("l2_reg", training_cfg.get("weight_decay", training_cfg.get("l2_reg", 0.0))),
        )
    )

    gradient_clipping_raw = optimizer_cfg.get("gradient_clipping", training_cfg.get("gradient_clipping"))
    gradient_clipping = None if gradient_clipping_raw in (None, "", "null") else float(gradient_clipping_raw)

    bumps_before_tpt = _as_bool(dynamics_cfg.get("bumps_before_tpt", False))
    bumps_at_tpt = _as_bool(dynamics_cfg.get("bumps_at_tpt", False))

    period_length = int(dynamics_cfg.get("period_length", 250))
    w_max, w_max_source = _resolve_w_max(dynamics_cfg)

    if period_length <= 0:
        raise ValueError("dynamics.period_length must be > 0")
    if w_max < 1.0:
        raise ValueError("dynamics.w_max must be >= 1.0")

    dataset_bundle: DatasetBundle = build_dataset_bundle(config, run_dir)
    num_classes = int(data_cfg.get("num_classes", dataset_bundle.num_classes))
    if num_classes != dataset_bundle.num_classes:
        raise ValueError(
            f"Configured num_classes ({num_classes}) does not match dataset labels ({dataset_bundle.num_classes})."
        )

    bump_order = _validate_bump_order(_parse_bump_order(dynamics_cfg.get("bump_order")), num_classes)

    metric_inputs = dataset_bundle.train_inputs
    metric_targets = dataset_bundle.train_targets
    test_metric_inputs = dataset_bundle.test_inputs
    test_metric_targets = dataset_bundle.test_targets
    metric_subset_size: int | None = None
    if dataset_key in {"mnist", "cifar10", "cifar100", "tiny_imagenet"}:
        metric_subset_size = int(
            analysis_cfg.get("metric_subset_size", data_cfg.get("metric_subset_size", 5000))
        )
        metric_subset_size = max(metric_subset_size, num_classes)

        train_subset_size = min(metric_subset_size, dataset_bundle.train_inputs.shape[0])
        if train_subset_size < dataset_bundle.train_inputs.shape[0]:
            metric_rng = np.random.default_rng(random_seed + 101)
            metric_inputs, metric_targets, _ = _build_metric_subset(
                inputs=dataset_bundle.train_inputs,
                targets=dataset_bundle.train_targets,
                class_to_indices=dataset_bundle.class_to_indices,
                num_classes=num_classes,
                subset_size=train_subset_size,
                rng=metric_rng,
            )

        test_subset_size = min(metric_subset_size, dataset_bundle.test_inputs.shape[0])
        if test_subset_size < dataset_bundle.test_inputs.shape[0]:
            test_metric_rng = np.random.default_rng(random_seed + 202)
            test_class_to_indices = _build_class_to_indices(dataset_bundle.test_targets, num_classes)
            test_metric_inputs, test_metric_targets, _ = _build_metric_subset(
                inputs=dataset_bundle.test_inputs,
                targets=dataset_bundle.test_targets,
                class_to_indices=test_class_to_indices,
                num_classes=num_classes,
                subset_size=test_subset_size,
                rng=test_metric_rng,
            )

    architecture_str = str(model_cfg.get("architecture", "mlp")).strip().lower()
    use_pytorch_cnn = architecture_str in {"resnet18_cnn", "cnn"}
    built_model = build_model(
        model_cfg=model_cfg, input_shape=dataset_bundle.input_shape,
        num_classes=num_classes, random_seed=random_seed,
    )
    model = built_model.model
    params = jax.device_put(built_model.params, device=device)
    initial_params = params
    initial_classifier_weight = np.asarray(model.classifier_weight_matrix(initial_params), dtype=np.float64,)
    last_classifier_weight = initial_classifier_weight.copy()
    cumulative_weight_distance = 0.0
    prev_params = params
    grad_mask = _build_grad_mask(params, None)
    reinit_key = jax.random.PRNGKey(int(random_seed) + 1)

    total_epochs = int(math.ceil(total_steps / save_metrics_every_n_steps))

    
    if optimizer_name == "adam":
        base_optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)
    else:
        base_optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    optimizer = optax.chain(optax.clip_by_global_norm(gradient_clipping), base_optimizer) if gradient_clipping is not None else base_optimizer
    opt_state = optimizer.init(params)

    csv_path = run_dir / "training_metrics.csv"
    figure_path = run_dir / "training_report.png"
    plot_version = _normalize_plot_version(analysis_cfg.get("plot_version"))

    enable_switching_metrics = _analysis_output_enabled(
        analysis_cfg, ("switching_metrics",), default=False
    )
    enable_modularity_metrics = enable_switching_metrics and _analysis_output_enabled(
        analysis_cfg, ("switching_modularity_metrics",), default=True
    )
    switching_regime_filter = str(analysis_cfg.get("switching_regime_filter", "auto")).strip().lower()
    switching_topk = int(analysis_cfg.get("switching_topk", DEFAULT_TOPK))
    switching_null_run_dir = analysis_cfg.get("switching_null_run_dir")

    # ---- Dynamics banner: make the actual regime of this run explicit. ----
    any_bumps = (bumps_before_tpt or bumps_at_tpt) and w_max > 1.0
    peak_focus, peak_other = _batch_composition(w_max, batch_size, num_classes)
    centre_focus, centre_other = _batch_composition(0.5 * (1.0 + w_max), batch_size, num_classes)
    central_single_class = any_bumps and centre_other == 0
    print(
        f"[dynamics] w_max={w_max:g} (from {w_max_source}), period_length={period_length}, "
        f"bumps_before_tpt={bumps_before_tpt}, bumps_at_tpt={bumps_at_tpt} -> "
        f"{'BUMP run' if any_bumps else 'UNIFORM (null) run'}"
    )
    if any_bumps:
        print(
            f"[dynamics] batch at bump peak: {peak_focus} focus + {peak_other}/other class; "
            f"at central-window edge (w={(1 + w_max) / 2:g}): {centre_focus} + {centre_other}/other. "
            f"Central window single-class: {central_single_class} "
            f"(needs w > B - C + 1 = {batch_size - num_classes + 1})."
        )

    switching_csv_path: Path | None = None
    switching_figure_path: Path | None = None
    modularity_figure_path: Path | None = None
    switching_tracker: SwitchingTracker | None = None
    class_pairs = build_class_pairs(num_classes)

    if enable_switching_metrics:
        cycle_order = list(bump_order) if bump_order else list(range(num_classes))
        if sorted(cycle_order) != list(range(num_classes)):
            raise ValueError(
                "Switching metrics require dynamics.bump_order to be a permutation of all classes "
                f"(got {cycle_order}); repeated classes would overwrite each other's snapshots."
            )
        switching_csv_path = run_dir / "switching_metrics.csv"
        switching_figure_path = run_dir / "switching_metrics.png"
        if enable_modularity_metrics:
            modularity_figure_path = run_dir / "modularity_metrics.png"
        probe_sets = _build_probe_sets(
            dataset_bundle.train_inputs,
            dataset_bundle.train_targets,
            num_classes,
            int(analysis_cfg.get("switching_probe_size_per_class", 256)),
            random_seed + 303,
            device,
        )
        switching_tracker = SwitchingTracker(
            model=model,
            params=params,
            csv_path=switching_csv_path,
            run_dir=run_dir,
            class_pairs=class_pairs,
            num_classes=num_classes,
            period_length=period_length,
            cycle_order=cycle_order,
            tau=float(analysis_cfg.get("switching_tau", 0.1)),
            topk=switching_topk,
            snapshot_mode=str(analysis_cfg.get("switching_snapshot", "central_mean")).strip().lower(),
            probe_sets=probe_sets,
            modularity_enabled=enable_modularity_metrics,
            modularity_quantile=float(analysis_cfg.get("switching_modularity_quantile", 0.9)),
            save_snapshots=_as_bool(analysis_cfg.get("switching_save_snapshots", True)),
            energy_q=float(analysis_cfg.get("switching_energy_quantile", DEFAULT_ENERGY_Q)),
            own_q=float(analysis_cfg.get("switching_ownership_quantile", DEFAULT_OWNERSHIP_Q)),
            detrend_modes=_as_detrend_modes(analysis_cfg.get("switching_detrend", DEFAULT_DETREND)),
        )
        print(
            f"[switching] proxy tensor for rho: {switching_tracker.proxy_name}; "
            f"snapshot mode: {switching_tracker.snapshot_mode}; top-k: {switching_topk}; "
            f"decomposition detrend: {', '.join(switching_tracker.detrend_modes)}"
        )

    train_step = _make_train_step(model, optimizer, weight_decay, num_classes)

    predict_step = jax.jit(lambda step_params, batch_x: model.apply(step_params, batch_x))

    last_classifier_grads = None
    global_step = 0
    tpt_reached = False
    tpt_step = -1
    rng = np.random.default_rng(random_seed)
    class_distribution_history: list[np.ndarray] = []
    prev_deep_state: Dict[str, Any] | None = None
    
    last_train_metrics: Dict[str, Any] = {
        "loss": np.nan,
        "accuracy": np.nan,
        "per_class_loss": np.full(num_classes, np.nan),
        "per_class_accuracy": np.full(num_classes, np.nan),
    }
    last_test_metrics: Dict[str, Any] = {
        "loss": np.nan,
        "accuracy": np.nan,
        "per_class_loss": np.full(num_classes, np.nan),
        "per_class_accuracy": np.full(num_classes, np.nan),
    }

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(num_classes, training_results))
        f.flush()

        for epoch in range(1, total_epochs + 1):
            steps_this_epoch = min(save_metrics_every_n_steps, total_steps - global_step)
            if steps_this_epoch <= 0:
                break

            bumps_enabled = bumps_at_tpt if tpt_reached else bumps_before_tpt

            params_before_block = params
            opt_state_before_block = opt_state
            (params, opt_state, _step_train_metrics, global_step, bump_state, step_distributions,
             cumulative_weight_distance, prev_params, last_classifier_grads,
             initial_params, grad_mask, reinit_key) = train_epoch(
            params=params,
            opt_state=opt_state,
            train_step=train_step,
            model=model,
            train_inputs=dataset_bundle.train_inputs,
            train_targets=dataset_bundle.train_targets,
            class_to_indices=dataset_bundle.class_to_indices,
            batch_size=batch_size,
            device=device,
            num_classes=num_classes,
            global_step=global_step,
            steps_this_epoch=steps_this_epoch,
            total_steps=total_steps,
            bumps_enabled=bumps_enabled,
            period_length=period_length,
            w_max=w_max,
            bump_order=bump_order,
            cumulative_weight_distance=cumulative_weight_distance,
            prev_params=prev_params,
            rng=rng,
            grad_mask=grad_mask,
            reinit_key=reinit_key,
            initial_params=initial_params,
            compute_per_class=compute_per_class_flag,
            switching_tracker=switching_tracker,
            )
            
            if step_distributions.size > 0:
                class_distribution_history.extend(step_distributions)

            train_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=metric_inputs,
                targets=metric_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
                compute_per_class=compute_per_class_flag,
            )
            test_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=test_metric_inputs,
                targets=test_metric_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
                compute_per_class=compute_per_class_flag,
            )

            if (not tpt_reached) and bool(train_metrics.get("zero_training_error", False)):
                tpt_reached = True
                tpt_step = global_step

            row = _metric_row(
                epoch=epoch,
                global_step=global_step,
                tpt_reached=tpt_reached,
                tpt_step=tpt_step,
                bump_state=bump_state,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                num_classes=num_classes,
                training_results=training_results,
            )
            writer.writerow(row)
            f.flush()

            last_train_metrics = train_metrics
            last_test_metrics = test_metrics


    architecture = str(model_cfg.get("architecture", "mlp"))
    plot_training_report(
        csv_path=csv_path,
        output_path=figure_path,
        num_classes=num_classes,
        title=f"{run_label} | {dataset_name} | {architecture}",
    )

    distribution_figure_path = run_dir / "example_distribution_dynamics.png"
    if class_distribution_history:
        distribution_history_arr = np.asarray(class_distribution_history, dtype=np.float64)
    else:
        distribution_history_arr = np.zeros((0, num_classes), dtype=np.float64)
    plot_example_distribution_dynamics(
        class_distribution_history=distribution_history_arr,
        output_path=distribution_figure_path,
        title=f"{run_label} | Example Distribution Dynamics",
        tpt_step=tpt_step if tpt_reached else -1,
    )
        
    if enable_switching_metrics:
        finalize_switching_metrics(
            switching_csv_path=switching_csv_path,
            output_path=switching_figure_path,
            class_pairs=class_pairs,
            num_classes=num_classes,
            regime_filter=switching_regime_filter,
        )
        if enable_modularity_metrics:
            finalize_modularity_metrics(
                base_dir=run_dir,
                output_path=modularity_figure_path,
                classes=list(range(num_classes)),
                regime_filter=switching_regime_filter,
            )
        print(f"[switching] cycles logged by regime: {switching_tracker.logged}")
        if switching_null_run_dir:
            try:
                compare_to_null(run_dir, Path(switching_null_run_dir), class_pairs=class_pairs, topk=switching_topk)
            except Exception as exc:  # the null run may not exist yet
                print(f"[switching] compare_to_null skipped: {exc}")

    boundary_figure_path: Path | None = None
    if dataset_key in {"spiral", "gaussian_blobs", "blobs", "rings", "checkerboard", "random_checkerboard", "dartboard"}:
        boundary_figure_path = run_dir / f"{dataset_key}_decision_boundaries.png"
        plot_2d_decision_boundaries(
            model=model,
            params=params,
            train_inputs=dataset_bundle.train_inputs,
            train_targets=dataset_bundle.train_targets,
            test_inputs=dataset_bundle.test_inputs,
            test_targets=dataset_bundle.test_targets,
            output_path=boundary_figure_path,
            title=f"{run_label} | {dataset_name} | {architecture} | Decision Boundaries",
        )

    summary = {
        "run_label": run_label,
        "device": _device_label(device),
        "optimizer": optimizer_name,
        "init_type": model.init_type,
        "total_steps": total_steps,
        "save_metrics_every_n_steps": save_metrics_every_n_steps,
        "epochs": total_epochs,
        "tpt_reached": bool(tpt_reached),
        "tpt_step": int(tpt_step),
        "metric_subset_size": int(metric_subset_size) if metric_subset_size is not None else None,
        "metric_subset_count": int(metric_inputs.shape[0]),
        "csv_path": str(csv_path),
        "figure_path": str(figure_path),
        "distribution_figure_path": str(distribution_figure_path),
        "switching_csv_path": str(switching_csv_path) if switching_csv_path is not None else None,
        "switching_figure_path": str(switching_figure_path) if switching_figure_path is not None else None,
        "modularity_figure_path": str(modularity_figure_path) if modularity_figure_path is not None else None,
        "decision_boundary_path": str(boundary_figure_path) if boundary_figure_path is not None else None,
        "final_train_loss": float(last_train_metrics["loss"]),
        "final_train_accuracy": float(last_train_metrics["accuracy"]),
        "final_test_loss": float(last_test_metrics["loss"]),
        "final_test_accuracy": float(last_test_metrics["accuracy"]),
        "w_max": float(w_max),
        "w_max_source": w_max_source,
        "bumps_before_tpt": bool(bumps_before_tpt),
        "bumps_at_tpt": bool(bumps_at_tpt),
        "bump_run": bool(any_bumps),
        "central_window_single_class": bool(central_single_class),
        "batch_at_peak": {"focus": peak_focus, "per_other_class": peak_other},
        "switching_proxy": switching_tracker.proxy_name if switching_tracker else None,
        "switching_snapshot_mode": switching_tracker.snapshot_mode if switching_tracker else None,
        "switching_detrend_modes": list(switching_tracker.detrend_modes) if switching_tracker else None,
        "switching_cycles_logged": dict(switching_tracker.logged) if switching_tracker else None,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        run_dir / "dataset_bundle.npz",
        train_inputs=dataset_bundle.train_inputs,
        train_targets=dataset_bundle.train_targets,
        test_inputs=dataset_bundle.test_inputs,
        test_targets=dataset_bundle.test_targets
    )

    with open(run_dir / "jax_params.pkl", "wb") as f:
        pickle.dump(params, f)
            
    return summary