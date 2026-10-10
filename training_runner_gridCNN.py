"""
training_runner_gridCNN.py

Grid-search version of training_runner.py for the PyTorch CNN path only
(CIFAR-10; resnet18-family / simple_cnn / myrtle backbones). Outputs per run:

    training_metrics.csv   epoch, global_step, tpt/bump state, train/test loss+acc
    summary.json           final numbers, tpt step, wall time (also used by the
                           sbatch script to skip already-finished runs)
    training_report.png    (optional, output.plot, default true; made once at the end)
    model_weights.pt       (optional, output.save_weights, default false)

Training math kept identical to the original CNN path:
  * same bump schedule (_compute_focus_weight / _compute_class_probabilities),
    same deterministic per-class counts, same rng.choice / rng.permutation calls
    on the same np.random.default_rng(seed) stream -> same batch indices
  * uniform tail over the last 5% of steps, bumps_before_tpt / bumps_at_tpt
  * torch SGD(lr, momentum, weight_decay), CrossEntropyLoss (mean)
  * MultiStepLR(gamma=0.1) at 1/3 and 2/3 of training (now expressed in steps,
    so it no longer depends on how often metrics are saved -- see README note)
  * train / test metrics on the same stratified 5000-example subsets
    (seed+101 / seed+202), model in eval() mode, TPT = 100% train-subset accuracy

What makes it fast:
  * dataset lives on the GPU as uint8; only the 128 batch indices go host->GPU
  * no .item() / host sync inside the training loop (the original synced twice
    per step for running train stats that were never written anywhere)
  * no forward hook cloning features on every forward pass
  * no NC / separability / PCA / classifier / per-class metrics, no jax, no
    per-run dataset visualisation, no 600 MB dataset_bundle.npz dump per run
  * evaluation in large batches under inference_mode with a single sync
  * cudnn.benchmark, optional fused SGD, optional bf16/fp16 autocast + channels_last
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from generate_dataset_gridCNN import CIFAR10_MEAN, CIFAR10_STD, build_class_index_map, load_cifar10
from model_cnn_gridCNN import build_model_from_config


# ---------------------------------------------------------------------------
# small helpers (copied from training_runner.py)
# ---------------------------------------------------------------------------

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_bump_order(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        text = text.strip("[]")
        return [int(p.strip()) for p in text.split(",") if p.strip()]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [int(p) for p in raw]
    raise ValueError("dynamics.bump_order must be null, a list of ints, or a comma-separated string.")


def _validate_bump_order(bump_order: list[int] | None, num_classes: int) -> list[int] | None:
    if bump_order is None:
        return None
    if len(bump_order) != num_classes:
        raise ValueError(f"dynamics.bump_order must have length {num_classes}, got {len(bump_order)}.")
    for idx, c in enumerate(bump_order):
        if c < 0 or c >= num_classes:
            raise ValueError(f"dynamics.bump_order has invalid class id {c} at index {idx}.")
    return bump_order


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_focus_weight(period_step: int, period_length: int, w_max: float) -> tuple[float, float]:
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    return focus_weight, t / period


def _compute_class_probabilities(step, num_classes, period_length, w_max, bump_order=None):
    if bump_order is None:
        focus_class = (step // period_length) % num_classes
    else:
        focus_class = int(bump_order[(step // period_length) % len(bump_order)])
    focus_weight, bump_phase = _compute_focus_weight(step % period_length, period_length, w_max)
    weights = np.ones(num_classes, dtype=np.float64)
    weights[focus_class] = focus_weight
    return weights / weights.sum(), int(focus_class), float(focus_weight), float(bump_phase)


def _sample_indices_by_class_counts(class_to_indices, class_counts, rng) -> np.ndarray:
    """Same rng.choice calls, in the same order, as the original
    _sample_batch_by_class_counts -- but returns only the indices."""
    parts = []
    for class_id, count in enumerate(np.asarray(class_counts, dtype=np.int64)):
        if count <= 0:
            continue
        parts.append(rng.choice(class_to_indices[class_id], size=int(count), replace=True).astype(np.int64, copy=False))
    if not parts:
        raise ValueError("Deterministic class_counts produced an empty batch.")
    return np.concatenate(parts, axis=0)


class _EpochShuffler:
    """Identical to model_cnn._EpochShuffler (DataLoader(shuffle=True, drop_last=True))."""

    def __init__(self, num_samples: int, batch_size: int, rng: np.random.Generator):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.rng = rng
        self._order = self.rng.permutation(num_samples)
        self._pos = 0

    def next_batch(self) -> np.ndarray:
        if self._pos + self.batch_size > self.num_samples:
            self._order = self.rng.permutation(self.num_samples)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx


def _metric_subset_indices(targets, class_to_indices, num_classes, subset_size, rng) -> np.ndarray:
    """Same selection (same rng calls) as training_runner._build_metric_subset,
    returning the chosen indices instead of copies of the arrays."""
    total = int(targets.shape[0])
    subset_size = min(int(subset_size), total)
    if subset_size < num_classes:
        return rng.choice(total, size=subset_size, replace=False)
    counts = np.array([len(class_to_indices[c]) for c in range(num_classes)], dtype=np.int64)
    per_class = np.minimum(subset_size // num_classes, counts)
    remaining = subset_size - int(per_class.sum())
    if remaining > 0:
        available = counts - per_class
        if int(available.sum()) > 0:
            per_class += rng.multinomial(remaining, available / available.sum())
    chosen = []
    for c, count in enumerate(per_class):
        if count <= 0:
            continue
        idx = class_to_indices[c]
        chosen.append(rng.choice(idx, size=int(count), replace=int(count) > idx.shape[0]).astype(np.int64, copy=False))
    return rng.permutation(np.concatenate(chosen, axis=0))


CSV_HEADER = [
    "epoch", "global_step", "tpt_reached", "tpt_step",
    "bumps_active", "bump_focus_class", "bump_focus_weight", "bump_phase",
    "learning_rate",
    "train_loss", "train_accuracy", "train_error",
    "test_loss", "test_accuracy",
]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_training(config: dict, run_dir: Path, run_label: str) -> Dict[str, Any]:
    t_start = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    dynamics_cfg = config.get("dynamics", {}) or {}
    optimizer_cfg = config.get("optimizer", {}) or {}
    analysis_cfg = config.get("analysis", {}) or {}
    output_cfg = config.get("output", {}) or {}
    perf_cfg = config.get("performance", {}) or {}

    dataset_name = str(data_cfg.get("dataset_name", "cifar10")).strip().lower()
    if dataset_name != "cifar10":
        raise ValueError(f"gridCNN runner only supports cifar10 (got '{dataset_name}').")
    num_classes = int(data_cfg.get("num_classes", 10))
    if num_classes != 10:
        raise ValueError("gridCNN runner: CIFAR-10 requires data.num_classes = 10.")

    # ---------------- device / performance knobs ----------------
    use_gpu = _as_bool(config.get("use_gpu", True))
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("use_gpu: true but CUDA is not available to PyTorch.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(int(perf_cfg.get("num_threads", 1)))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = _as_bool(perf_cfg.get("cudnn_benchmark", True))

    amp_mode = str(perf_cfg.get("amp", "none")).strip().lower()
    if amp_mode in {"false", "off", "0", ""}:
        amp_mode = "none"
    if amp_mode not in {"none", "bf16", "fp16"}:
        raise ValueError("performance.amp must be none, bf16 or fp16.")
    if device.type != "cuda":
        amp_mode = "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_mode)
    channels_last = _as_bool(perf_cfg.get("channels_last", False))
    memory_format = torch.channels_last if channels_last else torch.contiguous_format

    # ---------------- training config ----------------
    random_seed = int(training_cfg.get("random_seed", config.get("random_seed", 42)))
    _set_global_seed(random_seed)

    total_steps = int(training_cfg.get("training_steps", training_cfg.get("total_steps", 25000)))
    save_every = int(training_cfg.get("save_metrics_every_n_steps", 50))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(perf_cfg.get("eval_batch_size", training_cfg.get("eval_batch_size", 1000)))
    if total_steps <= 0 or save_every <= 0 or batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("training_steps, save_metrics_every_n_steps, batch_size, eval_batch_size must be > 0")

    loss_name = str(training_cfg.get("loss", "cross_entropy")).strip().lower()
    if loss_name not in {"cross_entropy", "cross-entropy", "ce"}:
        raise ValueError(f"Only cross-entropy is supported, got '{loss_name}'.")

    learning_rate = float(optimizer_cfg.get("learning_rate", 0.002))
    momentum = float(optimizer_cfg.get("momentum", 0.0))
    weight_decay = float(optimizer_cfg.get("weight_decay", optimizer_cfg.get("l2_reg", 0.0)))
    nesterov = _as_bool(optimizer_cfg.get("nesterov", False))
    clip_raw = optimizer_cfg.get("gradient_clipping")
    gradient_clipping = None if clip_raw in (None, "", "null") else float(clip_raw)
    lr_gamma = float(optimizer_cfg.get("lr_gamma", 0.1))
    milestones_raw = optimizer_cfg.get("lr_milestone_steps")
    if milestones_raw is None:
        lr_milestones = [total_steps // 3, (2 * total_steps) // 3]
    else:
        lr_milestones = [int(m) for m in (milestones_raw or [])]
    lr_milestone_set = set(lr_milestones)

    bumps_before_tpt = _as_bool(dynamics_cfg.get("bumps_before_tpt", False))
    bumps_at_tpt = _as_bool(dynamics_cfg.get("bumps_at_tpt", False))
    period_length = int(dynamics_cfg.get("period_length", 250))
    w_max = float(dynamics_cfg.get("w_max", 50.0))
    tpt_threshold = float(dynamics_cfg.get("tpt_accuracy_threshold", 1.0))
    uniform_tail_fraction = float(dynamics_cfg.get("uniform_tail_fraction", 0.05))
    if period_length <= 0:
        raise ValueError("dynamics.period_length must be > 0")
    if w_max < 1.0:
        raise ValueError("dynamics.w_max must be >= 1.0")
    bump_order = _validate_bump_order(_parse_bump_order(dynamics_cfg.get("bump_order")), num_classes)

    # ---------------- data (GPU resident, uint8) ----------------
    data = load_cifar10(data_cfg.get("data_dir", "./data"))
    train_targets_np = data.train_targets
    class_to_indices = data.class_to_indices

    metric_subset_size = max(int(analysis_cfg.get("metric_subset_size", data_cfg.get("metric_subset_size", 5000))), num_classes)
    n_train, n_test = train_targets_np.shape[0], data.test_targets.shape[0]
    train_eval_idx = np.arange(n_train, dtype=np.int64)
    test_eval_idx = np.arange(n_test, dtype=np.int64)
    if min(metric_subset_size, n_train) < n_train:
        train_eval_idx = _metric_subset_indices(
            train_targets_np, class_to_indices, num_classes, metric_subset_size,
            np.random.default_rng(random_seed + 101),
        )
    if min(metric_subset_size, n_test) < n_test:
        test_eval_idx = _metric_subset_indices(
            data.test_targets, build_class_index_map(data.test_targets, num_classes), num_classes,
            metric_subset_size, np.random.default_rng(random_seed + 202),
        )

    train_x_u8 = torch.from_numpy(data.train_images_u8).to(device)
    train_y = torch.from_numpy(train_targets_np).to(device)
    # eval subsets gathered once (fixed across the run)
    train_eval_x_u8 = train_x_u8[torch.from_numpy(train_eval_idx).to(device)]
    train_eval_y = train_y[torch.from_numpy(train_eval_idx).to(device)]
    test_eval_x_u8 = torch.from_numpy(data.test_images_u8[test_eval_idx]).to(device)
    test_eval_y = torch.from_numpy(data.test_targets[test_eval_idx]).to(device)
    del data

    mean_t = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def _normalize(x_u8: torch.Tensor) -> torch.Tensor:
        # same ops/dtype as generate_dataset._normalize_image_batch: (x/255 - mean)/std
        x = (x_u8.float() / 255.0 - mean_t) / std_t
        return x.contiguous(memory_format=memory_format)

    # ---------------- model / optimizer ----------------
    model = build_model_from_config(model_cfg, num_classes=num_classes, input_ch=3)
    model = model.to(device=device, memory_format=memory_format)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    sgd_kwargs = dict(lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    optimizer = None
    if device.type == "cuda" and _as_bool(perf_cfg.get("fused_optimizer", True)):
        try:
            optimizer = torch.optim.SGD(params, fused=True, **sgd_kwargs)
        except (TypeError, RuntimeError, ValueError):
            optimizer = None
    if optimizer is None:
        optimizer = torch.optim.SGD(params, **sgd_kwargs)

    scaler = torch.amp.GradScaler("cuda") if amp_mode == "fp16" else None
    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
        if amp_dtype is not None
        else (lambda: torch.autocast(device_type=device.type, enabled=False))
    )

    def _train_step(idx_np: np.ndarray) -> None:
        idx = torch.from_numpy(idx_np).to(device, non_blocking=True)
        x = _normalize(train_x_u8.index_select(0, idx))
        y = train_y.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(x)
            loss = F.cross_entropy(out, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(params, gradient_clipping)
            optimizer.step()

    @torch.inference_mode()
    def _evaluate(x_u8: torch.Tensor, y: torch.Tensor) -> tuple[float, float, bool]:
        model.eval()
        n = int(y.shape[0])
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.int64, device=device)
        for s in range(0, n, eval_batch_size):
            xb = _normalize(x_u8[s:s + eval_batch_size])
            yb = y[s:s + eval_batch_size]
            with autocast_ctx():
                logits = model(xb)
            logits = logits.float()
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").double()
            correct += (logits.argmax(dim=1) == yb).sum()
        model.train()
        loss_v, correct_v = torch.stack([loss_sum, correct.double()]).tolist()  # one sync
        correct_i = int(round(correct_v))
        return loss_v / max(1, n), correct_i / max(1, n), correct_i == n

    # ---------------- loop ----------------
    rng = np.random.default_rng(random_seed)
    shuffler = _EpochShuffler(n_train, batch_size, rng)
    total_epochs = int(math.ceil(total_steps / save_every))
    # original: uniform sampling once global_step >= 0.95 * total_steps
    tail_start = (1.0 - uniform_tail_fraction) * float(total_steps) if uniform_tail_fraction > 0 else float("inf")

    global_step = 0
    tpt_reached = False
    tpt_step = -1
    current_lr = learning_rate
    last = {"train_loss": float("nan"), "train_acc": float("nan"), "test_loss": float("nan"), "test_acc": float("nan")}
    print_every = max(1, total_epochs // 20)
    csv_path = run_dir / "training_metrics.csv"

    print(
        f"[{run_label}] device={device} amp={amp_mode} channels_last={channels_last} "
        f"steps={total_steps} eval_every={save_every} w_max={w_max} period={period_length} seed={random_seed} "
        f"lr_milestones={lr_milestones} params={sum(p.numel() for p in params):,}",
        flush=True,
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_HEADER)

        for epoch in range(1, total_epochs + 1):
            steps_this_epoch = min(save_every, total_steps - global_step)
            if steps_this_epoch <= 0:
                break
            bumps_enabled = bumps_at_tpt if tpt_reached else bumps_before_tpt
            bump_state = {"active": False, "focus_class": -1, "focus_weight": 1.0, "phase": 0.0}

            for _ in range(steps_this_epoch):
                if global_step in lr_milestone_set:
                    current_lr = learning_rate * (lr_gamma ** sum(1 for m in lr_milestones if m <= global_step))
                    for g in optimizer.param_groups:
                        g["lr"] = current_lr

                if bumps_enabled and global_step < tail_start:
                    probs, focus_class, focus_weight, phase = _compute_class_probabilities(
                        global_step, num_classes, period_length, w_max, bump_order
                    )
                    bump_state = {"active": True, "focus_class": focus_class, "focus_weight": focus_weight, "phase": phase}
                    counts = (probs * float(batch_size)).astype(np.int64)
                    idx_np = _sample_indices_by_class_counts(class_to_indices, counts, rng)
                else:
                    idx_np = shuffler.next_batch()

                _train_step(idx_np)
                global_step += 1

            tr_loss, tr_acc, tr_perfect = _evaluate(train_eval_x_u8, train_eval_y)
            te_loss, te_acc, _ = _evaluate(test_eval_x_u8, test_eval_y)

            if not tpt_reached:
                hit = tr_perfect if tpt_threshold >= 1.0 else (tr_acc >= tpt_threshold)
                if hit:
                    tpt_reached = True
                    tpt_step = global_step

            writer.writerow([
                epoch, global_step, int(tpt_reached), tpt_step,
                int(bump_state["active"]), int(bump_state["focus_class"]),
                f"{bump_state['focus_weight']:.6g}", f"{bump_state['phase']:.6g}",
                f"{current_lr:.6g}",
                f"{tr_loss:.6g}", f"{tr_acc:.6g}", f"{1.0 - tr_acc:.6g}",
                f"{te_loss:.6g}", f"{te_acc:.6g}",
            ])
            last = {"train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc}

            if epoch % print_every == 0 or epoch == total_epochs:
                f.flush()
                el = time.time() - t_start
                print(
                    f"[{run_label}] step {global_step}/{total_steps} | train {tr_loss:.4f}/{tr_acc:.4f} "
                    f"| test {te_loss:.4f}/{te_acc:.4f} | tpt={tpt_step} | {global_step / max(el, 1e-9):.1f} it/s",
                    flush=True,
                )

    elapsed = time.time() - t_start

    if _as_bool(output_cfg.get("plot", True)):
        try:
            from metrics_gridCNN import plot_training_report
            plot_training_report(csv_path, run_dir / "training_report.png",
                                 title=f"{run_label} | cifar10 | {model_cfg.get('architecture')}")
        except Exception as exc:  # a plot must never kill a finished run
            print(f"[{run_label}] plot failed: {exc}", flush=True)

    if _as_bool(output_cfg.get("save_weights", False)):
        torch.save({"torch_model": model.state_dict()}, run_dir / "model_weights.pt")

    summary = {
        "run_label": run_label,
        "status": "completed",
        "device": str(device),
        "amp": amp_mode,
        "random_seed": random_seed,
        "w_max": w_max,
        "period_length": period_length,
        "total_steps": total_steps,
        "save_metrics_every_n_steps": save_every,
        "lr_milestone_steps": lr_milestones,
        "tpt_reached": bool(tpt_reached),
        "tpt_step": int(tpt_step),
        "metric_subset_count_train": int(train_eval_idx.shape[0]),
        "metric_subset_count_test": int(test_eval_idx.shape[0]),
        "final_train_loss": float(last["train_loss"]),
        "final_train_accuracy": float(last["train_acc"]),
        "final_test_loss": float(last["test_loss"]),
        "final_test_accuracy": float(last["test_acc"]),
        "wall_time_seconds": float(elapsed),
        "csv_path": str(csv_path),
    }
    # written last and atomically: its existence == "run finished"
    tmp = run_dir / "summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, run_dir / "summary.json")
    return summary
