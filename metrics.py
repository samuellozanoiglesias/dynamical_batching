from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from model import JAXModel, ParamTree


def _to_float(value: str) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _desaturate_towards_white(color: tuple[float, float, float, float], mix: float = 0.6) -> tuple[float, float, float, float]:
    r, g, b, _a = color
    return (
        r + (1.0 - r) * mix,
        g + (1.0 - g) * mix,
        b + (1.0 - b) * mix,
        1.0,
    )


def _extract_2d_points(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)

    if x.ndim == 2 and x.shape[1] == 2:
        return x, y
    if x.ndim == 4 and x.shape[1] == 2 and x.shape[2] == 1 and x.shape[3] == 1:
        return x[:, :, 0, 0], y

    raise ValueError(
        "Decision boundary plot expects 2D inputs with shape (N,2) "
        "or (N,2,1,1)."
    )


def plot_2d_decision_boundaries(
    model: JAXModel,
    params: ParamTree,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    output_path: Path,
    title: str,
    grid_size: int = 400,
) -> None:
    train_xy, train_y = _extract_2d_points(train_inputs, train_targets)
    test_xy, test_y = _extract_2d_points(test_inputs, test_targets)

    all_xy = np.concatenate([train_xy, test_xy], axis=0)
    x_min, x_max = float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))
    y_min, y_max = float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))
    x_margin = max(0.05, 0.1 * (x_max - x_min))
    y_margin = max(0.05, 0.1 * (y_max - y_min))

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, grid_size, dtype=np.float32),
        np.linspace(y_min - y_margin, y_max + y_margin, grid_size, dtype=np.float32),
    )
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    input_rank = int(np.asarray(train_inputs).ndim)
    model_input = grid_xy
    if input_rank == 4:
        model_input = np.reshape(model_input, (-1, 2, 1, 1))

    logits = model.apply(params, model_input)
    pred = np.asarray(jnp.argmax(logits, axis=1), dtype=np.int64).reshape(xx.shape)

    max_class = int(max(np.max(train_y), np.max(test_y), np.max(pred)))
    num_classes = max_class + 1
    base_cmap = plt.get_cmap("tab10")
    point_colors = [base_cmap(class_id % 10) for class_id in range(num_classes)]
    background_colors = [_desaturate_towards_white(color, mix=0.62) for color in point_colors]
    background_cmap = ListedColormap(background_colors)
    points_cmap = ListedColormap(point_colors)
    levels = np.arange(-0.5, num_classes + 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    contour = ax.contourf(xx, yy, pred, levels=levels, cmap=background_cmap, alpha=1.0)
    if num_classes > 1:
        ax.contour(
            xx,
            yy,
            pred,
            levels=np.arange(0.5, num_classes, 1.0),
            colors="k",
            linewidths=0.5,
            alpha=0.45,
        )

    ax.scatter(
        train_xy[:, 0],
        train_xy[:, 1],
        c=train_y,
        cmap=points_cmap,
        vmin=0,
        vmax=max(0, num_classes - 1),
        s=20,
        alpha=0.98,
        edgecolors="black",
        linewidths=0.25,
        label="Train samples",
    )
    ax.scatter(
        test_xy[:, 0],
        test_xy[:, 1],
        c=test_y,
        cmap=points_cmap,
        vmin=0,
        vmax=max(0, num_classes - 1),
        marker="x",
        s=26,
        alpha=1.0,
        linewidths=1.0,
        label="Test samples",
    )

    cbar = fig.colorbar(contour, ax=ax, ticks=np.arange(num_classes))
    cbar.set_label("Predicted class")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spiral_decision_boundaries(
    model: JAXModel,
    params: ParamTree,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    output_path: Path,
    title: str,
    grid_size: int = 400,
) -> None:
    plot_2d_decision_boundaries(
        model=model,
        params=params,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        output_path=output_path,
        title=title,
        grid_size=grid_size,
    )


def plot_training_report(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    title: str,
) -> None:
    steps = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    train_loss_pc = [[] for _ in range(num_classes)]
    test_loss_pc = [[] for _ in range(num_classes)]
    train_acc_pc = [[] for _ in range(num_classes)]
    test_acc_pc = [[] for _ in range(num_classes)]
    tpt_step = -1.0
    
    has_per_class = False

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "train_loss_class_0" in reader.fieldnames:
            has_per_class = True
            
        for row in reader:
            steps.append(_to_float(row.get("global_step", "")))
            train_loss.append(_to_float(row.get("train_loss", "")))
            test_loss.append(_to_float(row.get("test_loss", "")))
            train_acc.append(_to_float(row.get("train_accuracy", "")))
            test_acc.append(_to_float(row.get("test_accuracy", "")))

            row_tpt_step = _to_float(row.get("tpt_step", ""))
            if tpt_step < 0.0 and np.isfinite(row_tpt_step) and row_tpt_step >= 0.0:
                tpt_step = row_tpt_step

            if has_per_class:
                for class_id in range(num_classes):
                    train_loss_pc[class_id].append(_to_float(row.get(f"train_loss_class_{class_id}", "")))
                    test_loss_pc[class_id].append(_to_float(row.get(f"test_loss_class_{class_id}", "")))
                    train_acc_pc[class_id].append(_to_float(row.get(f"train_accuracy_class_{class_id}", "")))
                    test_acc_pc[class_id].append(_to_float(row.get(f"test_accuracy_class_{class_id}", "")))

    if not steps:
        raise ValueError(f"No rows were found in metrics CSV: {csv_path}")

    steps_arr = np.array(steps, dtype=np.float64)
    
    if has_per_class:
        fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharex=True)
        axes = np.expand_dims(axes, axis=0) # Make it (1, 2) to maintain standard indexing

    fig.suptitle(title, fontsize=16)

    # Row 0: Overall Loss & Accuracy
    ax = axes[0, 0]
    ax.plot(steps_arr, train_loss, linewidth=2.0, label="Train")
    ax.plot(steps_arr, test_loss, linewidth=2.0, label="Test")
    ax.set_ylabel("Loss")
    ax.set_title("Overall Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps_arr, train_acc, linewidth=2.0, label="Train")
    ax.plot(steps_arr, test_acc, linewidth=2.0, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Overall Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Rows 1 & 2: Conditional Per-Class plots
    if has_per_class:
        ax = axes[1, 0]
        for class_id in range(num_classes):
            ax.plot(steps_arr, train_loss_pc[class_id], linewidth=1.5, label=f"C{class_id}")
        ax.set_ylabel("Loss")
        ax.set_title("Train Loss Per Class")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for class_id in range(num_classes):
            ax.plot(steps_arr, test_loss_pc[class_id], linewidth=1.5, label=f"C{class_id}")
        ax.set_ylabel("Loss")
        ax.set_title("Test Loss Per Class")
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        for class_id in range(num_classes):
            ax.plot(steps_arr, train_acc_pc[class_id], linewidth=1.5, label=f"C{class_id}")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Train Accuracy Per Class")
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        for class_id in range(num_classes):
            ax.plot(steps_arr, test_acc_pc[class_id], linewidth=1.5, label=f"C{class_id}")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Test Accuracy Per Class")
        ax.grid(True, alpha=0.3)

        if num_classes <= 12:
            axes[1, 0].legend(loc="best", ncol=2, fontsize=8)
            axes[1, 1].legend(loc="best", ncol=2, fontsize=8)
            axes[2, 0].legend(loc="best", ncol=2, fontsize=8)
            axes[2, 1].legend(loc="best", ncol=2, fontsize=8)

    if tpt_step >= 0.0:
        for axis in axes.flat:
            axis.axvline(
                x=tpt_step,
                color="black",
                linewidth=2.0,
                linestyle="-",
            )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_example_distribution_dynamics(
    class_distribution_history: np.ndarray,
    output_path: Path,
    title: str,
    tpt_step: int = -1,
) -> None:
    if class_distribution_history.ndim != 2:
        raise ValueError(
            "class_distribution_history must be a 2D array with shape "
            "(num_steps, num_classes)."
        )

    num_steps, num_classes = class_distribution_history.shape
    if num_steps == 0 or num_classes == 0:
        raise ValueError("class_distribution_history cannot be empty.")

    steps = np.arange(1, num_steps + 1, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(14, 7))
    for class_id in range(num_classes):
        ax.plot(
            steps,
            class_distribution_history[:, class_id],
            linewidth=1.2,
            alpha=0.9,
            label=f"Class {class_id}",
        )

    if tpt_step >= 0:
        ax.axvline(
            x=tpt_step,
            color="black",
            linewidth=2.0,
            linestyle="-",
            label=f"TPT start ({tpt_step})",
        )
    else:
        ax.text(
            0.98,
            0.98,
            "TPT not reached",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Fraction of Batch Examples")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    if num_classes <= 16:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_neural_collapse_differences(
    steps: np.ndarray,
    pairwise_differences: np.ndarray,
    class_pairs: list[tuple[int, int]],
    output_path: Path,
    title: str,
    tpt_step: int = -1,
) -> None:
    if steps.ndim != 1:
        raise ValueError("steps must be a 1D array.")
    if pairwise_differences.ndim != 2:
        raise ValueError("pairwise_differences must be a 2D array.")
    if steps.shape[0] != pairwise_differences.shape[0]:
        raise ValueError("steps and pairwise_differences must have the same number of rows.")
    if pairwise_differences.shape[1] != len(class_pairs):
        raise ValueError("Number of columns in pairwise_differences must match class_pairs length.")
    if steps.size == 0:
        raise ValueError("No points available to plot neural collapse differences.")

    fig, ax = plt.subplots(figsize=(14, 7))
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            pairwise_differences[:, pair_idx],
            linewidth=1.8,
            alpha=0.9,
            label=f"||mu_{left} - mu_{right}||",
        )

    if tpt_step >= 0:
        ax.axvline(
            x=tpt_step,
            color="black",
            linewidth=2.0,
            linestyle="-",
            label=f"TPT start ({tpt_step})",
        )

    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Norm of Class Mean Difference")
    ax.grid(True, alpha=0.3)

    if len(class_pairs) <= 16:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)