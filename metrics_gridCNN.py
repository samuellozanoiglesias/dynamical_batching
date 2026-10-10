"""
metrics_gridCNN.py

Only the overall train/test loss + accuracy report survives from metrics.py
(no per-class panels, no decision boundaries, no distribution / NC plots, no
jax import). matplotlib is imported lazily so runs with output.plot: false
never pay its import cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_training_report(csv_path: Path, output_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return
    data = np.atleast_1d(data)
    steps = data["global_step"]
    tpt_step = float(np.max(data["tpt_step"])) if "tpt_step" in data.dtype.names else -1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0]
    ax.plot(steps, data["train_loss"], lw=1.5, label="Train")
    ax.plot(steps, data["test_loss"], lw=1.5, label="Test")
    ax.set_title("Loss")
    ax.set_xlabel("Global Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(steps, data["train_accuracy"], lw=1.5, label="Train")
    ax.plot(steps, data["test_accuracy"], lw=1.5, label="Test")
    ax.set_title("Accuracy")
    ax.set_xlabel("Global Step")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if tpt_step >= 0:
        for a in axes:
            a.axvline(x=tpt_step, color="black", lw=2.0)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
