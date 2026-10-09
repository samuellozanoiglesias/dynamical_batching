from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import ListedColormap
import yaml
import jax.numpy as jnp

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import JAXModel, ParamTree
from fig2_style import style_axes

# ========================================================================
# CONFIG -- tweak sizes & fonts freely
# ========================================================================
PLOT_FIGSIZE = (10, 10)                 # 10x10 square plot canvas
CBAR_FIGSIZE = (10.0, 1.2)              # Standalone colorbar dimensions (width, height in inches)
CBAR_AX_RECT = [0.05, 0.40, 0.90, 0.45]  # colorbar axes inside CBAR_FIGSIZE (fractions)

TICK_FONTSIZE = 34            # Plot axis tick labels size
LABEL_FONTSIZE = 40           # Plot axis labels size
LEGEND_FONTSIZE = 34          # Legend font size
LEGEND_MARKERSCALE = 3        # Multiplier for legend marker sizes
CBAR_TICK_FONTSIZE = 40       # Colorbar tick numbers size
AXES_LINEWIDTH = 1.1

# Point / line sizes at standalone scale (the compound figure scales them)
TRAIN_S, TRAIN_EDGE_LW = 35, 0.25
TEST_S, TEST_LW = 45, 1.2
BOUNDARY_LW = 0.5
GRID_LW = 0.8
LEGEND_FRAME_LW = 1.0

RC = {
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": AXES_LINEWIDTH,
}

PATH_DYN = '/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-320_bumps/training_2026_09_24-02_48_56/'
PATH_STATIC = '/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_320_bumps/training_2026_09_24-02_39_50/'
PATH = PATH_STATIC


def _desaturate_towards_white(color, mix: float = 0.6):
    r, g, b, _a = color
    return (r + (1.0 - r) * mix, g + (1.0 - g) * mix, b + (1.0 - b) * mix, 1.0)


def _extract_2d_points(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim == 2 and x.shape[1] == 2:
        return x, y
    if x.ndim == 4 and x.shape[1] == 2 and x.shape[2] == 1 and x.shape[3] == 1:
        return x[:, :, 0, 0], y
    raise ValueError("Decision boundary plot expects 2D inputs with shape (N,2) or (N,2,1,1).")


# ========================================================================
# computation (model evaluation on a grid) -- separated from drawing
# ========================================================================
def compute_decision_boundary(
    model: JAXModel,
    params: ParamTree,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    grid_size: int = 400,
) -> dict:
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

    model_input = grid_xy
    if int(np.asarray(train_inputs).ndim) == 4:
        model_input = np.reshape(model_input, (-1, 2, 1, 1))

    logits = model.apply(params, model_input)
    pred = np.asarray(jnp.argmax(logits, axis=1), dtype=np.int64).reshape(xx.shape)

    num_classes = int(max(np.max(train_y), np.max(test_y), np.max(pred))) + 1
    return {"xx": xx, "yy": yy, "pred": pred,
            "train_xy": train_xy, "train_y": train_y,
            "test_xy": test_xy, "test_y": test_y,
            "num_classes": num_classes}


def class_colormaps(num_classes: int):
    """(points_cmap, background_cmap, levels) -- custom Red - Grey - Blue palette."""
    red_grey_blue = mcolors.LinearSegmentedColormap.from_list(
        "red_grey_blue", ["#d7191c", "#808080", "#2b83ba"])
    idx = [0.5] if num_classes == 1 else np.linspace(0.0, 1.0, num_classes)
    point_colors = [red_grey_blue(i) for i in idx]
    background_colors = [_desaturate_towards_white(c, mix=0.62) for c in point_colors]
    levels = np.arange(-0.5, num_classes + 0.5, 1.0)
    return ListedColormap(point_colors), ListedColormap(background_colors), levels


# ========================================================================
# drawing
# ========================================================================
def draw_decision_boundary(ax, db: dict, num_classes: int | None = None, scale: float = 1.0,
                           rasterized: bool = False):
    """Draw decision regions + samples into `ax`. `scale` shrinks everything uniformly."""
    num_classes = num_classes or db["num_classes"]
    points_cmap, background_cmap, levels = class_colormaps(num_classes)
    s2 = scale ** 2  # scatter sizes are areas

    contour_set = ax.contourf(db["xx"], db["yy"], db["pred"], levels=levels,
                              cmap=background_cmap, alpha=1.0,
                              zorder=-1 if rasterized else None)
    if rasterized:  # everything below zorder 0 (the filled regions) is rasterized in PDFs
        ax.set_rasterization_zorder(0)
    if num_classes > 1:
        ax.contour(db["xx"], db["yy"], db["pred"], levels=np.arange(0.5, num_classes, 1.0),
                   colors="k", linewidths=BOUNDARY_LW * scale, alpha=0.45)

    vmax = max(0, num_classes - 1)
    ax.scatter(db["train_xy"][:, 0], db["train_xy"][:, 1], c=db["train_y"], cmap=points_cmap,
               vmin=0, vmax=vmax, s=TRAIN_S * s2, alpha=0.98, edgecolors="black",
               linewidths=TRAIN_EDGE_LW * scale, label="Train samples")
    ax.scatter(db["test_xy"][:, 0], db["test_xy"][:, 1], c=db["test_y"], cmap=points_cmap,
               vmin=0, vmax=vmax, marker="x", s=TEST_S * s2, alpha=1.0,
               linewidths=TEST_LW * scale, label="Test samples")

    style_axes(ax, scale=scale, spine_lw=AXES_LINEWIDTH)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE * scale)
    ax.set_xlabel("x", fontsize=LABEL_FONTSIZE * scale)
    ax.set_ylabel("y", fontsize=LABEL_FONTSIZE * scale, rotation=0)
    ax.set_title("")
    ax.grid(True, alpha=0.2, linewidth=GRID_LW * scale)
    leg = ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE * scale,
                    markerscale=LEGEND_MARKERSCALE)
    leg.get_frame().set_linewidth(LEGEND_FRAME_LW * scale)
    return contour_set


def draw_class_colorbar(fig, cax, contour, num_classes: int, scale: float = 1.0):
    """Horizontal 'Predicted Class' colorbar into `cax`."""
    cbar = fig.colorbar(contour, cax=cax, orientation="horizontal", ticks=np.arange(num_classes))
    style_axes(cbar.ax, scale=scale, spine_lw=AXES_LINEWIDTH)
    cbar.outline.set_linewidth(AXES_LINEWIDTH * scale)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE * scale)
    cbar.ax.set_xlabel("Predicted Class", fontsize=CBAR_TICK_FONTSIZE * scale)
    return cbar


def save_standalone_colorbar(contour, num_classes: int, output_path: Path,
                             figsize: tuple[float, float] = CBAR_FIGSIZE) -> None:
    """Save the horizontal colorbar as a standalone PNG."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(CBAR_AX_RECT)
    draw_class_colorbar(fig, ax, contour, num_classes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


def plot_2d_decision_boundaries(model, params, train_inputs, train_targets,
                                test_inputs, test_targets, output_path: Path,
                                grid_size: int = 400):
    """Standalone 10x10 decision-boundary PNG. Returns (contour_set, num_classes)."""
    output_path = Path(output_path).with_suffix(".png")
    db = compute_decision_boundary(model, params, train_inputs, train_targets,
                                   test_inputs, test_targets, grid_size)
    return save_decision_boundary_plot(db, output_path)


def save_decision_boundary_plot(db: dict, output_path: Path, num_classes: int | None = None):
    num_classes = num_classes or db["num_classes"]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    contour_set = draw_decision_boundary(ax, db, num_classes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return contour_set, num_classes


plot_spiral_decision_boundaries = plot_2d_decision_boundaries


# ========================================================================
# loading
# ========================================================================
def _load_jax_params(run_dir: Path) -> ParamTree:
    params_path = run_dir / "jax_params.pkl"
    if not params_path.is_file():
        raise FileNotFoundError(f"No jax_params.pkl found in {run_dir}")
    with params_path.open("rb") as f:
        return pickle.load(f)


def _load_model(run_dir: Path) -> JAXModel:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}.")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]

    architecture = str(model_cfg.get("architecture", "mlp")).strip().lower()
    activation_name = {"mlp": "identity", "mlp_relu": "relu", "mlp_tanh": "tanh"}.get(architecture)
    if activation_name is None:
        raise ValueError(f"Unsupported model architecture '{architecture}'.")

    return JAXModel(
        architecture=architecture,
        input_shape=(2,),
        num_classes=int(data_cfg["num_classes"]),
        hidden_dim=int(model_cfg.get("nn_width", 10)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
        use_bias=bool(model_cfg.get("use_bias", True)),
        activation_name=activation_name,
        init_type=str(model_cfg.get("init_type", "glorot")),
    )


def _load_dataset(run_dir: Path):
    bundle_path = run_dir / "dataset_bundle.npz"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"No dataset_bundle.npz found in {run_dir}")
    with np.load(bundle_path) as bundle:
        return (bundle["train_inputs"], bundle["train_targets"],
                bundle["test_inputs"], bundle["test_targets"])


def load_decision_boundary_data(run_dir: Path, grid_size: int = 400) -> dict:
    """Load model + params + dataset of one run and evaluate it on the grid."""
    run_dir = Path(run_dir)
    params = _load_jax_params(run_dir)
    model = _load_model(run_dir)
    tr_x, tr_y, te_x, te_y = _load_dataset(run_dir)
    return compute_decision_boundary(model, params, tr_x, tr_y, te_x, te_y, grid_size)


def reobtain_decision_boundary_plot(
    run_dir: Path,
    grid_size: int = 400,
    output_plot_path: Path = Path("analysis/figures/fig2_decision_boundaries.png"),
    output_cbar_path: Path = Path("analysis/figures/colorbar_decision_boundaries.png"),
) -> tuple[Path, Path] | None:
    if not (run_dir / "jax_params.pkl").is_file():
        print(f"[reobtain] Missing jax_params.pkl in {run_dir}")
        return None

    db = load_decision_boundary_data(run_dir, grid_size)
    with plt.rc_context(RC):
        contour_set, num_classes = save_decision_boundary_plot(db, output_plot_path)
        save_standalone_colorbar(contour_set, num_classes, output_cbar_path)
    return output_plot_path, output_cbar_path


def _find_run_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent]
    if not root.exists():
        return []
    if root.is_dir():
        if root.name.lower().startswith("training_"):
            return [root]
        training_dirs: set[Path] = set()
        for pattern in ("training_*", "Training_*"):
            for path in root.rglob(pattern):
                if path.is_dir():
                    training_dirs.add(path)
        if training_dirs:
            return sorted(training_dirs, key=str)
        if (root / "jax_params.pkl").is_file():
            return [root]
        return sorted({p.parent for p in root.rglob("jax_params.pkl")}, key=str)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reobtain the decision-boundary plot as square PNG with a separate colorbar.")
    parser.add_argument("path", nargs="?", default=PATH, help="Run directory containing jax_params.pkl.")
    parser.add_argument("--grid-size", type=int, default=400, help="Resolution of the decision-boundary grid.")
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser()
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"No run directories found under {root}", file=sys.stderr)
        return 1

    exit_code = 0
    for run_dir in run_dirs:
        try:
            res = reobtain_decision_boundary_plot(run_dir, grid_size=args.grid_size)
        except Exception as exc:
            print(f"[reobtain] Failed for {run_dir}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if res is not None:
            print(f"[reobtain] Saved plot to {res[0]}")
            print(f"[reobtain] Saved colorbar to {res[1]}")
        else:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())