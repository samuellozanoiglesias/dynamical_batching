#!/usr/bin/env python3
"""
Launch script for config-driven training runs.

How to run:

1) Single run using the config as-is
    python launch.py --config config/spiral_mlp_1layer.yaml --mode single

2) Run all four bump setups automatically
    python launch.py --config config/spiral_mlp_1layer.yaml --mode all

3) Run a specific setup
    python launch.py --config config/spiral_mlp_1layer.yaml --mode with_bumps
    python launch.py --config config/spiral_mlp_1layer.yaml --mode without_bumps
    python launch.py --config config/spiral_mlp_1layer.yaml --mode with_bumps_at_tpt
    python launch.py --config config/spiral_mlp_1layer.yaml --mode with_bumps_before_tpt

4) Override values from the command line (can be repeated)
    python launch.py --config config/spiral_mlp_1layer.yaml --mode single \
         --override training.training_steps=5000 \
         --override training.batch_size=128 \
         --override optimizer.learning_rate=0.001 \
         --override use_gpu=true \
         --override data.points_per_class=2000 \
         --override data.noise_std=0.05

5) Spiral custom initial angles (degrees)
    python launch.py --config config/spiral_mlp_1layer.yaml --mode single \
         --override data.angular_offsets="[0,20,45,70,110,160,210,250,300,340]"

nohup example (run in background with a config file):

    nohup python launch.py --config config/spiral_mlp_1layer.yaml --mode single \
         > launch_spiral_mlp_1layer.log 2>&1 &

Useful monitoring commands:
    tail -f launch_spiral_mlp_1layer.log
    ps aux | grep "python launch.py"

Notes:
- Results are written under output.output_dir / inferred_experiment_name / output.config_name / training_<training_id>.
- A fresh training_id is generated at each launcher invocation to avoid overwriting old runs.
- In mode=single, dynamics flags come from the config file itself and infer the experiment name.
- In mode=all (or one of the named setup modes), dynamics flags are set by mode.
- output.experiment_name, output.config_name, and output.experiment_timestamp are always generated at runtime and saved into each run's config.yaml.
- Device selection can be controlled with either use_gpu=true/false (preferred) or device=auto|cuda|cpu.
- Optimizer hyperparameters are read from optimizer.* (optimizer_type, learning_rate, beta1, beta2, eps, gradient_clipping, l2_reg/weight_decay).
"""
from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def _parse_override_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_override(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor: Dict[str, Any] = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_config(config_path: Path, overrides: Iterable[str]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        _apply_override(config, key.strip(), _parse_override_value(raw_value.strip()))

    return config


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _infer_experiment_name(dynamics_cfg: Dict[str, Any]) -> str:
    bumps_before_tpt = _as_bool(dynamics_cfg.get("bumps_before_tpt", False))
    bumps_at_tpt = _as_bool(dynamics_cfg.get("bumps_at_tpt", False))

    if bumps_before_tpt and bumps_at_tpt:
        return "with_bumps"
    if (not bumps_before_tpt) and bumps_at_tpt:
        return "with_bumps_at_tpt"
    if bumps_before_tpt and (not bumps_at_tpt):
        return "with_bumps_before_tpt"
    return "without_bumps"


def _ensure_output_section(config: Dict[str, Any], config_path: Path, run_timestamp: str) -> None:
    output_cfg = config.setdefault("output", {})
    output_cfg.setdefault("output_dir", "./outputs")
    output_cfg["config_name"] = config_path.stem
    output_cfg["experiment_timestamp"] = run_timestamp
    output_cfg.pop("experiment_name", None)


def _normalize_device_config(config: Dict[str, Any]) -> None:
    if "use_gpu" in config:
        config["device"] = "cuda" if _as_bool(config.get("use_gpu")) else "cpu"
        return

    device_raw = config.get("device", "auto")
    if isinstance(device_raw, bool):
        config["device"] = "cuda" if device_raw else "cpu"
        return

    normalized = str(device_raw).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    elif normalized in {"1", "true", "yes", "on"}:
        normalized = "cuda"
    elif normalized in {"0", "false", "no", "off"}:
        normalized = "cpu"

    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            "Invalid device setting. Use use_gpu=true/false or device=auto|cuda|cpu. "
            f"Got '{device_raw}'."
        )
    config["device"] = normalized


def _build_training_id() -> str:
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


def _mode_plan(mode: str) -> List[Tuple[str, bool | None, bool | None]]:
    if mode == "with_bumps_tpt":
        mode = "with_bumps_at_tpt"

    if mode == "single":
        return [("single", None, None)]

    named_modes: List[Tuple[str, bool, bool]] = [
        ("with_bumps", True, True),
        ("without_bumps", False, False),
        ("with_bumps_at_tpt", False, True),
        ("with_bumps_before_tpt", True, False),
    ]
    if mode == "all":
        return [(name, before, after) for name, before, after in named_modes]

    for name, before, after in named_modes:
        if mode == name:
            return [(name, before, after)]

    raise ValueError(f"Unsupported mode '{mode}'.")


def _build_run_dir(config: Dict[str, Any], training_id: str) -> Path:
    output_cfg = config["output"]
    output_base = Path(str(output_cfg.get("output_dir", "./outputs"))).expanduser().resolve()
    experiment_name = str(output_cfg.get("experiment_name", "dynamical_sgd"))
    config_name = str(output_cfg.get("config_name", "config"))
    run_leaf = f"training_{training_id}"

    run_dir = output_base / experiment_name / config_name / run_leaf
    suffix = 1
    while run_dir.exists():
        run_dir = output_base / experiment_name / config_name / f"{run_leaf}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_effective_config(run_dir: Path, config: Dict[str, Any]) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven training launcher for dynamical-sgd")
    parser.add_argument(
        "--config",
        type=str,
        default="config/spiral_mlp_1layer.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=[
            "single",
            "with_bumps",
            "without_bumps",
            "with_bumps_at_tpt",
            "with_bumps_tpt",
            "with_bumps_before_tpt",
            "all",
        ],
        help="Run mode. 'single' uses dynamics flags from config; 'all' executes all four bump setups.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with dot notation, e.g. --override training.training_steps=5000",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = _load_config(config_path, args.override)
    run_timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    _ensure_output_section(base_config, config_path, run_timestamp=run_timestamp)
    _normalize_device_config(base_config)

    # JAX may fail CPU fallback if CUDA plugins are broken; force the selected backend early.
    selected_device = str(base_config.get("device", "auto")).strip().lower()
    if selected_device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    elif selected_device == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["JAX_PLATFORM_NAME"] = "cuda"

    # Delay importing the JAX training stack until backend preference is pinned.
    from training_runner import run_training

    setup_plan = _mode_plan(args.mode)
    training_id = _build_training_id()

    print("=" * 90)
    print("DYNAMICAL-SGD TRAINING")
    print("=" * 90)
    print(f"Config: {config_path}")
    print(f"Mode: {args.mode}")
    print(f"Device: {base_config.get('device', 'auto')}")
    print(f"Experiment timestamp: {run_timestamp}")
    print(f"Training ID: {training_id}")
    print("=" * 90)

    all_summaries: List[Dict[str, Any]] = []
    run_dirs: List[Path] = []
    for idx, (setup_name, bumps_before, bumps_after) in enumerate(setup_plan, start=1):
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("dynamics", {})
        cfg.setdefault("output", {})

        if bumps_before is not None and bumps_after is not None:
            cfg["dynamics"]["bumps_before_tpt"] = bool(bumps_before)
            cfg["dynamics"]["bumps_at_tpt"] = bool(bumps_after)

        inferred_experiment_name = _infer_experiment_name(cfg["dynamics"])
        cfg["output"]["experiment_name"] = inferred_experiment_name
        cfg["output"]["config_name"] = config_path.stem
        cfg["output"]["experiment_timestamp"] = run_timestamp

        run_dir = _build_run_dir(cfg, training_id=training_id)
        _write_effective_config(run_dir, cfg)

        print()
        print("-" * 90)
        print(f"Run {idx}/{len(setup_plan)}: requested={setup_name} | inferred={inferred_experiment_name}")
        print(f"Output: {run_dir}")
        print(
            "Dynamics: "
            f"bumps_before_tpt={cfg['dynamics'].get('bumps_before_tpt', False)} | "
            f"bumps_at_tpt={cfg['dynamics'].get('bumps_at_tpt', False)}"
        )
        print("-" * 90)

        summary = run_training(cfg, run_dir=run_dir, run_label=inferred_experiment_name)
        all_summaries.append(summary)
        run_dirs.append(run_dir)

    print("\nAll requested runs completed.")


if __name__ == "__main__":
    main()
