"""
launch_gridCNN.py

    python launch_gridCNN.py --config cfg.yaml \
        --override dynamics.w_max=20 --override training.random_seed=1 ...

--override takes dotted.key=value; the value is parsed as YAML (so 20 -> int,
true -> bool, [1,2] -> list, null -> None). The run directory is exactly
output.output_dir (the sbatch script gives every (w_max, period, seed) its own).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# One CPU thread per process: with 15+ concurrent runs, default torch/OpenMP
# threading oversubscribes the node's cores and slows every run down.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import yaml  # noqa: E402


def _apply_override(cfg: dict, spec: str) -> None:
    if "=" not in spec:
        raise ValueError(f"--override expects key=value, got '{spec}'")
    key, raw = spec.split("=", 1)
    value = yaml.safe_load(raw) if raw.strip() != "" else None
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run_label", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for spec in args.override:
        _apply_override(cfg, spec)

    run_dir = Path(str((cfg.get("output", {}) or {}).get("output_dir", "./runs"))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label or f"{run_dir.parent.name}/{run_dir.name}"

    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    from training_runner_gridCNN import run_training
    summary = run_training(cfg, run_dir, run_label)
    print(
        f"[{run_label}] done in {summary['wall_time_seconds'] / 60:.1f} min | "
        f"final test acc {summary['final_test_accuracy']:.4f} | tpt_step {summary['tpt_step']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
