# Dynamical Batching

Code, configuration files and analysis scripts for the article

> N. Ratier-Werbin, S. Lozano and M. Ruiz-García,
> **Dynamical Batching: How Class-Cycling Training Shapes the Internal Structure of Neural Networks**,
> arXiv:XXXX.XXXXX (2026).

<!-- After the first Zenodo release, replace XXXXXXX with the concept-DOI number:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

**Dynamical Batching (DB)** is a sampling protocol that periodically emphasizes each class in the mini-batches, while leaving the loss and the optimizer unchanged. One focal class is emphasized per period of `T` steps, with a triangular ramp of amplitude `A`. A *cycle* is `C` consecutive periods, one per class. `A = 1` recovers **Uniform Batching (UB)**, in which every mini-batch has equal class proportions. In the last 5% of training steps, all batches are uniform.

The repository contains:

- **MLP experiments** on synthetic 2D datasets (spiral, blobs, checkerboard, dartboard, rings). These are written in JAX and include the weight-level switching-set analysis.
- **CNN grid searches** on CIFAR-10 over amplitude and period. These are written in PyTorch.
- **The scripts that produce the figures** of the paper.

## Repository structure

| Path | Content |
|---|---|
| `launch.py` | Launcher for the MLP experiments (config-driven, with `--override key=value`). |
| `training_runner.py` | MLP training loop (JAX/Optax): DB batch schedule, metrics, class-state snapshots. |
| `model.py` | MLP definition and initialization (JAX). |
| `generate_dataset.py` | Synthetic datasets and dataset loaders. |
| `metrics.py` | Training reports and decision-boundary plots for MLP runs. |
| `switching_metrics.py` | Switching set, persistence, switching distance, participation ratio and probe-gradient cosines. Can also be run as a script to recompute the decomposition of existing runs. |
| `launch_gridCNN.py` | Launcher for one CIFAR-10 CNN run. |
| `training_runner_gridCNN.py` | CNN training loop (PyTorch), with the same DB batch schedule. |
| `model_cnn_gridCNN.py` | CNN backbones: ResNet-10/18 family with a width multiplier, plain CNN with a bottleneck, Myrtle5. |
| `generate_dataset_gridCNN.py` | CIFAR-10 download and one-time uint8 cache. |
| `metrics_gridCNN.py` | Training report for CNN runs. |
| `config/ICLR_STUDY/` | Configs of the MLP experiments. |
| `config/CIFAR10/` | Configs of the CIFAR-10 grid searches. |
| `analysis/` | Figure scripts and the summary CSVs of the spiral grid searches. |
| `brigit/`, `cuenca/` | Example SLURM and bash scripts used on our clusters (paths must be adapted). |

## Installation

The code was run with Python 3.10.

```bash
git clone https://github.com/samuellozanoiglesias/dynamical_batching.git
cd dynamical_batching
conda create -n dynamical-batching python=3.10
conda activate dynamical-batching
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 12 builds of JAX and PyTorch.

- **MLP experiments** run on CPU (`use_gpu: false` in the configs).
- **CIFAR-10 grid searches** need a CUDA GPU.

On a machine without CUDA, install the CPU builds of `jax` and `torch` instead of the `nvidia-*`, `jax-cuda12-*` and `triton` packages.

## Reproducing the results

Every config has an `output.output_dir` and a `data.data_dir` that point to our cluster. Override both, as in the examples below.

### 1. MLP experiments on synthetic data (Secs. III–IV, Appendixes A–B)

| Experiment | DB config | UB control |
|---|---|---|
| Spiral, small network (`H = 50`), `(A, T) = (20, 320)`, Fig. 3 and Table I | `spiral-320_bumps.yaml` | `spiral-1_320_bumps.yaml` |
| Spiral, small network, doubled budget (250,000 steps) | `spiral-320_bumps_long.yaml` | `spiral-1_320_bumps_long.yaml` |
| Spiral, large network (`H = 500`), `(A, T)= (10, 150)` | `spiral-150_bumps_large.yaml` | `spiral-1_150_bumps_large.yaml` |
| Blobs, checkerboard, dartboard, rings, `(A, T) = (20, 320)` | `<dataset>-320_bumps.yaml` | `<dataset>-1_320_bumps.yaml` |

All configs are in `config/ICLR_STUDY/`. They train a one-hidden-layer ReLU MLP with Adam (learning rate 0.002) for 125,000 steps with batch size 50.

To run one seed:

```bash
python launch.py --config config/ICLR_STUDY/spiral-320_bumps.yaml --mode single \
    --override use_gpu=false \
    --override data.data_dir=./data \
    --override output.output_dir=./runs \
    --override training.random_seed=0 \
    --override data.random_seed=0
```

The paper uses 20 seeds, `0, 5, 10, …, 95`, for every configuration (see `cuenca/launch.sh`, which runs them in parallel).

Each run writes the following to its output directory:
- training metrics (`training_metrics.csv`);
- the per-cycle switching measures (`switching_metrics.csv`, `switch_decomposition.csv`, `switch_decomposition_layerwise.csv`);
- the class-state snapshots (`switching_snapshots/`);
- the final parameters.

To recompute the switching-set decomposition of existing runs, for example with a different energy threshold:

```bash
python switching_metrics.py runs/ --energy-q 0.8
```

The spiral grid searches over amplitude and period (Fig. 2a and the large-batch appendix figure) are included as summary CSVs in `analysis/`:
- `results_new.csv` and `results_large_nn_new.csv`: batch size 50;
- `results_large_bs_new.csv` and `results_large_bs_large_nn_new.csv`: batch size 200.

### 2. CIFAR-10 grid searches (Sec. V, Appendix C)

| Architecture | Config | Paper |
|---|---|---|
| ResNet-10, width 0.15 | `cifar10_ICLR_resnet10narrow_gridCNN.yaml` | main text (Fig. 4) |
| Plain CNN, 50-unit fully connected bottleneck | `cifar10_ICLR_plaincnn_bottleneck50_gridCNN.yaml` | Appendix C |
| Myrtle5, base width 64 | `cifar10_ICLR_myrtle5_gridCNN.yaml` | Appendix C |
| ResNet-10, width 0.5 | `cifar10_ICLR_resnet10_gridCNN.yaml` | Appendix C |

The other configs in `config/CIFAR10/` were not used in the paper.

All four configs share the same training setup:
- **Optimizer:** SGD with momentum 0.9 and weight decay 5e-4, for 100,000 steps with batch size 128.
- **Learning rate:** 0.0679, or 0.05 for the plain CNN, divided by 10 at 1/3 and 2/3 of training.
- **Evaluation:** on fixed, stratified 5,000-example subsets of the training and test sets.

First, build the CIFAR-10 cache (it downloads the dataset if needed):

```bash
python generate_dataset_gridCNN.py --data_dir ./data
```

Then run one cell of the grid:

```bash
python launch_gridCNN.py --config config/CIFAR10/cifar10_ICLR_resnet10narrow_gridCNN.yaml \
    --override use_gpu=true \
    --override data.data_dir=./data \
    --override output.output_dir=./runs_cifar/resnet10narrow/w125_T200_s0 \
    --override dynamics.w_max=125 \
    --override dynamics.period_length=200 \
    --override training.random_seed=0
```

The full grid uses:
- **Amplitudes `A`:** 1, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250.
- **Periods `T`:** 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 400, 800, 1000, 1200, 1500.
- **Seeds:** 0, 10, 20, 30, 40.

`brigit/gridsearch_gridCNN.sbatch` runs the whole grid on one GPU, with several runs in parallel, and skips runs that have already finished.

Each run writes `training_metrics.csv` and `summary.json`, plus optionally `training_report.png` and `model_weights.pt`.

### 3. Figures

| Figure | Script (in `analysis/`) |
|---|---|
| Fig. 1 (protocol) | `fig1_full_panel.py` |
| Fig. 2 (grid search and decision boundaries) | `fig2_compound.py` |
| Fig. 3 (switching set, persistence, distance) | `fig3_subpanels.py` |
| Fig. 4 and Appendix C (CIFAR-10 heatmaps) | `fig4_subpanels.py` |
| Synthetic datasets | `fig_sm_datasets.py` |
| Spiral accuracy curves | `fig_sm_spiral_accuracy.py` |
| Probe-gradient cosine, participation ratio, threshold robustness | `fig_sm_spiral_metrics.py` |
| Large network (`H = 500`) | `fig_sm_large.py` |
| Large-batch grid search | `fig_sm_gridmap.py` |

Run the scripts from inside `analysis/`. Scripts that read run directories have the paths at the top of the file (marked `EDIT`) or as command-line options (`--help`). Point them to your own output directories.

## Citation

If you use this code, please cite the article:

```bibtex
@article{ratierwerbin2026dynamical,
  author  = {Ratier-Werbin, Nicol{\'a}s and Lozano, Samuel and Ruiz-Garc{\'i}a, Miguel},
  title   = {Dynamical Batching: How Class-Cycling Training Shapes the Internal Structure of Neural Networks},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

GitHub's "Cite this repository" button, generated from `CITATION.cff`, gives the same reference.

## License

This code is released under the MIT License (see `LICENSE`).

## Acknowledgments

This work received the support of a fellowship from Fundación Ramón Areces. We acknowledge support from:
- the Ramón y Cajal program (RYC2021-032055-I), funded by MCIN/AEI/10.13039/501100011033 and by European Union NextGenerationEU/PRTR;
- an HFSP Research Grant (RGEC33/2024);
- grant PID2023-147067NB-I00, funded by MCIU/AEI/10.13039/501100011033 and by ERDF/EU.

## Contact

Nicolás Ratier-Werbin (nratier@ucm.es) and Samuel Lozano (samuel.lozano@ucm.es).
