#!/bin/bash
#SBATCH --job-name=generate_reduce_dim_plot_mlp_launcher
#SBATCH --time=30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

# ==========================================
# Environment Setup
# ==========================================
# Activar entorno conda
source ~/.bashrc
conda activate dynamical-sgd

cd /mnt/lustre/home/samuloza/dynamical-sgd/


# ==========================================
# Execution Command
# ==========================================

echo "Starting job..."

# One run_dir per trained MLP checkpoint (each containing config.yaml,
# jax_params.pkl, dataset_bundle.npz as saved by training_runner.py).
# Add/remove entries here to cover whichever datasets you want plotted.
run_dirs=(
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-always_bumps/training_2026_08_13-16_31_00"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-always_bumps/training_2026_08_13-19_12_42"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/with_bumps/rings-always_bumps/training_2026_08_13-16_31_29"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/with_bumps/rings-always_bumps/training_2026_08_13-19_10_18"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/training_2026_08_13-13_22_11"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/training_2026_08_13-16_30_16"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/training_2026_08_13-16_30_54"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/training_2026_08_13-22_13_17"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/training_2026_08_13-23_01_09"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/training_2026_08_13-13_22_11"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/training_2026_08_13-16_31_29"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/training_2026_08_13-18_56_28"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/training_2026_08_13-22_12_41"
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/training_2026_08_13-22_12_43"
)

reduce_methods=(pca umap tsne)
reduce_dims=(2 3 5)
plot_dims=(2 3)
feature_layer="pre_classifier"

for run_dir in "${run_dirs[@]}"; do
    echo "Generating feature-space reduced-dim plots for run_dir=$run_dir, feature_layer=$feature_layer, reduce_methods=${reduce_methods[*]}, reduce_dims=${reduce_dims[*]}, plot_dims=${plot_dims[*]}"
    python generate_reduce_dim_plot_mlp.py \
        --run_dir "$run_dir" \
        --reduce_dims "${reduce_dims[@]}" \
        --plot_dims "${plot_dims[@]}" \
        --reduce_methods "${reduce_methods[@]}" \
        --feature_layer "$feature_layer"
done

echo "Job finished."
