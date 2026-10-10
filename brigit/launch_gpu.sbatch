#!/bin/bash
#SBATCH --job-name=dynamical_sgd_launcher
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
# We enforce the GPU override explicitly just to be safe, though your script 
# handles auto-detection. You can change '--mode all' to 'single' or a specific 
# configuration as outlined in your Python file's docstring.

config_file="config/OTHER_DATASETS/spiral_big-always_bumps.yaml"
seed=20

echo "Starting training job with configuration: $config_file"
python launch.py \
    --config $config_file \
    --override use_gpu=true \
    --override data.data_dir=/mnt/lustre/home/samuloza/dynamical-sgd/data \
    --override output.output_dir=/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd \
    --override training.random_seed="${seed}" \
    --override data.random_seed="${seed}"
echo "Job finished."