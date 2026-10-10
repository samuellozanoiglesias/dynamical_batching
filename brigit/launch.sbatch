#!/bin/bash
#SBATCH --job-name=dynamical_sgd_launcher
#SBATCH -t 02:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
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

echo "Starting training job..."
python launch.py \
    --config config/mnist_mlp_1layer-always_bumps.yaml \
    --override use_gpu=false \
    --override data.data_dir=/mnt/lustre/home/samuloza/dynamical-sgd/data \
    --override output.output_dir=/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd

echo "Job finished."