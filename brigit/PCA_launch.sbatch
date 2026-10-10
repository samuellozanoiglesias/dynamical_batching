#!/bin/bash
#SBATCH --job-name=PCA_analysis
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

#/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_baseline_cnn/training_2026_07_23-20_54_10/
#/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_metaclasses_scratch/training_2026_07_23-17_04_15/
#/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_metaclasses_scratch_intuitive/training_2026_07_23-17_04_15/
#/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_metaclasses_finetune/training_2026_07_23-19_55_54/
#/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_metaclasses_finetune_intuitive/training_2026_07_23-19_56_28/

folder_dir="/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_metaclasses_finetune/training_2026_07_23-19_55_54/"

echo "Starting PCA analysis..."
python PCA_explained_from_checkpoint.py $folder_dir

echo "Job finished."