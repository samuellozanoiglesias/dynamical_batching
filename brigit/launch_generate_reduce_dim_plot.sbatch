#!/bin/bash
#SBATCH --job-name=generate_reduce_dim_plot_launcher
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

echo "Starting training job..."

checkpoint_type="cifar10_resnet10-always_bumps"
checkpoint_id="training_2026_08_13-14_43_35"
pretrained_num_classes=10
plotting_num_classes=3
metaclass_mapping_type="intuitive"
reduce_method="umap"
reduce_dims=(2 3 5)
plot_dims=(2 3)

for dim in "${reduce_dims[@]}"; do
    for plot_dim in "${plot_dims[@]}"; do
        if [ "$dim" -ge "$plot_dim" ]; then
            echo "Generating plot for checkpoint_type=$checkpoint_type, checkpoint_id=$checkpoint_id, pretrained_num_classes=$pretrained_num_classes, metaclass_mapping_type=$metaclass_mapping_type, reduce_method=$reduce_method, reduce_dim=$dim, plot_dim=$plot_dim"
            python generate_reduce_dim_plot.py \
                --checkpoint_type "$checkpoint_type" \
                --checkpoint_id "$checkpoint_id" \
                --plotting_num_classes $plotting_num_classes \
                --pretrained_num_classes $pretrained_num_classes \
                --metaclass_mapping_type "$metaclass_mapping_type" \
                --reduce_dim $dim \
                --reduce_method $reduce_method \
                --plot_dim $plot_dim
        fi
    done
done

echo "Job finished."