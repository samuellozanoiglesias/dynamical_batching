#!/bin/bash
#SBATCH --job-name=gridsearch_wmaxs_periods
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
source ~/.bashrc
conda activate dynamical-sgd
cd /mnt/lustre/home/samuloza/dynamical-sgd/

config_file="${1:-config/CIFAR10/ICLR/cifar10_ICLR_resnet10standard_gridCNN.yaml}"
config_name="$(basename "${config_file}" .yaml)"
DATA_DIR=/mnt/lustre/home/samuloza/dynamical-sgd/data
BASE_OUTPUT_DIR=/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods/${config_name}
mkdir -p "$BASE_OUTPUT_DIR"

# ==========================================
# Grid definition — edit these
# ==========================================
W_MAXS=(1 5 10 20 30 50 75 100 125 150 175 200 225 250)                 # -> dynamics.w_max
PERIODS=(10 15 20 30 40 50 75 100 150 200 400 800 1000 1200 1500)       # -> dynamics.period_length
SEEDS=(0 10 20 30 40)                                                  # -> training.random_seed

# ==========================================
# Concurrency
# ==========================================
MAX_PARALLEL=15      # concurrent python processes sharing the GPU
LAUNCH_DELAY=10       # seconds to wait after each launch (staggers CUDA init / data loading)
USE_MPS=0            # 1 = start NVIDIA MPS so concurrent processes truly share the GPU (if the cluster allows it)

# ==========================================
# CPU / GPU sharing behaviour (PyTorch — this runner no longer imports JAX)
# ==========================================
export OMP_NUM_THREADS=1              # 15 procs x default threads would oversubscribe 24 cores
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with many procs

if [ "$USE_MPS" -eq 1 ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_${SLURM_JOB_ID}
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${SLURM_JOB_ID}
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d && echo "MPS daemon started"
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
fi

# ==========================================
# Build the CIFAR-10 uint8 cache ONCE (avoids 15 procs racing to extract it)
# ==========================================
python generate_dataset_gridCNN.py --data_dir "${DATA_DIR}" || { echo "Dataset preparation failed"; exit 1; }

# ==========================================
# Build the combination list (seed outermost: a full grid for the first
# seed finishes before the second seed starts, so early results are usable)
# ==========================================
combos=()
for S in "${SEEDS[@]}"; do
    for W in "${W_MAXS[@]}"; do
        for P in "${PERIODS[@]}"; do
            combos+=("${W}:${P}:${S}")
        done
    done
done

echo "Config: ${config_file}"
echo "Total runs: ${#combos[@]}  (${#W_MAXS[@]} w_maxs x ${#PERIODS[@]} periods x ${#SEEDS[@]} seeds)"
echo "Running with MAX_PARALLEL=${MAX_PARALLEL}, LAUNCH_DELAY=${LAUNCH_DELAY}s"

run_one () {
    local W=$1 P=$2 S=$3
    local run_name="w${W}_p${P}/seed${S}"
    local out_dir="${BASE_OUTPUT_DIR}/w${W}_p${P}/seed${S}"
    mkdir -p "$out_dir"
    echo "[$(date)] Starting ${run_name}"
    python launch_gridCNN.py \
        --config "$config_file" \
        --run_label "${config_name}/${run_name}" \
        --override use_gpu=true \
        --override data.data_dir="${DATA_DIR}" \
        --override output.output_dir="${out_dir}" \
        --override dynamics.w_max="${W}" \
        --override dynamics.period_length="${P}" \
        --override training.random_seed="${S}" \
        > "${out_dir}/run.log" 2>&1
    local status=$?
    echo "[$(date)] Finished ${run_name} (exit code ${status})"
}

# ==========================================
# Launch with a throttled parallel pool
# ==========================================
pids=()
n_skipped=0
for combo in "${combos[@]}"; do
    IFS=':' read -r W P S <<< "$combo"

    if [ -f "${BASE_OUTPUT_DIR}/w${W}_p${P}/seed${S}/summary.json" ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    run_one "$W" "$P" "$S" &
    pids+=($!)
    sleep "$LAUNCH_DELAY"

    if [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; then
        wait -n
        alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        pids=("${alive[@]}")
    fi
done

wait
echo "Skipped ${n_skipped} already-finished runs."
echo "Grid search finished. Results in ${BASE_OUTPUT_DIR}/w<W>_p<P>/seed<S>/"
