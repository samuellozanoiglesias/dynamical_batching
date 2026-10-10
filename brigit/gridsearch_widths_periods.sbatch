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

config_file="config/CIFAR10/cifar10_plaincnn_bottleneck50-always_bumps.yaml"
BASE_OUTPUT_DIR=/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods
mkdir -p "$BASE_OUTPUT_DIR"

# ==========================================
# Grid definition — edit these
# ==========================================
W_MAXS=(1 5 10 20 30 50 75 100 125 150 175 200 225 250)        # -> dynamics.w_max
PERIODS=(10 15 20 30 40 50 75 100 150 200 400 800 1000 1200 1500)     # -> dynamics.period_length

# ==========================================
# Concurrency — set this from probe_gpu_parallelism.sbatch results
# ==========================================
MAX_PARALLEL=15

# ==========================================
# JAX memory behavior
# ==========================================
# By default JAX preallocates ~75% of GPU memory PER PROCESS on init, which
# alone is enough to starve concurrent runs regardless of actual model size.
# Disable that, and give each process a memory ceiling sized to MAX_PARALLEL
# (with a bit of slack) so they share the card instead of fighting over it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk "BEGIN{printf \"%.2f\", 0.85/${MAX_PARALLEL}}")
echo "Per-process GPU memory fraction cap: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# ==========================================
# Build the combination list
# ==========================================
combos=()
for W in "${W_MAXS[@]}"; do
    for P in "${PERIODS[@]}"; do
        combos+=("${W}:${P}")
    done
done

echo "Total combinations: ${#combos[@]}  (${#W_MAXS[@]} w_maxs x ${#PERIODS[@]} periods)"
echo "Running with MAX_PARALLEL=${MAX_PARALLEL}"

run_one () {
    local W=$1
    local P=$2
    local run_name="w${W}_p${P}"
    local out_dir="${BASE_OUTPUT_DIR}/${run_name}"
    mkdir -p "$out_dir"
    echo "[$(date)] Starting ${run_name}"
    python launch.py \
        --config $config_file \
        --override use_gpu=true \
        --override data.data_dir=/mnt/lustre/home/samuloza/dynamical-sgd/data \
        --override output.output_dir=${out_dir} \
        --override dynamics.w_max=${W} \
        --override dynamics.period_length=${P} \
        > "${out_dir}/run.log" 2>&1
    local status=$?
    echo "[$(date)] Finished ${run_name} (exit code ${status})"
}

# ==========================================
# Launch with a throttled parallel pool
# ==========================================
pids=()
for combo in "${combos[@]}"; do
    IFS=':' read -r W P <<< "$combo"
    run_one "$W" "$P" &
    pids+=($!)

    if [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; then
        wait -n
        # drop finished pids from the tracking array
        alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        pids=("${alive[@]}")
    fi
done

wait
echo "Grid search finished. Results in ${BASE_OUTPUT_DIR}, one subfolder per (width, period) combo."
