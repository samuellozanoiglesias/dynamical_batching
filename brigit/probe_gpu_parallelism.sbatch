#!/bin/bash
#SBATCH --job-name=gpu_parallel_probe
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-probe-%x-%j.out
#SBATCH --error=error-probe-%x-%j.err

# ==========================================
# Purpose
# ==========================================
# Finds out how many copies of your training script can run at once on the
# single GPU allocated to this job, by launching an increasing number of
# short training runs in parallel and watching for OOM / crashes.
# Read $PROBE_DIR/nvidia-smi.log afterwards to see memory/utilization headroom.

source ~/.bashrc
conda activate dynamical-sgd
cd /mnt/lustre/home/samuloza/dynamical-sgd/

config_file="config/CIFAR10/cifar10_plaincnn_bottleneck50-always_bumps.yaml"
PROBE_DIR=/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gpu_probe/${SLURM_JOB_ID}
mkdir -p "$PROBE_DIR"

# ==========================================
# JAX memory behavior
# ==========================================
# By default JAX preallocates ~75% of GPU memory PER PROCESS the moment it
# initializes, regardless of actual model size. That alone is enough to make
# 2-3 processes fight over the whole card before any of them do real work.
# Disabling preallocation makes each process grow its memory usage on demand
# instead of grabbing a fixed chunk upfront.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ==========================================
# Tunables
# ==========================================
MAX_TO_TEST=20       # highest concurrency level to try; GPU memory was NOT the limiter last run
STEPS_FOR_PROBE=200  # a bit longer than before so timing differences are measurable, not noise

# Background GPU memory/utilization logger (every 2s)
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 2 > "$PROBE_DIR/nvidia-smi.log" &
MONITOR_PID=$!

for N in $(seq 1 $MAX_TO_TEST); do
    echo "=== [$(date)] Testing with $N concurrent process(es) ==="
    RUN_DIR="$PROBE_DIR/n${N}"
    mkdir -p "$RUN_DIR"

    t_start=$(date +%s)
    pids=()
    for i in $(seq 1 $N); do
        python launch.py \
            --config $config_file \
            --override use_gpu=true \
            --override data.data_dir=/mnt/lustre/home/samuloza/dynamical-sgd/data \
            --override output.output_dir=${RUN_DIR}/proc${i} \
            --override training.training_steps=${STEPS_FOR_PROBE} \
            --override training.save_metrics_every_n_steps=100 \
            > "${RUN_DIR}/proc${i}.log" 2>&1 &
        pids+=($!)
    done

    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    t_end=$(date +%s)
    elapsed=$((t_end - t_start))
    per_run_equiv=$(awk "BEGIN{printf \"%.1f\", ${elapsed}/${N}}")

    if [ $fail -eq 1 ]; then
        echo "!!! N=$N had at least one failure (check ${RUN_DIR}/proc*.log for OOM/errors) — stopping probe"
        break
    else
        echo "N=$N: all completed in ${elapsed}s wall-clock (${per_run_equiv}s per run, normalized) — steps/s across the batch: $(awk "BEGIN{printf \"%.1f\", (${N}*${STEPS_FOR_PROBE})/${elapsed}}")"
    fi
    sleep 5
done

kill $MONITOR_PID 2>/dev/null
echo "Probe finished."
echo "  - Per-N logs: $PROBE_DIR/n<N>/proc<i>.log"
echo "  - GPU memory/util over time: $PROBE_DIR/nvidia-smi.log"
echo ""
echo "Since memory is unlikely to be the limiter on a 46GB card, pick MAX_PARALLEL"
echo "by looking at the 'steps/s across the batch' numbers above: increasing N should"
echo "roughly increase that number too. Pick the N just before it flattens out or drops -"
echo "beyond that point you're paying for more processes without getting more real throughput"
echo "(CPU/dataloader or GPU compute contention, not memory)."
