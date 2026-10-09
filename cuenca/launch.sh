#USE: bash launch.sh

# =====================================================
# Configuration
# =====================================================

CONFIG_NAME="dartboard-1_320_bumps"
CONFIG="config/ICLR_STUDY/${CONFIG_NAME}.yaml"
DATA_DIR="/home/samuel_lozano/dynamical-sgd/data"
OUTPUT_DIR="/data/samuel_lozano/dynamical-sgd"

# Seeds to run
SEEDS=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)
#SEEDS=(0)

# Maximum simultaneous jobs
MAX_PARALLEL=10

# =====================================================
# Launch function
# =====================================================

LOG="launch_${CONFIG_NAME}.log"

# If we're not already running under nohup, restart ourselves.
if [[ -z "$LAUNCH_INTERNAL" ]]; then
    export LAUNCH_INTERNAL=1
    nohup bash "$0" >> "$LOG" 2>&1 &
    echo "Launched in background. Log: $LOG"
    exit 0
fi

cd /home/samuel_lozano/dynamical_batching || exit 1

run_job () {
    seed=$1

    echo "Starting seed ${seed}"

    python launch.py \
        --config "${CONFIG}" \
        --override use_gpu=false \
        --override data.data_dir="${DATA_DIR}" \
        --override output.output_dir="${OUTPUT_DIR}" \
        --override training.random_seed="${seed}" \
        --override data.random_seed="${seed}"

    echo "Finished seed ${seed}"
}

# =====================================================
# Parallel execution
# =====================================================

running=0

for seed in "${SEEDS[@]}"; do
    run_job "${seed}" &
    ((running++))
    
    sleep 10

    if (( running >= MAX_PARALLEL )); then
        wait -n
        ((running--))
    fi
done

wait

echo "All jobs finished."