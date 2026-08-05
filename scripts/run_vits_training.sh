#!/usr/bin/env bash
# Convenience controller for background four-GPU VITS training.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VITS_PYTHON:-$REPO_ROOT/.venv/bin/python}"
RUN_ROOT="${VITS_RUN_ROOT:-$HOME/tts_runs}"
CONTROL_DIR="$RUN_ROOT/control"
PID_FILE="$CONTROL_DIR/vits.pid"
CURRENT_LOG_FILE="$CONTROL_DIR/current-log"
CURRENT_GPU_LOG_FILE="$CONTROL_DIR/current-gpu-log"
LAST_STATUS_FILE="$CONTROL_DIR/last-exit-status"

mkdir -p "$RUN_ROOT" "$CONTROL_DIR"

usage() {
    echo "Usage: $0 {start|smoke|status|logs|gpu-logs|stop}"
}

is_running() {
    [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start_background() {
    local mode="$1"
    if is_running; then
        echo "VITS is already running (PID $(cat "$PID_FILE"))."
        exit 1
    fi
    rm -f "$PID_FILE" "$LAST_STATUS_FILE"
    local output="$RUN_ROOT/vits_ro_${mode}"
    local log_dir="$output/logs"
    mkdir -p "$log_dir"
    nohup setsid "$0" _run "$mode" > "$log_dir/launcher.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    echo "Started VITS $mode run in background (PID $pid)."
    echo "Run directory: $output"
    echo "Training log: $log_dir/training-latest.log"
    echo "Follow it with: $0 logs"
}

run_training() {
    local mode="$1"
    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    local output="$RUN_ROOT/vits_ro_${mode}"
    local log_dir="$output/logs"
    local train_log="$log_dir/training-${timestamp}.log"
    local gpu_log="$log_dir/gpu-${timestamp}.csv"
    local -a mode_args=()

    if [ "$mode" = "smoke" ]; then
        mode_args=(--small_run 256)
        output="$RUN_ROOT/vits_ro_smoke"
        log_dir="$output/logs"
        train_log="$log_dir/training-${timestamp}.log"
        gpu_log="$log_dir/gpu-${timestamp}.csv"
    fi

    mkdir -p "$log_dir"
    ln -sfn "$train_log" "$log_dir/training-latest.log"
    ln -sfn "$gpu_log" "$log_dir/gpu-latest.csv"
    echo "$train_log" > "$CURRENT_LOG_FILE"
    echo "$gpu_log" > "$CURRENT_GPU_LOG_FILE"
    echo "$$" > "$PID_FILE"

    (
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader
            sleep 10
        done
    ) > "$gpu_log" 2>&1 &
    local monitor_pid=$!

    cleanup() {
        kill "$monitor_pid" 2>/dev/null || true
        rm -f "$PID_FILE"
    }
    trap cleanup EXIT INT TERM

    cd "$REPO_ROOT"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export PYTHONUNBUFFERED=1

    set +e
    "$PYTHON" -m trainer.distribute \
        --gpus "${VITS_GPUS:-0,1,2,3}" \
        "${mode_args[@]}" \
        --script src/tts_training/train.py \
        --manifest out/mara.manifest \
        --corpus-root datasets/MARA \
        --manifest out/swara_train.manifest \
        --corpus-root datasets/SWARA \
        --output "$output" \
        --run-name "vits_ro_mara_swara_${mode}" \
        --batch-size "${VITS_BATCH_SIZE:-8}" \
        --eval-batch-size "${VITS_EVAL_BATCH_SIZE:-8}" \
        --num-loader-workers "${VITS_WORKERS:-4}" \
        --num-eval-loader-workers "${VITS_EVAL_WORKERS:-2}" \
        --epochs "$([ "$mode" = "smoke" ] && echo 1 || echo "${VITS_EPOCHS:-1000}")" \
        --print-step "$([ "$mode" = "smoke" ] && echo 5 || echo 25)" \
        --save-step "$([ "$mode" = "smoke" ] && echo 100 || echo 5000)" \
        --save-n-checkpoints "$([ "$mode" = "smoke" ] && echo 2 || echo 5)" \
        2>&1 | tee "$train_log"
    local status=${PIPESTATUS[0]}
    set -e

    echo "$status" > "$LAST_STATUS_FILE"
    echo "VITS $mode run finished with exit status $status" | tee -a "$train_log"
    return "$status"
}

case "${1:-}" in
    start)
        start_background full
        ;;
    smoke)
        start_background smoke
        ;;
    status)
        if is_running; then
            echo "RUNNING (PID $(cat "$PID_FILE"))"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
        else
            echo "NOT RUNNING"
            [ -f "$LAST_STATUS_FILE" ] && echo "Last exit status: $(cat "$LAST_STATUS_FILE")"
            true
        fi
        ;;
    logs)
        [ -f "$CURRENT_LOG_FILE" ] || { echo "No training log yet."; exit 1; }
        tail -n 100 -F "$(cat "$CURRENT_LOG_FILE")"
        ;;
    gpu-logs)
        [ -f "$CURRENT_GPU_LOG_FILE" ] || { echo "No GPU log yet."; exit 1; }
        tail -n 40 -F "$(cat "$CURRENT_GPU_LOG_FILE")"
        ;;
    stop)
        if is_running; then
            pid="$(cat "$PID_FILE")"
            kill -TERM -- "-$pid"
            echo "Sent TERM to VITS process group $pid."
        else
            echo "VITS is not running."
        fi
        ;;
    _run)
        run_training "${2:?missing run mode}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
