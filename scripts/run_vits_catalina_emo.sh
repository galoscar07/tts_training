#!/usr/bin/env bash
# Emotion fine-tune of VITS on CATALINA, with the four emotions as speaker
# slots (catalina_angry / _calm / _happy / _neutral).
#
#   bash scripts/run_vits_catalina_emo.sh prepare   # manifest + warm-start ckpt
#   bash scripts/run_vits_catalina_emo.sh start     # train in the background
#   bash scripts/run_vits_catalina_emo.sh status | logs | stop
#
# `prepare` warm-starts the speaker embedding from the single-speaker Catalina
# fine-tune so each emotion slot begins from the voice the model already knows.
# Without it coqui's partial restore drops emb_g.weight and the run relearns
# the speaker from scratch on ~1.4k utterances — see vits_expand_speakers.py.
#
# Knobs (env):
#   BASE_CKPT=<pth>   warm-start source (default: highest checkpoint_*.pth in
#                     the latest out/training_runs/vits_catalina_ft run)
#   RESUME=1          continue the emotion run itself (restore its own latest
#                     checkpoint, speaker table already matches — no warm-start)
#   GPUS=0,1,2,3      GPUs to train on
#   LR=1e-4           generator+discriminator learning rate
#   BATCH_SIZE=8      per-GPU batch size
#   EPOCHS=1000  SAVE_STEP=2000  SAVE_N=3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/out/training_runs}"
OUT_DIR="$RUN_ROOT/vits_catalina_emo"
FT_ROOT="${FT_ROOT:-$RUN_ROOT/vits_catalina_ft}"
MANIFEST="${MANIFEST:-$REPO_ROOT/out/catalina_emotions.manifest}"
CORPUS="${CORPUS:-$REPO_ROOT/datasets/CATALINA}"
INIT_CKPT="${INIT_CKPT:-$REPO_ROOT/out/vits_catalina_emo_init.pth}"

CONTROL_DIR="$RUN_ROOT/control"
PID_FILE="$CONTROL_DIR/vits_emo.pid"
CURRENT_LOG_FILE="$CONTROL_DIR/current-emo-log"
mkdir -p "$CONTROL_DIR"

GPUS="${GPUS:-0,1,2,3}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1000}"
SAVE_STEP="${SAVE_STEP:-2000}"
SAVE_N="${SAVE_N:-3}"

is_running() { [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

latest_ckpt() {  # highest-numbered checkpoint under a run root
    local root="$1" run
    run="$(ls -1dt "$root"/*/ 2>/dev/null | head -1)"; run="${run%/}"
    [ -n "$run" ] || return 1
    ls -1 "$run"/checkpoint_*.pth 2>/dev/null \
        | sed 's/.*checkpoint_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

prepare() {
    echo "=== 1/2  manifest (emotion as speaker) ==="
    if [ -s "$MANIFEST" ]; then
        echo "reusing $MANIFEST"
    else
        "$PYTHON" -m tts_training.data.manifest \
            --dataset catalina_emotions --corpus-root "$CORPUS" --out "$MANIFEST"
    fi
    echo "rows per speaker:"
    cut -d'|' -f3 "$MANIFEST" | sort | uniq -c

    echo
    echo "=== 2/2  warm-start checkpoint ==="
    if [ "${RESUME:-0}" = "1" ]; then
        echo "RESUME=1 — skipping warm-start, will restore the emotion run's own checkpoint"
        return
    fi
    local base="${BASE_CKPT:-$(latest_ckpt "$FT_ROOT" || true)}"
    [ -n "$base" ] || { echo "ERROR: no source checkpoint; set BASE_CKPT=<pth>" >&2; exit 1; }
    "$PYTHON" scripts/vits_expand_speakers.py \
        --checkpoint "$base" \
        --speakers-file "$(dirname "$base")/speakers.pth" \
        --manifest "$MANIFEST" \
        --out "$INIT_CKPT"
}

start_background() {
    if is_running; then echo "already running (PID $(cat "$PID_FILE"))"; exit 1; fi
    [ -s "$MANIFEST" ] || { echo "ERROR: $MANIFEST missing — run 'prepare' first" >&2; exit 1; }
    local log_dir="$OUT_DIR/logs"
    mkdir -p "$log_dir"
    nohup setsid "$0" _run > "$log_dir/launcher.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo "started (PID $(cat "$PID_FILE"))"
    echo "run dir: $OUT_DIR"
    echo "follow:  bash $0 logs"
}

run_training() {
    local ts log
    ts="$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUT_DIR/logs"
    log="$OUT_DIR/logs/training-${ts}.log"
    ln -sfn "$log" "$OUT_DIR/logs/training-latest.log"
    echo "$log" > "$CURRENT_LOG_FILE"
    echo $$ > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT INT TERM

    local restore
    if [ "${RESUME:-0}" = "1" ]; then
        restore="$(latest_ckpt "$OUT_DIR" || true)"
        [ -n "$restore" ] || { echo "ERROR: RESUME=1 but no checkpoint in $OUT_DIR" >&2; exit 1; }
    else
        restore="$INIT_CKPT"
        [ -s "$restore" ] || { echo "ERROR: $restore missing — run 'prepare' first" >&2; exit 1; }
    fi

    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" PYTHONUNBUFFERED=1

    # Hyphenated flags are unknown to trainer.distribute, so they are forwarded
    # to train.py rather than consumed (same trick as run_vits_training.sh).
    local -a cmd=(
        --script src/tts_training/train.py
        --manifest "$MANIFEST" --corpus-root "$CORPUS"
        --output "$OUT_DIR" --run-name vits_catalina_emo
        --restore-path "$restore"
        --batch-size "$BATCH_SIZE" --eval-batch-size "$BATCH_SIZE"
        --lr "$LR"
        --epochs "$EPOCHS" --print-step 25
        --save-step "$SAVE_STEP" --save-n-checkpoints "$SAVE_N"
    )

    echo "restore-path: $restore"
    echo "gpus: $GPUS   lr: $LR   batch: $BATCH_SIZE"
    if [[ "$GPUS" == *,* ]]; then
        "$PYTHON" -m trainer.distribute --gpus "$GPUS" "${cmd[@]}" 2>&1 | tee "$log"
    else
        CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m tts_training.train "${cmd[@]:2}" 2>&1 | tee "$log"
    fi
}

case "${1:-}" in
    prepare) prepare ;;
    start)   start_background ;;
    status)
        if is_running; then
            echo "RUNNING (PID $(cat "$PID_FILE"))"
        else
            echo "NOT RUNNING"
        fi
        RUN_ROOT="$OUT_DIR" bash "$REPO_ROOT/scripts/vits_catalina_status.sh" || true
        ;;
    logs)
        [ -f "$CURRENT_LOG_FILE" ] || { echo "no log yet"; exit 1; }
        tail -n 100 -F "$(cat "$CURRENT_LOG_FILE")"
        ;;
    stop)
        if is_running; then kill -TERM -- "-$(cat "$PID_FILE")"; echo "stopped"; else echo "not running"; fi
        ;;
    _run) run_training ;;
    *) echo "Usage: $0 {prepare|start|status|logs|stop}"; exit 2 ;;
esac
