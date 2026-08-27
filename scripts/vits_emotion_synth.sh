#!/usr/bin/env bash
# Synthesize the four emotion sentence sets with the VITS Catalina fine-tune.
#
#   bash scripts/vits_emotion_synth.sh
#
# Runs on CPU by default so it does not disturb the training job on the GPUs
# (VITS is fast on CPU). Overrides:
#   CKPT=<...pth>      pin a checkpoint (default: highest-step one in the latest run)
#   CONFIG=<...json>   pin the config (default: alongside CKPT)
#   SPEAKER=catalina   speaker slot to synthesize
#   GPU=0              use CUDA device 0 instead of CPU
#   OUT=out/vits_emotions

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/out/training_runs/vits_catalina_ft}"
SPEAKER="${SPEAKER:-catalina}"
OUT="${OUT:-out/vits_emotions}"
SENT_DIR="${SENT_DIR:-data/evaluation/emotion_sentences}"
EMOTIONS="${EMOTIONS:-angry happy neutral calm}"

if [ -z "${CKPT:-}" ]; then
    RUN_DIR="$(ls -1dt "$RUN_ROOT"/*/ 2>/dev/null | head -1)"
    RUN_DIR="${RUN_DIR%/}"
    [ -n "$RUN_DIR" ] || { echo "ERROR: no run dir under $RUN_ROOT" >&2; exit 1; }
    CKPT="$(ls -1 "$RUN_DIR"/checkpoint_*.pth 2>/dev/null \
        | sed 's/.*checkpoint_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
    [ -n "$CKPT" ] || { echo "ERROR: no checkpoint_*.pth in $RUN_DIR" >&2; exit 1; }
fi
CONFIG="${CONFIG:-$(dirname "$CKPT")/config.json}"
SPEAKERS_FILE="${SPEAKERS_FILE:-$(dirname "$CKPT")/speakers.pth}"

echo "checkpoint: $CKPT"
echo "config:     $CONFIG"
echo "out dir:    $OUT"

DEVICE_ARGS=(--no-cuda)
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    DEVICE_ARGS=()
    echo "device:     cuda:$GPU"
else
    echo "device:     cpu (training is using the GPUs; set GPU=N to override)"
fi

SPK_ARGS=()
[ -f "$SPEAKERS_FILE" ] && SPK_ARGS=(--speakers-file "$SPEAKERS_FILE")

echo
echo "=== speakers in this checkpoint ==="
"$PYTHON" -m tts_training.vits.batch_synthesize \
    --checkpoint "$CKPT" --config "$CONFIG" "${SPK_ARGS[@]}" \
    --sentences "$SENT_DIR/neutral.txt" --out-dir "$OUT/_probe" \
    "${DEVICE_ARGS[@]}" --list-speakers

echo
for emo in $EMOTIONS; do
    sent="$SENT_DIR/$emo.txt"
    [ -f "$sent" ] || { echo "skip $emo (missing $sent)"; continue; }
    echo "=== $emo -> $OUT/$emo ==="
    "$PYTHON" -m tts_training.vits.batch_synthesize \
        --checkpoint "$CKPT" --config "$CONFIG" "${SPK_ARGS[@]}" \
        --sentences "$sent" \
        --speakers "$SPEAKER" \
        --out-dir "$OUT/$emo" \
        "${DEVICE_ARGS[@]}" \
        --postprocess
done

echo
echo "done. wavs:"
find "$OUT" -name '*.wav' | sort | head -50
find "$OUT" -name '*.wav' | wc -l
