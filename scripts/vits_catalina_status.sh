#!/usr/bin/env bash
# Status of the VITS Catalina fine-tune (run on the GPU box).
#   bash scripts/vits_catalina_status.sh
# Override the run root with:  RUN_ROOT=... bash scripts/vits_catalina_status.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/out/training_runs/vits_catalina_ft}"

echo "=================== PROCESS ==================="
pgrep -af "train.py|trainer.distribute" | grep -v pgrep || echo "no VITS training process found"

echo
echo "=================== RUN DIR ==================="
if [ ! -d "$RUN_ROOT" ]; then
    echo "run root missing: $RUN_ROOT"; exit 1
fi
RUN_DIR="$(ls -1dt "$RUN_ROOT"/*/ 2>/dev/null | head -1)"
RUN_DIR="${RUN_DIR%/}"
echo "latest run: $RUN_DIR"
echo "last modified: $(date -r "$RUN_DIR" 2>/dev/null)"

echo
echo "=================== CHECKPOINTS ==================="
ls -lh --time-style=+"%Y-%m-%d %H:%M" "$RUN_DIR"/*.pth 2>/dev/null | awk '{print $5, $6, $7, $NF}'
echo "-- highest-step checkpoint --"
ls -1 "$RUN_DIR"/checkpoint_*.pth 2>/dev/null \
    | sed 's/.*checkpoint_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-

echo
echo "=================== TRAINING LOG (tail) ==================="
LOG="$(ls -1t "$RUN_DIR"/logs/training-*.log "$RUN_DIR"/trainer_*_log.txt "$RUN_ROOT"/logs/*.log 2>/dev/null | head -1)"
if [ -n "${LOG:-}" ]; then
    echo "log: $LOG"
    echo "-- last GLOBAL_STEP / epoch --"
    grep -aE "GLOBAL_STEP|EPOCH:" "$LOG" | tail -5
    echo "-- last losses --"
    grep -aE "avg_loss|loss_gen|loss_disc" "$LOG" | tail -12
    echo "-- last 15 raw lines --"
    tail -15 "$LOG"
else
    echo "no training log found under $RUN_DIR/logs or $RUN_DIR"
fi

echo
echo "=================== SPEAKERS IN THIS RUN ==================="
# Answers whether the fine-tune used one 'catalina' slot or per-emotion slots.
if [ -f "$RUN_DIR/config.json" ]; then
    python3 - "$RUN_DIR/config.json" <<'PY' 2>/dev/null || grep -o '"num_speakers":[^,]*' "$RUN_DIR/config.json"
import json, sys
cfg = json.load(open(sys.argv[1]))
ma = cfg.get("model_args", {})
print("use_speaker_embedding:", ma.get("use_speaker_embedding"))
print("num_speakers:", ma.get("num_speakers", cfg.get("num_speakers")))
print("speakers_file:", cfg.get("speakers_file"))
print("run_name:", cfg.get("run_name"))
PY
fi
ls -1 "$RUN_DIR"/speakers.pth 2>/dev/null

echo
echo "=================== DISK ==================="
df -h /media/DATA | tail -1

echo
echo "=================== GPU ==================="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
