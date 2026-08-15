#!/usr/bin/env bash
#
# One-shot setup: from a bare machine to "ready to train".
#   system packages -> Python 3.10 -> venv -> all deps -> POS model -> manifest
#
# Run from the repo root.
#
# GPU box (Ubuntu, needs sudo for the system step):
#   INSTALL_SYSTEM=1 CUDA=cu121 ./scripts/setup_training_env.sh
#   # ADAPT CUDA to your driver (nvidia-smi): cu121, cu118, ...
#
# Local dev / data-prep only (e.g. macOS: no GPU, no training deps):
#   PYTHON=python3 CUDA=cpu EXTRAS="dev,linguistic,ai" ./scripts/setup_training_env.sh
#
# Knobs (env vars, with defaults):
#   PYTHON=python3.10   interpreter for the venv (matches .python-version)
#   CUDA=cu121          torch build: a CUDA tag (cu121/cu118/...) or "cpu"
#   VENV=.venv          venv directory
#   EXTRAS=dev,linguistic,ai,training   pip extras to install (the 'requirements')
#   DATASET=mara        corpus to build a manifest for
#   INSTALL_SYSTEM=0    1 = apt-get espeak/ffmpeg + deadsnakes Python 3.10 (sudo)
#   BUILD_MANIFEST=1    1 = build out/<DATASET>.manifest after install
#   START_TRAIN=0       1 = launch training at the end

set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
CUDA="${CUDA:-cu121}"
VENV="${VENV:-.venv}"
EXTRAS="${EXTRAS:-dev,linguistic,ai,training}"
DATASET="${DATASET:-mara}"
INSTALL_SYSTEM="${INSTALL_SYSTEM:-0}"
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"
START_TRAIN="${START_TRAIN:-0}"

# Always operate from the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "ERROR: run from the repo root (no pyproject.toml)." >&2; exit 1; }

# --- 1. system packages (optional, needs sudo, Debian/Ubuntu) --------------
if [ "$INSTALL_SYSTEM" = "1" ]; then
    echo ">> [1/5] Installing system packages (sudo)"
    sudo apt-get update
    sudo apt-get install -y espeak ffmpeg git rsync build-essential software-properties-common
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo ">>       $PYTHON not found — adding deadsnakes PPA"
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
    fi
else
    echo ">> [1/5] Skipping system packages (INSTALL_SYSTEM=0)"
fi

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "ERROR: '$PYTHON' not found. Install it (deadsnakes) or set PYTHON=..." >&2; exit 1; }

# --- 2. venv ----------------------------------------------------------------
echo ">> [2/5] Creating venv ($VENV) with $($PYTHON --version)"
"$PYTHON" -m venv "$VENV"
PY="./$VENV/bin/python"
PIP="./$VENV/bin/pip"

# Fail fast if the interpreter is missing stdlib C modules (a source-built
# Python without libbz2/liblzma/libffi/libsqlite dev headers). Stanza and
# Trankit both need _bz2 — catch it here, before a multi-GB install.
if ! "$PY" -c "import bz2, lzma, ctypes, sqlite3" 2>/dev/null; then
    echo "ERROR: '$PYTHON' is missing stdlib C modules (bz2/lzma/ctypes/sqlite3)." >&2
    echo "       That interpreter can't run Stanza/Trankit. Use a complete Python:" >&2
    echo "         deadsnakes:  PYTHON=/usr/bin/python3.10 ... $0" >&2
    echo "         miniconda:   PYTHON=\$HOME/miniconda3/envs/tts/bin/python3.10 ... $0" >&2
    rm -rf "$VENV"
    exit 1
fi

"$PIP" install -U pip wheel

# --- 3. torch + project deps ------------------------------------------------
if [ "$CUDA" = "cpu" ]; then
    echo ">> [3/5] Installing CPU torch + project extras [${EXTRAS}]"
    "$PIP" install torch torchaudio
else
    echo ">> [3/5] Installing CUDA torch ($CUDA) + project extras [${EXTRAS}]"
    "$PIP" install torch torchaudio --index-url "https://download.pytorch.org/whl/${CUDA}"
fi
# Editable install of the project with the requested extras (the 'requirements').
"$PIP" install -e ".[${EXTRAS}]"

# --- 4. POS model -----------------------------------------------------------
# Prefer Trankit (transformer) on Python <=3.10; fall back to Stanza (neural)
# if Trankit can't be loaded/downloaded — e.g. a Python built without _bz2
# (its langid dep needs it) or Python 3.11+. The pipeline itself makes the
# same fallback at runtime, so Stanza-only is fully functional. Base training
# needs only this model, not the emotion model.
echo ">> [4/5] Downloading the POS/linguistic model"
if "$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] < (3,11) else 1)" \
   && "$PY" -c "import trankit" 2>/dev/null; then
    "$PY" scripts/download_trankit_model.py
else
    echo ">>       Using Stanza backend (Trankit unavailable — e.g. Python without _bz2, or Py3.11+)."
    "$PY" scripts/download_stanza_model.py
fi

# --- 5. manifest ------------------------------------------------------------
MANIFEST="out/${DATASET}.manifest"
if [ "$BUILD_MANIFEST" = "1" ]; then
    if [ -d "datasets/${DATASET}" ] || [ -n "${TTS_DATASETS_DIR:-}" ]; then
        echo ">> [5/5] Building manifest -> ${MANIFEST}"
        mkdir -p out
        "$PY" -m tts_training.data.manifest --dataset "$DATASET" --out "$MANIFEST"
    else
        echo ">> [5/5] Skipping manifest: datasets/${DATASET} not found (rsync it, or set TTS_DATASETS_DIR / use --corpus-root)."
        BUILD_MANIFEST=0
    fi
else
    echo ">> [5/5] Skipping manifest (BUILD_MANIFEST=0)"
fi

# --- summary ----------------------------------------------------------------
echo
echo ">> Done."
"$PY" --version
"$PY" - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("torch check failed:", e)
PY

TRAIN_CMD="$PY -m tts_training.train --manifest ${MANIFEST} --corpus-root datasets/${DATASET} --output out/vits_ro_base"
echo
if [ "$START_TRAIN" = "1" ] && [ "$BUILD_MANIFEST" = "1" ]; then
    echo ">> START_TRAIN=1 -> launching training"
    eval "$TRAIN_CMD"
else
    echo ">> To start training:"
    echo "   $TRAIN_CMD"
fi
