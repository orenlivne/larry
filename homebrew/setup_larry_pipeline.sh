#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# setup_larry_pipeline_full.sh
# Full idempotent pipeline (ARM64 macOS, TF+Metal)
# Move map created before training, random sampling to 2.5M
# ==========================================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
SCRIPTS="$ROOT_DIR/scripts"
VENV_DIR="$ROOT_DIR/.venv"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"

# Model dirs
MAIA_DIR="$DATA_DIR/maia-2200"
BASE_H5="$MAIA_DIR/maia-2200.h5"
BASE_SAVED_MODEL_DIR="$MAIA_DIR/saved_model"
SAVE_DIR="$DATA_DIR/larrybot"
LARRY_H5="$SAVE_DIR/larrybot.h5"
LARRY_SAVED_MODEL_DIR="$SAVE_DIR/saved_model"

# Lichess
LICHESS_RAW="$DATA_DIR/lichess_raw"
YEAR="${YEAR:-2025}"
MONTH="01"
LICHESS_URL="https://database.lichess.org/standard/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
LICHESS_JSONL="$DATA_DIR/lichess_${YEAR}.jsonl"
LICHESS_TRAIN_JSONL="$DATA_DIR/lichess_train.jsonl"
SAMPLE_GAMES=${SAMPLE_GAMES:-200000}
SAMPLED_JSONL="$DATA_DIR/lichess_train_sampled.jsonl"

# Larry dataset
LARRY_PGN="$DATA_DIR/larry_games.pgn"
LARRY_JSONL="$DATA_DIR/larry_dataset.jsonl"
LARRY_TRAIN_JSONL="$DATA_DIR/larry_train.jsonl"

# Move map
MOVE_MAP="$SAVE_DIR/move_index_map.json"
REQUIRED_TF="2.16.1"

# -----------------------------
# 0) Validate Python & venv
# -----------------------------
echo "=== 0) Validate ARM64 Python & create/activate virtualenv ==="
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ ERROR: ARM64 Python not found at $PYTHON_BIN"; exit 1
fi
ARCH=$("$PYTHON_BIN" -c 'import platform; print(platform.machine())')
if [ "$ARCH" != "arm64" ]; then
  echo "❌ ERROR: Python at $PYTHON_BIN is not arm64 (found: $ARCH)"; exit 1
fi
mkdir -p "$DATA_DIR" "$SCRIPTS" "$LICHESS_RAW"
if [ ! -d "$VENV_DIR" ]; then
  echo "⚙️ Creating venv at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  PYTHON="$(which python)"
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
else
  source "$VENV_DIR/bin/activate"
  PYTHON="$(which python)"
fi

# -----------------------------
# 0.5) Install TF + Metal
# -----------------------------
echo "=== 0.5) Install TensorFlow-macos + Metal ==="
INSTALLED_TF="$($PYTHON -c 'import tensorflow as tf; print(tf.__version__)')"
if [ "$INSTALLED_TF" = "$REQUIRED_TF" ]; then
  echo "✅ tensorflow $REQUIRED_TF already installed."
else
  "$PYTHON" -m pip install --upgrade --force-reinstall \
    "tensorflow-macos==$REQUIRED_TF" "tensorflow-metal==1.1.0" \
    numpy<2 pandas python-chess tqdm requests zstandard scikit-learn matplotlib seaborn
fi

# -----------------------------
# 0.75) TF sanity check
# -----------------------------
echo "=== 0.75) TensorFlow + GPU check ==="
"$PYTHON" - <<'PY'
import sys, tensorflow as tf
print("Python:", sys.executable)
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if not gpus: print("⚠️ No GPU detected")
else: print("✅ GPU detected:", gpus)
PY

# -----------------------------
# 1) Download Lichess PGN
# -----------------------------
echo "=== 1) Download & filter Lichess PGN ==="
if [ -s "$LICHESS_JSONL" ]; then
  echo "✅ Found $LICHESS_JSONL — skipping download."
else
  "$PYTHON" "$SCRIPTS/download_lichess_games.py" \
    --url "$LICHESS_URL" --out "$LICHESS_JSONL" \
    --max-games "$SAMPLE_GAMES" --workers "$(sysctl -n hw.ncpu)"
fi

# -----------------------------
# 2) Convert JSONL → board/move JSONL
# -----------------------------
echo "=== 2) Convert JSONL → board/move JSONL ==="
if [ ! -s "$LICHESS_TRAIN_JSONL" ]; then
  CORES=$(($(sysctl -n hw.ncpu)-1))
  "$PYTHON" "$SCRIPTS/convert_to_train_jsonl.py" \
    --input "$LICHESS_JSONL" --output "$LICHESS_TRAIN_JSONL" --workers "$CORES"
fi

# -----------------------------
# 3) Generate move_index_map.json BEFORE training
# -----------------------------
echo "=== 3) Generate move_index_map.json ==="
mkdir -p "$SAVE_DIR"
if [ ! -f "$MOVE_MAP" ]; then
  "$PYTHON" "$SCRIPTS/generate_move_index_map.py" \
    --data "$LICHESS_TRAIN_JSONL" --out "$MOVE_MAP"
else
  echo "✅ Found $MOVE_MAP — skipping."
fi

# -----------------------------
# 4) Optional: sample 2.5M moves for training
# -----------------------------
echo "=== 4) Random sample 2.5M moves for training ==="
if [ ! -s "$SAMPLED_JSONL" ]; then
  "$PYTHON" - <<PY
import json, random
with open("$LICHESS_TRAIN_JSONL") as f:
    lines = list(f)
sampled = random.sample(lines, min(len(lines), 2500000))
with open("$SAMPLED_JSONL", "w") as out_f:
    for l in sampled:
        out_f.write(l)
print("✅ Sampled 2.5M moves to $SAMPLED_JSONL")
PY
else
  echo "✅ Found sampled train set — skipping."
fi

# -----------------------------
# 5) Train base Maia model
# -----------------------------
echo "=== 5) Train base Maia model ==="
if [ -f "$BASE_H5" ]; then
  echo "✅ Found base model $BASE_H5 — skipping."
else
  mkdir -p "$MAIA_DIR"
  "$PYTHON" "$SCRIPTS/train_base_fallback.py" \
    --data "$SAMPLED_JSONL" --move-map "$MOVE_MAP" --out "$MAIA_DIR"
  if [ -d "$BASE_SAVED_MODEL_DIR" ]; then
    "$PYTHON" - <<PY
import tensorflow as tf
model = tf.keras.models.load_model("$BASE_SAVED_MODEL_DIR")
model.save("$BASE_H5", save_format="h5")
print("✅ Saved base model:", "$BASE_H5")
PY
  fi
fi

# -----------------------------
# 6) Prepare Larry dataset
# -----------------------------
echo "=== 6) Prepare Larry dataset ==="
if [ ! -s "$LARRY_JSONL" ] && [ -s "$LARRY_PGN" ]; then
  "$PYTHON" "$SCRIPTS/pgn_to_training.py" --pgn "$LARRY_PGN" --out "$LARRY_JSONL"
fi
if [ -s "$LARRY_JSONL" ] && [ ! -s "$LARRY_TRAIN_JSONL" ]; then
  CORES=$(($(sysctl -n hw.ncpu)-1))
  "$PYTHON" "$SCRIPTS/convert_to_train_jsonl.py" \
    --input "$LARRY_JSONL" --output "$LARRY_TRAIN_JSONL" --workers "$CORES"
fi

# -----------------------------
# 7) Fine-tune LarryBot
# -----------------------------
echo "=== 7) Fine-tune base → LarryBot ==="
if [ -f "$LARRY_H5" ]; then
  echo "✅ Found fine-tuned $LARRY_H5 — skipping."
else
  mkdir -p "$SAVE_DIR"
  "$PYTHON" "$SCRIPTS/train_base_fallback.py" \
    --data "$LARRY_TRAIN_JSONL" --move-map "$MOVE_MAP" --init-from "$BASE_H5" \
    --out "$SAVE_DIR" --epochs 3 --batch-size 128 --lr 1e-5
  if [ -d "$LARRY_SAVED_MODEL_DIR" ]; then
    "$PYTHON" - <<PY
import tensorflow as tf
model = tf.keras.models.load_model("$LARRY_SAVED_MODEL_DIR")
model.save("$LARRY_H5", save_format="h5")
print("✅ Saved fine-tuned LarryBot:", "$LARRY_H5")
PY
  fi
fi

# -----------------------------
# Done
# -----------------------------
echo ""
echo "🎯 LarryBot pipeline complete!"
echo "  Base model:    $BASE_H5"
echo "  Fine-tuned:    $LARRY_H5"
echo "  Move map:      $MOVE_MAP"