#!/usr/bin/env bash
set -euo pipefail

echo "==== 0) Larry-Maia Setup Starting ===="
VENV_DIR="./.venv"

# -------------------- Parameters --------------------
DOWNSAMPLE_COUNT=5000   # Number of Lichess moves to use in training

# -------------------- Environment --------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "⚡ Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  NEW_VENV=1
else
  echo "==== 0.5) Virtual environment already exists ===="
  NEW_VENV=0
fi

source "$VENV_DIR/bin/activate"

if [ "$NEW_VENV" -eq 1 ]; then
  echo "==== 0.75) Installing TensorFlow and dependencies ===="
  pip install --upgrade pip setuptools wheel
  pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 "numpy<2" pandas python-chess tqdm requests zstandard scikit-learn matplotlib seaborn pytest
else
  echo "==== 0.75) Dependencies already installed ===="
fi

# -------------------- Datasets --------------------
DATA_DIR="./data"
SAMPLED="$DATA_DIR/lichess_train_sampled.jsonl"
DOWNSAMPLED="$DATA_DIR/lichess_train_sampled_${DOWNSAMPLE_COUNT}.jsonl"

if [ ! -f "$DOWNSAMPLED" ]; then
  echo "⚡ Downsampling $SAMPLED → $DOWNSAMPLED ($DOWNSAMPLE_COUNT)"
  python - <<END
import json, random
with open("$SAMPLED") as f_in, open("$DOWNSAMPLED", "w") as f_out:
    lines = f_in.readlines()
    random.seed(42)
    for line in random.sample(lines, min($DOWNSAMPLE_COUNT, len(lines))):
        f_out.write(line)
END
else
  echo "✅ Downsampled dataset already exists: $DOWNSAMPLED"
fi

# -------------------- Models --------------------
mkdir -p ./models
BASE_MODEL="./models/maia_larry_base.keras"
FINE_TUNE_MODEL="./models/maia_larry_finetuned.keras"
MOVE_MAP_JSON="$DATA_DIR/move_map.json"

# -------------------- Base Training --------------------
if [ -f "$BASE_MODEL" ]; then
    echo "✅ Base model already exists: $BASE_MODEL, skipping training"
else
    echo "==== 2) Training Maia-Larry base model ===="
    python "$PWD/scripts/train_base_maia.py" \
      --data "$DOWNSAMPLED" \
      --out_model "$BASE_MODEL" \
      --epochs 3 \
      --batch_size 256 \
      --lr 1e-3
fi

# -------------------- Fine-tuning on Larry's games --------------------
LARRY_PGN="$DATA_DIR/larry_games.pgn"
LARRY_JSONL="$DATA_DIR/larry_games.jsonl"

# Convert PGN → JSONL if needed
if [ -f "$LARRY_PGN" ] && [ ! -f "$LARRY_JSONL" ]; then
  echo "⚡ Converting $LARRY_PGN → $LARRY_JSONL"
  python "$PWD/scripts/pgn_to_training.py" --pgn "$LARRY_PGN" --out "$LARRY_JSONL"
fi

if [ -f "$LARRY_JSONL" ]; then
    if [ -f "$FINE_TUNE_MODEL" ]; then
        echo "✅ Fine-tuned model already exists: $FINE_TUNE_MODEL, skipping fine-tuning"
    else
        echo "✅ Found Larry's dataset: $LARRY_JSONL"
        echo "⚡ Fine-tuning Maia-Larry on Larry's games..."
        python "$PWD/scripts/train_base_maia.py" \
          --data "$LARRY_JSONL" \
          --out_model "$FINE_TUNE_MODEL" \
          --epochs 2 \
          --batch_size 128 \
          --lr 5e-4
        echo "✅ Fine-tuned model saved to $FINE_TUNE_MODEL"
    fi
else
    echo "⚠️ Skipping fine-tuning: $LARRY_JSONL not found."
fi

# -------------------- Save move map for tests --------------------
if [ ! -f "$MOVE_MAP_JSON" ]; then
    echo "⚡ Saving move map for tests → $MOVE_MAP_JSON"
    python - <<END
import json, numpy as np
from scripts.train_base_maia import load_dataset, build_move_map

X, y_raw = load_dataset("$DOWNSAMPLED")
move_map = build_move_map(y_raw)
with open("$MOVE_MAP_JSON", "w") as f:
    json.dump(move_map, f)
END
else
    echo "✅ Move map already exists: $MOVE_MAP_JSON"
fi

# -------------------- Done --------------------
echo "🎯 Pipeline complete!"
echo "✅ Models:"
echo "   Base model: $BASE_MODEL"
echo "   Fine-tuned model: $FINE_TUNE_MODEL (if available)"
echo "✅ Move map: $MOVE_MAP_JSON"