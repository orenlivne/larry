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
  echo "==== 0.75) TensorFlow and dependencies installing ===="
  pip install --upgrade pip setuptools wheel
  pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 "numpy<2" pandas python-chess tqdm requests zstandard scikit-learn matplotlib seaborn pytest
else
  echo "==== 0.75) TensorFlow and dependencies already installed ===="
fi

# -------------------- Datasets --------------------
DATA_DIR="./data"
SAMPLED="$DATA_DIR/lichess_train_sampled.jsonl"
DOWNSAMPLED="$DATA_DIR/lichess_train_sampled_${DOWNSAMPLE_COUNT}.jsonl"

# Downsample Lichess dataset to DOWNSAMPLE_COUNT if needed
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

# -------------------- Main Training --------------------
echo "==== 1) Checking dataset ===="
if [ -f "$DOWNSAMPLED" ]; then
  echo "✅ Found downsampled dataset: $DOWNSAMPLED"
else
  echo "⚠️  Missing downsampled dataset: $DOWNSAMPLED"
  exit 1
fi

mkdir -p ./models

echo "==== 2) Training Maia-Larry model ===="
python "$PWD/scripts/train_base_maia.py" \
  --data "$DOWNSAMPLED" \
  --out_model "./models/maia_larry_base.keras" \
  --epochs 3 \
  --batch_size 256 \
  --lr 1e-3

# -------------------- Fine-tuning --------------------
echo "==== 3) Fine-tuning on Larry's games ===="

# Larry’s PGN file
LARRY_PGN="$DATA_DIR/larry_games.pgn"
LARRY_JSONL="$DATA_DIR/larry_games.jsonl"
FINE_TUNE_MODEL="./models/maia_larry_finetuned.keras"

# Convert PGN → JSONL if needed
if [ -f "$LARRY_PGN" ] && [ ! -f "$LARRY_JSONL" ]; then
  echo "⚡ Converting $LARRY_PGN → $LARRY_JSONL"
  python "$PWD/scripts/pgn_to_training.py" --pgn "$LARRY_PGN" --output "$LARRY_JSONL"
fi

# Fine-tune if JSONL exists
if [ -f "$LARRY_JSONL" ]; then
  echo "✅ Found Larry's dataset: $LARRY_JSONL"
  echo "⚡ Fine-tuning Maia-Larry on Larry's games..."
  python "$PWD/scripts/train_base_maia.py" \
    --data "$LARRY_JSONL" \
    --out_model "$FINE_TUNE_MODEL" \
    --epochs 2 \
    --batch_size 128 \
    --lr 5e-4
  echo "✅ Fine-tuned model saved to $FINE_TUNE_MODEL"
else
  echo "⚠️  Skipping fine-tuning: $LARRY_JSONL not found."
fi

# -------------------- Done --------------------
echo "🎯 Pipeline complete!"
echo "✅ Models:"
echo "   Base model: ./models/maia_larry_base.keras"
echo "   Fine-tuned model: ./models/maia_larry_finetuned.keras (if available)"