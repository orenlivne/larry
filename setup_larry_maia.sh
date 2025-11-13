#!/usr/bin/env bash
set -euo pipefail

echo "==== 0) Larry-Maia Setup Starting ===="
VENV_DIR="./.venv"

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
  pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 numpy<2 pandas python-chess tqdm requests zstandard scikit-learn matplotlib seaborn
else
  echo "==== 0.75) TensorFlow and dependencies already installed ===="
fi

echo "==== 1) Checking dataset ===="
DATA_DIR="./data"
SAMPLED="$DATA_DIR/lichess_train_sampled.jsonl"
if [ -f "$SAMPLED" ]; then
  echo "✅ Found dataset: $SAMPLED"
else
  echo "⚠️  Missing dataset: $SAMPLED. Please ensure it is present."
fi

echo "==== 2) Training Maia-Larry model ===="
python "$PWD/scripts/train_base_maia.py" \
  --data "$SAMPLED" \
  --out_model "./models/maia_larry.h5" \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3

echo "🎯 Pipeline complete!"
echo "✅ Model saved to ./models/maia_larry.h5"