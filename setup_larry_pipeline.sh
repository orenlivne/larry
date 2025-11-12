#!/usr/bin/env bash
set -euo pipefail

# ===============================
# LarryBot Full Setup Script
# ===============================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
SCRIPTS="$ROOT_DIR/scripts"
MAIA_DIR="$DATA_DIR/maia-2200"
LICHESS_JSONL="$DATA_DIR/lichess_2025.jsonl"
TRAIN_JSONL="$DATA_DIR/train.jsonl"
VAL_JSONL="$DATA_DIR/val.jsonl"
SAVE_DIR="$DATA_DIR/larrybot"

echo "=== 5) Convert PGN → JSONL dataset (legacy, skip if JSONL exists) ==="
if [ -f "$LICHESS_JSONL" ]; then
  echo "✅ Found $LICHESS_JSONL — skipping conversion."
else
  echo "⚙️ Converting lichess PGN to JSONL..."
  uv run python "$SCRIPTS/download_lichess_games.py" \
    --zst "$DATA_DIR/lichess_raw/lichess_db_standard_rated_2025-01.pgn.zst" \
    --out "$LICHESS_JSONL" \
    --min-elo 2400 \
    --max-elo 2800 \
    --max-games 200000
fi

echo "=== 6) Obtain base Maia model (optional) ==="
if [ -d "$MAIA_DIR" ]; then
  echo "✅ Found existing $MAIA_DIR — skipping download."
else
  echo "⚙️ Downloading Maia 2200 base model..."
  mkdir -p "$MAIA_DIR"
  curl -L -o "$MAIA_DIR/maia-2200.h5" https://storage.googleapis.com/maia-models/maia-2200.h5
fi

echo "=== 6.5) Split dataset into train/val sets ==="
if [ -f "$TRAIN_JSONL" ] && [ -f "$VAL_JSONL" ]; then
  echo "✅ Found $TRAIN_JSONL and $VAL_JSONL — skipping split."
else
  echo "⚙️ Splitting $LICHESS_JSONL into train/val..."
  uv run python "$SCRIPTS/split_dataset.py" \
    --input "$LICHESS_JSONL" \
    --train-out "$TRAIN_JSONL" \
    --val-out "$VAL_JSONL" \
    --val-fraction 0.1
fi

echo "=== 7) Train base model on Lichess ==="
if [ ! -d "$MAIA_DIR/maia-2200.h5/saved_model" ]; then
  echo "⚙️ Training base Maia-style model from Lichess..."
  uv run python "$SCRIPTS/train_base_fallback.py" \
    --data "$TRAIN_JSONL" \
    --out "$MAIA_DIR/maia-2200.h5" \
    --epochs 3 \
    --batch-size 256 \
    --lr 1e-3
else
  echo "✅ Found existing $MAIA_DIR/maia-2200.h5 — skipping base training."
fi

echo "=== 7.5) Convert Larry PGN → JSONL ==="
LARRY_JSONL="$DATA_DIR/larry_games.jsonl"
if [ -f "$LARRY_JSONL" ]; then
  echo "✅ Found $LARRY_JSONL — skipping conversion."
else
  echo "⚙️ Converting Larry PGN → JSONL..."
  uv run python "$SCRIPTS/pgn_to_jsonl.py" \
    --pgn "$DATA_DIR/larry_games.pgn" \
    --out "$LARRY_JSONL"
fi

echo "=== 8) Fine-tune on Larry's games ==="
if [ -d "$SAVE_DIR/larrybot" ]; then
  echo "✅ Found existing $SAVE_DIR/larrybot — skipping fine-tuning."
else
  echo "⚙️ Fine-tuning Maia → LarryBot ..."
  uv run python "$SCRIPTS/train_base_fallback.py" \
    --data "$DATA_DIR/larry_games.jsonl" \
    --out "$SAVE_DIR/larrybot" \
    --init-from "$MAIA_DIR/maia-2200.h5" \
    --epochs 1 \
    --batch-size 128 \
    --lr 1e-5
fi

echo "✅ LarryBot setup completed successfully!"
