#!/usr/bin/env bash
set -euo pipefail

echo "==== 0) Larry-Maia Setup Starting ===="
VENV_DIR="./.venv"

# -------------------- Parameters --------------------
MIN_ELO=2400
MAX_ELO=2800
MAX_GAMES=10000         # limit downloaded Lichess games
DOWNSAMPLE_COUNT=2500000
LARRY_PGN_DEFAULT="./data/larry_games.pgn"

# Allow overriding via command line:
#   ./setup.sh --larry-pgn myfile.pgn --min-elo 2200 …
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
      --min-elo) MIN_ELO="$2"; shift; shift ;;
      --max-elo) MAX_ELO="$2"; shift; shift ;;
      --max-games) MAX_GAMES="$2"; shift; shift ;;
      --larry-pgn) LARRY_PGN_DEFAULT="$2"; shift; shift ;;
      *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "📌 Using parameters:"
echo "    MIN_ELO=$MIN_ELO"
echo "    MAX_ELO=$MAX_ELO"
echo "    MAX_GAMES=$MAX_GAMES"
echo "    LARRY_PGN=$LARRY_PGN_DEFAULT"

# -------------------- Environment --------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "⚡ Creating virtual environment at $VENV_DIR with Python 3.10"
  python3.10 -m venv "$VENV_DIR"
  NEW_VENV=1
else
  echo "==== 0.5) Virtual environment already exists ===="
  NEW_VENV=0
fi

source "$VENV_DIR/bin/activate"

if [ "$NEW_VENV" -eq 1 ]; then
  echo "==== 0.75) Installing TensorFlow-Metal and dependencies ===="
  pip install --upgrade pip setuptools wheel
  pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 "numpy<2"
  pip install pandas python-chess tqdm requests zstandard scikit-learn matplotlib seaborn berserk pytest
else
  echo "==== 0.75) Dependencies already installed ===="
fi

# -------------------- Datasets --------------------
DATA_DIR="./data"
SAMPLED="$DATA_DIR/lichess_train_sampled.jsonl"
DOWNSAMPLED="$DATA_DIR/lichess_train_sampled_${DOWNSAMPLE_COUNT}.jsonl"

mkdir -p "$DATA_DIR"

# -------------------- 1) Download Lichess (filtered, JSONL) --------------------
if [ ! -f "$SAMPLED" ]; then
  echo "⚡ Downloading + filtering Lichess games → $SAMPLED"
  uv run python scripts/download_lichess_games.py \
    --url "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst" \
    --out "$SAMPLED" \
    --min-elo "$MIN_ELO" \
    --max-elo "$MAX_ELO" \
    --max-games "$MAX_GAMES"
else
  echo "✅ JSONL already exists: $SAMPLED"
fi

# -------------------- 2) Downsample to desired size --------------------
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
  echo "✅ Downsampled dataset exists: $DOWNSAMPLED"
fi

# -------------------- Models --------------------
mkdir -p ./models
BASE_MODEL="./models/maia_larry_base.keras"
FINE_TUNE_MODEL="./models/maia_larry_finetuned.keras"
MOVE_MAP_JSON="$DATA_DIR/move_map.json"

# -------------------- 3) Train Base Model --------------------
if [ -f "$BASE_MODEL" ]; then
    echo "✅ Base model already exists: $BASE_MODEL"
else
    echo "==== 3) Training Maia-Larry base model ===="
    python scripts/train_base_maia.py \
      --data "$DOWNSAMPLED" \
      --out_model "$BASE_MODEL" \
      --epochs 10 \
      --batch_size 256 \
      --lr 1e-3
fi

# -------------------- 4) Fine-tuning on Larry's games --------------------
LARRY_PGN="$LARRY_PGN_DEFAULT"
LARRY_JSONL="$DATA_DIR/larry_games.jsonl"

if [ -f "$LARRY_PGN" ] && [ ! -f "$LARRY_JSONL" ]; then
  echo "⚡ Converting $LARRY_PGN → $LARRY_JSONL"
  python scripts/pgn_to_training.py \
    --pgn "$LARRY_PGN" \
    --out "$LARRY_JSONL"
fi

if [ -f "$LARRY_JSONL" ]; then
    if [ -f "$FINE_TUNE_MODEL" ]; then
        echo "✅ Fine-tuned model exists: $FINE_TUNE_MODEL"
    else
        echo "⚡ Fine-tuning on Larry’s games..."
        # <-- RUN IN VENV USING python directly
        python scripts/fine_tune_larry.py \
          --base_model "$BASE_MODEL" \
          --larry_jsonl "$LARRY_JSONL" \
          --out_model "$FINE_TUNE_MODEL" \
          --epochs 2 \
          --batch_size 128 \
          --lr 5e-4 \
          --freeze_conv
    fi
else
    echo "⚠️ No Larry PGN found. Skipping fine-tune."
fi

# -------------------- 5) Save move map --------------------
if [ ! -f "$MOVE_MAP_JSON" ]; then
    echo "⚡ Saving move map → $MOVE_MAP_JSON"
    python - <<END
import json
from scripts.train_base_maia import load_dataset, build_move_map
X, y_raw = load_dataset("$DOWNSAMPLED")
move_map = build_move_map(y_raw)
with open("$MOVE_MAP_JSON", "w") as f:
    json.dump(move_map, f)
END
else
    echo "✅ Move map already exists: $MOVE_MAP_JSON"
fi

echo "🎯 Pipeline complete!"
echo "   Base model:       $BASE_MODEL"
echo "   Fine-tuned model: $FINE_TUNE_MODEL"
echo "   Move map:         $MOVE_MAP_JSON"