#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/data"
SCRIPTS="$ROOT/scripts"
MAIA_DIR="$DATA/maia-2200"
MAIA_2200_URL="https://github.com/CallOn84/LeelaNets/raw/refs/heads/main/Nets/Maia%202200/maia-2200.pb.gz"
PYPROJECT="$ROOT/pyproject_maia_dummy.toml"

mkdir -p "$DATA" "$SCRIPTS" "$DATA/lichess_raw"

echo "=== 1) Create & activate virtual environment ==="
if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"

echo "=== 2) Install dependencies ==="
pip install --upgrade pip
pip install tensorflow==2.15.0 pandas numpy python-chess tqdm requests zstandard scikit-learn matplotlib seaborn

echo "=== 3) Link dummy pyproject.toml into maia-chess ==="
if [ -d "$ROOT/maia-chess" ]; then
  ln -sf "$PYPROJECT" "$ROOT/maia-chess/pyproject.toml"
else
  echo "⚠️  Warning: maia-chess directory not found. Skipping link."
fi

echo "=== 4) Download & filter Lichess games (Jan 2025 only, Elo 2400–2800) ==="
YEAR="${YEAR:-2025}"
MONTH="01"
LICHESS_DIR="$DATA/lichess_raw"
LICHESS_ZST="$LICHESS_DIR/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
LICHESS_PGN="$DATA/lichess_${YEAR}.pgn"
LICHESS_JSONL="$DATA/lichess_${YEAR}.jsonl"
LICHESS_URL="https://database.lichess.org/standard/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"

mkdir -p "$LICHESS_DIR"

# 1️⃣ Skip if final JSONL exists and non-empty
if [ -s "$LICHESS_JSONL" ]; then
  echo "✅ Found existing $LICHESS_JSONL — skipping download/filter."
else
  # 2️⃣ Download or resume .zst
  if [ -f "$LICHESS_ZST" ]; then
    echo "🔁 Resuming partial download for $LICHESS_ZST ..."
    curl -C - -L -o "$LICHESS_ZST" "$LICHESS_URL" || echo "⚠️ Warning: download interrupted, continuing with partial file."
  else
    echo "⬇️   Downloading $LICHESS_URL ..."
    curl -C - -L -o "$LICHESS_ZST" "$LICHESS_URL" || { echo "❌ Failed to download"; exit 1; }
  fi

  # 3️⃣ Sanity check
  if [ ! -s "$LICHESS_ZST" ]; then
    echo "❌ $LICHESS_ZST is empty or missing after download."
    exit 1
  fi

  # 4️⃣ Filter Elo 2400–2800 directly from .zst (limit to 200k output games)
  echo "📦 Filtering Elo 2400–2800 (max 200k games)..."
  uv run python "$SCRIPTS/download_lichess_games.py" \
    --zst "$LICHESS_ZST" \
    --min-elo 2400 \
    --max-elo 2800 \
    --max-games 200000 \
    --out "$LICHESS_JSONL"

  # 5️⃣ Verify output
  if [ ! -s "$LICHESS_JSONL" ]; then
    echo "❌ Filtering failed: $LICHESS_JSONL is empty."
    exit 1
  else
    echo "✅ Filtered dataset ready: $LICHESS_JSONL"
  fi
fi

echo "=== 5) Convert PGN → JSONL dataset (legacy, skip if JSONL exists) ==="
if [ -s "$LICHESS_JSONL" ]; then
  echo "✅ Found $LICHESS_JSONL — skipping conversion."
else
  uv run python "$SCRIPTS/pgn_to_training.py" \
    --pgn "$LICHESS_PGN" \
    --out "$LICHESS_JSONL"
fi

echo "=== 6) Obtain base Maia model (optional) ==="
if [ -d "$MAIA_DIR" ] && [ -f "$MAIA_DIR/maia-2200.pb" ]; then
  echo "✅ Found existing $MAIA_DIR — skipping download."
else
  echo "⬇️  Downloading Maia-2200 .pb.gz..."
  mkdir -p "$MAIA_DIR"
  curl -L -o "$MAIA_DIR/maia-2200.pb.gz" "$MAIA_2200_URL"
  gunzip -f "$MAIA_DIR/maia-2200.pb.gz"
  echo "✅ Decompressed to $MAIA_DIR/maia-2200.pb"
fi

echo "=== 7) Train base model (if no H5 checkpoint yet) ==="
if [ ! -f "$MAIA_DIR/maia-2200.h5" ]; then
  echo "⚙️  Training base Maia-style model from lichess_dataset.jsonl..."
  uv run python "$SCRIPTS/train_base_fallback.py" \
    --data "$LICHESS_JSONL" \
    --out "$MAIA_DIR/maia-2200.h5"
else
  echo "✅ Found existing $MAIA_DIR/maia-2200.h5 — skipping base training."
fi

echo "=== 8) Fine-tune Maia → LarryBot ==="
uv run python "$SCRIPTS/train_larry.py" \
  --train-data "$DATA/larry_dataset.jsonl" \
  --val-data "$LICHESS_JSONL" \
  --init-from "$MAIA_DIR/maia-2200.h5" \
  --save-dir "$DATA/larrybot" \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --use-gpu

echo "✅ All steps complete. LarryBot is ready!"
