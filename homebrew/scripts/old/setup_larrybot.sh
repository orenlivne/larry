#!/usr/bin/env bash
# ================================================================
# setup_larrybot.sh
# Full setup & fine-tuning of Maia-LarryBot on macOS (M1/M2/M3)
# Uses uv to manage the Python environment
# ================================================================

set -e

# ---- CONFIG ----------------------------------------------------
BASE_DIR="$HOME/gh/larry"
MAIA_DIR="$BASE_DIR/maia-chess"
DATA_DIR="$BASE_DIR/data"
SCRIPTS_DIR="$BASE_DIR/scripts"
MODEL_DIR="$BASE_DIR/models/larry_maia"
PGN_FILE="$DATA_DIR/larry_games.pgn"
DATASET_FILE="$DATA_DIR/larry_dataset.txt"
PB_GZ_FILE="$MAIA_DIR/data/maia-1900.pb.gz"
PB_FILE="$MAIA_DIR/data/maia-1900.pb"
H5_FILE="$MAIA_DIR/data/maia-1900.h5"
MAIA_PB_URL="https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1900.pb.gz"
EPOCHS=3
BATCH_SIZE=32
LR=1e-4
# ---------------------------------------------------------------

echo "🚀 Setting up Larry fine-tuning environment (uv-managed)"

# ---- STEP 0. Create directories --------------------------------
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$MAIA_DIR/data" "$SCRIPTS_DIR"

# ---- STEP 1. Clone Maia if missing -----------------------------
if [ ! -d "$MAIA_DIR/.git" ]; then
    echo "📥 Cloning Maia repo..."
    git clone https://github.com/CSSLab/maia-chess.git "$MAIA_DIR"
else
    echo "✅ Maia repo already present."
fi

# ---- STEP 2. Initialize uv environment ------------------------
cd "$BASE_DIR"
if [ ! -d ".venv" ]; then
    echo "🐍 Creating uv environment..."
    uv init
fi
source .venv/bin/activate

echo "📚 Installing Python dependencies..."
uv pip install --upgrade pip
uv pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 chess numpy tqdm

# ---- STEP 3. Download Maia 1900 PB -----------------------------
cd "$MAIA_DIR/data"
if [ ! -f "$PB_GZ_FILE" ]; then
    echo "⬇️  Downloading Maia 1900 model..."
    curl -L -o "$PB_GZ_FILE" "$MAIA_PB_URL"
else
    echo "✅ PB.gz file already exists."
fi

# ---- STEP 4. Unzip PB ------------------------------------------
if [ ! -f "$PB_FILE" ]; then
    echo "🗜️  Unzipping PB file..."
    gzip -d -f "$PB_GZ_FILE"
else
    echo "✅ PB already unzipped."
fi

# ---- STEP 5. Convert PB to H5 ----------------------------------
if [ ! -f "$H5_FILE" ]; then
    echo "🔄 Converting PB → H5..."
    uv run python "$SCRIPTS_DIR/convert_pb_to_h5.py" \
        --input "$PB_FILE" \
        --output "$H5_FILE" \
        --input-name input_placeholder:0 \
        --output-name policy_head/Logits:0
else
    echo "✅ H5 file already exists."
fi

# ---- STEP 6. Convert PGN to dataset ---------------------------
if [ ! -f "$PGN_FILE" ]; then
    echo "❌ ERROR: PGN file not found at $PGN_FILE"
    echo "Please copy your larry_games.pgn to $PGN_FILE"
    exit 1
fi

echo "♟️  Converting PGN → training dataset..."
uv run python "$SCRIPTS_DIR/pgn_to_training.py" "$PGN_FILE" --out "$DATASET_FILE"

# ---- STEP 7. Fine-tune Maia -----------------------------------
echo "🏋️  Starting fine-tuning..."
mkdir -p "$MODEL_DIR"

uv run python "$SCRIPTS_DIR/train_larry.py" \
    --train-data "$DATASET_FILE" \
    --val-data "$DATASET_FILE" \
    --init-from "$H5_FILE" \
    --save-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --use-gpu

echo
echo "==============================================================="
echo "✅ Training complete!"
echo "Your fine-tuned model is at: $MODEL_DIR/best.h5"
echo "==============================================================="
