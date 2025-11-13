#!/usr/bin/env python3
"""
Train a base Maia-style chess model from board/move JSONL.
FEN strings are converted to 12x64 one-hot arrays.
Supports move_index_map and optional max_samples.
"""

import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import chess

# -----------------------------
# Piece mapping for 12x64 encoding
# -----------------------------
PIECE_TO_IDX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_array(fen):
    """Convert FEN to 12x64 one-hot array (flattened to 768)."""
    board = chess.Board(fen)
    arr = np.zeros((12, 64), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = PIECE_TO_IDX[piece.symbol()]
            arr[idx, square] = 1.0
    return arr.flatten()

# -----------------------------
# Load JSONL
# -----------------------------
def load_jsonl(path, move_map_path, max_samples=None, random_seed=42):
    print(f"📂 Loading dataset from {path}...")
    with open(move_map_path) as f:
        move_map = json.load(f)
    move_to_idx = {v: int(k) for k, v in move_map.items()}

    X, y = [], []
    with open(path) as f:
        for i, line in enumerate(tqdm(f)):
            if max_samples and i >= max_samples:
                break
            entry = json.loads(line)
            fen = entry["board"]
            move = entry["move"]
            if move not in move_to_idx:
                continue
            X.append(fen_to_array(fen))
            y.append(move_to_idx[move])

    # Random shuffle / subsample if max_samples < len(X)
    if max_samples and len(X) > max_samples:
        random.seed(random_seed)
        idxs = random.sample(range(len(X)), max_samples)
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"📦 Dataset loaded: {len(X)} samples, {len(move_to_idx)} unique moves")
    return X, y

# -----------------------------
# Build model
# -----------------------------
def build_model(input_dim=768, output_dim=2000):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(output_dim, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train base Maia model from JSONL")
    parser.add_argument("--data", required=True, help="JSONL with board/move examples")
    parser.add_argument("--move-map", required=True, help="move_index_map.json")
    parser.add_argument("--out", required=True, help="Output directory for saved model")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max number of samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-from", type=str, default=None, help="Optional H5 model to fine-tune")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X, y = load_jsonl(args.data, args.move_map, max_samples=args.max_samples)
    n_moves = max(y) + 1  # output dimension

    if args.init_from:
        print(f"📦 Loading pretrained model from {args.init_from} ...")
        model = tf.keras.models.load_model(args.init_from)
    else:
        print(f"🏗 Building new model ({X.shape[1]} input, {n_moves} outputs)")
        model = build_model(input_dim=X.shape[1], output_dim=n_moves)
        model.optimizer.learning_rate = args.lr

    # Train/validation split
    val_split = 0.05
    n_val = int(len(X) * val_split)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]
    print(f"🏋️ Training: {len(X_train)} train, {len(X_val)} val")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    save_path = os.path.join(args.out, "saved_model")
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()