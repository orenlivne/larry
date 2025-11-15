#!/usr/bin/env python3
import argparse
import json
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from train_base_maia import load_dataset, build_move_map, encode_labels, fen_to_planes

# -------------------- Arguments --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Path to base model (.keras)")
    p.add_argument("--larry_jsonl", required=True, help="Larry games JSONL")
    p.add_argument("--out_model", required=True, help="Output fine-tuned model (.keras)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze_conv", action="store_true", help="Freeze conv layers during fine-tuning")
    return p.parse_args()

# -------------------- Dataset Loader --------------------
def load_larry_dataset(path, move_map):
    X, y_raw = load_dataset(path)
    # Add any new moves to move_map
    new_moves = set(y_raw) - set(move_map.keys())
    if new_moves:
        max_idx = max(move_map.values())
        for i, m in enumerate(sorted(new_moves), start=max_idx + 1):
            move_map[m] = i
    y = encode_labels(y_raw, move_map)
    # Ensure y is 1D int32 array for Keras
    y = np.array(y, dtype=np.int32).reshape(-1)
    return np.array(X, np.float32), y, move_map

# -------------------- Main --------------------
def main():
    args = parse_args()

    print(f"📦 Loading base model from {args.base_model} ...")
    model = load_model(args.base_model)

    if args.freeze_conv:
        print("🧊 Freezing convolutional layers...")
        for layer in model.layers:
            if not isinstance(layer, Dense):
                layer.trainable = False

    print(f"📦 Loading Larry dataset from {args.larry_jsonl} ...")
    # Placeholder move_map from base model output
    num_moves_base = model.output_shape[-1]
    move_map = {i: i for i in range(num_moves_base)}
    X, y, move_map = load_larry_dataset(args.larry_jsonl, move_map)

    # Resize output layer if new moves were added
    num_moves = len(move_map)
    if num_moves != model.output_shape[-1]:
        print(f"⚡ Adjusting output layer for {num_moves} moves (was {model.output_shape[-1]})")
        x = model.layers[-2].output  # penultimate layer
        new_out = Dense(num_moves, activation="softmax", name="output_dense")(x)
        model = Model(model.input, new_out)

    print("⚡ Compiling model ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    print("🎯 Fine-tuning ...")
    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.05,
        shuffle=True
    )

    print(f"✅ Saving fine-tuned model to {args.out_model}")
    model.save(args.out_model)

    # Save updated move map
    move_map_json = args.out_model.replace(".keras", "_move_map.json")
    with open(move_map_json, "w") as f:
        json.dump(move_map, f)
    print(f"✅ Saved updated move map to {move_map_json}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()