#!/usr/bin/env python3
import argparse
import json
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from train_base_maia import fen_to_planes

# -------------------- Dataset Loader --------------------

def load_larry_dataset(jsonl_path, move_map):
    X, y_raw = [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)

            # expected format: {"board": "<FEN>", "move": "<SAN>"}
            fen = obj["board"]
            san = obj["move"]

            X.append(fen_to_planes(fen))
            y_raw.append(san)

    # Update move map with new SAN moves
    new_moves = set(y_raw) - set(move_map.keys())
    if new_moves:
        max_idx = max(move_map.values())
        for i, mv in enumerate(sorted(new_moves), start=max_idx + 1):
            move_map[mv] = i

    # Encode labels
    y = np.array([move_map[mv] for mv in y_raw], dtype=np.int32)
    X = np.array(X, dtype=np.float32)

    return X, y, move_map


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--larry_jsonl", required=True)
    ap.add_argument("--out_model", required=True)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_conv", action="store_true")

    args = ap.parse_args()

    print(f"📦 Loading base model: {args.base_model}")
    model = load_model(args.base_model)

    # Create initial move map from base model output dimensionality
    num_moves_base = model.output_shape[-1]
    move_map = {mv: mv for mv in range(num_moves_base)}

    if args.freeze_conv:
        print("🧊 Freezing convolutional layers.")
        for layer in model.layers:
            if not isinstance(layer, Dense):
                layer.trainable = False

    print(f"📦 Loading Larry dataset: {args.larry_jsonl}")
    X, y, move_map = load_larry_dataset(args.larry_jsonl, move_map)

    # Resize output if new moves were added
    if len(move_map) != model.output_shape[-1]:
        new_dim = len(move_map)
        print(f"⚡ Adjusting output layer ({model.output_shape[-1]} → {new_dim})")

        x = model.layers[-2].output
        new_out = Dense(new_dim, activation="softmax", name="output_dense")(x)
        model = Model(model.input, new_out)

    print("⚡ Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    )

    print("🎯 Fine-tuning...")
    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.05,
        shuffle=True
    )

    print(f"💾 Saving fine-tuned model to {args.out_model}")
    model.save(args.out_model)

    # Save updated move map
    mm_path = args.out_model.replace(".keras", "_move_map.json")
    with open(mm_path, "w") as f:
        json.dump(move_map, f)
    print(f"💾 Saved move map: {mm_path}")


if __name__ == "__main__":
    main()