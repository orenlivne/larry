#!/usr/bin/env python3
"""
Train Maia base model + fine-tune on Larry dataset
Exports H5 models and creates move_index_map.json
Supports optional random sampling of Lichess moves.
"""

import argparse
import json
import random
from maia_chess.data_generators.move_prediction import MoveDataset
from maia_chess.maia_models.model import build_move_model
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lichess", required=True, help="Path to Lichess train JSONL")
    parser.add_argument("--larry", required=True, help="Path to Larry train JSONL")
    parser.add_argument("--move-map", required=True, help="Path to move_index_map.json")
    parser.add_argument("--base-h5", required=True, help="Output H5 path for base Maia")
    parser.add_argument("--larry-h5", required=True, help="Output H5 path for fine-tuned LarryBot")
    parser.add_argument("--sample-size", type=int, default=None, help="Random subset of Lichess samples")
    args = parser.parse_args()

    # -----------------------------
    # Load move map
    # -----------------------------
    move_map = json.load(open(args.move_map))
    num_moves = len(move_map)
    print(f"✅ Move map loaded: {num_moves} moves")

    # -----------------------------
    # Optional random sampling
    # -----------------------------
    lichess_file = args.lichess
    if args.sample_size is not None:
        print(f"📦 Sampling {args.sample_size} moves from Lichess dataset...")
        sampled_lines = []
        with open(lichess_file) as f:
            all_lines = f.readlines()
            if args.sample_size < len(all_lines):
                sampled_lines = random.sample(all_lines, args.sample_size)
            else:
                sampled_lines = all_lines
        lichess_file = lichess_file.replace(".jsonl", "_sampled.jsonl")
        with open(lichess_file, "w") as f:
            f.writelines(sampled_lines)
        print(f"✅ Sampled dataset saved to {lichess_file}")

    # -----------------------------
    # Train base Maia
    # -----------------------------
    print(f"🏗 Building base Maia model from {lichess_file}...")
    dataset = MoveDataset(lichess_file, move_index_map=move_map)
    train_ds, val_ds = dataset.get_tf_dataset(split=0.95)

    model = build_move_model(input_dim=768, output_dim=num_moves)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.save(args.base_h5, save_format="h5")
    print(f"✅ Base Maia model saved to {args.base_h5}")

    # -----------------------------
    # Fine-tune on Larry games
    # -----------------------------
    print("🏗 Fine-tuning on Larry dataset...")
    dataset = MoveDataset(args.larry, move_index_map=move_map)
    train_ds, val_ds = dataset.get_tf_dataset(split=0.95)

    model = tf.keras.models.load_model(args.base_h5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.save(args.larry_h5, save_format="h5")
    print(f"✅ Fine-tuned LarryBot saved to {args.larry_h5}")

if __name__ == "__main__":
    main()