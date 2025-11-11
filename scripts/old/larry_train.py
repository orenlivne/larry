#!/usr/bin/env python3
"""
train_larry.py

Fine-tune a Maia model on Larry Christian's games.
Assumes the base model (.pb) is already unzipped and ready.
"""

import argparse
from pathlib import Path
import tensorflow as tf

def load_maia_model(init_model_path):
    """
    Load an unzipped SavedModel (.pb) or SavedModel directory.
    """
    # init_model_path must be the string path to the unzipped model
    return tf.keras.models.load_model(init_model_path)

def load_dataset(path):
    """Load the training/validation dataset from a text file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Dummy tokenization for demonstration: convert first 100 chars to ordinal values
    x = [list(map(ord, l[:100])) for l in lines]
    y = [0] * len(x)  # placeholder labels
    return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("📦 Loading base model...")
    model = load_maia_model(args.init_from)

    print("📚 Loading datasets...")
    x_train, y_train = load_dataset(args.train_data)
    x_val, y_val = load_dataset(args.val_data)

    print("🏋️ Starting fine-tuning...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    save_path = Path(args.save_dir) / "best.pb"
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()
