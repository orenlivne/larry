#!/usr/bin/env python3
import tensorflow as tf
import argparse
import os
import json
import numpy as np

def train_model(train_data, val_data, init_from, save_dir, epochs=3, batch_size=512, lr=1e-4):
    print(f"📥 Loading base Maia model from {init_from} ...")
    model = tf.keras.models.load_model(init_from)

    print(f"📚 Loading dataset: {train_data}")
    with open(train_data, "r") as f:
        data = [json.loads(line) for line in f]

    # Placeholder: later we can replace this with real board encodings
    X = np.zeros((len(data), 8, 8, 12), dtype=np.float32)
    y = np.zeros((len(data), 1), dtype=np.float32)

    if len(X.shape) == 4:
        X = X.reshape((X.shape[0], -1))

    print(f"🧠 Fine-tuning Maia on {len(X)} samples for {epochs} epochs...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    out_dir = os.path.join(save_dir, "larrybot.keras")
    model.save(out_dir)
    print(f"✅ Saved fine-tuned model to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(**vars(args))
