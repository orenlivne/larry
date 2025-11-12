#!/usr/bin/env python3
"""
Lightweight trainer: trains a small sequence model on JSONL moves.
This is a fallback for quick prototyping; not strictly the Maia architecture.
Supports continued training from pretrained model.
"""

import argparse, json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_jsonl(path, max_examples=None):
    moves = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples: break
            d = json.loads(line)
            moves.append(d["moves"])
    return moves

def build_vocab(seqs, min_count=1):
    from collections import Counter
    c = Counter(m for seq in seqs for m in seq)
    tokens = [tok for tok, cnt in c.items() if cnt >= min_count]
    token_index = {tok: i+1 for i, tok in enumerate(tokens)}  # 0 reserved for pad
    return token_index

def encode_seqs(seqs, token_index, maxlen=120):
    encoded = [[token_index.get(m, 0) for m in seq] for seq in seqs]
    return pad_sequences(encoded, maxlen=maxlen, padding="post")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seqs = load_jsonl(args.data)
    token_index = build_vocab(seqs)
    X = encode_seqs(seqs, token_index, maxlen=120)
    # next-token targets
    y = np.zeros_like(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = 0

    vocab_size = max(token_index.values()) + 1

    if args.init_from:
        print(f"📥 Loading pretrained model from {args.init_from} ...")
        model = tf.keras.models.load_model(os.path.join(args.init_from, "saved_model"))
    else:
        print("⚙️ Building new model from scratch ...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(120,)),
            tf.keras.layers.Embedding(vocab_size, 128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy")
    print(f"🧠 Training on {len(X)} sequences for {args.epochs} epochs ...")
    model.fit(X, np.expand_dims(y, -1), batch_size=args.batch_size, epochs=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "saved_model"))
    print("✅ Saved model to", args.out)

if __name__ == "__main__":
    main()
