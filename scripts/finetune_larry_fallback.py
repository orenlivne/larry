#!/usr/bin/env python3
import argparse, os, json, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_jsonl(path, max_examples=None):
    seqs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples: break
            d = json.loads(line); seqs.append(d["moves"])
    return seqs

def build_vocab(seqs):
    from collections import Counter
    c = Counter(m for seq in seqs for m in seq)
    tk = {tok:i+1 for i,tok in enumerate(sorted(c.keys()))}
    return tk

def encode_seqs(seqs, tk, maxlen=120):
    enc = [[tk.get(m,0) for m in s] for s in seqs]
    return pad_sequences(enc, maxlen=maxlen, padding='post')

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="base model dir")
ap.add_argument("--data", required=True, help="larry jsonl")
ap.add_argument("--out", required=True, help="out model dir")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4)
args = ap.parse_args()

seqs = load_jsonl(args.data)
tk = build_vocab(seqs)
X = encode_seqs(seqs, tk, maxlen=120)
y = np.zeros_like(X); y[:,:-1] = X[:,1:]

# load base model
model = tf.keras.models.load_model(args.base)
# compile with low lr for fine-tune
model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy")
model.fit(X, np.expand_dims(y, -1), batch_size=args.batch_size, epochs=args.epochs)

os.makedirs(args.out, exist_ok=True)
model.save(os.path.join(args.out, "saved_model"))
print("Saved fine-tuned model to", args.out)
