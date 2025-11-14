#!/usr/bin/env python3
import argparse, json, numpy as np, tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, ReLU, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()

def fen_to_planes(fen: str):
    planes = np.zeros((8,8,18), np.float32)
    ranks = fen.split()[0].split("/")
    piece_map = {"P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,
                 "p":6,"n":7,"b":8,"r":9,"q":10,"k":11}
    for r, rank in enumerate(ranks):
        f = 0
        for c in rank:
            if c.isdigit():
                f += int(c)
            else:
                planes[r,f,piece_map[c]] = 1
                f += 1
    return planes

def load_dataset(path, max_samples=None):
    X, y = [], []
    with open(path) as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {path}")):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            # unchanged: your JSONL has "board" (FEN) and "move" (uci like e2e4 / or c3e2)
            X.append(fen_to_planes(obj["board"]))
            y.append(obj["move"])
    return np.array(X, np.float32), np.array(y, object)

def build_move_map(y): 
    # stable sorting and deterministic mapping
    return {m:i for i,m in enumerate(sorted(set(y)))}

def encode_labels(y, move_map):
    # ✅ Return int32 array suitable for sparse_categorical_crossentropy
    return np.array([move_map[m] for m in y], dtype=np.int32)

def residual_block(x, filters, dropout_rate=0.2):
    y = Conv2D(filters, 3, padding="same", activation="relu")(x)
    y = Dropout(dropout_rate)(y)
    y = Conv2D(filters, 3, padding="same")(y)
    x = Add()([x, y])
    x = ReLU()(x)
    return x

def build_model(input_shape, num_moves, lr):
    inp = Input(shape=input_shape)
    x = Conv2D(128, 3, padding="same", activation="relu")(inp)
    for _ in range(5):
        x = residual_block(x, 128, dropout_rate=0.25)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(num_moves, activation="softmax")(x)

    model = Model(inp, out)
    # use learning_rate keyword (safer) and SparseTopK for sparse labels
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")
        ])
    return model

if __name__ == "__main__":
    args = parse_args()
    print(f"==== Loading {args.data} ====")
    X, y_raw = load_dataset(args.data, args.max_samples)
    move_map = build_move_map(y_raw)
    y = encode_labels(y_raw, move_map)
    print(f"📦 Dataset: X={X.shape}, y={y.shape}, moves={len(move_map)}")

    model = build_model(X.shape[1:], len(move_map), args.lr)
    model.summary()

    # Checkpoint naming: append epoch suffix to the requested out_model name
    # support .keras or any extension by inserting before extension
    base_out = args.out_model
    if "." in base_out:
        base, ext = base_out.rsplit(".", 1)
        checkpoint_path = f"{base}_epoch_{{epoch:02d}}.{ext}"
    else:
        checkpoint_path = base_out + "_epoch_{epoch:02d}.keras"

    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_freq="epoch")

    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.05,
        callbacks=[checkpoint_cb]
    )

    model.save(args.out_model)
    print(f"✅ Saved final model to {args.out_model}")