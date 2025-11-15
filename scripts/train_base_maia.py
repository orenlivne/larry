#!/usr/bin/env python3
import argparse, json, numpy as np, tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, ReLU, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import os

# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--val_split", type=float, default=0.05)
    return p.parse_args()

# -----------------------------
# FEN -> planes
# -----------------------------
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
    stm_plane = np.full((8,8,1), 1.0 if " w " in fen else 0.0, np.float32)
    planes = np.concatenate([planes, stm_plane], axis=-1)
    return planes

# -----------------------------
# Dataset generator
# -----------------------------
def dataset_generator(path, move_map, max_samples=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            X = fen_to_planes(obj["board"])
            y = move_map[obj["move"]]
            yield X, y

# -----------------------------
# Move map builder
# -----------------------------
def build_move_map(path, max_samples=None):
    moves = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            moves.add(obj["move"])
    moves = sorted(moves)
    return {m:i for i,m in enumerate(moves)}

# -----------------------------
# Model
# -----------------------------
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")
        ])
    return model

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    print(f"==== Building move map ====")
    move_map = build_move_map(args.data, args.max_samples)
    num_moves = len(move_map)
    print(f"Found {num_moves} unique moves")

    # Save move map
    os.makedirs("./data", exist_ok=True)
    with open("./data/move_map.json","w") as f:
        json.dump(move_map,f)

    # -----------------------------
    # Build dataset using tf.data
    # -----------------------------
    print(f"==== Creating dataset pipeline ====")
    sample_X, _ = next(dataset_generator(args.data, move_map))
    input_shape = sample_X.shape
    full_ds = tf.data.Dataset.from_generator(
        lambda: dataset_generator(args.data, move_map, args.max_samples),
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Split into train/validation
    val_count = int(args.val_split * (args.max_samples or 1_000_000))
    val_ds = full_ds.take(val_count).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = full_ds.skip(val_count).shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # -----------------------------
    # Build model
    # -----------------------------
    model = build_model(input_shape, num_moves, args.lr)
    model.summary()

    # -----------------------------
    # Checkpoint: save best only
    # -----------------------------
    base_out = args.out_model
    if "." in base_out:
        base, ext = base_out.rsplit(".",1)
        checkpoint_path = f"{base}_best.{ext}"
    else:
        checkpoint_path = base_out + "_best.keras"

    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb]
    )

    # Save final model anyway
    model.save(args.out_model)
    print(f"✅ Saved final model to {args.out_model}")