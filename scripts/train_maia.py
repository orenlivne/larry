#!/usr/bin/env python3
"""
Fine-tune a Maia model on mixed Larry + 2400–2800 games.
"""

import os
import pandas as pd
import tensorflow as tf
from maia_chess.train.train_utils import load_data, build_model, train_model

# --- Paths ---
BASE_DIR = os.path.expanduser("~/gh/larry")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MAIA_BASE = os.path.join(DATA_DIR, "maia-2200.h5")
LICHESS_PGN = os.path.join(DATA_DIR, "raw_lichess_games_2024.pgn")
LARRY_PGN = os.path.join(DATA_DIR, "larry_games.pgn")
MERGED_CSV = os.path.join(DATA_DIR, "merged_train_data.csv")
FINETUNED_MODEL = os.path.join(MODEL_DIR, "maia_2200_finetuned_larry.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Step 1: Convert PGN to training CSV ---
from maia_chess.data.build_training_data import build_training_data_from_pgn

if not os.path.exists(MERGED_CSV):
    print("📘 Converting PGNs to training format...")
    df1 = build_training_data_from_pgn(LARRY_PGN)
    df2 = build_training_data_from_pgn(LICHESS_PGN)
    df = pd.concat([df1, df2]).sample(frac=1, random_state=42)
    df.to_csv(MERGED_CSV, index=False)
else:
    print("✅ Using existing merged training file.")
    df = pd.read_csv(MERGED_CSV)

# --- Step 2: Load Maia base model ---
print("🧠 Loading base Maia model...")
base_model = tf.keras.models.load_model(MAIA_BASE, compile=False)

# --- Step 3: Prepare data ---
X, y = load_data(MERGED_CSV)

# --- Step 4: Fine-tune ---
print("🚀 Fine-tuning on combined dataset...")
base_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history = base_model.fit(
    X, y,
    batch_size=4096,
    epochs=3,
    validation_split=0.1,
    shuffle=True,
)

# --- Step 5: Save new checkpoint ---
base_model.save(FINETUNED_MODEL)
print(f"🎯 Fine-tuned model saved to {FINETUNED_MODEL}")
