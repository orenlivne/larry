#!/usr/bin/env python3
"""
split_dataset.py — Split a JSONL chess dataset into training and validation sets.

Usage:
  python scripts/split_dataset.py \
      --input data/lichess_2025.jsonl \
      --train-out data/train.jsonl \
      --val-out data/val.jsonl \
      --val-fraction 0.1
"""

import argparse
import random
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train/val subsets.")
    parser.add_argument("--input", required=True, help="Input JSONL file (e.g. lichess_2025.jsonl)")
    parser.add_argument("--train-out", required=True, help="Output path for training JSONL")
    parser.add_argument("--val-out", required=True, help="Output path for validation JSONL")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data for validation (default: 0.1)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with open(input_path, "r") as f:
        lines = f.readlines()

    random.seed(42)
    random.shuffle(lines)

    n_val = int(len(lines) * args.val_fraction)
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    os.makedirs(Path(args.train_out).parent, exist_ok=True)

    with open(args.train_out, "w") as f:
        f.writelines(train_lines)

    with open(args.val_out, "w") as f:
        f.writelines(val_lines)

    print(f"✅ Split complete: {len(train_lines)} train, {len(val_lines)} val")

if __name__ == "__main__":
    main()
