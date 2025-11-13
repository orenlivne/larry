#!/usr/bin/env python3
"""
Generate move_index_map.json for LarryBot from a JSONL dataset.
This matches the token indexing used in train_base_fallback.py.
"""

import argparse
import json
from collections import Counter

def load_jsonl(path, max_examples=None):
    moves = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples: 
                break
            d = json.loads(line)
            moves.append(d["move"])   # <- use append, not extend
    return moves

def build_vocab(seqs, min_count=1):
    c = Counter(seqs)
    tokens = [tok for tok, cnt in c.items() if cnt >= min_count]
    token_index = {tok: i+1 for i, tok in enumerate(tokens)}  # 0 reserved for pad
    return token_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", required=True, help="Path to output JSON move index map")
    args = parser.parse_args()

    seqs = load_jsonl(args.data)
    token_index = build_vocab(seqs)
    # Invert the mapping so index -> move
    move_index_map = {str(idx): move for move, idx in token_index.items()}

    with open(args.out, "w") as f:
        json.dump(move_index_map, f, indent=2)
    print(f"✅ Saved move_index_map.json to {args.out}")

if __name__ == "__main__":
    main()
