#!/usr/bin/env python3
"""
Optimized conversion of Lichess JSONL → board/move training JSONL.
Parallelized with multiprocessing + streaming + tqdm progress bar.
"""

import argparse
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import io

def game_to_examples(game_json):
    """Convert a single game dict to (board, move) examples."""
    import chess
    examples = []
    board = chess.Board()
    for move_san in game_json.get("moves", []):
        try:
            move = board.parse_san(move_san)
            examples.append({"board": board.fen(), "move": move.uci()})
            board.push(move)
        except Exception:
            continue  # skip illegal moves
    return examples

def process_line(line):
    """Worker function: parse JSON line and convert to examples."""
    try:
        game_json = json.loads(line)
        return game_to_examples(game_json)
    except Exception:
        return []

def convert_jsonl_to_train(input_path, output_path, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Count total lines for tqdm
    with open(input_path, "r", encoding="utf-8") as f:
        total_games = sum(1 for _ in f)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile, \
         Pool(n_workers) as pool:

        print(f"🧠 Converting JSONL → board/move JSONL using {n_workers} workers...")
        results = pool.imap_unordered(process_line, infile, chunksize=100)

        pbar = tqdm(total=total_games, unit="games")
        for examples in results:
            for ex in examples:
                outfile.write(json.dumps(ex) + "\n")
            pbar.update(1)
        pbar.close()
    print(f"✅ Finished writing training JSONL to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Lichess JSONL → board/move JSONL for training")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output training JSONL path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    convert_jsonl_to_train(args.input, args.output, args.workers)

if __name__ == "__main__":
    main()