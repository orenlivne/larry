#!/usr/bin/env python3
"""
Convert PGN -> JSONL training dataset for Maia-style training.

Each line in the output JSONL:
    {"board": "<FEN before move>", "move": "<SAN move>"}

This is per-move format, not per-game.

Skips invalid or corrupt games.
"""

import argparse
import json
import chess.pgn

def pgn_to_dataset(pgn_path, out_path, max_games=None):
    good, bad = 0, 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break

            try:
                board = game.board()
                for mv in game.mainline_moves():
                    # ensure legality
                    if not board.is_legal(mv):
                        raise ValueError("Illegal move")

                    # write training record: FEN BEFORE MOVE, SAN MOVE
                    rec = {
                        "board": board.fen(),
                        "move": board.san(mv)
                    }
                    f_out.write(json.dumps(rec) + "\n")

                    board.push(mv)

                good += 1
                if max_games and good >= max_games:
                    break

            except Exception:
                bad += 1
                continue

    print(f"Converted {good} games; skipped {bad} bad games -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="input PGN")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    pgn_to_dataset(args.pgn, args.out, args.max_games)