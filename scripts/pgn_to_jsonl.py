#!/usr/bin/env python3
"""
Convert a PGN file to JSONL format compatible with train_base_fallback.py.
"""

import argparse
import chess.pgn
import json

def parse_game(game):
    """Return moves as list of SAN strings."""
    if game is None:
        return None
    moves = []
    board = game.board()
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    if not moves:
        return None
    return {"moves": moves}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True, help="Input PGN file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    args = parser.parse_args()

    with open(args.pgn) as f, open(args.out, "w", encoding="utf-8") as out_f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_json = parse_game(game)
            if game_json:
                out_f.write(json.dumps(game_json) + "\n")

if __name__ == "__main__":
    main()
