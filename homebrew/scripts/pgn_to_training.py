#!/usr/bin/env python3
"""
Convert PGN -> JSONL training dataset.
Each line: {"moves":["e4","e5",...], "result": 1/0/0.5 optional}
Skips invalid/corrupt games.
"""

import argparse, chess.pgn, json

def pgn_to_dataset(pgn_path, out_path, max_games=None):
    good, bad = 0, 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break
            try:
                board = game.board()
                moves = []
                for mv in game.mainline_moves():
                    if not board.is_legal(mv):
                        raise ValueError("illegal move")
                    moves.append(board.san(mv))
                    board.push(mv)
                result = game.headers.get("Result", "*")
                res_val = None
                if result == "1-0": res_val = 1
                elif result == "0-1": res_val = -1
                elif result == "1/2-1/2": res_val = 0
                f_out.write(json.dumps({"moves": moves, "result": res_val}) + "\n")
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
