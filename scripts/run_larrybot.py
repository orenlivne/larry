#!/usr/bin/env python3
"""
Run LarryBot locally or on Lichess (supports Keras 3 `.keras` model format).

This runtime is robust to:
 - move_map.json being either {move: index} or {index: move}
 - model input channel counts (it will pad planes if needed and put stm plane in channel 12 if available)
 - move string representation differences (it checks SAN and UCI when matching legal moves)
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import chess
import chess.engine

# lazy import berserk only if needed (not required for local tests)
try:
    import berserk
except Exception:
    berserk = None


# -----------------------------
# Move map loader (robust)
# -----------------------------
def load_move_map(path):
    """
    Load move map JSON and return (move_to_idx, idx_to_move).
    Accepts either:
      {"e2e4": 12, ...}  OR  {"0": "e2e4", "1": "d2d4", ...}
    Ensures types: move_to_idx: str->int, idx_to_move: int->str
    """
    with open(path) as f:
        data = json.load(f)

    # Detect format
    # If keys are numeric strings or ints -> probably idx->move
    all_keys_are_int_like = all(
        (isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()))
        for k in data.keys()
    )
    all_vals_are_int_like = all(
        (isinstance(v, int) or (isinstance(v, str) and v.lstrip("-").isdigit()))
        for v in data.values()
    )

    move_to_idx = {}
    idx_to_move = {}

    if all_keys_are_int_like and not all_vals_are_int_like:
        # keys are indices -> invert
        for k, v in data.items():
            idx = int(k)
            move = v
            idx_to_move[idx] = move
            move_to_idx[move] = idx
    elif all_vals_are_int_like:
        # values are indices -> straightforward
        for k, v in data.items():
            move = k
            idx = int(v)
            move_to_idx[move] = idx
            idx_to_move[idx] = move
    else:
        # Mixed / unknown: best-effort: treat keys as moves if many are non-digit
        # Fallback: keys are moves, values are indexes or strings representing indexes
        for k, v in data.items():
            if isinstance(v, int):
                move_to_idx[k] = v
                idx_to_move[v] = k
            else:
                # try parse v, else treat as move->move (we can't map)
                try:
                    idx = int(v)
                    move_to_idx[k] = idx
                    idx_to_move[idx] = k
                except Exception:
                    # value is a string that is not int -> maybe the file already maps idx->move but keys were strings like "0"
                    # attempt fallback: if key looks like int, swap
                    if isinstance(k, str) and k.lstrip("-").isdigit():
                        idx = int(k)
                        move = v
                        idx_to_move[idx] = move
                        move_to_idx[move] = idx
                    else:
                        # last resort: map move -> incremental indices
                        # (this is unlikely; notify the user)
                        pass

    if not move_to_idx:
        raise ValueError(f"Could not interpret move map at {path}. Contents sample: {list(data.items())[:6]}")

    # ensure idx->move is complete
    if not idx_to_move:
        for m, i in move_to_idx.items():
            idx_to_move[i] = m

    return move_to_idx, idx_to_move


# -----------------------------
# FEN -> planes (model-compatible)
# -----------------------------
def fen_to_planes_adapted(fen, required_channels):
    """
    Produce piece planes from FEN; pad/truncate to required_channels.
    - Base piece planes: 12 channels (P,N,B,R,Q,K for white then black)
    - If required_channels >= 13 we put a side-to-move plane in the next channel
    - Any remaining channels are zeros (padding)
    This lets us run models with different channel counts without changing training code.
    """
    board = chess.Board(fen)
    # 12 piece planes
    planes = np.zeros((8, 8, 12), dtype=np.float32)
    piece_planes = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)   # convert 0..63 to board row (training uses rank order)
        col = sq % 8
        planes[row, col, piece_planes[piece.symbol()]] = 1.0

    extras = []
    # side-to-move plane (put in next channel if space)
    stm_plane = np.full((8, 8, 1), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32)
    extras.append(stm_plane)

    # concatenate base + extras
    arr = np.concatenate([planes] + extras, axis=-1)  # shape (8,8,12 + len(extras))
    cur_ch = arr.shape[-1]
    if cur_ch == required_channels:
        return arr
    elif cur_ch < required_channels:
        # pad with zero planes
        pad = np.zeros((8, 8, required_channels - cur_ch), dtype=np.float32)
        return np.concatenate([arr, pad], axis=-1)
    else:
        # truncate (rare)
        return arr[:, :, :required_channels]


# -----------------------------
# Prediction helpers
# -----------------------------
def predict_maia_move(model, board, move_to_idx, idx_to_move):
    """
    Given a loaded keras model and a chess.Board instance, predict the best legal move.
    - model: tf.keras Model
    - board: chess.Board
    - move_to_idx: dict move_str -> index
    - idx_to_move: dict index -> move_str
    Returns: chess.Move or None
    """
    # infer expected input channels from model
    try:
        input_shape = model.input_shape  # typically (None, 8, 8, C)
    except Exception:
        input_shape = getattr(model, "inputs", None)
    if isinstance(input_shape, tuple):
        channels = input_shape[-1]
    elif isinstance(input_shape, (list, tuple)) and len(input_shape) and hasattr(input_shape[0], 'shape'):
        # fallback
        channels = model.input_shape[-1]
    else:
        # final fallback
        channels = 13

    planes = fen_to_planes_adapted(board.fen(), required_channels=channels)
    X = np.expand_dims(planes, 0)  # (1,8,8,C)

    preds = model.predict(X, verbose=0)[0]  # (num_moves,)
    # preds may be numpy array or list; assume array

    legal_moves = list(board.legal_moves)
    best = None
    best_score = -1.0

    # For each legal move, generate canonical strings to look up in move_to_idx:
    # - SAN (e.g. Nf3)
    # - UCI (e2e4)
    # - Long algebraic (fallback)
    for mv in legal_moves:
        try:
            san = board.san(mv)
        except Exception:
            san = None
        uci = mv.uci()
        candidates = []
        if san:
            candidates.append(san)
        candidates.append(uci)

        # check each candidate string in move_to_idx
        found_idx = None
        for cand in candidates:
            if cand in move_to_idx:
                found_idx = move_to_idx[cand]
                break
        if found_idx is None:
            # maybe idx_to_move values use SANs with move numbers or annotations; skip
            continue

        score = float(preds[found_idx]) if found_idx < preds.shape[0] else -1.0
        if score > best_score:
            best_score = score
            best = mv

    return best


# -----------------------------
# Local test runner
# -----------------------------
def run_local_test(model, move_to_idx, idx_to_move, stockfish_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print("🎯 Local test: LarryBot (White) vs Stockfish (Black)")

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            mv = predict_maia_move(model, board, move_to_idx, idx_to_move)
            if mv is None:
                print("⚠️ LarryBot predicted no legal move; giving up.")
                break
            board.push(mv)
            print(f"🤖 LarryBot: {mv.uci()}")
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))
            board.push(result.move)
            print(f"💡 Stockfish: {result.move.uci()}")

        print(board, "\n")

    engine.quit()
    print(f"🏁 Game over: {board.result()} ({board.outcome()})")


# -----------------------------
# Lichess handling (minimal)
# -----------------------------
def handle_lichess_game(client, model, move_to_idx, idx_to_move, game_id):
    board = chess.Board()
    for event in client.bots.stream_game_state(game_id):
        if event["type"] not in ("gameFull", "gameState"):
            continue
        moves = event.get("moves", "")
        if not moves:
            continue
        board = chess.Board()
        for mv in moves.split():
            board.push_uci(mv)

        if board.turn == chess.WHITE:
            mv = predict_maia_move(model, board, move_to_idx, idx_to_move)
            if mv is None:
                print("⚠️ LarryBot had no legal move.")
                continue
            client.bots.make_move(game_id, mv.uci())
            print(f"🤖 LarryBot played {mv.uci()}")


def run_live(token_file, model, move_to_idx, idx_to_move):
    if berserk is None:
        raise RuntimeError("berserk not installed; cannot run in Lichess mode.")
    with open(token_file) as f:
        token = f.read().strip()
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    print("♟️ Connected to Lichess. Waiting for challenges...")
    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            client.bots.accept_challenge(event["challenge"]["id"])
        elif event["type"] == "gameStart":
            handle_lichess_game(client, model, move_to_idx, idx_to_move, event["game"]["id"])


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", help="Lichess bot token file (required for live mode)")
    ap.add_argument("--model-dir", required=True, help="Directory containing model file (maia_larry_finetuned.keras)")
    ap.add_argument("--move-map", required=True, help="Path to move_map json")
    ap.add_argument("--local-test", action="store_true", help="Play a local match vs stockfish")
    ap.add_argument("--stockfish", default="/usr/local/bin/stockfish", help="Path to stockfish")
    args = ap.parse_args()

    print("📂 Loading model and move map ...")
    # load model file inside model-dir; try several common names
    model_path_candidates = [
        os.path.join(args.model_dir, "maia_larry_finetuned.keras"),
        os.path.join(args.model_dir, "maia_larry_base.keras"),
        os.path.join(args.model_dir, "larrybot.keras"),
    ]
    model_path = None
    for p in model_path_candidates:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        # fallback: if model_dir is a file itself
        if os.path.exists(args.model_dir) and os.path.isfile(args.model_dir):
            model_path = args.model_dir
        else:
            raise FileNotFoundError(f"Could not find a model in {args.model_dir}; tried {model_path_candidates}")

    model = tf.keras.models.load_model(model_path)
    move_to_idx, idx_to_move = load_move_map(args.move_map)

    if args.local_test:
        run_local_test(model, move_to_idx, idx_to_move, args.stockfish)
    else:
        if not args.token_file:
            raise ValueError("Must specify --token-file when not in --local-test mode.")
        run_live(args.token_file, model, move_to_idx, idx_to_move)


if __name__ == "__main__":
    main()