#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import chess
import chess.engine

############################################################
# FEN → planes (must match training: 8x8x18)
############################################################
import chess
import numpy as np

def fen_to_planes(fen):
    """
    Converts FEN to 18-plane tensor:
      - 12 planes: piece type (P,N,B,R,Q,K x white/black)
      - 1 plane: side to move
      - 4 planes: castling rights (KQkq)
      - 1 plane: move count / dummy
    """
    board = chess.Board(fen)
    planes = np.zeros((8,8,18), dtype=np.float32)

    piece_planes = {
        'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
        'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
    }

    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        planes[row, col, piece_planes[piece.symbol()]] = 1

    # 13th plane: side to move
    planes[:,:,12] = 1.0 if board.turn == chess.WHITE else 0.0

    # 14-17: castling rights
    planes[:,:,13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[:,:,14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[:,:,15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[:,:,16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # 18th plane: dummy / optional (fill with zeros)
    planes[:,:,17] = 0.0

    return planes

############################################################
# Predict move
############################################################
def predict_maia_move(model, board, inv_map):
    X = np.expand_dims(fen_to_planes(board.fen()), axis=0)  # shape (1,8,8,18)
    preds = model.predict(X, verbose=0)[0]                   # (num_moves,)

    legal_moves = list(board.legal_moves)
    best = None
    best_score = -1e9

    for mv in legal_moves:
        san = board.san(mv)
        for idx, san_move in inv_map.items():
            if san_move == san:
                if preds[idx] > best_score:
                    best_score = preds[idx]
                    best = mv
                break

    return best

############################################################
# Load move map
############################################################

def load_move_map(path):
    with open(path) as f:
        move_to_idx = json.load(f)
    inv = {int(v): k for k, v in move_to_idx.items()}  # idx → SAN
    return move_to_idx, inv


############################################################
# Local Stockfish test
############################################################

def run_local_test(model, inv_map, stockfish_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    print("🎯 Local test: LarryBot (White) vs Stockfish (Black)")

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            mv = predict_maia_move(model, board, inv_map)
            if mv is None:
                print("⚠️ LarryBot predicted no legal move; resigning.")
                break
            board.push(mv)
            print(f"🤖 LarryBot plays: {mv.uci()}")

        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            print(f"💡 Stockfish plays: {result.move.uci()}")

        print(board, "\n")

    engine.quit()
    print(f"🏁 Game over: {board.result()} ({board.outcome()})")


############################################################
# Lichess BOT play
############################################################

def handle_lichess_game(client, model, inv_map, game_id):
    board = chess.Board()

    for event in client.bots.stream_game_state(game_id):
        if event["type"] not in ("gameFull", "gameState"):
            continue

        moves = event["moves"].split()
        board = chess.Board()
        for mv in moves:
            board.push_uci(mv)

        if board.turn == chess.WHITE:
            mv = predict_maia_move(model, board, inv_map)
            if mv is None:
                print("⚠️ LarryBot had no move.")
                continue
            client.bots.make_move(game_id, mv.uci())
            print(f"🤖 LarryBot played {mv.uci()}")


def run_live(token_file, model, inv_map):
    with open(token_file) as f:
        token = f.read().strip()

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    print("♟️ Connected to Lichess. Waiting for challenges...")

    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            client.bots.accept_challenge(event["challenge"]["id"])
        elif event["type"] == "gameStart":
            handle_lichess_game(client, model, inv_map, event["game"]["id"])


############################################################
# main()
############################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--move-map", required=True)
    ap.add_argument("--local-test", action="store_true")
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    args = ap.parse_args()

    print("📂 Loading model and move map...")
    model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "maia_larry_finetuned.keras")
    )
    move_to_idx, inv_map = load_move_map(args.move_map)

    if args.local_test:
        run_local_test(model, inv_map, args.stockfish)
    else:
        if not args.token_file:
            raise ValueError("Need --token-file for Lichess mode.")
        run_live(args.token_file, model, inv_map)

if __name__ == "__main__":
    main()