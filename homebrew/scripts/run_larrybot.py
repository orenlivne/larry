#!/usr/bin/env python3
"""
Run LarryBot locally or on Lichess (supports Keras 3 `.keras` model format)
"""

import argparse
import json
import os
import tensorflow as tf
import numpy as np
import chess
import chess.engine
import berserk

def load_move_map(move_map_path):
    with open(move_map_path) as f:
        move_index_map = json.load(f)
    inv_map = {int(k): v for k, v in move_index_map.items()}
    token_index = {v: int(k) for k, v in inv_map.items()}
    return token_index, inv_map

def encode_moves(moves, token_index, maxlen=120):
    seq = [token_index.get(m, 0) for m in moves]
    seq = seq[-maxlen:]
    seq = np.pad(seq, (0, maxlen - len(seq)))
    return seq[np.newaxis, :]

def predict_move(model, token_index, inv_map, moves):
    X = encode_moves(moves, token_index)
    preds = model.predict(X, verbose=0)[0, -1, :]
    top_idx = int(np.argmax(preds))
    return inv_map.get(top_idx)

def run_local_test(model, token_index, inv_map, stockfish_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print("🎯 Local test: LarryBot (White) vs Stockfish (Black)")

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            moves = [m.uci() for m in board.move_stack]
            move_uci = predict_move(model, token_index, inv_map, moves)
            if not move_uci:
                print("⚠️ LarryBot predicted no move; resigning.")
                break
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                print(f"🤖 LarryBot plays: {move_uci}")
            else:
                print(f"⚠️ Illegal move predicted: {move_uci}")
                break
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            print(f"💡 Stockfish plays: {result.move.uci()}")

        print(board, "\n")

    engine.quit()
    print(f"🏁 Game over: {board.result()} ({board.outcome()})")

def handle_lichess_game(client, model, token_index, inv_map, game_id):
    board = chess.Board()
    for event in client.bots.stream_game_state(game_id):
        if event["type"] not in ("gameFull", "gameState"):
            continue
        moves = event["moves"].split()
        if board.turn == chess.WHITE:
            move_uci = predict_move(model, token_index, inv_map, moves)
            if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves:
                client.bots.make_move(game_id, move_uci)
                board.push_uci(move_uci)
                print(f"🤖 LarryBot played {move_uci}")

def run_live(token_file, model, token_index, inv_map):
    with open(token_file) as f:
        token = f.read().strip()

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    print("♟️ Connected to Lichess, waiting for challenges...")

    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            client.bots.accept_challenge(event["challenge"]["id"])
        elif event["type"] == "gameStart":
            handle_lichess_game(client, model, token_index, inv_map, event["game"]["id"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", help="Path to Lichess API token")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--move-map", required=True)
    ap.add_argument("--local-test", action="store_true")
    ap.add_argument("--stockfish", default="/usr/local/bin/stockfish")
    args = ap.parse_args()

    print("📂 Loading model and move map ...")
    model = tf.keras.models.load_model(os.path.join(args.model_dir, "larrybot.keras"))
    token_index, inv_map = load_move_map(args.move_map)

    if args.local_test:
        run_local_test(model, token_index, inv_map, args.stockfish)
    else:
        if not args.token_file:
            raise ValueError("Must specify --token-file when not in --local-test mode.")
        run_live(args.token_file, model, token_index, inv_map)

if __name__ == "__main__":
    main()
