#!/usr/bin/env python3
"""
Fully automated Lichess bot using a fine-tuned LarryBot model.
Usage:
    python run_larrybot.py --token-file path/to/token.txt --model-dir path/to/larrybot
"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import chess
import berserk

def load_token(token_file):
    with open(token_file, "r") as f:
        return f.read().strip()

def encode_board(board):
    """
    Encode a chess.Board into 8x8x12 tensor.
    Order of pieces: P, N, B, R, Q, K for white, then black.
    """
    X = np.zeros((8,8,12), dtype=np.float32)
    piece_map = board.piece_map()
    piece_to_idx = {
        "P":0, "N":1, "B":2, "R":3, "Q":4, "K":5,
        "p":6, "n":7, "b":8, "r":9, "q":10, "k":11
    }
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_idx[piece.symbol()]
        X[row, col, idx] = 1
    return X

def board_to_input(board):
    """Flatten to match your trained model input (120 or 768)."""
    X = encode_board(board)
    return X.flatten()[np.newaxis, :]  # shape (1, 768)

def pick_move(model, board, legal_moves, move_index_map):
    """
    Predict next move from model.
    `move_index_map` maps output index to SAN or UCI string.
    """
    X = board_to_input(board)
    preds = model.predict(X, verbose=0)
    # Convert predictions to legal move indices
    legal_indices = [i for i, move in enumerate(move_index_map) if move in [m.uci() for m in legal_moves]]
    if not legal_indices:
        return np.random.choice(list(legal_moves))
    probs = preds[0][legal_indices]
    move_idx = legal_indices[np.argmax(probs)]
    return chess.Move.from_uci(move_index_map[move_idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", required=True, help="Path to Lichess API token")
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned LarryBot model saved_model dir")
    parser.add_argument("--move-map", required=True, help="Path to JSON file mapping model outputs to UCI moves")
    args = parser.parse_args()

    token = load_token(args.token_file)
    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    print(f"📥 Loading model from {args.model_dir}")
    model = tf.keras.models.load_model(args.model_dir)

    print(f"📥 Loading move map from {args.move_map}")
    with open(args.move_map, "r") as f:
        move_index_map = json.load(f)

    print("🤖 LarryBot ready! Listening for incoming challenges...")

    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            challenge_id = event["challenge"]["id"]
            print(f"✅ Accepting challenge {challenge_id}")
            client.bots.accept_challenge(challenge_id)

        elif event["type"] == "gameStart":
            game_id = event["game"]["id"]
            print(f"🎮 Game started: {game_id}")

            for state in client.bots.stream_game_state(game_id):
                board = chess.Board(state["state"]["fen"])
                if board.turn == chess.WHITE and state["state"]["moves"].count(" ") % 2 == 0:
                    # bot is white
                    legal_moves = list(board.legal_moves)
                    move = pick_move(model, board, legal_moves, move_index_map)
                    print(f"♟️ Playing move {move.uci()}")
                    client.bots.make_move(game_id, move.uci())
                elif board.turn == chess.BLACK and state["state"]["moves"].count(" ") % 2 == 1:
                    # bot is black
                    legal_moves = list(board.legal_moves)
                    move = pick_move(model, board, legal_moves, move_index_map)
                    print(f"♟️ Playing move {move.uci()}")
                    client.bots.make_move(game_id, move.uci())

if __name__ == "__main__":
    main()
