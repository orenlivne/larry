import os
import chess
import chess.engine
import numpy as np
import pytest
import tensorflow as tf


from run_larrybot import (
    predict_maia_move,
)

MODEL_PATH = "./models/active/maia_larry_finetuned.keras"
MOVE_MAP_PATH = "./data/move_map.json"
STOCKFISH = "/opt/homebrew/bin/stockfish"   # brew install stockfish

@pytest.fixture(scope="session")
def model():
    assert os.path.exists(MODEL_PATH), "Model not found"
    return tf.keras.models.load_model(MODEL_PATH)

@pytest.fixture(scope="session")
def move_map():
    import json
    with open(MOVE_MAP_PATH) as f:
        move_to_idx = json.load(f)
    inv = {int(v): k for k, v in move_to_idx.items()}  # idx → SAN
    return move_to_idx, inv

def test_larry_vs_stockfish(model, move_map):
    move_to_idx, inv_map = move_map

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

    larry_move_count = 0

    for ply in range(20):  # short test — 20 plies max
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            mv = predict_maia_move(model, board, move_to_idx, inv_map)
            assert mv is not None, "LarryBot returned no legal move"
            assert mv in board.legal_moves, "LarryBot returned illegal move"
            board.push(mv)
            larry_move_count += 1
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))
            board.push(result.move)

    engine.quit()

    assert larry_move_count > 0, "LarryBot never played a move"
    assert larry_move_count >= 3, "LarryBot too weak or failed early"
