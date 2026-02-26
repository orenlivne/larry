"""Tests for the Stockfish engine wrapper."""

import chess
import pytest

from larrybot.config import MAX_UCI_ELO, MIN_UCI_ELO
from larrybot.engine import StockfishEngine
from conftest import get_stockfish_path, requires_stockfish


@requires_stockfish
class TestStockfishEngine:

    def test_init_and_elo(self):
        with StockfishEngine(get_stockfish_path(), elo=1500) as eng:
            assert eng.elo == 1500

    def test_set_elo(self):
        with StockfishEngine(get_stockfish_path(), elo=1500) as eng:
            eng.set_elo(2000)
            assert eng.elo == 2000

    def test_elo_too_low(self):
        with pytest.raises(ValueError, match="UCI_Elo must be between"):
            StockfishEngine(get_stockfish_path(), elo=500)

    def test_elo_too_high(self):
        with pytest.raises(ValueError, match="UCI_Elo must be between"):
            StockfishEngine(get_stockfish_path(), elo=4000)

    def test_elo_boundaries(self):
        with StockfishEngine(get_stockfish_path(), elo=MIN_UCI_ELO) as eng:
            assert eng.elo == MIN_UCI_ELO
        with StockfishEngine(get_stockfish_path(), elo=MAX_UCI_ELO) as eng:
            assert eng.elo == MAX_UCI_ELO

    def test_get_candidates_returns_legal_moves(self):
        board = chess.Board()
        with StockfishEngine(get_stockfish_path(), elo=2000) as eng:
            candidates = eng.get_candidates(board, num_moves=3)
            assert len(candidates) > 0
            assert len(candidates) <= 3
            for c in candidates:
                assert c.move in board.legal_moves

    def test_get_candidates_sorted_by_score(self):
        board = chess.Board()
        with StockfishEngine(get_stockfish_path(), elo=2000) as eng:
            candidates = eng.get_candidates(board, num_moves=5)
            if len(candidates) >= 2:
                # First should be best or equal
                assert candidates[0].score_cp >= candidates[-1].score_cp

    def test_get_candidates_empty_board(self):
        """No candidates when game is over (stalemate/checkmate)."""
        # Scholar's mate position — Black is checkmated
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push_uci(uci)
        assert board.is_game_over()
        with StockfishEngine(get_stockfish_path(), elo=2000) as eng:
            candidates = eng.get_candidates(board)
            assert len(candidates) == 0

    def test_play_move_returns_legal(self):
        board = chess.Board()
        with StockfishEngine(get_stockfish_path(), elo=1500) as eng:
            move = eng.play_move(board)
            assert move in board.legal_moves

    def test_context_manager(self):
        with StockfishEngine(get_stockfish_path(), elo=1500) as eng:
            move = eng.play_move(chess.Board())
            assert move is not None
        # Engine should be closed — no error expected
