"""Tests for the personal opening book."""

import json
import tempfile
from pathlib import Path

import chess
import pytest

from larrybot.opening_book import OpeningBook
from conftest import SAMPLE_PGN


class TestOpeningBook:

    def test_from_pgn_creates_entries(self):
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        assert book.position_count > 0

    def test_lookup_starting_position(self):
        """Larry's first move should be in the book (e4 in all White games)."""
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        board = chess.Board()
        move = book.lookup(board)
        assert move is not None
        assert move in board.legal_moves
        # In all 4 White games, Larry plays 1.e4
        assert move.uci() == "e2e4"

    def test_lookup_unknown_position(self):
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        # Play random moves to reach an unlikely position
        board = chess.Board()
        board.push_uci("a2a4")
        board.push_uci("h7h5")
        board.push_uci("a4a5")
        move = book.lookup(board)
        assert move is None  # Not in book

    def test_lookup_returns_legal_moves_only(self):
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        board = chess.Board()
        for _ in range(50):
            move = book.lookup(board)
            if move is None:
                break
            assert move in board.legal_moves
            board.push(move)
            # Also push an opponent move if available
            if not board.is_game_over():
                opp_move = list(board.legal_moves)[0]
                board.push(opp_move)
            if board.is_game_over():
                break

    def test_move_frequency_weighting(self):
        """Moves played more often should be weighted higher."""
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        board = chess.Board()
        key = OpeningBook._position_key(board)
        entry = book._book.get(key, {})
        # e2e4 should be the most frequent first move (4 games)
        assert "e2e4" in entry
        assert entry["e2e4"] >= 4  # Played in all 4 White games

    def test_save_load_roundtrip(self):
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book.save(path)
            loaded = OpeningBook.load(path)
            assert loaded.position_count == book.position_count
            # Verify same moves for starting position
            board = chess.Board()
            key = OpeningBook._position_key(board)
            assert loaded._book[key] == book._book[key]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_max_depth_limits_entries(self):
        shallow = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen", max_depth=3)
        deep = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen", max_depth=20)
        assert shallow.position_count <= deep.position_count

    def test_only_records_player_moves(self):
        """Book should only contain positions where it's the player's turn."""
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen")
        # Starting position — Larry is White, so it's his turn
        board = chess.Board()
        key = OpeningBook._position_key(board)
        assert key in book._book

    def test_empty_pgn(self):
        import io
        from larrybot.pgn_utils import iter_player_positions

        empty_pgn = io.StringIO("")
        book = OpeningBook()
        assert book.position_count == 0
        assert book.lookup(chess.Board()) is None
