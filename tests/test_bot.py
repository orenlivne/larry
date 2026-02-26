"""Tests for the main PlayerBot."""

import chess
import pytest

from larrybot.bot import PlayerBot
from larrybot.config import PlayerConfig, StyleVector
from larrybot.engine import StockfishEngine
from larrybot.opening_book import OpeningBook
from conftest import SAMPLE_PGN, get_stockfish_path, requires_stockfish


@requires_stockfish
class TestPlayerBot:

    def _make_bot(self, elo=2000, with_book=True, with_style=True):
        sf_path = get_stockfish_path()
        config = PlayerConfig(
            player_name="Christiansen",
            classical_elo=elo,
            stockfish_path=sf_path,
        )
        engine = StockfishEngine(sf_path, config.clamped_elo)
        book = OpeningBook.from_pgn(SAMPLE_PGN, "Christiansen") if with_book else None
        style = StyleVector(aggression=0.7, material_weight=0.4, activity_weight=0.6, centrality=0.5) if with_style else None
        return PlayerBot(config, engine, book, style)

    def test_select_move_returns_legal(self):
        bot = self._make_bot()
        try:
            board = chess.Board()
            move = bot.select_move(board)
            assert move in board.legal_moves
        finally:
            bot.close()

    def test_uses_opening_book(self):
        """First move should come from the book (e4)."""
        bot = self._make_bot()
        try:
            board = chess.Board()
            move = bot.select_move(board)
            # In sample PGN, Larry always plays 1.e4
            assert move.uci() == "e2e4"
        finally:
            bot.close()

    def test_falls_back_to_engine_without_book(self):
        bot = self._make_bot(with_book=False)
        try:
            board = chess.Board()
            move = bot.select_move(board)
            assert move in board.legal_moves
        finally:
            bot.close()

    def test_plays_short_game(self):
        """Play 20 plies without crashing."""
        bot = self._make_bot(elo=1500)
        opponent = StockfishEngine(get_stockfish_path(), elo=1500)
        try:
            board = chess.Board()
            moves_played = 0
            for _ in range(20):
                if board.is_game_over():
                    break
                if board.turn == chess.WHITE:
                    move = bot.select_move(board)
                else:
                    move = opponent.play_move(board)
                assert move in board.legal_moves
                board.push(move)
                moves_played += 1
            assert moves_played >= 10
        finally:
            bot.close()
            opponent.close()

    def test_never_returns_none(self):
        bot = self._make_bot()
        try:
            board = chess.Board()
            for _ in range(10):
                if board.is_game_over():
                    break
                move = bot.select_move(board)
                assert move is not None
                board.push(move)
                if not board.is_game_over():
                    # Push a simple opponent move
                    opp = list(board.legal_moves)[0]
                    board.push(opp)
        finally:
            bot.close()

    def test_engine_only_no_style(self):
        """Works with engine only (no style, no book)."""
        bot = self._make_bot(with_book=False, with_style=False)
        try:
            board = chess.Board()
            move = bot.select_move(board)
            assert move in board.legal_moves
        finally:
            bot.close()
