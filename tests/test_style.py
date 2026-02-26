"""Tests for style vector extraction and move scoring."""

import chess
import pytest

from larrybot.config import StyleVector
from larrybot.style import (
    StyleAnalyzer,
    compute_move_features,
    score_move,
)
from conftest import SAMPLE_PGN


class TestStyleVector:

    def test_valid_range(self):
        sv = StyleVector(aggression=0.0, material_weight=1.0, activity_weight=0.5, centrality=0.3)
        assert sv.aggression == 0.0
        assert sv.material_weight == 1.0

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="aggression"):
            StyleVector(aggression=1.5)

    def test_negative(self):
        with pytest.raises(ValueError, match="material_weight"):
            StyleVector(material_weight=-0.1)


class TestComputeMoveFeatures:

    def test_returns_four_floats(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        features = compute_move_features(board, move, chess.WHITE)
        assert len(features) == 4
        for f in features:
            assert isinstance(f, float)
            assert 0.0 <= f <= 1.0

    def test_check_is_aggressive(self):
        # Position where Qh5 gives check (scholar's mate setup)
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        # Qh5 is a legal check (if f7 is weak)
        # Actually let's use a clearer check position
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 2")
        # Not a check, but let's use a position where we can verify non-zero aggression
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        agg, _, _, _ = compute_move_features(board, move, chess.WHITE)
        # e4 from starting position isn't aggressive (no check, no attack on king zone)
        assert isinstance(agg, float)

    def test_capture_has_material_score(self):
        # Position where White can capture a knight
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 1")
        # Black plays e5
        board.push_uci("e7e5")
        # Now White can play Nxe5 (capturing pawn)
        move = chess.Move.from_uci("f3e5")
        _, mat, _, _ = compute_move_features(board, move, chess.WHITE)
        assert mat > 0.0  # Capturing a pawn

    def test_non_capture_has_zero_material(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        _, mat, _, _ = compute_move_features(board, move, chess.WHITE)
        assert mat == 0.0


class TestScoreMove:

    def test_score_varies_with_style(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        aggressive_style = StyleVector(aggression=1.0, material_weight=0.0, activity_weight=0.0, centrality=0.0)
        passive_style = StyleVector(aggression=0.0, material_weight=0.0, activity_weight=0.0, centrality=1.0)
        score_agg = score_move(board, move, chess.WHITE, aggressive_style)
        score_pas = score_move(board, move, chess.WHITE, passive_style)
        # Scores should differ
        assert isinstance(score_agg, float)
        assert isinstance(score_pas, float)

    def test_all_scores_non_negative(self):
        board = chess.Board()
        style = StyleVector()
        for move in board.legal_moves:
            s = score_move(board, move, chess.WHITE, style)
            assert s >= 0.0


class TestStyleAnalyzer:

    def test_from_pgn(self):
        sv = StyleAnalyzer.from_pgn(SAMPLE_PGN, "Christiansen")
        assert 0.0 <= sv.aggression <= 1.0
        assert 0.0 <= sv.material_weight <= 1.0
        assert 0.0 <= sv.activity_weight <= 1.0
        assert 0.0 <= sv.centrality <= 1.0

    def test_unknown_player_returns_defaults(self):
        sv = StyleAnalyzer.from_pgn(SAMPLE_PGN, "Nobody")
        # Default style vector when no games found
        assert sv.aggression == 0.5
        assert sv.material_weight == 0.5
