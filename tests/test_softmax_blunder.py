"""Tests for softmax blunder injection: _softmax, temperature sampling, and
ELO-correlated blunder rates.

These tests use mocked engines so they run fast and without Stockfish.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from unittest.mock import MagicMock, patch

import chess
import pytest

from larrybot.bot import PlayerBot, _softmax
from larrybot.config import PlayerConfig, StyleVector, elo_to_temperature
from larrybot.engine import CandidateMove


# ---------------------------------------------------------------------------
# _softmax unit tests
# ---------------------------------------------------------------------------


class TestSoftmax:

    def test_empty_input(self):
        assert _softmax([], 1.0) == []

    def test_uniform_values(self):
        """Equal inputs → equal probabilities."""
        probs = _softmax([1.0, 1.0, 1.0], 1.0)
        assert len(probs) == 3
        for p in probs:
            assert abs(p - 1 / 3) < 1e-6

    def test_sums_to_one(self):
        probs = _softmax([3.0, 1.0, 0.5, -2.0], 0.5)
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_all_positive(self):
        probs = _softmax([10.0, -10.0], 1.0)
        for p in probs:
            assert p > 0

    def test_high_temperature_flattens(self):
        """Very high temperature → near-uniform distribution."""
        probs = _softmax([10.0, 0.0, -10.0], temperature=100.0)
        assert max(probs) - min(probs) < 0.1

    def test_low_temperature_sharpens(self):
        """Very low temperature → nearly all mass on the max."""
        probs = _softmax([10.0, 5.0, 0.0], temperature=0.01)
        assert probs[0] > 0.99

    def test_preserves_ordering(self):
        """Higher input values should get higher probabilities."""
        probs = _softmax([3.0, 2.0, 1.0], temperature=1.0)
        assert probs[0] > probs[1] > probs[2]

    def test_two_values_known(self):
        """Verify against hand-computed softmax for two values."""
        # softmax([a, b], T=1) = [e^a / (e^a + e^b), e^b / (e^a + e^b)]
        probs = _softmax([2.0, 0.0], temperature=1.0)
        expected_0 = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
        assert abs(probs[0] - expected_0) < 1e-9
        assert abs(probs[1] - (1.0 - expected_0)) < 1e-9

    def test_numerical_stability_large_values(self):
        """Should not overflow with large values (uses max-subtraction trick)."""
        probs = _softmax([1000.0, 999.0, 998.0], temperature=1.0)
        assert abs(sum(probs) - 1.0) < 1e-9
        assert probs[0] > probs[1] > probs[2]

    def test_near_zero_temperature(self):
        """Near-zero temperature should not crash (clamped internally)."""
        probs = _softmax([5.0, 3.0, 1.0], temperature=1e-12)
        assert abs(sum(probs) - 1.0) < 1e-6
        assert probs[0] > 0.99


# ---------------------------------------------------------------------------
# Helpers for mocked bot tests
# ---------------------------------------------------------------------------


def _make_candidates(scores_cp: list[int]) -> list[CandidateMove]:
    """Create fake CandidateMoves with the given centipawn scores.

    Uses legal moves from the starting position so the moves are real.
    """
    board = chess.Board()
    legal = list(board.legal_moves)
    return [
        CandidateMove(move=legal[i], score_cp=cp, depth=20)
        for i, cp in enumerate(scores_cp)
    ]


def _make_bot(elo: int, style: StyleVector | None = None) -> PlayerBot:
    """Create a PlayerBot with a mocked engine."""
    config = PlayerConfig(
        player_name="Test",
        classical_elo=elo,
        stockfish_path="/fake/stockfish",
        num_candidates=5,
    )
    engine = MagicMock()
    return PlayerBot(config, engine, book=None, style=style)


# ---------------------------------------------------------------------------
# select_move with mocked engine — softmax sampling behavior
# ---------------------------------------------------------------------------


class TestSelectMoveSampling:

    def test_single_candidate_returns_it(self):
        """With one candidate, always returns that move."""
        bot = _make_bot(2000)
        cands = _make_candidates([100])
        bot.engine.get_candidates.return_value = cands

        board = chess.Board()
        for _ in range(10):
            assert bot.select_move(board) == cands[0].move

    def test_always_returns_legal_move(self):
        """Every sampled move must be in the candidate list."""
        bot = _make_bot(1500)
        cands = _make_candidates([300, 200, 100, 0, -100])
        bot.engine.get_candidates.return_value = cands
        legal_moves = {c.move for c in cands}

        board = chess.Board()
        for _ in range(50):
            move = bot.select_move(board)
            assert move in legal_moves

    def test_high_elo_favors_best_move(self):
        """At high ELO (low temperature), the best move should dominate."""
        bot = _make_bot(3100)  # temp ≈ 0.05
        cands = _make_candidates([300, 100, 0, -50, -200])
        bot.engine.get_candidates.return_value = cands

        board = chess.Board()
        random.seed(42)
        counts = Counter()
        n = 500
        for _ in range(n):
            move = bot.select_move(board)
            counts[move] += 1

        best_move = cands[0].move
        assert counts[best_move] / n > 0.90, (
            f"At 3100 ELO, best move should be picked >90% of the time, "
            f"got {counts[best_move] / n:.1%}"
        )

    def test_low_elo_spreads_probability(self):
        """At low ELO (high temperature), non-best moves should appear often."""
        bot = _make_bot(1320)  # temp ≈ 0.85
        cands = _make_candidates([300, 200, 100, 0, -100])
        bot.engine.get_candidates.return_value = cands

        board = chess.Board()
        random.seed(42)
        counts = Counter()
        n = 1000
        for _ in range(n):
            move = bot.select_move(board)
            counts[move] += 1

        best_move = cands[0].move
        non_best_pct = 1 - (counts[best_move] / n)
        assert non_best_pct > 0.15, (
            f"At 1320 ELO, non-best moves should appear >15% of the time, "
            f"got {non_best_pct:.1%}"
        )
        # At least 3 different moves should be chosen
        assert len(counts) >= 3, (
            f"At 1320 ELO, should pick at least 3 distinct moves, got {len(counts)}"
        )

    def test_style_bonus_shifts_preference(self):
        """With a strong style vector, style-aligned moves should be preferred
        even if not the engine's top pick."""
        # Use a high-aggression style
        style = StyleVector(aggression=1.0, material_weight=0.0,
                            activity_weight=0.0, centrality=0.0)
        bot = _make_bot(2000, style=style)

        # Give candidate 0 the best engine score but candidates are close
        cands = _make_candidates([110, 100, 105, 95, 90])
        bot.engine.get_candidates.return_value = cands

        board = chess.Board()
        random.seed(42)
        counts = Counter()
        n = 500
        for _ in range(n):
            move = bot.select_move(board)
            counts[move] += 1

        # With close engine scores, style bonus should cause non-trivial
        # variation — not just always the engine's #1 pick
        assert len(counts) >= 2, "Style bonus should cause move diversity"


# ---------------------------------------------------------------------------
# Blunder rate: the core "feels human" test
# ---------------------------------------------------------------------------


class TestBlunderRate:
    """Verify that lower-ELO bots blunder more than higher-ELO bots.

    A "blunder" here = choosing a move whose engine eval is significantly
    worse than the best candidate. This is a statistical test over many
    samples to confirm temperature calibration produces human-like rates.
    """

    BLUNDER_THRESHOLD_CP = 100  # ≥1 pawn worse than best = blunder

    @staticmethod
    def _blunder_rate(elo: int, candidates_cp: list[int], n_trials: int = 2000) -> float:
        """Measure how often the bot picks a move ≥ BLUNDER_THRESHOLD_CP worse
        than the best available."""
        bot = _make_bot(elo)
        cands = _make_candidates(candidates_cp)
        bot.engine.get_candidates.return_value = cands

        best_cp = max(candidates_cp)
        cp_by_move = {cands[i].move: candidates_cp[i] for i in range(len(cands))}

        board = chess.Board()
        blunders = 0
        for _ in range(n_trials):
            move = bot.select_move(board)
            if best_cp - cp_by_move[move] >= TestBlunderRate.BLUNDER_THRESHOLD_CP:
                blunders += 1

        return blunders / n_trials

    def test_low_elo_blunders_more_than_high_elo(self):
        """1400-ELO bot should blunder significantly more than a 2800 bot."""
        random.seed(123)
        # Realistic-ish candidate spread: best +300, then declining
        candidates_cp = [300, 200, 100, 0, -150]

        low_rate = self._blunder_rate(1400, candidates_cp)
        high_rate = self._blunder_rate(2800, candidates_cp)

        assert low_rate > high_rate, (
            f"Low ELO blunder rate ({low_rate:.1%}) should exceed "
            f"high ELO rate ({high_rate:.1%})"
        )
        # Low ELO should blunder at least 5% of the time
        assert low_rate > 0.05, f"1400 ELO blunder rate too low: {low_rate:.1%}"
        # High ELO should blunder less than low
        assert high_rate < low_rate * 0.5, (
            f"2800 ELO ({high_rate:.1%}) should be <50% of 1400 ELO ({low_rate:.1%})"
        )

    def test_blunder_rate_monotonically_decreasing(self):
        """Blunder rate should decrease as ELO increases."""
        random.seed(456)
        candidates_cp = [250, 150, 50, -50, -200]
        elos = [1400, 1800, 2200, 2600, 3000]

        rates = [self._blunder_rate(elo, candidates_cp, n_trials=1500) for elo in elos]

        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                f"Blunder rate at {elos[i]} ELO ({rates[i]:.1%}) should be >= "
                f"rate at {elos[i + 1]} ELO ({rates[i + 1]:.1%})"
            )

    def test_mid_elo_blunder_rate_reasonable(self):
        """A ~2000 ELO bot should blunder roughly 5-25% with a spread of
        candidates — similar to human club players."""
        random.seed(789)
        candidates_cp = [200, 120, 50, -30, -150]

        rate = self._blunder_rate(2000, candidates_cp, n_trials=3000)
        assert 0.03 < rate < 0.35, (
            f"2000 ELO blunder rate should be 3-35%, got {rate:.1%}"
        )

    def test_extreme_low_elo_blunders_often(self):
        """At the minimum ELO, blunders should be frequent."""
        random.seed(321)
        candidates_cp = [300, 150, 0, -100, -300]

        rate = self._blunder_rate(1320, candidates_cp, n_trials=2000)
        assert rate > 0.10, f"1320 ELO should blunder >10%, got {rate:.1%}"
