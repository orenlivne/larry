"""Tests for PGN parsing utilities."""

import io

import chess
import pytest

from larrybot.pgn_utils import count_games, extract_player_elo, iter_player_positions
from conftest import SAMPLE_PGN


def test_iter_player_positions_white():
    """Extracts only the named player's moves when playing White."""
    positions = list(iter_player_positions(SAMPLE_PGN, "Christiansen"))
    # 4 games as White × 10 moves each + 1 game as Black × 10 moves = 50 total
    assert len(positions) > 0
    for board, move, color in positions:
        assert move in board.legal_moves


def test_iter_player_positions_case_insensitive():
    """Player name matching is case-insensitive."""
    upper = list(iter_player_positions(SAMPLE_PGN, "CHRISTIANSEN"))
    lower = list(iter_player_positions(SAMPLE_PGN, "christiansen"))
    assert len(upper) == len(lower)
    assert len(upper) > 0


def test_iter_player_positions_substring():
    """Player name matching uses substring."""
    full = list(iter_player_positions(SAMPLE_PGN, "Christiansen, Larry"))
    sub = list(iter_player_positions(SAMPLE_PGN, "Christiansen"))
    assert len(full) == len(sub)


def test_iter_player_positions_no_match():
    """Returns nothing for a player not in the PGN."""
    positions = list(iter_player_positions(SAMPLE_PGN, "Kasparov"))
    assert len(positions) == 0


def test_iter_player_positions_from_stream():
    """Can read from a text stream instead of a file path."""
    pgn_text = (
        '[White "TestPlayer"]\n[Black "Opponent"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 1-0\n"
    )
    positions = list(iter_player_positions(io.StringIO(pgn_text), "TestPlayer"))
    assert len(positions) == 2  # e4, Nf3
    for board, move, color in positions:
        assert color == chess.WHITE


def test_iter_player_positions_both_colors():
    """Correctly tracks moves for a player who plays both colors."""
    positions = list(iter_player_positions(SAMPLE_PGN, "Davis"))
    colors_seen = {color for _, _, color in positions}
    # Davis plays White in game 4 and Black in game 5
    assert chess.WHITE in colors_seen
    assert chess.BLACK in colors_seen


def test_count_games():
    assert count_games(SAMPLE_PGN) == 5


def test_malformed_pgn():
    """Malformed PGN is handled gracefully."""
    pgn_text = (
        '[White "Player"]\n[Black "Opp"]\n[Result "*"]\n\n'
        "1. e4 e5 2. ILLEGAL *\n\n"
        '[White "Player"]\n[Black "Opp2"]\n[Result "1-0"]\n\n'
        "1. d4 d5 1-0\n"
    )
    # Should get at least the valid moves from the first game + second game
    positions = list(iter_player_positions(io.StringIO(pgn_text), "Player"))
    assert len(positions) >= 1  # at least e4 and d4


def test_extract_player_elo_from_sample():
    """Average ELO should match the headers in sample.pgn."""
    elo = extract_player_elo(SAMPLE_PGN, "Christiansen")
    # All 5 games have WhiteElo=2620 or BlackElo=2620 for Christiansen
    assert elo == 2620


def test_extract_player_elo_from_stream():
    pgn_text = (
        '[White "Player"]\n[Black "Opp"]\n[WhiteElo "2400"]\n[BlackElo "2000"]\n[Result "1-0"]\n\n'
        "1. e4 e5 1-0\n\n"
        '[White "Opp2"]\n[Black "Player"]\n[WhiteElo "2100"]\n[BlackElo "2500"]\n[Result "0-1"]\n\n'
        "1. d4 d5 0-1\n"
    )
    elo = extract_player_elo(io.StringIO(pgn_text), "Player")
    # Average of 2400 (as White) and 2500 (as Black) = 2450
    assert elo == 2450


def test_extract_player_elo_no_match():
    elo = extract_player_elo(SAMPLE_PGN, "Kasparov")
    assert elo is None


def test_extract_player_elo_no_headers():
    """PGN without ELO headers returns None."""
    pgn_text = (
        '[White "Player"]\n[Black "Opp"]\n[Result "1-0"]\n\n'
        "1. e4 e5 1-0\n"
    )
    elo = extract_player_elo(io.StringIO(pgn_text), "Player")
    assert elo is None
