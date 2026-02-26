"""PGN parsing utilities for extracting player positions and moves."""

from __future__ import annotations

import io
from typing import Iterator

import chess
import chess.pgn


def _player_matches(header_name: str, player_name: str) -> bool:
    """Case-insensitive substring match for player names."""
    return player_name.lower() in header_name.lower()


def iter_player_positions(
    pgn_source: str | io.TextIOBase,
    player_name: str,
) -> Iterator[tuple[chess.Board, chess.Move, chess.Color]]:
    """Yield (board_before_move, player_move, player_color) for each of the
    player's moves across all games in the PGN.

    ``pgn_source`` can be a file path or an open text stream.
    Player name matching is case-insensitive substring.
    Corrupted or illegal-move games are silently skipped.
    """
    if isinstance(pgn_source, str):
        handle = open(pgn_source, errors="replace")
        should_close = True
    else:
        handle = pgn_source
        should_close = False

    try:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")

            colors: list[chess.Color] = []
            if _player_matches(white, player_name):
                colors.append(chess.WHITE)
            if _player_matches(black, player_name):
                colors.append(chess.BLACK)
            if not colors:
                continue

            board = game.board()
            try:
                for move in game.mainline_moves():
                    if board.turn in colors:
                        yield board.copy(), move, board.turn
                    board.push(move)
            except (ValueError, IllegalMoveError):
                continue
    finally:
        if should_close:
            handle.close()


class IllegalMoveError(Exception):
    pass


def extract_player_elo(
    pgn_source: str | io.TextIOBase,
    player_name: str,
) -> int | None:
    """Compute the average ELO of a player from PGN header fields.

    Looks at WhiteElo/BlackElo headers for games where the player appears.
    Returns the rounded mean, or ``None`` if no ratings are found.
    """
    if isinstance(pgn_source, str):
        handle = open(pgn_source, errors="replace")
        should_close = True
    else:
        handle = pgn_source
        should_close = False

    elos: list[int] = []
    try:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if _player_matches(white, player_name):
                raw = game.headers.get("WhiteElo", "")
                if raw.isdigit():
                    elos.append(int(raw))
            if _player_matches(black, player_name):
                raw = game.headers.get("BlackElo", "")
                if raw.isdigit():
                    elos.append(int(raw))
    finally:
        if should_close:
            handle.close()

    if not elos:
        return None
    return round(sum(elos) / len(elos))


def count_games(pgn_source: str | io.TextIOBase) -> int:
    """Count the number of games in a PGN source."""
    if isinstance(pgn_source, str):
        handle = open(pgn_source, errors="replace")
        should_close = True
    else:
        handle = pgn_source
        should_close = False

    count = 0
    try:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            count += 1
    finally:
        if should_close:
            handle.close()
    return count
