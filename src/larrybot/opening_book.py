"""Personal opening book built from a player's PGN games."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import chess

from .pgn_utils import iter_player_positions


class OpeningBook:
    """Weighted opening book keyed by position FEN (board-only part).

    Each entry maps ``{move_uci: play_count}`` so the most frequently
    played moves are chosen more often, keeping the bot's opening
    repertoire faithful to the player's real games.
    """

    def __init__(self) -> None:
        # position_key -> {move_uci: count}
        self._book: dict[str, dict[str, int]] = {}

    @staticmethod
    def _position_key(board: chess.Board) -> str:
        """Use the board FEN (pieces + side + castling + ep) as key."""
        return board.fen()

    @classmethod
    def from_pgn(
        cls,
        pgn_path: str,
        player_name: str,
        max_depth: int = 20,
    ) -> OpeningBook:
        """Build an opening book from a PGN file.

        Only the player's own moves (up to ``max_depth`` half-moves into
        each game) are recorded.
        """
        book = cls()
        for board, move, _color in iter_player_positions(pgn_path, player_name):
            if board.fullmove_number > max_depth:
                continue
            key = cls._position_key(board)
            uci = move.uci()
            entry = book._book.setdefault(key, {})
            entry[uci] = entry.get(uci, 0) + 1
        return book

    def lookup(self, board: chess.Board) -> Optional[chess.Move]:
        """Return a move weighted by play frequency, or ``None``."""
        key = self._position_key(board)
        entry = self._book.get(key)
        if not entry:
            return None

        # Filter to currently-legal moves only
        legal_ucis = {m.uci() for m in board.legal_moves}
        candidates = {uci: cnt for uci, cnt in entry.items() if uci in legal_ucis}
        if not candidates:
            return None

        moves = list(candidates.keys())
        weights = [candidates[m] for m in moves]
        chosen = random.choices(moves, weights=weights, k=1)[0]
        return chess.Move.from_uci(chosen)

    @property
    def position_count(self) -> int:
        return len(self._book)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._book, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> OpeningBook:
        book = cls()
        book._book = json.loads(Path(path).read_text())
        return book
