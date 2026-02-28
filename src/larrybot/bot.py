"""Main bot: combines Stockfish engine, opening book, and style scoring.

Move selection uses softmax sampling over candidate moves, with temperature
calibrated to the target ELO so the bot blunders at human-realistic rates.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import chess

from .config import PlayerConfig, StyleVector, elo_to_temperature
from .engine import CandidateMove, StockfishEngine
from .opening_book import OpeningBook
from .style import score_move


def _softmax(values: list[float], temperature: float) -> list[float]:
    """Softmax with temperature. Lower temp = more deterministic."""
    if not values:
        return []
    scaled = [v / max(temperature, 1e-8) for v in values]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


class PlayerBot:
    """A chess bot that plays in a specific player's style at a target ELO.

    Move selection:
      1. Opening book (if position found) — weighted random by frequency.
      2. Stockfish candidates at target ELO, scored by a blend of engine
         evaluation and style alignment, then sampled via softmax with
         ELO-calibrated temperature for human-like imperfection.
    """

    def __init__(
        self,
        config: PlayerConfig,
        engine: StockfishEngine,
        book: Optional[OpeningBook] = None,
        style: Optional[StyleVector] = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.book = book
        self.style = style
        self._temperature = elo_to_temperature(config.clamped_elo)

    @classmethod
    def from_player_dir(cls, player_dir: str) -> PlayerBot:
        """Load a fully configured bot from a player directory."""
        from pathlib import Path

        pdir = Path(player_dir)
        config = PlayerConfig.load(pdir / "config.json")

        book = None
        book_file = pdir / "opening_book.json"
        if book_file.exists():
            book = OpeningBook.load(book_file)

        engine = StockfishEngine(config.stockfish_path, config.clamped_elo)
        return cls(config, engine, book, config.style)

    def select_move(self, board: chess.Board) -> chess.Move:
        """Pick a move for the current position."""
        # Phase 1: opening book
        if self.book is not None:
            book_move = self.book.lookup(board)
            if book_move is not None:
                return book_move

        # Phase 2: engine candidates + style bias + human-like sampling
        candidates = self.engine.get_candidates(
            board,
            num_moves=self.config.num_candidates,
        )

        if not candidates:
            return next(iter(board.legal_moves))

        if len(candidates) == 1:
            return candidates[0].move

        # Compute combined scores: engine eval (in pawns) + style bonus
        color = board.turn
        combined: list[float] = []
        for c in candidates:
            engine_score = c.score_cp / 100.0  # centipawns → pawns
            style_bonus = 0.0
            if self.style is not None:
                style_bonus = score_move(board, c.move, color, self.style)
            combined.append(engine_score + style_bonus)

        # Sample via softmax with ELO-calibrated temperature
        probs = _softmax(combined, self._temperature)
        chosen_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[chosen_idx].move

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> PlayerBot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
