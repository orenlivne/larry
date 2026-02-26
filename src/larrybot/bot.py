"""Main bot: combines Stockfish engine, opening book, and style scoring."""

from __future__ import annotations

from typing import Optional

import chess

from .config import PlayerConfig, StyleVector
from .engine import StockfishEngine
from .opening_book import OpeningBook
from .style import score_move


class PlayerBot:
    """A chess bot that plays in a specific player's style at a target ELO.

    Move selection:
      1. Opening book (if position found) — weighted random by frequency.
      2. Stockfish candidates at target ELO, biased by style scoring when
         several moves are within ``candidate_threshold_cp`` of the best.
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

        # Phase 2: engine candidates + style bias
        candidates = self.engine.get_candidates(
            board,
            num_moves=self.config.num_candidates,
        )

        if not candidates:
            # Fallback: pick any legal move
            return next(iter(board.legal_moves))

        if len(candidates) == 1 or self.style is None:
            return candidates[0].move

        best_cp = candidates[0].score_cp
        threshold = self.config.candidate_threshold_cp

        # Moves within threshold of the best
        close = [c for c in candidates if best_cp - c.score_cp <= threshold]
        if len(close) <= 1:
            return candidates[0].move

        # Score by style alignment
        color = board.turn
        scored = [
            (c, score_move(board, c.move, color, self.style))
            for c in close
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].move

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> PlayerBot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
