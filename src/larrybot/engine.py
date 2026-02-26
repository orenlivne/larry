"""Stockfish wrapper with UCI_Elo control for real ELO calibration."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine

from .config import MAX_UCI_ELO, MIN_UCI_ELO


@dataclass
class CandidateMove:
    """A candidate move with its engine evaluation."""

    move: chess.Move
    score_cp: int
    depth: int


class StockfishEngine:
    """Thin wrapper around Stockfish with ELO-limited play."""

    def __init__(
        self,
        stockfish_path: str,
        elo: int,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})
        self.set_elo(elo)

    def set_elo(self, elo: int) -> None:
        if elo < MIN_UCI_ELO or elo > MAX_UCI_ELO:
            raise ValueError(
                f"UCI_Elo must be between {MIN_UCI_ELO} and {MAX_UCI_ELO}, got {elo}"
            )
        self._elo = elo
        self._engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": elo}
        )

    @property
    def elo(self) -> int:
        return self._elo

    def get_candidates(
        self,
        board: chess.Board,
        num_moves: int = 5,
        time_limit: float = 0.1,
    ) -> list[CandidateMove]:
        """Return top-N candidate moves with evaluations via MultiPV."""
        legal_count = board.legal_moves.count()
        multipv = min(num_moves, legal_count)
        if multipv == 0:
            return []

        results = self._engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=multipv,
        )

        candidates = []
        for info in results:
            move = info.get("pv", [None])[0]
            if move is None:
                continue
            score = info.get("score")
            cp = 0
            if score is not None:
                pov = score.white() if board.turn == chess.WHITE else score.relative
                cp = pov.score(mate_score=10000)
            depth = info.get("depth", 0)
            candidates.append(CandidateMove(move=move, score_cp=cp, depth=depth))

        return candidates

    def play_move(
        self, board: chess.Board, time_limit: float = 0.1
    ) -> chess.Move:
        """Let Stockfish pick a single move at the configured ELO."""
        result = self._engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> StockfishEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
