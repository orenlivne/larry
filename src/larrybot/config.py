"""Player configuration: name, ELO, style, paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


MIN_UCI_ELO = 1320
MAX_UCI_ELO = 3190

RATING_SYSTEMS = ["fide", "lichess", "chesscom", "uscf"]


def convert_to_fide(rating: int, system: str) -> int:
    """Convert a rating from *system* to FIDE-equivalent.

    Stockfish's UCI_Elo is calibrated against FIDE/CCRL. Ratings from other
    systems must be converted so the bot plays at the correct strength.

    Approximate linear fits derived from cross-rating statistical analysis
    (ChessGoals.com, 2025).  Use ``--elo-offset`` for further fine-tuning.
    """
    if system == "fide":
        return rating
    if system == "lichess":
        return round(0.84 * rating + 65)
    if system == "chesscom":
        return round(0.93 * rating + 65)
    if system == "uscf":
        return round(0.94 * rating + 70)
    raise ValueError(
        f"Unknown rating system {system!r}. "
        f"Valid systems: {', '.join(RATING_SYSTEMS)}"
    )


def elo_to_temperature(elo: int) -> float:
    """Map ELO to a softmax temperature for human-like move sampling.

    Lower ELO → higher temperature → more "blunders" (picks non-best moves).
    Higher ELO → lower temperature → near-deterministic (picks the best).

    Calibrated so that:
      - 1320 ELO ≈ 0.90 (often picks 2nd/3rd best, ~15-20% blunder-like)
      - 2000 ELO ≈ 0.55 (occasionally picks non-best, ~8-10%)
      - 2500 ELO ≈ 0.30 (rarely deviates, ~3-5%)
      - 3190 ELO ≈ 0.05 (near-deterministic)
    """
    return max(0.05, (3200 - elo) / 2200)


@dataclass
class StyleVector:
    """Captures a player's stylistic tendencies (each value 0-1)."""

    aggression: float = 0.5
    material_weight: float = 0.5
    activity_weight: float = 0.5
    centrality: float = 0.5

    def __post_init__(self) -> None:
        for name in ("aggression", "material_weight", "activity_weight", "centrality"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass
class PlayerConfig:
    """Everything needed to instantiate a player bot."""

    player_name: str
    classical_elo: int
    elo_offset: int = 0
    stockfish_path: str = "/opt/homebrew/bin/stockfish"
    book_path: Optional[str] = None
    style: Optional[StyleVector] = None
    candidate_threshold_cp: int = 80
    num_candidates: int = 5
    book_depth: int = 20

    @property
    def target_elo(self) -> int:
        return self.classical_elo + self.elo_offset

    @property
    def clamped_elo(self) -> int:
        """Target ELO clamped to Stockfish's UCI_Elo range."""
        return max(MIN_UCI_ELO, min(MAX_UCI_ELO, self.target_elo))

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        data["target_elo"] = self.target_elo
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PlayerConfig:
        data = json.loads(Path(path).read_text())
        data.pop("target_elo", None)
        style_data = data.pop("style", None)
        style = StyleVector(**style_data) if style_data else None
        return cls(style=style, **data)
