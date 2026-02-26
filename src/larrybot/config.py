"""Player configuration: name, ELO, style, paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


MIN_UCI_ELO = 1320
MAX_UCI_ELO = 3190


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
