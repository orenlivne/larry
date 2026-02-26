"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

SAMPLE_PGN = str(Path(__file__).parent / "data" / "sample.pgn")
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def stockfish_available() -> bool:
    return shutil.which(STOCKFISH_PATH) is not None or shutil.which("stockfish") is not None


def get_stockfish_path() -> str:
    if shutil.which(STOCKFISH_PATH):
        return STOCKFISH_PATH
    found = shutil.which("stockfish")
    if found:
        return found
    raise FileNotFoundError("Stockfish not found")


requires_stockfish = pytest.mark.skipif(
    not stockfish_available(),
    reason="Stockfish not installed",
)
