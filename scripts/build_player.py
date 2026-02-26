#!/usr/bin/env python3
"""Build a player model from a PGN database.

Usage:
    python scripts/build_player.py \
        --pgn data/larry_games.pgn \
        --player "Christiansen" \
        --classical-elo 2620 \
        --elo-offset -200 \
        --output players/larry_christiansen/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from larrybot.config import PlayerConfig, StyleVector
from larrybot.opening_book import OpeningBook
from larrybot.pgn_utils import count_games
from larrybot.style import StyleAnalyzer


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a player model from PGN games")
    ap.add_argument("--pgn", required=True, help="Path to PGN file")
    ap.add_argument("--player", required=True, help="Player name (substring match)")
    ap.add_argument("--classical-elo", type=int, required=True, help="Player's classical ELO")
    ap.add_argument("--elo-offset", type=int, default=0, help="ELO offset (e.g. -200 for simul)")
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish", help="Stockfish path")
    ap.add_argument("--book-depth", type=int, default=20, help="Opening book depth (half-moves)")
    ap.add_argument("--output", required=True, help="Output directory for player model")
    args = ap.parse_args()

    pgn_path = args.pgn
    if not Path(pgn_path).exists():
        print(f"Error: PGN file not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_games = count_games(pgn_path)
    print(f"PGN contains {total_games} games")

    # Build opening book
    print(f"Building opening book for '{args.player}' (depth={args.book_depth})...")
    book = OpeningBook.from_pgn(pgn_path, args.player, max_depth=args.book_depth)
    book_path = out_dir / "opening_book.json"
    book.save(book_path)
    print(f"Opening book: {book.position_count} positions saved to {book_path}")

    # Extract style
    print(f"Analyzing play style for '{args.player}'...")
    style = StyleAnalyzer.from_pgn(pgn_path, args.player)
    print(f"Style: aggression={style.aggression:.2f}, material={style.material_weight:.2f}, "
          f"activity={style.activity_weight:.2f}, centrality={style.centrality:.2f}")

    # Save config
    config = PlayerConfig(
        player_name=args.player,
        classical_elo=args.classical_elo,
        elo_offset=args.elo_offset,
        stockfish_path=args.stockfish,
        book_path=str(book_path),
        style=style,
        book_depth=args.book_depth,
    )
    config_path = out_dir / "config.json"
    config.save(config_path)
    print(f"Config saved to {config_path}")
    print(f"Target ELO: {config.target_elo} (classical {args.classical_elo} + offset {args.elo_offset})")
    print(f"Clamped ELO (Stockfish): {config.clamped_elo}")
    print("Done!")


if __name__ == "__main__":
    main()
