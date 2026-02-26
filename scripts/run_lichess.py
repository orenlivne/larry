#!/usr/bin/env python3
"""Run the bot on Lichess.

Usage:
    python scripts/run_lichess.py \
        --player-dir players/larry_christiansen/ \
        --token-file data/larrybot/token.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from larrybot.bot import PlayerBot
from larrybot.lichess_client import LichessClient


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chess bot on Lichess")
    ap.add_argument("--player-dir", required=True, help="Player model directory")
    ap.add_argument("--token-file", required=True, help="Lichess bot API token file")
    args = ap.parse_args()

    bot = PlayerBot.from_player_dir(args.player_dir)
    print(f"Player: {bot.config.player_name}")
    print(f"Target ELO: {bot.config.target_elo} (clamped: {bot.config.clamped_elo})")

    try:
        client = LichessClient(args.token_file, bot)
        client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bot.close()


if __name__ == "__main__":
    main()
