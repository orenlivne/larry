#!/usr/bin/env python3
"""Play local games: PlayerBot vs Stockfish at a given opponent ELO.

Usage:
    python scripts/play_local.py \
        --player-dir players/larry_christiansen/ \
        --opponent-elo 2000 \
        --num-games 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chess

from larrybot.bot import PlayerBot
from larrybot.engine import StockfishEngine


def play_game(bot: PlayerBot, opponent: StockfishEngine, bot_color: chess.Color) -> str:
    """Play one game, return result from bot's perspective."""
    board = chess.Board()
    print(f"\nNew game: Bot plays {'White' if bot_color == chess.WHITE else 'Black'}")
    print(f"Bot ELO: {bot.config.clamped_elo}, Opponent ELO: {opponent.elo}")
    print("-" * 40)

    while not board.is_game_over():
        if board.turn == bot_color:
            move = bot.select_move(board)
            label = "Bot"
        else:
            move = opponent.play_move(board, time_limit=0.1)
            label = "Opp"

        san = board.san(move)
        move_num = board.fullmove_number
        if board.turn == chess.WHITE:
            print(f"{move_num}. {san}", end=" ")
        else:
            print(san)
        board.push(move)

    print(f"\nResult: {board.result()}")
    outcome = board.outcome()
    if outcome is None:
        return "1/2-1/2"
    if outcome.winner == bot_color:
        return "1-0"
    elif outcome.winner is None:
        return "1/2-1/2"
    else:
        return "0-1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Play local games vs Stockfish")
    ap.add_argument("--player-dir", required=True, help="Player model directory")
    ap.add_argument("--opponent-elo", type=int, default=2000, help="Opponent Stockfish ELO")
    ap.add_argument("--num-games", type=int, default=1, help="Number of games to play")
    ap.add_argument("--stockfish", help="Override Stockfish path")
    args = ap.parse_args()

    bot = PlayerBot.from_player_dir(args.player_dir)
    sf_path = args.stockfish or bot.config.stockfish_path
    opponent = StockfishEngine(sf_path, elo=args.opponent_elo)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    try:
        for i in range(args.num_games):
            bot_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            result = play_game(bot, opponent, bot_color)
            results[result] = results.get(result, 0) + 1
    finally:
        bot.close()
        opponent.close()

    print(f"\n{'='*40}")
    print(f"Results after {args.num_games} game(s):")
    print(f"  Wins:   {results['1-0']}")
    print(f"  Losses: {results['0-1']}")
    print(f"  Draws:  {results['1/2-1/2']}")


if __name__ == "__main__":
    main()
