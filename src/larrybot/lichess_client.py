"""Lichess bot integration using the berserk library."""

from __future__ import annotations

import sys
from pathlib import Path

import chess

try:
    import berserk
except ImportError:
    berserk = None  # type: ignore[assignment]

from .bot import PlayerBot


class LichessClient:
    """Connects to Lichess as a BOT account, accepts challenges, and plays."""

    def __init__(self, token_path: str, bot: PlayerBot) -> None:
        if berserk is None:
            raise RuntimeError(
                "berserk is not installed. Run: pip install berserk"
            )
        token = Path(token_path).read_text().strip()
        session = berserk.TokenSession(token)
        self._client = berserk.Client(session=session)
        self._bot = bot
        self._username = self._client.account.get()["username"].lower()

    def run(self) -> None:
        """Main event loop: accept challenges and play games."""
        print(f"Connected to Lichess as {self._username}. Waiting for challenges...")
        for event in self._client.bots.stream_incoming_events():
            etype = event.get("type")
            if etype == "challenge":
                cid = event["challenge"]["id"]
                print(f"Accepting challenge {cid}")
                self._client.bots.accept_challenge(cid)
            elif etype == "gameStart":
                game_id = event["game"]["id"]
                print(f"Game started: {game_id}")
                self._handle_game(game_id)

    def _handle_game(self, game_id: str) -> None:
        my_color: chess.Color | None = None

        for event in self._client.bots.stream_game_state(game_id):
            etype = event.get("type")

            if etype == "gameFull":
                white_id = event["white"].get("id", "").lower()
                black_id = event["black"].get("id", "").lower()
                if white_id == self._username:
                    my_color = chess.WHITE
                elif black_id == self._username:
                    my_color = chess.BLACK
                else:
                    print(f"Cannot determine color in game {game_id}", file=sys.stderr)
                    return
                moves_str = event.get("state", {}).get("moves", "")
            elif etype == "gameState":
                moves_str = event.get("moves", "")
            else:
                continue

            board = chess.Board()
            if moves_str:
                for uci in moves_str.split():
                    board.push_uci(uci)

            if board.is_game_over():
                print(f"Game {game_id} over: {board.result()}")
                return

            if my_color is None or board.turn != my_color:
                continue

            move = self._bot.select_move(board)
            self._client.bots.make_move(game_id, move.uci())
            color_name = "White" if my_color == chess.WHITE else "Black"
            print(f"[{game_id}] {color_name} played {move.uci()}")
