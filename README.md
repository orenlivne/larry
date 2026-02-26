# LarryBot

Chess bot that models any player's style at any ELO level. Give it a PGN database of someone's games and a target ELO, and it plays like them.

Built for the specific use case of simulating Larry Christiansen in a simul (classical ELO minus 200 points), but works for any player.

## How It Works

- **Stockfish** with `UCI_LimitStrength` / `UCI_Elo` provides real, FIDE-calibrated playing strength (range 1320-3190)
- **Opening book** built from the player's games gives them their real repertoire
- **Style vector** (aggression, material preference, piece activity, centrality) biases move selection among close Stockfish candidates toward the player's tendencies

No neural network training required. No GPU required.

## Requirements

- Python 3.10+
- [Stockfish](https://stockfishchess.org/) (`brew install stockfish` on macOS)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A PGN file of the player's games

## Setup

```bash
git clone https://github.com/orenlivne/larry.git
cd larry
uv sync
```

Or with pip:

```bash
pip install -e .
```

Verify Stockfish is installed:

```bash
which stockfish
# Should print something like /opt/homebrew/bin/stockfish
```

## Step 1: Get Game Data

You need a PGN file containing the player's games. Sources:

### From ChessBase
1. Open ChessBase
2. Search for the player (e.g., "Christiansen, Larry")
3. Select the games you want (e.g., White games only for simul prep)
4. File > Export > PGN
5. Save as `data/larry_games.pgn`

### From Chess.com

Download the last 12 months of a player's games:

```bash
# Download all games from the last year (replace USERNAME)
for month in $(seq -f "%02g" 1 12); do
  curl -s "https://api.chess.com/pub/player/USERNAME/games/2025/${month}/pgn" >> data/player_games.pgn
done
```

Or a single month:
```bash
curl "https://api.chess.com/pub/player/USERNAME/games/2025/01/pgn" -o data/player_games.pgn
```

### From Lichess

Download the last year of a player's games:

```bash
# All games from the last 365 days
curl "https://lichess.org/api/games/user/USERNAME?since=$(date -v-1y +%s000)&pgnInJson=false" \
  -o data/player_games.pgn

# Or with a specific date range (millisecond timestamps)
curl "https://lichess.org/api/games/user/USERNAME?since=1704067200000&until=1735689600000&pgnInJson=false" \
  -o data/player_games.pgn
```

Put the PGN file anywhere accessible; you'll pass the path to the build script.

## Step 2: Build the Player Model

```bash
python scripts/build_player.py \
    --pgn data/larry_games.pgn \
    --player "Christiansen" \
    --classical-elo 2620 \
    --elo-offset -200 \
    --output players/larry_christiansen/
```

| Flag | Description |
|---|---|
| `--pgn` | Path to the PGN file |
| `--player` | Player name (case-insensitive substring match against PGN headers) |
| `--classical-elo` | The player's ELO rating, or `auto` to extract from PGN headers |
| `--elo-offset` | ELO adjustment (e.g., `-200` for simul, `0` for full strength) |
| `--stockfish` | Path to Stockfish binary (default: `/opt/homebrew/bin/stockfish`) |
| `--book-depth` | How many half-moves deep to build the opening book (default: 20) |
| `--output` | Directory to save the player model |

This produces:
```
players/larry_christiansen/
  config.json        # Player config (ELO, style vector, paths)
  opening_book.json  # Opening repertoire from games
```

### More Examples

Auto-detect ELO from PGN headers (chess.com and Lichess exports include ratings):
```bash
python scripts/build_player.py \
    --pgn data/friend_games.pgn \
    --player "friend_username" \
    --classical-elo auto \
    --output players/my_friend/
```

Model a chess.com friend at a specific rating:
```bash
python scripts/build_player.py \
    --pgn data/friend_games.pgn \
    --player "friend_username" \
    --classical-elo 1800 \
    --output players/my_friend/
```

Model a GM at rapid strength (-100 ELO):
```bash
python scripts/build_player.py \
    --pgn data/nakamura_games.pgn \
    --player "Nakamura" \
    --classical-elo 2785 \
    --elo-offset -100 \
    --output players/nakamura_rapid/
```

## Step 3: Play Locally

Test the bot against Stockfish:

```bash
python scripts/play_local.py \
    --player-dir players/larry_christiansen/ \
    --opponent-elo 2000 \
    --num-games 3
```

| Flag | Description |
|---|---|
| `--player-dir` | Player model directory (from step 2) |
| `--opponent-elo` | Stockfish opponent ELO (1320-3190) |
| `--num-games` | Number of games to play (alternates White/Black) |
| `--stockfish` | Override Stockfish path |

The bot alternates colors each game and prints full move lists with results.

## Step 4: Deploy to Lichess

### Get a Lichess BOT token
1. Create or use a Lichess account dedicated to the bot
2. Upgrade the account to BOT: `https://lichess.org/api/bot/account/upgrade` (irreversible)
3. Generate an API token at `https://lichess.org/account/oauth/token` with the **Play games with the bot API** scope
4. Save the token to a file:
   ```bash
   echo "lip_your_token_here" > data/larrybot/token.txt
   ```

### Run the bot
```bash
python scripts/run_lichess.py \
    --player-dir players/larry_christiansen/ \
    --token-file data/larrybot/token.txt
```

The bot will:
- Connect to Lichess and print the account name
- Accept all incoming challenges
- Play each game using the player model at the configured ELO
- Print moves as they're played

Stop with `Ctrl+C`.

## ELO Calibration

Playing strength comes directly from Stockfish's `UCI_Elo` parameter, which is calibrated against CCRL/FIDE ratings:

| Target | `classical_elo` | `elo_offset` | Effective |
|---|---|---|---|
| Larry in simul | 2620 | -200 | 2420 |
| Larry full strength | 2620 | 0 | 2620 |
| Club player | 1600 | 0 | 1600 |
| Beginner | 1320 | 0 | 1320 |

The valid range is 1320-3190. Values outside this range are clamped automatically.

## Running Tests

```bash
pytest tests/ -v
```

Tests require Stockfish to be installed. If Stockfish is not found, engine-dependent tests are skipped automatically.

## Project Structure

```
src/larrybot/
  config.py           # PlayerConfig + StyleVector dataclasses
  pgn_utils.py        # PGN parsing (ChessBase, chess.com, Lichess formats)
  engine.py           # Stockfish wrapper with UCI_Elo control
  opening_book.py     # Personal opening book from PGN games
  style.py            # Style extraction + move scoring
  bot.py              # Main bot: engine + book + style
  lichess_client.py   # Lichess BOT API integration

scripts/
  build_player.py     # Build player model from PGN
  play_local.py       # Play locally vs Stockfish
  run_lichess.py      # Run on Lichess

tests/                # 54 tests covering all components
```
