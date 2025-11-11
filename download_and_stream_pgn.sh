uv run python scripts/download_lichess_games.py \
    --zst-file data/lichess_raw/lichess_db_standard_rated_2025-01.pgn.zst \
    --out data/lichess_2025.pgn \
    --min-elo 2400 \
    --max-elo 2800 \
    --max-games 10000
