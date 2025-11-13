import chess.pgn, os, tqdm

in_pgn = "data/raw_lichess_games.pgn"
out_txt = "data/larry_dataset.txt"

print("🧩 Converting PGN → Maia dataset format...")
with open(in_pgn) as f_in, open(out_txt, "w") as f_out:
    for game in tqdm.tqdm(iter(lambda: chess.pgn.read_game(f_in), None)):
        rating = int(game.headers.get("WhiteElo", 0))
        if 2400 <= rating <= 2800:
            f_out.write(str(game) + "\n")

print(f"✅ Wrote {out_txt}")
