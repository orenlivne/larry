#!/usr/bin/env python3
import requests, io, zstandard as zstd
import chess.pgn, json, argparse, multiprocessing as mp
from pathlib import Path

def process_game(pgn_text):
    """Parse a single game, return JSON dict with moves (as UCI)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        return None
    moves = [move.uci() for move in game.mainline_moves()]
    if not moves:
        return None
    return {"moves": moves}

def worker(input_queue, tmp_file):
    """Worker: get PGN text from queue, write JSON lines."""
    with open(tmp_file, "w", encoding="utf-8") as f:
        while True:
            pgn = input_queue.get()
            if pgn is None:
                break
            game_json = process_game(pgn)
            if game_json:
                f.write(json.dumps(game_json) + "\n")

def stream_zst(url, out_path, max_games=200_000, num_workers=mp.cpu_count()):
    """Download, decompress, parse PGN in parallel, stop at max_games."""
    tmp_dir = Path(out_path).parent / "tmp_workers"
    tmp_dir.mkdir(exist_ok=True)
    input_queue = mp.Queue(maxsize=5000)
    workers = []

    for i in range(num_workers):
        tmp_file = tmp_dir / f"worker_{i}.jsonl"
        p = mp.Process(target=worker, args=(input_queue, tmp_file))
        p.start()
        workers.append((p, tmp_file))

    games_count = 0
    resp = requests.get(url, stream=True)
    dctx = zstd.ZstdDecompressor()
    reader = io.TextIOWrapper(dctx.stream_reader(resp.raw), encoding="utf-8", errors="ignore")

    buffer = ""
    for line in reader:
        if line.startswith("[Event "):
            if buffer:
                input_queue.put(buffer)
                games_count += 1
                if games_count >= max_games:
                    break
                buffer = ""
        buffer += line
    if buffer and games_count < max_games:
        input_queue.put(buffer)
        games_count += 1

    # Stop workers
    for _ in workers:
        input_queue.put(None)
    for p, _ in workers:
        p.join()

    # Merge per-worker files
    with open(out_path, "w", encoding="utf-8") as f_out:
        for _, tmp_file in workers:
            with open(tmp_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
            tmp_file.unlink()
    tmp_dir.rmdir()
    print(f"✅ Done! {games_count} games written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-games", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    stream_zst(args.url, args.out, max_games=args.max_games, num_workers=args.workers)