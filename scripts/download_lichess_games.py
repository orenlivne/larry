#!/usr/bin/env python3
import zstandard as zstd
import chess.pgn
import json
import io
import argparse
import multiprocessing as mp
import time
from collections import deque

def parse_game(pgn_text):
    """Parse PGN and return JSON dict with moves as list of SAN strings."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))

        board = game.board()
        moves = []
        for move in game.mainline_moves():
            try:
                moves.append(board.san(move))
                board.push(move)
            except Exception:
                continue
        if not moves:
            return None

        return {
            "WhiteElo": white_elo,
            "BlackElo": black_elo,
            "moves": moves,
        }
    except Exception:
        return None

def worker(input_queue, output_queue, min_elo, max_elo):
    """Worker process to parse PGN text into JSON."""
    while True:
        pgn_text = input_queue.get()
        if pgn_text is None:  # Sentinel — exit
            break
        game_data = parse_game(pgn_text)
        if not game_data:
            continue
        if min(game_data["WhiteElo"], game_data["BlackElo"]) < min_elo:
            continue
        if max(game_data["WhiteElo"], game_data["BlackElo"]) > max_elo:
            continue
        output_queue.put({"moves": game_data["moves"]})

def stream_zst_pgn_to_jsonl(zst_path, out_path, min_elo, max_elo, max_games, num_workers=8):
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    input_queue = mp.Queue(maxsize=5000)
    output_queue = mp.Queue()
    counter = mp.Value("i", 0)

    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(input_queue, output_queue, min_elo, max_elo))
        p.start()
        workers.append(p)

    start_time = time.time()
    last_time = start_time
    last_count = 0
    recent_rates = deque(maxlen=5)

    def print_status():
        nonlocal last_time, last_count
        now = time.time()
        elapsed = now - start_time
        count = counter.value
        rate = (count - last_count) / (now - last_time + 1e-9)
        recent_rates.append(rate)
        avg_rate = sum(recent_rates) / len(recent_rates)
        eta_min = (max_games - count) / avg_rate / 60 if avg_rate > 0 else 0
        print(f"📦 Output {count:,} games written ({avg_rate:.1f}/s, ETA {eta_min:.1f} min)", flush=True)
        last_time, last_count = now, count

    with open(out_path, "w", encoding="utf-8") as f:
        with open(zst_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                buffer = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                pgn_text = ""
                for line in buffer:
                    if line.startswith("[Event "):
                        if pgn_text.strip():
                            input_queue.put(pgn_text)
                            pgn_text = ""
                    pgn_text += line

                    # Drain output queue
                    while not output_queue.empty():
                        game = output_queue.get()
                        f.write(json.dumps(game) + "\n")
                        f.flush()
                        with counter.get_lock():
                            counter.value += 1
                            if counter.value % 10000 == 0:
                                print_status()
                            if counter.value >= max_games:
                                # Stop feeding workers
                                for _ in workers:
                                    input_queue.put(None)
                                # Drain remaining output
                                while not output_queue.empty():
                                    game = output_queue.get()
                                    f.write(json.dumps(game) + "\n")
                                    f.flush()
                                    with counter.get_lock():
                                        counter.value += 1
                                print_status()
                                for w in workers:
                                    w.join(timeout=5)
                                return

        # Last game
        if pgn_text.strip():
            input_queue.put(pgn_text)
        # Final drain
        for _ in workers:
            input_queue.put(None)
        while not output_queue.empty():
            game = output_queue.get()
            f.write(json.dumps(game) + "\n")
            f.flush()
            with counter.get_lock():
                counter.value += 1
        print_status()
        for w in workers:
            w.join(timeout=5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-elo", type=int, default=2400)
    parser.add_argument("--max-elo", type=int, default=2800)
    parser.add_argument("--max-games", type=int, default=200000)
    args = parser.parse_args()

    stream_zst_pgn_to_jsonl(
        zst_path=args.zst,
        out_path=args.out,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        max_games=args.max_games,
    )

if __name__ == "__main__":
    main()
