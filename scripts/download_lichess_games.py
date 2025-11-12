#!/usr/bin/env python3
import argparse
import io
import json
import multiprocessing as mp
import os
import time
import tempfile
import zstandard as zstd
import chess.pgn

def process_game_text(game_text, min_elo, max_elo):
    """Parse PGN text and return list of SAN moves if valid, else None."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if not game:
            return None
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
            return None
        board = game.board()
        moves_list = []
        for move in game.mainline_moves():
            try:
                moves_list.append(board.san(move))
                board.push(move)
            except Exception:
                continue
        if not moves_list:
            return None
        return moves_list
    except Exception:
        return None

def worker(input_queue, output_path, min_elo, max_elo, shared_counter, max_games):
    """Worker process: consumes PGN text, validates, writes JSONL."""
    with open(output_path, "w", encoding="utf-8") as out:
        while True:
            game_text = input_queue.get()
            if game_text is None:
                break

            # Stop if global limit reached
            with shared_counter.get_lock():
                if max_games and shared_counter.value >= max_games:
                    break

            moves = process_game_text(game_text, min_elo, max_elo)
            if moves:
                out.write(json.dumps({"moves": moves}) + "\n")
                out.flush()
                with shared_counter.get_lock():
                    shared_counter.value += 1
                    if shared_counter.value % 1000 == 0:
                        print(f"📦 Output {shared_counter.value} games written", flush=True)

                # Stop if reached global cap
                if max_games and shared_counter.value >= max_games:
                    break

def stream_zst_pgn_to_jsonl(zst_path, output_file, min_elo=2400, max_elo=2800, max_games=None, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    input_queue = mp.Queue(maxsize=5000)
    shared_counter = mp.Value("i", 0)

    # Temporary files for each worker
    temp_files = [tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") for _ in range(num_workers)]
    temp_paths = [tf.name for tf in temp_files]
    for tf in temp_files:
        tf.close()

    # Start workers
    pool = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(input_queue, temp_paths[i], min_elo, max_elo, shared_counter, max_games))
        p.start()
        pool.append(p)

    decompressed_bytes = 0
    processed_count = 0
    buffer = []
    in_game = False
    start_time = time.time()

    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                decompressed_bytes += len(line.encode("utf-8"))

                # Stop early if output limit reached
                with shared_counter.get_lock():
                    if max_games and shared_counter.value >= max_games:
                        break

                if line.startswith("[Event "):
                    if buffer:
                        input_queue.put("".join(buffer))
                        processed_count += 1
                        buffer = []
                    in_game = True

                if in_game:
                    buffer.append(line)

    if buffer:
        input_queue.put("".join(buffer))

    # Signal workers to exit
    for _ in range(num_workers):
        input_queue.put(None)

    for p in pool:
        p.join()

    # Merge worker outputs
    output_count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for path in temp_paths:
            with open(path, "r", encoding="utf-8") as tf:
                for line in tf:
                    if max_games and output_count >= max_games:
                        break
                    out.write(line)
                    output_count += 1
            os.remove(path)

    elapsed = time.time() - start_time
    print(f"✅ Finished. Processed {processed_count} PGNs, output {shared_counter.value} games, decompressed={decompressed_bytes / (1024*1024):.1f} MB, elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst", required=True, help="Input Lichess .pgn.zst file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--min-elo", type=int, default=2400)
    parser.add_argument("--max-elo", type=int, default=2800)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    stream_zst_pgn_to_jsonl(
        args.zst, args.out, args.min_elo, args.max_elo, args.max_games, args.num_workers
    )
