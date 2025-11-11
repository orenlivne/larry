#!/usr/bin/env python3
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
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
                continue  # skip illegal/bad moves
        if not moves_list:
            return None
        return moves_list
    except Exception:
        return None

def worker(input_queue, output_path, min_elo, max_elo):
    """Worker process: consumes PGN text, validates, writes JSONL."""
    count = 0
    skipped = 0
    with open(output_path, "w") as out:
        while True:
            game_text = input_queue.get()
            if game_text is None:
                break
            moves = process_game_text(game_text, min_elo, max_elo)
            if moves:
                out.write(json.dumps({"moves": moves}) + "\n")
                count += 1
            else:
                skipped += 1
    return (count, skipped)

def stream_zst_pgn_to_jsonl(zst_path, output_file, min_elo=2400, max_elo=2800, max_games=None, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    input_queue = mp.Queue(maxsize=5000)

    # Create temporary files for workers
    temp_files = [tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') for _ in range(num_workers)]
    temp_paths = [tf.name for tf in temp_files]
    for tf in temp_files:
        tf.close()

    # Start worker processes
    pool = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(input_queue, temp_paths[i], min_elo, max_elo))
        p.start()
        pool.append(p)

    count = 0
    skipped = 0
    decompressed_bytes = 0
    buffer = []
    in_game = False
    start_time = time.time()

    # Decompress .zst file using multiple threads
    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                decompressed_bytes += len(line.encode("utf-8"))

                # Detect start of a game
                if line.startswith("[Event "):
                    if buffer:
                        input_queue.put("".join(buffer))
                        count += 1
                        if max_games and count >= max_games:
                            break
                        buffer = []
                    in_game = True

                if in_game:
                    buffer.append(line)

                # Periodic progress
                if count % 10000 == 0 and count > 0:
                    elapsed = time.time() - start_time
                    print(f"📦 Processed ~{count} games, decompressed={decompressed_bytes / (1024*1024):.1f} MB, elapsed={elapsed:.1f}s, ~{elapsed/count:.6f}s/game")

            # enqueue last game
            if buffer and (not max_games or count < max_games):
                input_queue.put("".join(buffer))

    # Signal workers to exit
    for _ in range(num_workers):
        input_queue.put(None)

    # Wait for workers
    for p in pool:
        p.join()

    # Merge temporary files
    with open(output_file, "w", encoding="utf-8") as out:
        for path in temp_paths:
            with open(path, "r", encoding="utf-8") as tf:
                for line in tf:
                    out.write(line)
            os.remove(path)

    elapsed = time.time() - start_time
    print(f"✅ Finished. Decompressed {decompressed_bytes / (1024*1024):.1f} MB, elapsed={elapsed:.1f}s")
    print(f"📦 Filtered dataset saved to: {output_file}")

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
