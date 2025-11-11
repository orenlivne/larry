#!/usr/bin/env python3
import argparse
import io
import json
import time
import zstandard as zstd
import chess.pgn

def process_game_text(game_text, min_elo, max_elo):
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if not game:
            return None
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
            return None

        # Robust move extraction
        board = game.board()
        moves_list = []
        for move in game.mainline_moves():
            try:
                moves_list.append(board.san(move))
                board.push(move)
            except Exception:
                continue  # Skip illegal/bad moves

        if not moves_list:
            return None
        return moves_list
    except Exception:
        return None

def stream_zst_pgn_to_jsonl(zst_path, output_file, min_elo=2400, max_elo=2800, max_games=None):
    count = 0
    skipped = 0
    decompressed_bytes = 0
    last_printed_count = 0
    start_time = time.time()

    buffer = []
    in_game = False

    with open(zst_path, "rb") as fh, open(output_file, "w") as out:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                decompressed_bytes += len(line.encode("utf-8"))

                if line.startswith("[Event "):
                    if buffer:
                        moves = process_game_text("".join(buffer), min_elo, max_elo)
                        if moves:
                            out.write(json.dumps({"moves": moves}) + "\n")
                            count += 1
                            if max_games and count >= max_games:
                                elapsed = time.time() - start_time
                                print(f"✅ Reached max_games={max_games}, stopping.")
                                print(f"Elapsed time: {elapsed:.1f}s, {elapsed/count:.3f}s per game")
                                return
                        else:
                            skipped += 1
                        buffer = []
                    in_game = True

                if in_game:
                    buffer.append(line)

                # Print progress only once per 1000 processed games
                if count // 100 > last_printed_count:
                    last_printed_count = count // 100
                    elapsed = time.time() - start_time
                    per_game = elapsed / count if count else 0
                    print(f"📦 Processed {count} games, skipped {skipped}, "
                          f"decompressed={decompressed_bytes / (1024*1024):.1f} MB, "
                          f"elapsed={elapsed:.1f}s, {per_game:.3f}s/game")

            # Process last game
            if buffer:
                moves = process_game_text("".join(buffer), min_elo, max_elo)
                if moves:
                    out.write(json.dumps({"moves": moves}) + "\n")
                    count += 1
                else:
                    skipped += 1

    elapsed = time.time() - start_time
    per_game = elapsed / count if count else 0
    print(f"✅ Finished. Processed {count} games, skipped {skipped}, "
          f"decompressed={decompressed_bytes / (1024*1024):.1f} MB, "
          f"elapsed={elapsed:.1f}s, {per_game:.3f}s/game")
    print(f"📦 Filtered dataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst", required=True, help="Input Lichess .pgn.zst file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--min-elo", type=int, default=2400)
    parser.add_argument("--max-elo", type=int, default=2800)
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    stream_zst_pgn_to_jsonl(args.zst, args.out, args.min_elo, args.max_elo, args.max_games)
