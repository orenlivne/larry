#!/usr/bin/env python3
"""
Sample N lines from a JSONL file.
Usage:
  python sample_jsonl.py --input input.jsonl --output sampled.jsonl --samples 1000000
"""

import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()

    # Count total lines first
    print(f"Reading {args.input} ...")
    with open(args.input, "r") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total lines: {total}, sampling {args.samples} lines ...")
    if args.samples >= total:
        sampled_lines = lines
    else:
        sampled_lines = random.sample(lines, args.samples)

    with open(args.output, "w") as f:
        f.writelines(sampled_lines)

    print(f"✅ Sampled {len(sampled_lines)} lines → {args.output}")

if __name__ == "__main__":
    main()
