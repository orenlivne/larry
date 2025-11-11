#!/usr/bin/env python3
"""
Inspect a TensorFlow .pb file and list all operation names.
Helps you determine input/output tensors for conversion to H5.
"""

import tensorflow as tf
import argparse

def list_operations(pb_file, max_ops=None):
    print(f"🔍 Loading PB file: {pb_file}")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    print("\n📝 Operations in graph:")
    nodes = graph_def.node
    if max_ops:
        nodes = nodes[:max_ops]
    for i, op in enumerate(nodes):
        print(f"{i+1:04d}: {op.name} [{op.op}]")
    print(f"\nTotal operations: {len(graph_def.node)}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a TensorFlow .pb file")
    parser.add_argument("--pb", required=True, help="Path to the .pb file")
    parser.add_argument("--max", type=int, default=50, help="Max operations to print initially")
    args = parser.parse_args()

    list_operations(args.pb, args.max)

if __name__ == "__main__":
    main()
