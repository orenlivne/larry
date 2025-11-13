#!/usr/bin/env python3
import tensorflow as tf, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

print(f"🔄 Converting {args.input} → {args.output}")

# Try to load the pb as SavedModel or GraphDef
try:
    model = tf.keras.models.load_model(args.input)
except Exception:
    with tf.io.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.io.write_graph(graph_def, ".", args.output, as_text=False)

print("✅ Conversion complete")
