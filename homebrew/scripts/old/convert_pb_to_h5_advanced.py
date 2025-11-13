#!/usr/bin/env python3
"""
Convert a frozen .pb inference graph into a Keras Model, if possible.

Usage:
  python scripts/convert_pb_to_h5_advanced.py --input path/to/model.pb --output path/to/model.h5 \
       --input-tensor “input_tensor_name:0” --output-tensor “output_tensor_name:0”
"""

import argparse
import tensorflow as tf
import os

def load_frozen_graph(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def wrap_graph_as_keras(graph_def, input_tensor_name, output_tensor_name):
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

def convert(pb_path, h5_path, input_name, output_name):
    print(f"Loading graph from {pb_path}")
    graph_def = load_frozen_graph(pb_path)
    print("Wrapping graph into Keras model...")
    model = wrap_graph_as_keras(graph_def, input_name, output_name)
    print(f"Saving to {h5_path}")
    model.save(h5_path)
    print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-tensor", required=True)
    parser.add_argument("--output-tensor", required=True)
    args = parser.parse_args()

    convert(args.input, args.output, args.input_tensor, args.output_tensor)

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
