#!/usr/bin/env python
import argparse

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
parser.add_argument(
    "-o", "--output", type=str, default = "./modified.onnx", help = "Output file name"
)
parser.add_argument(
    "--nodes", nargs="+", help="List of nodes to remove"
)
parser.add_argument(
    "--layer", nargs="+", help="List of output layers to remove"
)
args = parser.parse_args()

##
# Modify model
import onnx
from onnx import helper, version_converter
from onnx import shape_inference

RM_NODE = ["yolonms_layer_1/non_max_suppression/NonMaxSuppressionV3", "Cast"]
RM_OUTPUT = ["yolonms_layer_1/concat_2:0"]

model = onnx.load(args.model)
graph = model.graph

target_node = [node for node in graph.node if node.name in args.nodes]
for node in target_node:
    graph.node.remove(node)

new_output = [o for o in graph.output if o.name not in args.layer]
graph.ClearField('output')
graph.output.extend(new_output)


onnx.save(model, args.output)