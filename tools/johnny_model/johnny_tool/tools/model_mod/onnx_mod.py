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
    "--op", type=int, help = "Change opset"
)
parser.add_argument(
    "--debug", action='store_true'
)
args = parser.parse_args()

##
# Modify model
import onnx
from onnx import helper, version_converter
from onnx import shape_inference
import onnxruntime as ort

model = onnx.load(args.model)
graph = model.graph
session = ort.InferenceSession(args.model)

if args.debug:
    print("===Before===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)


# Remove Dropout Layers
dropout_nodes = [node for node in graph.node if node.op_type == "Dropout"]
if len(dropout_nodes):
    print("Removing `Dropout` nodes")
    for dropout_node in dropout_nodes:
        dropout_input = dropout_node.input[0]
        dropout_output = dropout_node.output[0]

        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == dropout_output:
                    node.input[i] = dropout_input
        graph.node.remove(dropout_node)


# Remove Softmax Layers
softmax_nodes = [node for node in graph.node if node.op_type == "Softmax"]
if len(softmax_nodes):
    print("Removing `Softmax` nodes")
    for softmax_node in softmax_nodes:
        softmax_input = softmax_node.input[0]
        softmax_output = softmax_node.output[0]

        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == softmax_output:
                    node.input[i] = softmax_input

        for output in graph.output:
            if output.name == softmax_output:
                output.name = softmax_input

        graph.node.remove(softmax_node)


# Remove unnecessary model.graph.input
original_inputs = {input.name for input in graph.input}
used_inputs = set()
for node in graph.node:
    for input_name in node.input:
        used_inputs.add(input_name)
new_inputs = [input for input in graph.input if input.name in used_inputs]

if sorted([n for n in original_inputs]) != sorted([input.name for input in new_inputs]):
    print("Remove unnecessary model.graph.input")
    graph.ClearField('input')
    graph.input.extend(new_inputs)


# Change model.graph.input order
ort_input_names = [input.name for input in session.get_inputs()]
input_mapping = {input.name: input for input in graph.input}

reordered_inputs = [input_mapping[name] for name in ort_input_names if name in input_mapping]
reordered_inputs += [input for input in graph.input if input.name not in ort_input_names]

if [input.name for input in reordered_inputs] != [input.name for input in model.graph.input]:
    print("Input order has changed.")
    graph.ClearField('input')
    graph.input.extend(reordered_inputs)


# Change opset
if args.op:
    print("Change OP set")
    model = version_converter.convert_version(model, args.op)

# Save model
onnx.save(model, args.output)


if args.debug:
    print("===After===")
    for input in model.graph.input:
        print(input.name, input.type)
    for output in model.graph.output:
        print(output.name, output.type)
