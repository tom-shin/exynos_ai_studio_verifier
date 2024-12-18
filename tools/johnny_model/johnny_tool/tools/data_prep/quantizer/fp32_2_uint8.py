#!/usr/bin/env python
import argparse
import os

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, required=True, help = "Input file path"
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help = "Output file name"
)
parser.add_argument(
    "-r", "--range", type=float, nargs=2, default=[0.0, 1.0], help="Range of input data [min, max]"
)
parser.add_argument(
    "-f", "--format", nargs='+', choices = ("bin", "npy"), default = "npy", help = "Output format"
)
args = parser.parse_args()

##
# Data Export Functions
def export_bin(data, dir):
    with open(dir, "wb") as file:
        file.write(data.tobytes())

def export_npy(data, dir):
    with open(dir, "wb") as file:
        np.save(file, data)

##
# Process
import numpy as np

fp_data = np.load(args.input)
fp_data = (fp_data - args.range[0]) / (args.range[1] - args.range[0]) * 255
uint_data = np.array(fp_data, dtype=np.uint8)

if "bin" in args.format:
    export_bin(uint_data, args.output)
if "npy" in args.format:
    export_npy(uint_data, args.output)