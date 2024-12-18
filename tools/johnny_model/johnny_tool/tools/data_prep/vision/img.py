#!/usr/bin/env python
import argparse
import os
from PIL import Image
import numpy as np

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--image", type=str, required=True, help = "Image file to convert to raw data"
)
parser.add_argument(
    "-o", "--output", type=str, default = "./", help = "Output directory"
)
parser.add_argument(
    "-f", "--format", nargs='+', choices = ("bin", "npy"), default = "npy", help = "Output format"
)
parser.add_argument(
    "-l", "--layout", type=str, choices = ("nhwc", "nchw"), default = "nchw", help = "Tensor layout of output data"
)
parser.add_argument(
    "-s", "--input-shape", type=int, required=True, nargs=2, help = "Image size"
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
# Process image
image = Image.open(args.image)
image_resized = image.resize((args.input_shape[0], args.input_shape[1]))
image_data = np.array(image_resized, dtype=np.uint8)
image_data = np.expand_dims(image_data, axis=0)
if args.layout == "nchw":
    image_data = np.transpose(image_data, (0, 3, 1, 2))

if "bin" in args.format:
    image_file = os.path.join(args.output, f"image.bin") 
    export_bin(image_data, image_file)
if "npy" in args.format:
    image_file = os.path.join(args.output, f"image.npy") 
    export_npy(image_data, image_file)