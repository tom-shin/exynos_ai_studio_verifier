#!/usr/bin/env python
import argparse
import os

##
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True, help = "Model to create data for"
)
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
    "--scale", type = float, help = "Scale for uint8 -> fp32 conversion, required when datatype is set to fp32"
)
parser.add_argument(
    "--offset", type = float, help = "Offset for uint8 -> fp32 conversion, required when datatype is set to fp32"
)
args = parser.parse_args()

##
# Create raw data
import onnxruntime
import numpy as np
from PIL import Image

def img_2_uint(image):
    return np.array(image, dtype=np.uint8)

def img_2_fp(image, scale, offest):
    data = np.array(image, dtype=np.float32)
    return (data / scale) - offest

def hwc_2_chw(data):
    return np.transpose(data, (0, 3, 1, 2))

def export_bin(data, dir):
    with open(dir, "wb") as file:
        file.write(data.tobytes())

def export_npy(data, dir):
    with open(dir, "wb") as file:
        np.save(file, data)

session = onnxruntime.InferenceSession(args.model)

# Data preparation
input_details = session.get_inputs()
output_details = session.get_outputs()

input_shape = input_details[0].shape
input_name = input_details[0].name
input_dtype = input_details[0].type

image = Image.open(args.image)
image_resized = image.resize((input_shape[2], input_shape[3]))
if input_dtype == "tensor(float)":
    if args.scale is None or args.offset is None:
        raise ValueError("scale and offset must not be None for float32 input model")
    input_data = img_2_fp(image_resized, args.scale, args.offset)
elif args.datatype == "tensor(uint8)":
    input_data = img_2_uint(image_resized)
input_data = np.expand_dims(input_data, axis=0)
input_data = hwc_2_chw(input_data)

# Execute model
output_names = [output.name for output in output_details]
output = session.run(output_names, {input_name: input_data})

for idx, o in enumerate(output):
    if "bin" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.bin")
        export_bin(o, output_file)
    if "npy" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.npy")
        export_npy(o, output_file)
