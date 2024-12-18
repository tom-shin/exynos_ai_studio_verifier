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
import tensorflow as tf
import numpy as np
from PIL import Image

def img_2_uint(image):
    return np.array(image, dtype=np.uint8)

def img_2_fp(image, scale, offest):
    data = np.array(image, dtype=np.float32)
    return (data / scale) - offest

def export_bin(data, dir):
    with open(dir, "wb") as file:
        file.write(data.tobytes())

def export_npy(data, dir):
    with open(dir, "wb") as file:
        np.save(file, data)

interpreter = tf.lite.Interpreter(model_path=args.model)

# Data preparation
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_name = input_details[0]["name"]
input_dtype = input_details[0]["dtype"]

image = Image.open(args.image)
image_resized = image.resize((input_shape[1], input_shape[2]))
if input_dtype == np.float32:
    if args.scale is None or args.offset is None:
        raise ValueError("scale and offset must not be None for float32 input model")
    input_data = img_2_fp(image_resized, args.scale, args.offset)
elif args.datatype == np.uint8:
    input_data = img_2_uint(image_resized)
input_data = np.expand_dims(input_data, axis=0)

# Execute model
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

for idx, output_detail in enumerate(output_details):
    output_data = interpreter.get_tensor(output_detail['index'])
    if "bin" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.bin")
        export_bin(output_data, output_file)
    if "npy" in args.format:
        output_file = os.path.join(args.output, f"output{idx}.npy")
        export_npy(output_data, output_file)
