import onnx
from onnx import helper, version_converter
from onnx import shape_inference

MODEL = "yolov3-12.onnx"
LAYER_DICT = {
    "input_1": [1,3,416,416],
    "image_shape":[1,2],
    "yolonms_layer_1/ExpandDims_1:0": [1, 10647, 4],
    "yolonms_layer_1/ExpandDims_3:0": [1, 80, 10647],
}
MODIFIED_MODEL = "./modified.onnx"

model = onnx.load(MODEL)
print("===Before===")
for input in model.graph.input:
    print(input.name, input.type)
for output in model.graph.output:
    print(output.name, output.type)


graph = model.graph

for input_tensor in graph.input:
    if input_tensor.name not in LAYER_DICT:
        continue
    new_dims = LAYER_DICT[input_tensor.name]
    for idx, d in enumerate(new_dims):
        input_tensor.type.tensor_type.shape.dim[idx].dim_value = d

for output_tensor in graph.output:
    if output_tensor.name not in LAYER_DICT:
        continue
    new_dims = LAYER_DICT[output_tensor.name]
    for idx, d in enumerate(new_dims):
        output_tensor.type.tensor_type.shape.dim[idx].dim_value = d


print("===After===")
for input in model.graph.input:
    print(input.name, input.type)
for output in model.graph.output:
    print(output.name, output.type)


onnx.save(model, MODIFIED_MODEL)