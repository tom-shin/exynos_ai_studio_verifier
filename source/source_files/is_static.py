import onnx


def is_static_input(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Extract the model's input information
    input_info = model.graph.input

    # Iterate over each input tensor
    for input_tensor in input_info:
        if input_tensor.type.tensor_type.HasField("shape"):
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_param:  # Dynamic dimensions
                    return False
                elif not dim.dim_value:  # Undefined dimensions
                    return False
    return True


# Example usage
model_path = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\test_model_repo\ml_group\ml_group_all_checked_simplify\mobilenetv2-12-modify.onnx"
if is_static_input(model_path):
    print("The model has static input shapes.")
else:
    print("The model has dynamic input shapes.")
