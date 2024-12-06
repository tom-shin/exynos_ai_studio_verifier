from onnxsim import simplify
import onnx
import os
import traceback
from onnx import helper


class ONNX_INFO:
    def __init__(self):
        self.model = None
        self.model_path = None

    def load_onnx_model(self, model_path):
        self.model = onnx.load(model_path)
        self.model_path = model_path

    def is_static_input(self, model_path=None):
        if model_path is not None:
            self.model = model_path
            self.model_path = model_path

        # Extract the model's input information
        input_info = self.model.graph.input

        # Iterate over each input tensor
        for input_tensor in input_info:
            if input_tensor.type.tensor_type.HasField("shape"):
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_param:  # Dynamic dimensions
                        return False
                    elif not dim.dim_value:  # Undefined dimensions
                        return False
        return True

    def print_input_shapes(self):
        print("[Model Input Shape]")

        # Extract the model's input information
        input_info = self.model.graph.input
        for input_tensor in input_info:
            name = input_tensor.name
            shape = []
            if input_tensor.type.tensor_type.HasField("shape"):
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_param:  # Dynamic dimensions
                        shape.append(dim.dim_param)
                    elif dim.dim_value:  # Static dimensions
                        shape.append(dim.dim_value)
                    else:
                        shape.append("?")  # Undefined dimension

            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_tensor.type.tensor_type.elem_type]
            print(f"Input Name: {name}")
            print(f"Shape: {shape}")
            print(f"Data Type: {dtype}")
            print("-" * 30)

    def print_model_version_info(self):
        print("\n[Model Version && Opset Version]")

        # Model IR version (ONNX version)
        ir_version = self.model.ir_version

        # Opset version(s)
        opset_imports = self.model.opset_import

        print(f"ONNX IR Version: {ir_version}")  # Model format version
        for opset in opset_imports:
            print(f"Opset Domain: {opset.domain or 'ai.onnx'}, Opset Version: {opset.version}")  # Operator set version

    def remove_op(self, op_name):
        print(f"\n[Remove {op_name}]")

        # Dropout 연산 제거
        nodes = self.model.graph.node
        new_nodes = []

        for node in nodes:
            if node.op_type == op_name:  # "Dropout":
                input_name = node.input[0]
                output_name = node.output[0]

                # 제거할 노드의 출력을 사용하는 다른 노드 업데이트
                for next_node in nodes:
                    next_node.input[:] = [
                        input_name if inp == output_name else inp for inp in next_node.input
                    ]
            else:
                # 제거할 노드가 아닌 노드는 그대로 유지
                new_nodes.append(node)

        # 수정된 노드로 모델 그래프 갱신
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)

        # 수정된 모델 저장
        dir_path, filename = os.path.split(self.model)
        name, ext = os.path.splitext(filename)

        # new_name = rf"{name}_modified{ext}"
        new_name = rf"{name}-modified{ext}"
        onnx.save(self.model, os.path.join(dir_path, new_name))

    def execute_onnx_simplifier(self, model_path=None, save=False):
        print("\n[Execute ONNX Simplifier]")

        if model_path is not None:
            self.model = model_path
            self.model_path = model_path

        try:
            # Simplify the model
            model_simplified, check = simplify(self.model)

            # Verify the simplification worked correctly
            if check:
                print(f"{os.path.basename(self.model_path)}: Simplified model is valid!")
                # Save the simplified model
                if save:
                    dir_path, filename = os.path.split(self.model_path)
                    root, ext = os.path.splitext(filename)
                    new_name = rf"{root}-modified{ext}"
                    onnx.save(model_simplified, os.path.join(dir_path, new_name))
            else:
                print(f"{os.path.basename(self.model_path)}: Simplification invalid, original model retained.")

        except Exception as e:
            print(f"{os.path.basename(self.model_path)}: Error")
            # Print exception details
            print(traceback.format_exc())

    def modify_input_tensor(self, model_path=None, input_tensor_shape=None):
        print("\n[Modify Input Tensor]")

        if model_path is not None:
            self.model = model_path
            self.model_path = model_path

        # 입력 텐서의 이름 및 정보를 추출
        # input_shapes = {}
        # for input_tensor in self.model.graph.input:
        #     tensor_name = input_tensor.name
        #     # 예: 모든 입력 텐서를 (1, 3, 128, 128)으로 설정
        #     input_shapes[tensor_name] = input_tensor_shape
        #     print(f"Input Name: {tensor_name}, Shape Set to: {input_shapes[tensor_name]}")

        # 모델 간소화
        model_simplified, check = simplify(self.model, overwrite_input_shapes=input_tensor_shape,
                                           dynamic_input_shape=False)
        # model_simplified, check = simplify(self.model, overwrite_input_shapes=input_shapes, dynamic_input_shape=False)

        # 검증 및 저장
        if check:
            print("Simplified model is valid!")
            dir_path, filename = os.path.split(self.model_path)
            root, ext = os.path.splitext(filename)
            new_name = rf"{root}_modify{ext}"
            onnx.save(model_simplified, os.path.join(dir_path, new_name))
            print(f"Simplified model saved to {os.path.join(dir_path, new_name)}")

        else:
            print("Simplification failed. The original model is retained.")


if __name__ == "__main__":
    model_instance = ONNX_INFO()
    #
    path = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\test_model_repo\ml_group\ml_group_all_checked_simplify\mobilenetv2-12-modify.onnx"
    model_instance.load_onnx_model(model_path=path)


    ret = model_instance.is_static_input()
    if ret:
        print("The model has static input shapes.")
    else:
        print("The model has dynamic input shapes.")



    # model_instance.print_input_shapes()

    # python -m onnxsim mobilenetv2-10.onnx mobilenetv2-10_checked.onnx --overwrite-input-shape 1,3,224,224
    # shape = {"input": [1, 3, 416, 416], "image_shape": [416, 416]}   #yolo3
    shape = {"input": [1, 3, 224, 224]}  # yolo4 보통 NCHW방식인데 이 모델은 보니 NHWC 방식으로 채널이 뒤에 있음. 이경우 보통 모델에 TRANSPOSE가 있어서
                                             # 내부적으로 다시 NCHW방식으로 변경하는 경우가 많음.

    model_instance.modify_input_tensor(input_tensor_shape=shape)

    #
    # model_instance.print_model_version_info()
    # model_instance.print_input_shapes()
    # model_instance.execute_onnx_simplifier()

    # directory = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\test_model_repo\ml_group\ml_group_all_checked_simplify"
    # cnt = 0
    # for root, dirs, files in os.walk(directory, topdown=False):
    #     for name in files:
    #         file_path = os.path.join(root, name)
    #
    #         if "tflite" in file_path:
    #             continue
    #
    #         cnt += 1
    #         print(name, "==========================================================================")
    #         model_instance.load_onnx_model(model_path=file_path)
    #         model_instance.print_input_shapes()
    #         print("\n")
