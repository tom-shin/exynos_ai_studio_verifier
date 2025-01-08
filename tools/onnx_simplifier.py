from onnxsim import simplify
import onnx
import os
import traceback
from onnx import helper
import onnxsim
from typing import Tuple, List, Dict
import numpy as np
import onnxruntime
import tflite2onnx
from onnx import version_converter


def tflite2onnxF():
    tflite_path = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\v_1_0_model_pamir\auto_portal\request\zero_dce_lite_160x160_iter8_30.tflite"
    onnx_path = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\v_1_0_model_pamir\auto_portal\request\zero_dce_lite_160x160_iter8_30.onnx"

    tflite2onnx.convert(tflite_path, onnx_path)


class ONNX_INFO:
    def __init__(self, model_path=None):
        self.model = onnx.load(model_path)
        self.model_path = model_path

    def load_onnx_model(self, model_path=None):
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

        nodes = self.model.graph.node
        new_nodes = []

        # 현재 그래프의 출력 이름 추출
        output_names = {output.name for output in self.model.graph.output}

        for node in nodes:
            if node.op_type == op_name:
                input_name = node.input[0]
                output_name = node.output[0]

                # Softmax가 그래프 출력에 포함된 경우 출력 이름 대체
                if output_name in output_names:
                    for output in self.model.graph.output:
                        if output.name == output_name:
                            output.name = input_name  # 이전 노드의 출력을 최종 출력으로 설정

                # Softmax 노드의 출력을 사용하는 노드 업데이트
                for next_node in nodes:
                    next_node.input[:] = [
                        input_name if inp == output_name else inp for inp in next_node.input
                    ]
            else:
                new_nodes.append(node)

        # 수정된 노드로 모델 그래프 갱신
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)

        # 수정된 모델 저장
        dir_path, filename = os.path.split(self.model_path)
        name, ext = os.path.splitext(filename)

        new_name = rf"{name}_{op_name}_removed{ext}"
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

        # 모델 간소화
        model_simplified, check = simplify(self.model, overwrite_input_shapes=input_tensor_shape,
                                           dynamic_input_shape=False)

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

    def test_model_integrity(self, model_path=None):
        print("\n[ONNX Inference 테스트 실행 결과]")
        try:
            # ONNX Runtime 세션 생성
            if model_path is not None:
                self.model_path = model_path

            session = onnxruntime.InferenceSession(self.model_path)

            # 입력 텐서 정보 가져오기
            input_tensors = session.get_inputs()

            # 입력 데이터 생성 (zeros로 초기화)
            input_data = []
            for tensor in input_tensors:
                shape = tuple(map(int, tensor.shape))  # 문자열을 정수로 변환
                data = np.zeros(shape, dtype=np.float32)
                input_data.append(data)

            # 推论 수행
            outputs = session.run(None, dict(zip([t.name for t in input_tensors], input_data)))

            return True

        except Exception as e:
            print(f"에러 발생: {str(e)}")
            return False

    def onnx_version_converter(self):

        # 기존 ONNX 모델 불러오기
        model_path = self.model_path
        directory, file_name = os.path.split(model_path)
        file_name, ext = os.path.splitext(file_name)

        output_path = os.path.join(directory, file_name + "_opset13" + ext)

        model = onnx.load(model_path)

        # Opset 버전을 13으로 변환
        target_opset = 13
        print(f"Converting ONNX model to opset version {target_opset}...")
        converted_model = version_converter.convert_version(model, target_opset)

        # 변환된 모델 저장
        onnx.save(converted_model, output_path)
        print(f"Converted model saved to {output_path}")

    def check_input_tensor(self):

        model_path = self.model_path

        # ONNX 세션 생성
        session = onnxruntime.InferenceSession(model_path)

        # 입력 정보 출력
        input_name = session.get_inputs()[0].name
        print(f"Input Name: {input_name}")
        print(f"Expected Shape: {session.get_inputs()[0].shape}")

        # 올바른 입력 텐서 준비 (예: float32[1, 3, 224, 224])
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # 모델 실행
        outputs = session.run(None, {input_name: dummy_input})

    def force_change_input_tensor(self):
        model_path = self.model_path
        directory, file_name = os.path.split(model_path)
        file_name, ext = os.path.splitext(file_name)

        model = onnx.load(model_path)

        # 모델 입력 수정
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1  # 배치 크기
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224

        # 수정된 모델 저장
        output_path = os.path.join(directory, file_name + "_force_input_tensor_change" + ext)
        onnx.save(model, rf"{output_path}")


if __name__ == "__main__":
    IND_TEST = False

    path = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\ml_group\dropout\squeezenet1.0-3_opset13.onnx"
    model_instance = ONNX_INFO(model_path=path)
    model_instance.force_change_input_tensor()
    # model_instance.remove_op(op_name="Log")  # LOG Dropout

    if IND_TEST:
        model_instance.onnx_version_converter()

        # check model version
        model_instance.print_model_version_info()

        # check model simplify
        model_instance.execute_onnx_simplifier()

        # model static/ dynamic check
        ret = model_instance.is_static_input()
        if ret:
            print("The model has static input shapes.")
        else:
            print("The model has dynamic input shapes.")

        is_integrity_ok = model_instance.test_model_integrity()
        if is_integrity_ok:
            print("  → Model integrity test passed")
        else:
            print("  → Model integrity test failed")

        # check model input shape
        model_instance.print_input_shapes()

        # python -m onnxsim mobilenetv2-10.onnx mobilenetv2-10_checked.onnx --overwrite-input-shape 1,3,224,224
        # shape = {"input": [1, 3, 416, 416], "image_shape": [416, 416]}   #yolo3
        # yolo4 보통 NCHW방식인데 이 모델은 보니 NHWC 방식으로 채널이 뒤에 있음. 이경우 보통 모델에 TRANSPOSE가 있어서
        # 내부적으로 다시 NCHW방식으로 변경하는 경우가 많음.
        shape = {"input": [1, 3, 224, 224]}
        model_instance.modify_input_tensor(input_tensor_shape=shape)

        directory = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\ml_group\ml_group_all_checked_simplify"
        cnt = 0
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)

                if "tflite" in file_path:
                    continue

                cnt += 1
                # print(name, "==========================================================================")
                model_instance.load_onnx_model(model_path=file_path)
                # model_instance.print_input_shapes()
                is_integrity_ok = model_instance.test_model_integrity()
                if is_integrity_ok:
                    print("  → Model integrity test passed")
                else:
                    print("  → Model integrity test failed")
                print("\n")
