import argparse
import onnx
import onnxsim
import os
from typing import Tuple, List, Dict
import numpy as np
import onnxruntime


def analyze_tensor_shapes(tensor) -> Dict[str, List]:
    """텐서의 shape 정보를 분석합니다."""
    shape_info = []
    is_dynamic = False

    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_param:  # 심볼릭 차원
            shape_info.append(f"dynamic({dim.dim_param})")
            is_dynamic = True
        elif dim.dim_value:  # 고정 차원
            shape_info.append(str(dim.dim_value))
        else:  # 미정의 차원
            shape_info.append("dynamic")
            is_dynamic = True

    return {
        "name": tensor.name,
        "shape": shape_info,
        "is_dynamic": is_dynamic
    }


def check_model_directly(model_path: str) -> Tuple[bool, List[Dict]]:
    """모델을 직접 분석하여 동적 shape를 검사합니다."""
    model = onnx.load(model_path)

    try:
        onnx.checker.check_model(model)
        print("  • Model check passed ✓")
    except onnx.checker.ValidationError as e:
        print(f"  • Validation Error: {str(e)}")
    except Exception as e:
        print(f"  • Error: {str(e)}")

    # Check quantization type
    model_ops = set(node.op_type for node in model.graph.node)

    # Determine model type
    if "QLinearConv" in model_ops:
        print(f"  • Model Type: INT8 quantized model")
    elif "QuantizeLinear" in model_ops and "DequantizeLinear" in model_ops:
        print(f"  • Model Type: QDQ model")
    elif not any(op for op in model_ops if "Q" in op):
        print(f"  • Model Type: FP32 model")
    else:
        print(f"  • Model Type: Unknown quantization type")

    has_dynamic = False
    tensor_info = []

    # 입력 텐서 분석
    for input in model.graph.input:
        info = analyze_tensor_shapes(input)
        info["type"] = "input"
        tensor_info.append(info)
        if info["is_dynamic"]:
            has_dynamic = True

    # 출력 텐서 분석
    for output in model.graph.output:
        info = analyze_tensor_shapes(output)
        info["type"] = "output"
        tensor_info.append(info)
        if info["is_dynamic"]:
            has_dynamic = True

    return has_dynamic, tensor_info


def run_simplifier(model_path: str) -> List[str]:
    """ONNX Simplifier를 실행하고 발생하는 메시지를 수집합니다."""
    try:
        # simplified 폴더 생성 (현재 작업 디렉토리 기준)
        output_dir = os.path.join(os.getcwd(), "simplified")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            messages = [f"'simplified' 폴더가 생성되었습니다: {output_dir}"]
        else:
            messages = []

        # 출력 파일 경로 생성
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_simplified.onnx")
        messages.append(f"단순화된 모델이 다음 경로에 저장될 예정: {output_path}")

        model = onnx.load(model_path)

        try:
            simplified_model, check = onnxsim.simplify(
                model,
                skipped_optimizers=['eliminate_identity']
            )

            # 단순화된 모델 저장
            onnx.save(simplified_model, output_path)
            messages.append(f"Simplifier 실행 완료 (반환값: {check})")
            messages.append(f"단순화된 모델이 저장됨: {output_path}")

        except Exception as e:
            messages.append(f"단순화 중 예외 발생: {str(e)}")

        return messages

    except Exception as e:
        return [f"모델 로드 중 오류 발생: {str(e)}"]


def test_model_integrity(model_path: str) -> bool:
    try:
        # ONNX Runtime 세션 생성
        session = onnxruntime.InferenceSession(model_path)

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


def main(model_path: str):
    """메인 실행 함수"""
    print(f"\n🔍 모델 분석 시작: {model_path}\n")

    # 1. 직접 분석
    print("1. Direct Shape 분석:")
    has_dynamic_direct, tensor_info = check_model_directly(model_path)

    for info in tensor_info:
        shape_str = " × ".join(info["shape"])
        print(f"  • {info['type'].upper()}: {info['name']}")
        print(f"    Shape: [{shape_str}]")
    print(f"  → Analysis Result: {'Dynamic' if has_dynamic_direct else 'Static'} shape\n")

    # 2. Simplifier 실행
    print("2. ONNX Simplifier 실행 결과:")
    simplifier_messages = run_simplifier(model_path)
    for msg in simplifier_messages:
        print(f"  • {msg}")

    # 3. 모델 무결성 테스트
    print("3. ONNX Inference 테스트 실행 결과:")
    is_integrity_ok = test_model_integrity(model_path)
    if is_integrity_ok:
        print("  → Model integrity test passed")
    else:
        print("  → Model integrity test failed")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="ONNX 모델의 static/dynamic shape 여부를 검사합니다.")
    # parser.add_argument("model_path", type=str, help="입력 ONNX 모델 파일 경로")
    # args = parser.parse_args()
    # main(args.model_path)
    model = r"C:\Work\tom\python_project\exynos_ai_studio_verifier\source\source_files\simplified\yolov4_simplified.onnx"
    main(model_path=model)

    # directory = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\ml_group\ml_group_all_checked_simplify"
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
    #         main(model_path=file_path)
