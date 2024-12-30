import os
import onnx
import shutil
from collections import defaultdict
from huggingface_hub import hf_hub_download, list_models, list_repo_files

# 태그별 Opset 버전 히스토그램 저장
tag_opset_histogram = defaultdict(lambda: defaultdict(int))


# ONNX opset 버전 확인 함수
def check_opset_domain_version_f(p_model, tag):
    model = onnx.load(p_model)
    opset_imports = model.opset_import

    for opset in opset_imports:
        if opset.domain in ["", "ai.onnx"]:  # ONNX 기본 도메인만 필터링
            tag_opset_histogram[tag][opset.version] += 1

    # 태그별 Opset 버전 히스토그램 출력
    for tag, opset_histogram in tag_opset_histogram.items():
        print(f"Pipeline Tag: {tag}")
        for version, count in opset_histogram.items():
            print(f"  Opset Version {version}: {count} occurrence(s)")

    # 태그별 Opset 버전 히스토그램을 txt 파일에 기록
    file_txt = os.path.join(os.getcwd(), "opset_histogram.txt")
    with open(file_txt, "a") as file:
        for tag, opset_histogram in tag_opset_histogram.items():
            file.write(f"Pipeline Tag: {tag}\n")
            for version, count in opset_histogram.items():
                file.write(f"  Opset Version {version}: {count} occurrence(s)\n")
            file.write("\n")  # 각 태그별로 구분을 위한 공백 줄 추가

    # downloaded model list
    file_txt = os.path.join(os.getcwd(), "downloaded_model_list.txt")
    with open(file_txt, "a") as file:
        file.write(f"{p_model}\n")


# 필터링할 pipeline_tag 목록
pipeline_tags = [
    "depth-estimation", "image-classification", "object-detection",
    "image-segmentation", "text-to-image"
    # ,

    # "image-to-text",
    # "image-to-image", "image-to-video", "video-classification",
    # "text-to-video", "zero-shot-image-classification", "mask-generation",
    # "zero-shot-object-detection", "image-feature-extraction",
    # "keypoint-detection", "text-to-speech"
]

# pipeline_tag별로 모델을 필터링하고 처리
g_cnt = 0

save_download_model = True   # 계속 저장 할 것인가 ? 다운로드 받고 평가하고 지울 것인가 ? True이면 모델을 쭈욱 쌓아두지 않는다

for tag in pipeline_tags:

    if save_download_model:
        # 태그별 디렉터리 생성
        tag_dir = os.path.join(os.getcwd(), tag)
        os.makedirs(tag_dir, exist_ok=True)
        print(f"\nProcessing models for pipeline tag: {tag}")

    models = list_models(filter="onnx", pipeline_tag=tag)

    for model in models:
        model_id = model.modelId
        try:
            # 모델 저장소의 파일 목록 가져오기
            files = list_repo_files(model_id)

            # ONNX 파일이 있는지 확인
            onnx_files = [file for file in files if file.endswith(".onnx")]
            if not onnx_files:
                # print(f"No ONNX files found in model {model_id}. Skipping...")
                continue

            # ONNX 파일 다운로드
            m_down_cnt = 0

            if not save_download_model:
                tag_dir = os.path.join(os.getcwd(), "Temp")
                os.makedirs(tag_dir, exist_ok=True)

            for onnx_file in onnx_files:
                downloaded_file = hf_hub_download(repo_id=model_id, filename=onnx_file, cache_dir=tag_dir)
                g_cnt += 1
                m_down_cnt += 1
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(f"{g_cnt}  Downloaded ONNX model: {downloaded_file}")
                check_opset_domain_version_f(p_model=downloaded_file, tag=tag)

                # if m_down_cnt >= 2:  # 모델당 최대 2개 ONNX 파일만 확인
                #     break
            if not save_download_model:
                if os.path.exists(tag_dir):
                    shutil.rmtree(tag_dir)

        except Exception as e:
            print(f"Error processing model {model_id}: {e}")
