import tarfile
import json
import os
import shutil
import subprocess
from collections import OrderedDict
from ruamel.yaml import YAML
import re

from PyQt5.QtCore import pyqtSignal, QObject, QThread

from .. import CheckDir, separate_folders_and_files, separate_filename_and_extension


def get_image_info_from_dockerImg(image_path):
    # 이미지 파일에서 manifest.json을 추출하여 REPOSITORY와 TAG 확인
    try:
        with tarfile.open(image_path, 'r') as tar:
            manifest = tar.extractfile("manifest.json").read()
            manifest_data = json.loads(manifest)
            repo_tag = manifest_data[0]["RepoTags"]
            return repo_tag  # e.g., ["repository:tag"]
    except Exception as e:
        print(f"Error reading image metadata: {e}")
        return None


class Load_Target_Dir_Thread(QThread):
    send_scenario_update_ui_sig = pyqtSignal(int, str, str)
    send_finish_scenario_update_ui_sig = pyqtSignal()

    def __init__(self, file_path, grand_parent, model_config_data):
        super().__init__()
        self.test_file_paths = file_path
        self.grand_parent = grand_parent
        self.model_config_data = model_config_data

    def run(self):
        for cnt, test_path in enumerate(self.test_file_paths):
            target_model_path = test_path.replace("\\", "/")
            directory_, model = separate_folders_and_files(target_model_path)

            try:
                repo_src = self.model_config_data["model_config"][model]["repo"]
            except KeyError:
                repo_src = "Un-Known"  # 키가 없을 경우 기본값 설정 (필요에 따라 다른 값 설정 가능)

            self.send_scenario_update_ui_sig.emit(cnt, target_model_path, repo_src)

        self.send_finish_scenario_update_ui_sig.emit()


def organize_files(BASE_DIR, base_dir, files):
    current_dir = os.path.join(BASE_DIR, "Result")  # 현재 작업 디렉터리
    CheckDir(current_dir)

    for file_path in files:
        # 원본 파일의 dirname과 basename 추출
        relative_path = os.path.relpath(file_path, start=base_dir)
        dir_name = os.path.dirname(relative_path)
        base_name = os.path.basename(file_path)

        # 새로운 디렉터리 경로: 현재 작업 디렉터리 기준으로 상대 경로 설정
        new_dir_path = os.path.join(current_dir, dir_name, os.path.splitext(base_name)[0])
        os.makedirs(new_dir_path, exist_ok=True)

        # 파일 복사 경로 설정
        new_file_path = os.path.join(new_dir_path, base_name)
        shutil.copy(file_path, new_file_path)

        # caffemodel인 경우 prototxt 모델도 같이 복사 해야 함
        if base_name.split(".")[-1] == "caffemodel":
            src_prototxt_model = file_path.replace("caffemodel", "prototxt")
            dst_prototxt_model = new_file_path.replace("caffemodel", "prototxt")
            shutil.copy(src_prototxt_model, dst_prototxt_model)

        # print(f"Copied '{file_path}' to '{new_file_path}'")


def start_docker_desktop():
    docker_desktop_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"

    try:
        # subprocess.run을 사용하여 Docker Desktop 실행
        subprocess.run([docker_desktop_path], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker Desktop: {e}")
    except FileNotFoundError:
        print("Docker Desktop not found. Make sure it's installed.")


def X_set_model_config(grand_parent, src_config, target_config, model_name):
    # 파일 경로 설정
    model_config_json_path = src_config  # JSON 파일 경로
    target_yaml_path = target_config  # YAML 파일 경로

    # YAML 객체 초기화
    yaml = YAML()
    yaml.preserve_quotes = True  # 기존 따옴표 보존
    yaml.default_flow_style = None  # 블록 스타일로 저장
    yaml.allow_duplicate_keys = True  # 중복 키 허용 (필요시)

    if model_config_json_path.split(".")[-1] == "json":
        # JSON 파일에서 설정 읽기
        with open(model_config_json_path, 'r') as model_config_file:
            model_config_data = json.load(model_config_file)  # JSON 데이터 로드
    elif model_config_json_path.split(".")[-1] == "yaml":
        with open(model_config_json_path, 'r') as model_config_file:
            model_config_data = yaml.load(model_config_file)

    # YAML 파일에서 설정 읽기
    with open(target_yaml_path, 'r') as target_yaml_file:
        target_yaml_data = yaml.load(target_yaml_file)

    # 설정 저장용 OrderedDict 초기화
    settings = OrderedDict({
        "global_config": OrderedDict()
    })

    # 모델 설정 (model_config) 갱신
    model_settings = model_config_data.get("model_config", {}).get(model_name)
    if model_settings:
        for section_key, section_value in model_settings.items():
            # section_key가 target_yaml_data에 존재할 경우에만 갱신
            if section_key in target_yaml_data:
                for key, value in section_value.items():
                    # 정확히 이름이 일치하는 경우만 업데이트
                    if key in target_yaml_data[section_key]:
                        target_yaml_data[section_key][key] = value
                        if section_key not in settings:
                            settings[section_key] = OrderedDict()
                        settings[section_key][key] = value  # 설정 값 저장

    # global_config 설정을 model_config 적용 후에 갱신
    global_config_settings = model_config_data["global_config"]
    for key, value in global_config_settings.items():
        # 정확히 이름이 일치하는 경우만 업데이트
        def update_yaml_recursively(data):
            if isinstance(data, dict):
                for yaml_key in data:
                    if yaml_key == key:
                        # None 값이 null로 유지되도록 처리
                        if value is None:
                            data[yaml_key] = "null"  # 강제로 null을 문자열로 설정
                        else:
                            data[yaml_key] = value
                        settings["global_config"][key] = value  # 설정 값 저장
                    else:
                        update_yaml_recursively(data[yaml_key])
            elif isinstance(data, list):
                for item in data:
                    update_yaml_recursively(item)

        update_yaml_recursively(target_yaml_data)

    # 수정되지 않은 항목에서 null을 유지하도록 설정
    def enforce_nulls(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:  # null을 유지하도록 처리
                    data[key] = "temp_null"  # null을 강제로 문자열로 설정
                else:
                    enforce_nulls(value)
        elif isinstance(data, list):
            for item in data:
                enforce_nulls(item)

    # target_yaml_data에서 null 값 유지
    enforce_nulls(target_yaml_data)

    # YAML 파일에 저장 (주석 유지, null 값 유지)
    with open(target_yaml_path, 'w') as output_yaml_file:
        yaml.dump(target_yaml_data, output_yaml_file)

        # 저장된 파일에서 'null'을 실제 null 값으로 변경
    with open(target_yaml_path, 'r') as output_yaml_file:
        lines = output_yaml_file.readlines()

    with open(target_yaml_path, 'w') as output_yaml_file:
        for line in lines:
            # ': 'null''을 ': null'로 변경
            output_yaml_file.write(line.replace(": temp_null", ": null"))

    return settings


def set_model_config(grand_parent, my_src_config, target_config, model_name):
    # YAML 객체 초기화
    yaml = YAML()
    yaml.preserve_quotes = True  # 기존 따옴표 보존
    yaml.default_flow_style = None  # 블록 스타일로 저장
    yaml.allow_duplicate_keys = True  # 중복 키 허용 (필요시)

    with open(my_src_config, 'r') as model_config_file:
        model_config_data = yaml.load(model_config_file)

    # YAML 파일에서 설정 읽기
    with open(target_config, 'r') as target_yaml_file:
        target_yaml_data = yaml.load(target_yaml_file)

    # null 값 강제로 "temp_null"로 변환
    def replace_null_with_temp(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:  # null 값을 temp_null로 변경
                    data[key] = "temp_null"
                else:
                    replace_null_with_temp(value)
        elif isinstance(data, list):
            for item in data:
                replace_null_with_temp(item)

    # null 값을 temp_null로 변환
    replace_null_with_temp(target_yaml_data)

    # global_config를 먼저 처리
    global_config = model_config_data.get("global_config", {})

    # 변경된 항목을 저장할 리스트
    changed_items = []

    # global_config 항목 처리 (모든 모델에 공통)
    for global_section, global_values in global_config.items():
        if isinstance(global_values, dict):  # 섹션이 dict인 경우
            if global_section in target_yaml_data:
                # 모델별 섹션과 병합 (global_config 값 우선)
                for key, value in global_values.items():
                    if key in target_yaml_data[global_section]:
                        old_value = target_yaml_data[global_section][key]
                        if old_value != value:
                            changed_items.append({
                                'section': global_section,
                                'key': key,
                                'old_value': old_value,
                                'new_value': value
                            })
                        target_yaml_data[global_section][key] = value

    # 모델별 설정 처리
    model_settings = model_config_data.get("model_config", {}).get(model_name, {})
    if model_settings:
        for section_key, section_values in model_settings.items():
            if section_key in target_yaml_data:
                for key, value in section_values.items():
                    if key in target_yaml_data[section_key]:
                        old_value = target_yaml_data[section_key][key]
                        if old_value != value:
                            changed_items.append({
                                'section': section_key,
                                'key': key,
                                'old_value': old_value,
                                'new_value': value
                            })
                        target_yaml_data[section_key][key] = value

    # YAML 파일 저장
    with open(target_config, 'w') as output_yaml_file:
        yaml.dump(target_yaml_data, output_yaml_file)

    # 저장된 파일에서 "temp_null"을 실제 null로 변경
    with open(target_config, 'r') as output_yaml_file:
        lines = output_yaml_file.readlines()

    # 텍스트 처리: 첫 번째 줄은 그대로 두고, 이후 ':'로 끝나는 항목에서 두 줄 추가
    with open(target_config, 'w') as output_yaml_file:
        first_line = True  # 첫 번째 라인 여부를 체크하는 변수
        for line in lines:
            # ": temp_null"을 ": null"로 변경
            line = line.replace(": temp_null", ": null")

            # 첫 번째 라인은 그대로 작성
            if first_line:
                output_yaml_file.write(line)
                first_line = False
            else:
                # ':'로 끝나고 왼쪽에 공백이 없으면 두 줄을 추가
                if line.strip() and line.strip()[0] != " " and line.strip().endswith(':'):
                    output_yaml_file.write("\n")  # 두 줄 내려서 쓰기
                    output_yaml_file.write(line)  # 다시 그 라인 작성
                else:
                    output_yaml_file.write(line)

    # 변경된 항목 리스트 반환
    return changed_items
