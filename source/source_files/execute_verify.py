import tarfile
import json
import os
import shutil
import subprocess
from collections import OrderedDict
from ruamel.yaml import YAML
import re
import psutil
from PyQt5.QtCore import pyqtSignal, QObject, QThread

from .. import separate_folders_and_files
from source.__init__ import PRINT_


def get_image_info_from_dockerImg(image_path):
    # 이미지 파일에서 manifest.json을 추출하여 REPOSITORY와 TAG 확인
    try:
        with tarfile.open(image_path, 'r') as tar:
            manifest = tar.extractfile("manifest.json").read()
            manifest_data = json.loads(manifest)
            repo_tag = manifest_data[0]["RepoTags"]
            return repo_tag  # e.g., ["repository:tag"]
    except Exception as e:
        PRINT_(f"Error reading image metadata: {e}")
        return None


class Load_Target_Dir_Thread(QThread):
    send_scenario_update_ui_sig = pyqtSignal(int, str, str, str)
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
                repo_src = "unknown"  # 키가 없을 경우 기본값 설정 (필요에 따라 다른 값 설정 가능)

            try:
                License = self.model_config_data["model_config"][model]["license"]
            except KeyError:
                License = "unknown"  # 키가 없을 경우 기본값 설정 (필요에 따라 다른 값 설정 가능)

            self.send_scenario_update_ui_sig.emit(cnt, target_model_path, repo_src, License)

        self.send_finish_scenario_update_ui_sig.emit()


def start_docker_desktop():
    def check_docker_desktop():
        for process in psutil.process_iter(['pid', 'name']):
            try:
                # 프로세스 이름 확인
                if process.info['name'] and 'Docker Desktop.exe' in process.info['name']:
                    PRINT_(f"'Docker Desktop.exe' is running (PID: {process.info['pid']}).")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        PRINT_("'Docker Desktop.exe' is not running.")
        return False

    if check_docker_desktop():
        return

    docker_desktop_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"

    try:
        # subprocess.run을 사용하여 Docker Desktop 실행
        subprocess.run([docker_desktop_path], check=True)

    except subprocess.CalledProcessError as e:
        PRINT_(f"Failed to start Docker Desktop: {e}")
    except FileNotFoundError:
        PRINT_("Docker Desktop not found. Make sure it's installed.")


def set_model_config(grand_parent, my_src_config, target_config, model_name):
    # YAML 객체 초기화
    yaml = YAML()
    yaml.preserve_quotes = True  # 기존 따옴표 보존
    yaml.default_flow_style = None  # 블록 스타일로 저장
    yaml.allow_duplicate_keys = True  # 중복 키 허용 (필요시)

    with open(my_src_config, 'r') as model_config_file:
        model_config_data = yaml.load(model_config_file)

    with open(target_config, 'r') as target_yaml_file:
        target_yaml_data = yaml.load(target_yaml_file)

    if grand_parent.remoteradioButton.isChecked():
        device_id = grand_parent.sshdevicelineEdit.text().strip()
    else:
        device_id = grand_parent.localdeviceidlineEdit.text().strip()

    if "profiler" in model_config_data["global_config"]:
        model_config_data["global_config"]["profiler"]["device_id"] = str(device_id)
    else:
        model_config_data["global_config"]["profiler"] = {"device_id": f"{device_id}"}

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

    replace_null_with_temp(target_yaml_data)

    # global_config를 먼저 처리
    global_config = model_config_data.get("global_config", {})

    # 변경된 항목을 저장할 OrderedDict
    changed_items = OrderedDict()

    # global_config 항목 처리
    for global_section, global_values in global_config.items():
        if isinstance(global_values, dict):  # 섹션이 dict인 경우
            if global_section in target_yaml_data:
                # 모델별 섹션과 병합 (global_config 값 우선)
                for key, value in global_values.items():
                    if key in target_yaml_data[global_section]:
                        old_value = target_yaml_data[global_section][key]
                        if old_value != value:
                            if global_section not in changed_items:
                                changed_items[global_section] = OrderedDict()
                            changed_items[global_section][key] = {
                                'old_value': old_value,
                                'new_value': value
                            }
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
                            if section_key not in changed_items:
                                changed_items[section_key] = OrderedDict()
                            changed_items[section_key][key] = {
                                'old_value': old_value,
                                'new_value': value
                            }
                        target_yaml_data[section_key][key] = value

    # YAML 파일 저장
    with open(target_config, 'w') as output_yaml_file:
        yaml.dump(target_yaml_data, output_yaml_file)

    # 저장된 파일에서 "temp_null"을 실제 null로 변경
    with open(target_config, 'r') as output_yaml_file:
        lines = output_yaml_file.readlines()

    with open(target_config, 'w') as output_yaml_file:
        first_line = True
        for line in lines:
            line = line.replace(": temp_null", ": null")
            if first_line:
                output_yaml_file.write(line)
                first_line = False
            else:
                if line.strip() and line.strip()[0] != " " and line.strip().endswith(':'):
                    output_yaml_file.write("\n")
                    output_yaml_file.write(line)
                else:
                    output_yaml_file.write(line)

    # 변경된 항목 OrderedDict 반환
    return changed_items
