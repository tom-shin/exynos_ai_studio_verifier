#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import onnx
import logging
import easygui
import platform
import time
import pandas as pd
from collections import OrderedDict
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from ruamel.yaml import YAML
from io import StringIO
import re

from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5 import QtWidgets, QtCore, QtGui

from source.__init__ import *

from source.source_files.execute_verify import get_image_info_from_dockerImg, Load_Target_Dir_Thread, \
    start_docker_desktop, set_model_config

from source.source_files.main_enntest import run_enntest

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 일반 Python 스크립트 실행 시

logging.basicConfig(level=logging.INFO)


def PRINT_(*args):
    logging.info(args)


def load_module_func(module_name):
    mod = __import__(f"{module_name}", fromlist=[module_name])
    return mod


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class ImageLoadThread(QThread):
    send_finish_ui_sig = QtCore.pyqtSignal(bool, str)

    def __init__(self, parent, arg1, arg2):
        super().__init__(parent)
        self.parent = parent
        self.command = arg1
        self.run_time = arg2

    def run(self):
        start_time = time.time()
        load_result, _, _ = user_subprocess(cmd=self.command, run_time=self.run_time)
        elapsed_time = time.time() - start_time

        days = elapsed_time // (24 * 3600)
        remaining_secs = elapsed_time % (24 * 3600)
        hours = remaining_secs // 3600
        remaining_secs %= 3600
        minutes = remaining_secs // 60
        seconds = remaining_secs % 60

        total_time = f"{int(days)}day {int(hours)}h {int(minutes)}m {int(seconds)}s"

        if any("Loaded image:" in line for line in load_result):
            self.send_finish_ui_sig.emit(True, total_time)
        else:
            self.send_finish_ui_sig.emit(False, "")

    def stop(self):
        self.quit()
        self.wait(3000)


class Model_Analyze_Thread(QThread):
    output_signal = pyqtSignal(str, tuple, int, str)
    output_signal_2 = pyqtSignal(tuple, str, str, list, list, OrderedDict, list)

    timeout_output_signal = pyqtSignal(str, tuple, float)
    finish_output_signal = pyqtSignal(bool)

    send_max_progress_cnt = pyqtSignal(int)
    send_set_text_signal = pyqtSignal(str)
    send_onnx_opset_ver_signal = pyqtSignal(tuple, str, str, str, str)

    def __init__(self, parent=None, grand_parent=None, ssh_client=None, repo_tag=None):
        super().__init__()
        self.parent = parent
        self.grand_parent = grand_parent
        self._running = True
        self.ssh_client = ssh_client
        self.container_repo_tag = repo_tag
        self.timeout_expired = float(grand_parent.timeoutlineEdit.text().strip())
        self.container_trace = None

    def run(self):
        self.container_trace = []

        max_cnt = sum(1 for widget in self.parent.added_scenario_widgets if widget[0].scenario_checkBox.isChecked())
        self.send_max_progress_cnt.emit(max_cnt)

        executed_cnt = 0
        for target_widget in self.parent.added_scenario_widgets:
            if not self._running:
                break

            if not target_widget[0].scenario_checkBox.isChecked():
                continue

            executed_cnt += 1
            cwd, _ = separate_folders_and_files(target_widget[0].pathlineEdit.text())

            # 현재 경로를 WSL 경로로 변환
            drive, path = os.path.splitdrive(os.path.dirname(cwd))
            mnt_path = "/mnt/" + drive.lower().replace(":", "") + path.replace("\\", "/")

            try:
                container_pre_fix = self.container_repo_tag.split("/")[-1].replace(".", "_").replace("-", "_").replace(
                    ":", "_")
            except:
                container_pre_fix = "ai_studio"
            container_name = f"{container_pre_fix}_container_{executed_cnt}"
            self.container_trace.append(container_name)

            cmd = "docker rm -f $(docker ps -aq)"
            user_subprocess(cmd=f"{cmd}", run_time=False, log=False)
            # for container_name in self.container_trace:
            #     user_subprocess(cmd=f"docker stop {container_name}", run_time=False, log=False)
            #     user_subprocess(cmd=f"docker rm {container_name}", run_time=False, log=False)

            base_cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--security-opt", "seccomp:unconfined",  # 보안 프로필 해제 (2번 항목)
                "--cap-add=ALL",  # 모든 리눅스 기능 추가 (2번 항목)
                "--privileged",  # 권한 강화 (2번 항목)
                "--net", "host",  # 호스트 네트워크 사용 (3번 항목)
                "--ipc", "host",  # 호스트 IPC 네임스페이스 사용 (3번 항목)
                # "-e", f"http_proxy={http_proxy}",  # 환경 변수 전달 (4번 항목)
                # "-e", f"https_proxy={https_proxy}",  # 환경 변수 전달 (4번 항목)
                # "-e", f"DISPLAY={DISPLAY}",  # 환경 변수 전달 (4번 항목)
                # "-e", f"LOCAL_USER_ID=$(id -u $USER)",  # 환경 변수 전달 (4번 항목)
                "-v", f"{mnt_path}:/workspace",  # 볼륨 마운트 (이미 포함됨)
                "-v", "/etc/timezone:/etc/timezone",  # 타임존 설정 (6번 항목)
                "-w", "/workspace",  # 작업 디렉터리 설정 (6번 항목)
                # "-w", "/home/user/",  # 작업 디렉터리 설정 (6번 항목)
                self.container_repo_tag,
                "/bin/bash",
                "-c",
                "tail -f /dev/null"
            ]

            if self.grand_parent.cpuradioButton.isChecked():  # CPU
                # For CPU
                cmd = base_cmd[:3] + base_cmd[3:]  # No changes needed, just reuse the base command
            else:
                # For GPU
                cmd = base_cmd[:3] + ["--gpus", "all"] + base_cmd[3:]  # Add GPU specific option

            out, error, _ = user_subprocess(cmd=cmd, run_time=False, log=False, shell=False)

            if error:
                print("Error starting container:", error)
                continue  # 다음 루프로 넘어감

            cmd_fmt = []
            for i in range(0, 6):
                check_box_name = f"cmd{i}"  # 체크박스의 이름을 동적으로 생성
                check_box = getattr(self.grand_parent, check_box_name, None)  # 객체 가져오기
                if check_box and check_box.isChecked():  # 체크박스가 존재하고 선택되었는지 확인
                    cmd_fmt.append(check_box.text())  # 체크박스의 텍스트를 리스트에 추가

            start_T = time.time()
            init_success = False
            conversion_success = False
            compile_success = False

            # onnx information
            model = target_widget[0].pathlineEdit.text()
            name, ext = os.path.splitext(model)
            opset_ver = "Not ONNX Model"
            model_domain = "Not ONNX Model"
            onnx_validation = "Not ONNX Model"
            onnx_model_type = "Not ONNX Model"
            if ext.lower() == ".onnx":
                model = onnx.load(target_widget[0].pathlineEdit.text())
                opset_imports = model.opset_import
                opset_ver = ""
                model_domain = ""
                onnx_validation = ""
                onnx_model_type = ""

                for opset in opset_imports:
                    # s = f"{opset.version}\n"  # Operator set version
                    domain = f"Domain: {opset.domain or 'ai.onnx'}\n"  # Operator set version
                    model_domain += domain

                    ver = f"opset ver.: {opset.version}\n"  # Operator set version
                    opset_ver += ver

                try:
                    onnx.checker.check_model(model)
                    onnx_validation = "Passed"
                    # print("  • Model check passed ✓")
                except onnx.checker.ValidationError as e:
                    onnx_validation = f"[Error] {str(e)}"
                    # print(f"  • Validation Error: {str(e)}")
                except Exception as e:
                    onnx_validation = f"[Error] {str(e)}"
                    # print(f"  • Error: {str(e)}")

                # Check quantization type
                model_ops = set(node.op_type for node in model.graph.node)

                # Determine model type
                if "QLinearConv" in model_ops:
                    onnx_model_type = "INT8"
                    # print(f"  • Model Type: INT8 quantized model")
                elif "QuantizeLinear" in model_ops and "DequantizeLinear" in model_ops:
                    onnx_model_type = "QDQ"
                    # print(f"  • Model Type: QDQ model")
                elif not any(op for op in model_ops if "Q" in op):
                    onnx_model_type = "FP32"
                    # print(f"  • Model Type: FP32 model")
                else:
                    onnx_model_type = "Unknown quantization type"
                    # print(f"  • Model Type: Unknown quantization type")

            self.send_onnx_opset_ver_signal.emit(target_widget, onnx_validation, model_domain, opset_ver, onnx_model_type)

            for enntools_cmd in cmd_fmt:
                out = []
                error = []

                message = rf"{os.path.basename(cwd)} Executing {enntools_cmd} ...({executed_cnt}/{max_cnt})"
                self.send_set_text_signal.emit(message)

                cmd = ["docker", "exec", "-i", container_name, "/bin/bash", "-c",
                       f"cd /workspace/{os.path.basename(cwd)} && {enntools_cmd}"]
                if "enntest" not in enntools_cmd:
                    out, error, timeout_expired = user_subprocess(cmd=cmd, run_time=False, timeout=self.timeout_expired,
                                                                  shell=False, log=True)
                    if timeout_expired:
                        print("timeout_expired")
                        self.timeout_output_signal.emit(enntools_cmd, target_widget, self.timeout_expired)
                        break

                "실시간 yaml 수정: init 후 생성되는 yaml에 대해서 src_config을 참고해서 수정"
                failed_pairs = []
                parameter_set = OrderedDict()

                _directory_, _model_ = separate_folders_and_files(target_widget[0].pathlineEdit.text())
                _filename_, extension = separate_filename_and_extension(_model_)

                if "init" in enntools_cmd:
                    # 변경된 항목들을 텍스트로 변환

                    check_DATA_dir = os.path.join(_directory_, "DATA").replace("\\", "/")
                    target_config = os.path.join(_directory_, f"{_filename_}.yaml").replace("\\", "/")

                    if os.path.isdir(check_DATA_dir) and os.path.isfile(target_config):
                        init_success = True
                        print("init_success")

                    if self.grand_parent.modifiedradioButton.isChecked():
                        repo_tag = self.grand_parent.dockerimagecomboBox.currentText()
                        tag = int(repo_tag.split(":")[1].split(".")[0])

                        if tag >= 7:
                            src_config = os.path.join(BASE_DIR, "model_configuration",
                                                      "Ver2.0_model_config_new.yaml").replace("\\", "/")
                        else:
                            src_config = os.path.join(BASE_DIR, "model_configuration",
                                                      "Ver1.0_model_config_new.yaml").replace("\\", "/")

                        parameter_set = set_model_config(grand_parent=self.grand_parent, my_src_config=src_config,
                                                         target_config=target_config,
                                                         model_name=_model_)
                elif "conversion" in enntools_cmd:
                    check_log = os.path.join(cwd, "Converter_result", ".log")
                    error_keywords = keyword["error_keyword"]
                    if os.path.exists(check_log):
                        ret, error_contents_dict = upgrade_check_for_specific_string_in_files(check_log,
                                                                                              check_keywords=error_keywords)
                        if len(ret) == 0:
                            conversion_success = True
                            print("conversion_success")

                elif "compile" in enntools_cmd:
                    check_log = os.path.join(cwd, "Compiler_result", ".log")
                    error_keywords = keyword["error_keyword"]
                    if os.path.exists(check_log):
                        ret, error_contents_dict = upgrade_check_for_specific_string_in_files(check_log,
                                                                                              check_keywords=error_keywords)
                        if len(ret) == 0:
                            compile_success = True
                            print("compile_success")

                elif "enntest" in enntools_cmd:
                    if init_success and conversion_success and compile_success:
                        init_success = False
                        conversion_success = False
                        compile_success = False

                        nnc_model_path = os.path.join(_directory_, "Compiler_result").replace("\\", "/")
                        nnc_files = []
                        for file in os.listdir(nnc_model_path):
                            if file.endswith('.nnc') and os.path.isfile(os.path.join(nnc_model_path, file)):
                                filename = os.path.join(nnc_model_path, file).replace("\\", "/")
                                nnc_files.append(filename)

                        input_golden_path = os.path.join(_directory_, "Converter_result").replace("\\", "/")
                        # input_golden_pairs = find_paired_files(input_golden_path, mode=2)
                        input_golden_pairs = find_paired_files(input_golden_path, mode=1)

                        out_dir = os.path.join(_directory_, "Enntester_result").replace("\\", "/")
                        CheckDir(out_dir)

                        out = []
                        error = []
                        if len(nnc_files) != 0 and len(input_golden_pairs) != 0:
                            ret, failed_pairs = run_enntest(nnc_files, input_golden_pairs, out_dir,
                                                            self.grand_parent.enntestcomboBox.currentText())
                            if not ret:
                                out.append("Fail")
                                error.append("Fail")

                        else:
                            out.append("Skip(No nnc or input_golden)")
                            error.append("Skip(No nnc or input_golden)")
                    else:
                        out.append("Skip")
                        error.append("Skip")

                self.output_signal_2.emit(target_widget, enntools_cmd, cwd, out, error, parameter_set, failed_pairs)
                # time.sleep(3)

            for container_name in self.container_trace:
                user_subprocess(cmd=f"docker stop {container_name}", run_time=False, log=False)
                user_subprocess(cmd=f"docker rm {container_name}", run_time=False, log=False)

            # 전체 실행 시간 측정 종료
            elapsed_time = time.time() - start_T
            days = elapsed_time // (24 * 3600)
            remaining_secs = elapsed_time % (24 * 3600)
            hours = remaining_secs // 3600
            remaining_secs %= 3600
            minutes = remaining_secs // 60
            seconds = remaining_secs % 60
            total_time = f"{int(days)}day {int(hours)}h {int(minutes)}m {int(seconds)}s"

            self.output_signal.emit(total_time, target_widget, executed_cnt, cwd)

        self.finish_output_signal.emit(self._running)

    def re_config_yaml_file(self, target_widget):
        device = self.grand_parent.devicecomboBox.currentText()
        test_vector_gen = self.grand_parent.vectorcomboBox.currentText()
        debug = self.grand_parent.debugcomboBox.currentText()

        # 파일 경로 설정
        target_dir = target_widget[0].pathlineEdit
        filename = os.path.basename(target_dir)
        target_file = os.path.join(target_dir, filename).replace("\\", "/") + ".yaml"

        # 파일 열기 및 내용 수정
        with open(target_file, 'r') as file:
            lines = file.readlines()

        # 새로운 내용을 담을 리스트 생성
        new_lines = []
        for line in lines:
            # 'device:', 'test_vector_gen:', 'debug:' 부분을 수정
            if line.startswith('device:'):
                new_lines.append(f'device: {device}\n')
            elif line.startswith('test_vector_gen:'):
                new_lines.append(f'test_vector_gen: {test_vector_gen}\n')
            elif line.startswith('debug:'):
                new_lines.append(f'debug: {debug}\n')
            else:
                new_lines.append(line)

        # 수정된 내용을 다시 파일에 쓰기
        with open(target_file, 'w') as file:
            file.writelines(new_lines)

        print("파일 업데이트 완료")

    def stop(self):
        cmd = "docker rm -f $(docker ps -aq)"
        user_subprocess(cmd=f"{cmd}", run_time=False, log=False)
        # for container_name in self.container_trace:
        #     user_subprocess(cmd=f"docker stop {container_name}", run_time=False, log=False)
        #     user_subprocess(cmd=f"docker rm {container_name}", run_time=False, log=False)

        self._running = False
        self.quit()
        self.wait(3000)


class Model_Verify_Class(QObject):
    send_sig_delete_all_sub_widget = pyqtSignal()

    def __init__(self, parent, grand_parent):
        super().__init__()
        self.parent = parent
        self.grand_parent = grand_parent

        self.result_thread = None
        self.start_evaluation_time = None
        self.end_evaluation_time = None
        self.work_progress = None
        self.user_error_fmt = None
        self.model_analyze_thread_instance = None
        self.added_scenario_widgets = None

        self.insert_widget_thread = None
        self.insert_widget_progress = None

        self.send_sig_delete_all_sub_widget.connect(self.update_all_sub_widget)

    def open_file(self):
        # self.parent.mainFrame_ui.scenario_path_lineedit.setText(self.parent.directory.replace("\\", "/"))

        self.clear_sub_widget()

    def clear_sub_widget(self):
        while self.parent.mainFrame_ui.formLayout.count():
            item = self.parent.mainFrame_ui.formLayout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.send_sig_delete_all_sub_widget.emit()

    def finish_insert_each_widget(self):
        if self.insert_widget_progress is not None:
            self.insert_widget_progress.close()
            print(f"total loaded models: {len(self.added_scenario_widgets)}")

    def insert_each_widget(self, cnt, target_model_path, repo_src):
        if self.insert_widget_progress is not None:
            rt = load_module_func(module_name="source.ui_designer.main_widget")
            widget_ui = rt.Ui_Form()
            widget_instance = QtWidgets.QWidget()
            widget_ui.setupUi(widget_instance)

            model = os.path.basename(target_model_path)
            widget_ui.scenario_checkBox.setText(model)
            widget_ui.srclineEdit.setText(repo_src)

            widget_ui.pathlineEdit.setText(f"{target_model_path}")
            widget_ui.parametersetting_textEdit.setMinimumHeight(200)
            self.parent.mainFrame_ui.formLayout.setWidget(cnt, QtWidgets.QFormLayout.FieldRole,
                                                          widget_instance)

            self.added_scenario_widgets.append((widget_ui, widget_instance))
            self.insert_widget_progress.onCountChanged(value=cnt)

            widget_ui.inittextEdit.hide()
            widget_ui.conversiontextEdit.hide()
            widget_ui.compilertextEdit.hide()
            widget_ui.estimationtextEdit.hide()
            widget_ui.analysistextEdit.hide()
            widget_ui.profiletextEdit.hide()
            widget_ui.enntesttextEdit.hide()

            # 체크박스 신호 연결
            widget_ui.logonoffcheckBox.stateChanged.connect(
                lambda state, ui=widget_ui: self.toggle_text_edits(ui, state)
            )

    @staticmethod
    def toggle_text_edits(ui, state):
        """체크박스 상태에 따라 위젯 숨기기/보이기"""
        if state == QtCore.Qt.Checked:
            ui.inittextEdit.show()
            ui.conversiontextEdit.show()
            ui.compilertextEdit.show()
            ui.estimationtextEdit.show()
            ui.analysistextEdit.show()
            ui.profiletextEdit.show()
            ui.enntesttextEdit.show()
        else:
            ui.inittextEdit.hide()
            ui.conversiontextEdit.hide()
            ui.compilertextEdit.hide()
            ui.estimationtextEdit.hide()
            ui.analysistextEdit.hide()
            ui.profiletextEdit.hide()
            ui.enntesttextEdit.hide()

    def update_all_sub_widget(self):
        # BASE DIR 아래 Result 폴더 아래에 평가할 모델 복사
        user_fmt = [fmt.strip() for fmt in self.grand_parent.targetformat_lineedit.text().split(",") if fmt.strip()]
        get_test_model_path = get_directory(base_dir=self.parent.directory, user_defined_fmt=user_fmt,
                                            file_full_path=True)

        # Shared Volume 위치를 지정
        self.parent.directory = os.path.join(BASE_DIR, "Result").replace("\\", "/")
        self.parent.mainFrame_ui.scenario_path_lineedit.setText(self.parent.directory)
        CheckDir(self.parent.directory)

        all_test_path = []
        for test_model in get_test_model_path:
            directory, file_name = separate_folders_and_files(test_model)
            name, ext = separate_filename_and_extension(file_name)

            target_dir = os.path.join(self.parent.directory, name).replace("\\", "/")
            CheckDir(target_dir)

            src_file = test_model
            target_file = os.path.join(target_dir, file_name)
            shutil.copy2(src_file, target_file)

            all_test_path.append(target_file)

            if "caffemodel" in ext:
                src_file = os.path.join(directory, f"{name}.prototxt")
                target_file = os.path.join(target_dir, f"{name}.prototxt")
                shutil.copy2(src_file, target_file)

                # src_file = os.path.join(directory, f"{name}.protobin")
                # target_file = os.path.join(target_dir, f"{name}.protobin")
                # shutil.copy2(src_file, target_file)

        if self.parent.mainFrame_ui.popctrl_radioButton.isChecked():
            self.insert_widget_progress = ModalLess_ProgressDialog(message="Loading Scenario")
        else:
            self.insert_widget_progress = Modal_ProgressDialog(message="Loading Scenario")

        self.insert_widget_progress.setProgressBarMaximum(max_value=len(all_test_path))

        repo_tag = self.parent.mainFrame_ui.dockerimagecomboBox.currentText()
        tag = int(repo_tag.split(":")[1].split(".")[0])
        if tag >= 7:
            full_file = os.path.join(BASE_DIR, "model_configuration", "Ver2.0_model_config_new.yaml")
        else:
            full_file = os.path.join(BASE_DIR, "model_configuration", "Ver1.0_model_config_new.yaml")

        with open(full_file, 'r') as model_config_file:
            model_config_data = self.parent.yaml.load(model_config_file)

        self.added_scenario_widgets = []
        self.insert_widget_thread = Load_Target_Dir_Thread(all_test_path, self.parent,
                                                           model_config_data=model_config_data)
        self.insert_widget_thread.send_scenario_update_ui_sig.connect(self.insert_each_widget)
        self.insert_widget_thread.send_finish_scenario_update_ui_sig.connect(self.finish_insert_each_widget)

        self.insert_widget_thread.start()

        self.insert_widget_progress.showModal_less()

    def set_text_progress(self, string):
        if self.work_progress is not None:
            self.work_progress.onProgressTextChanged(string)

    @staticmethod
    def update_timeout_result(execute_cmd, target_widget, set_timeout):
        if "init" in execute_cmd:
            target_widget[0].initlineEdit.setText("Runtime Out")
        if "conversion" in execute_cmd:
            target_widget[0].conversionlineEdit.setText("Runtime Out")
        elif "compile" in execute_cmd:
            target_widget[0].compilelineEdit.setText("Runtime Out")
        elif "estimation" in execute_cmd:
            target_widget[0].estimationlineEdit.setText("Runtime Out")
        elif "analysis" in execute_cmd:
            target_widget[0].analysislineEdit.setText("Runtime Out")
        elif "profiling" in execute_cmd:
            target_widget[0].profilinglineEdit.setText("Runtime Out")

    @staticmethod
    def update_onnx_info(sub_widget, validation, model_domain, opset_ver, model_type):
        sub_widget[0].modelvalidaty.setText(validation)
        sub_widget[0].onnxdomain.setText(model_domain)
        sub_widget[0].onnxlineEdit.setText(opset_ver)
        sub_widget[0].modeltype.setText(model_type)

    def update_test_result_2(self, sub_widget, execute_cmd, cwd, out, error, parameter_setting, failed_pairs):
        if self.work_progress is not None:

            check_log = ""

            if "init" in execute_cmd:
                def convert_changed_items_to_text(changed_items):
                    # changed_items OrderedDict를 텍스트 형식으로 변환
                    result_text = ""
                    for section, keys in changed_items.items():  # OrderedDict의 섹션 순회
                        result_text += f"Section: {section}\n"  # 섹션 이름 추가
                        for key, values in keys.items():  # 각 섹션 내 키 순회
                            old_value = values['old_value']
                            new_value = values['new_value']
                            result_text += f"  Key: {key}\n"
                            result_text += f"    Old Value: {old_value}\n"
                            result_text += f"    New Value: {new_value}\n\n"

                    return result_text

                # 변경된 항목들을 텍스트로 변환
                changed_items_text = convert_changed_items_to_text(parameter_setting)
                sub_widget[0].parametersetting_textEdit.setText(changed_items_text)

                directory, model_name = separate_folders_and_files(sub_widget[0].pathlineEdit.text())
                filename, extension = separate_filename_and_extension(model_name)

                check_DATA_dir = os.path.join(directory, "DATA")
                check_init_yaml_file = os.path.join(directory, f"{filename}.yaml")
                if os.path.isdir(check_DATA_dir) and os.path.isfile(check_init_yaml_file):
                    sub_widget[0].initlineEdit.setText("Success")
                else:
                    sub_widget[0].initlineEdit.setText("Fail")

                # if len(out) == 0 and len(error) == 0:
                #     sub_widget[0].initlineEdit.setText("Success")
                # else:
                #     sub_widget[0].initlineEdit.setText("Fail")
                return

            elif "enntest" in execute_cmd:
                if len(out) == 0 and len(error) == 0:
                    sub_widget[0].enntestlineEdit.setText("Success")
                else:
                    sub_widget[0].enntestlineEdit.setText(out[0])
                    string_ = ""
                    for _in_bin, _golden_bin in failed_pairs:
                        string_ += f"[{os.path.basename(_in_bin)}, {os.path.basename(_golden_bin)}]\n"
                    sub_widget[0].enntesttextEdit.setText(string_)

                return

            elif "conversion" in execute_cmd:
                check_log = os.path.join(cwd, "Converter_result", ".log")

            elif "compile" in execute_cmd:
                check_log = os.path.join(cwd, "Compiler_result", ".log")

            elif "estimation" in execute_cmd:
                check_log = os.path.join(cwd, "Estimator_result", ".log")

            elif "analysis" in execute_cmd:
                check_log = os.path.join(cwd, "Analyzer_result", ".log")

            elif "profiling" in execute_cmd:
                check_log = os.path.join(cwd, "Profiler_result", ".log")

            error_keywords = keyword["error_keyword"]

            if os.path.exists(check_log):
                ret, error_contents_dict = upgrade_check_for_specific_string_in_files(check_log,
                                                                                      check_keywords=error_keywords)

                if len(ret) == 0:
                    if "conversion" in execute_cmd:
                        sub_widget[0].conversionlineEdit.setText("Success")
                    elif "compile" in execute_cmd:
                        sub_widget[0].compilelineEdit.setText("Success")
                    elif "estimation" in execute_cmd:
                        sub_widget[0].estimationlineEdit.setText("Success")
                    elif "analysis" in execute_cmd:
                        sub_widget[0].analysislineEdit.setText("Success")
                    elif "profiling" in execute_cmd:
                        sub_widget[0].profilinglineEdit.setText("Success")
                else:
                    text_to_display = ""

                    # context_data 내용을 문자열로 변환
                    for file, contexts in error_contents_dict.items():
                        text_to_display += f"File: {file}\n"
                        for context in contexts:
                            text_to_display += f"{context}\n{'-' * 40}\n"

                    if "conversion" in execute_cmd:
                        sub_widget[0].conversionlineEdit.setText("Fail")
                        sub_widget[0].conversiontextEdit.setText(text_to_display)

                    elif "compile" in execute_cmd:
                        sub_widget[0].compilelineEdit.setText("Fail")

                        profile_log = os.path.join(cwd, "Compiler_result", "profile_log.txt")
                        if os.path.exists(profile_log):
                            # 파일이 있으면 열기
                            text_to_display += f"\n\n[Profile Log]\n"
                            with open(profile_log, 'r') as file:
                                lines = file.readlines()

                            # Unsupported가 포함된 라인 찾기
                            for line in lines:
                                if "Unsupported" in line:  # "Unsupported" 키워드 포함 여부 확인
                                    text_to_display += line.strip() + "\n"  # 라인을 추가하고 줄 바꿈 추가

                        sub_widget[0].compilertextEdit.setText(text_to_display)

                    elif "estimation" in execute_cmd:
                        sub_widget[0].estimationlineEdit.setText("Fail")
                        sub_widget[0].estimationtextEdit.setText(text_to_display)

                    elif "analysis" in execute_cmd:
                        sub_widget[0].analysislineEdit.setText("Fail")
                        sub_widget[0].analysistextEdit.setText(text_to_display)

                    elif "profiling" in execute_cmd:
                        sub_widget[0].profilinglineEdit.setText("Fail")
                        sub_widget[0].profiletextEdit.setText(text_to_display)

    def update_test_result(self, elapsed_time, sub_widget, executed_cnt, cwd):
        if self.work_progress is not None:
            self.work_progress.onCountChanged(value=executed_cnt)
            sub_widget[0].elapsedlineEdit.setText(elapsed_time)
            self.save(self_saving=True)

    def error_update_test_result(self, error_message, sub_widget):
        if self.work_progress is not None:
            self.work_progress.close()

        sub_widget[0].contexts_textEdit.setText(error_message)
        sub_widget[0].lineEdit.setText("Error")

    def finish_update_test_result(self, normal_stop):
        if self.work_progress is not None:
            self.work_progress.close()

            self.end_evaluation_time = time.time()
            elapsed_time = self.end_evaluation_time - self.start_evaluation_time
            days = elapsed_time // (24 * 3600)
            remaining_secs = elapsed_time % (24 * 3600)
            hours = remaining_secs // 3600
            remaining_secs %= 3600
            minutes = remaining_secs // 60
            seconds = remaining_secs % 60

            total_time = f"{int(days)}day {int(hours)}h {int(minutes)}m {int(seconds)}s"
            msg_box = QtWidgets.QMessageBox()

            if normal_stop:
                msg_box.setWindowTitle("Test Done...")
                msg_box.setText(f"All Test Done !\nSave Button to store result data\nElapsed time: {total_time}")
            else:
                msg_box.setWindowTitle("Stop Test...")
                msg_box.setText(f"User forcibly terminated !")

            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)
            # Always show the message box on top
            if platform.system() == "Windows":
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

            # 메시지 박스를 최상단에 표시
            answer = msg_box.exec_()

    def set_max_progress_cnt(self, max_cnt):
        if self.work_progress is not None:
            self.work_progress.setProgressBarMaximum(max_value=max_cnt)

    def stop_analyze(self):
        if self.work_progress is not None:
            self.work_progress.close()

        if self.model_analyze_thread_instance is not None:
            self.model_analyze_thread_instance.stop()

    def analyze(self):
        if self.added_scenario_widgets is None:
            return

        if len(self.added_scenario_widgets) == 0:
            return

        check = False
        for cnt, target_widget in enumerate(self.added_scenario_widgets):
            if target_widget[0].scenario_checkBox.isChecked():
                check = True

                target_widget[0].modelvalidaty.setText("")
                target_widget[0].onnxdomain.setText("")
                target_widget[0].onnxlineEdit.setText("")
                target_widget[0].modeltype.setText("")

                # success/ fail lineedit 초기화
                target_widget[0].initlineEdit.setText("")
                target_widget[0].inittextEdit.setText("")

                target_widget[0].conversionlineEdit.setText("")
                target_widget[0].conversiontextEdit.setText("")

                target_widget[0].compilelineEdit.setText("")
                target_widget[0].compilertextEdit.setText("")

                target_widget[0].estimationlineEdit.setText("")
                target_widget[0].estimationtextEdit.setText("")

                target_widget[0].analysislineEdit.setText("")
                target_widget[0].analysistextEdit.setText("")

                target_widget[0].profilinglineEdit.setText("")
                target_widget[0].profiletextEdit.setText("")

                target_widget[0].elapsedlineEdit.setText("")
                target_widget[0].parametersetting_textEdit.setText("")

                target_widget[0].enntestlineEdit.setText("")
                target_widget[0].enntesttextEdit.setText("")

                directory, model_name = separate_folders_and_files(target_widget[0].pathlineEdit.text())
                filename, extension = separate_filename_and_extension(model_name)

                remove_alldata_files_except_specific_extension(directory=directory,
                                                               extension=extension.replace(".", ""))

        if not check:
            msg_box = QtWidgets.QMessageBox()  # QMessageBox 객체 생성
            msg_box.setWindowTitle("Check Test Model")  # 대화 상자 제목
            msg_box.setText(
                "test model is required.\nMark model")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)  # Yes/No 버튼 추가

            if platform.system() == "Windows":
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # 항상 위에 표시

            answer = msg_box.exec_()  # 대화 상자를 실행하고 사용자의 응답을 반환
            if answer == QtWidgets.QMessageBox.Yes:
                return

        self.start_evaluation_time = time.time()

        if self.parent.mainFrame_ui.popctrl_radioButton.isChecked():
            self.work_progress = ModalLess_ProgressDialog(message="Analyzing AI Model", show=True)
        else:
            self.work_progress = Modal_ProgressDialog(message="Analyzing AI Model", show=True)

        self.work_progress.send_user_close_event.connect(self.stop_analyze)

        self.user_error_fmt = [fmt.strip() for fmt in self.grand_parent.error_lineedit.text().split(",")]

        repo_tag = self.parent.mainFrame_ui.dockerimagecomboBox.currentText()
        self.model_analyze_thread_instance = Model_Analyze_Thread(self, self.grand_parent, repo_tag=repo_tag)
        self.model_analyze_thread_instance.output_signal.connect(self.update_test_result)
        self.model_analyze_thread_instance.output_signal_2.connect(self.update_test_result_2)
        self.model_analyze_thread_instance.send_onnx_opset_ver_signal.connect(self.update_onnx_info)
        self.model_analyze_thread_instance.timeout_output_signal.connect(self.update_timeout_result)

        self.model_analyze_thread_instance.send_set_text_signal.connect(self.set_text_progress)
        self.model_analyze_thread_instance.send_max_progress_cnt.connect(self.set_max_progress_cnt)
        self.model_analyze_thread_instance.finish_output_signal.connect(self.finish_update_test_result)

        self.model_analyze_thread_instance.start()

        self.work_progress.showModal_less()

    def select_all_scenario(self, check):
        if self.added_scenario_widgets is None or len(self.added_scenario_widgets) == 0:
            return

        for scenario_widget, scenario_widget_instance in self.added_scenario_widgets:
            scenario_widget.scenario_checkBox.setChecked(check)

    def save(self, self_saving=False):
        if self.added_scenario_widgets is None or len(self.added_scenario_widgets) == 0:
            return

        def clean_data(value):
            if isinstance(value, str):
                # Excel에서 허용되지 않는 문자 제거 (\x00 같은 제어 문자)
                # value = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", value)
                # \n과 \t는 제외하고 제어 문자 제거
                value = re.sub(r"[^\n\t\x20-\x7E]", "", value)
            return value

        result = []

        for cnt, target_widget in enumerate(self.added_scenario_widgets):
            # ret = target_widget[0].scenario_checkBox.text().strip().split(".")
            model, framework = os.path.splitext(target_widget[0].scenario_checkBox.text())
            #
            widget_result = {
                "Model": clean_data(model),
                "Framework": clean_data(framework.replace(".", "")),
                "Set parameter": clean_data(target_widget[0].parametersetting_textEdit.toPlainText().strip()),

                "Onnx_Opset Version": clean_data(target_widget[0].onnxlineEdit.text().strip()),
                "Onnx_Domain": clean_data(target_widget[0].onnxdomain.text().strip()),
                "Onnx_Model_Type": clean_data(target_widget[0].modeltype.text().strip()),
                "Onnx_Model_Validation": clean_data(target_widget[0].modelvalidaty.text().strip()),

                "init_Result": clean_data(target_widget[0].initlineEdit.text().strip()),
                "init_log": clean_data(""),  # 초기화된 값이 비어 있다면
                "conversion_Result": clean_data(target_widget[0].conversionlineEdit.text().strip()),
                "conversion_log": clean_data(target_widget[0].conversiontextEdit.toPlainText()),
                "compile_Result": clean_data(target_widget[0].compilelineEdit.text().strip()),
                "compile_log": clean_data(target_widget[0].compilertextEdit.toPlainText()),
                "estimation_Result": clean_data(target_widget[0].estimationlineEdit.text().strip()),
                "estimation_log": clean_data(target_widget[0].estimationtextEdit.toPlainText()),
                "analysis_Result": clean_data(target_widget[0].analysislineEdit.text().strip()),
                "analysis_log": clean_data(target_widget[0].analysistextEdit.toPlainText()),
                "profiling_Result": clean_data(target_widget[0].profilinglineEdit.text().strip()),
                "profiling_log": clean_data(target_widget[0].profiletextEdit.toPlainText()),
                "enntest_execute": clean_data(target_widget[0].enntestlineEdit.text().strip()),
                "enntest_fail(input, golden)": clean_data(target_widget[0].enntesttextEdit.toPlainText()),
                "model_source": clean_data(target_widget[0].srclineEdit.text().strip()),
                "elapsed_time": clean_data(target_widget[0].elapsedlineEdit.text().strip()),
            }
            result.append(widget_result)

        result_path = os.path.join(BASE_DIR, "Result", "result.json")
        json_dump_f(result_path, result)

        df = pd.DataFrame(result)

        # df = pd.DataFrame(result).applymap(clean_data)
        # 엑셀 파일로 저장
        excel_file = result_path.replace("json", "xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        # Load the workbook and access the active sheet
        wb = load_workbook(excel_file)
        ws = wb.active

        # Set wrap text for all cells and auto adjust column width and row height
        for row in ws.iter_rows():
            for cell in row:
                if cell.value:  # 셀 값이 존재할 경우에만 작업 수행
                    cell.value = str(cell.value).replace("\\n", "\n")  # 줄 바꿈 문자 처리
                    cell.alignment = Alignment(wrap_text=True)  # 줄 바꿈 설정

        # Auto-adjust column width
        for col in range(1, len(df.columns) + 1):
            max_length = 0
            column = get_column_letter(col)

            for row in ws.iter_rows(min_col=col, max_col=col):
                for cell in row:
                    if cell.value:  # 셀 값이 존재할 경우에만 길이 계산
                        max_length = max(max_length, len(str(cell.value)))

            ws.column_dimensions[column].width = max_length + 5  # Add padding to prevent truncation

        # Auto-adjust row height based on content (to ensure proper wrap)
        for row in ws.iter_rows():
            max_row_height = 0
            for cell in row:
                if cell.alignment.wrap_text and cell.value:  # 줄 바꿈이 설정된 셀에 대해 높이 조정
                    content_height = len(str(cell.value).split("\n"))
                    max_row_height = max(max_row_height, content_height)
            ws.row_dimensions[row[0].row].height = max_row_height * 15  # 행 높이 조정

        # Save the workbook with adjustments
        wb.save(excel_file)

        if not self_saving:
            msg_box = QtWidgets.QMessageBox()  # QMessageBox 객체 생성
            msg_box.setWindowTitle("Save Result")  # 대화 상자 제목
            msg_box.setText(
                "All saved.               \n")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes)  # Yes/No 버튼 추가

            if platform.system() == "Windows":
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # 항상 위에 표시

            answer = msg_box.exec_()  # 대화 상자를 실행하고 사용자의 응답을 반환


class Project_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.dialog = None
        self.directory = None
        self.single_op_ctrl = None
        self.wsl_shared_volume_name = None
        self.repo_tag = None
        self.load_progress = None
        self.load_thread = None
        """ for main frame & widget """
        self.mainFrame_ui = None
        self.widget_ui = None

        self.setupUi()

        self.yaml = YAML()
        self.yaml.preserve_quotes = True  # 기존 따옴표 보존
        self.yaml.default_flow_style = None  # 블록 스타일로 저장

    def setupUi(self):
        # Load the main window's UI module
        rt = load_module_func(module_name="source.ui_designer.main_frame")
        self.mainFrame_ui = rt.Ui_MainWindow()
        self.mainFrame_ui.setupUi(self)

        element_list = [fmt.strip() for fmt in keyword["element_1"]]
        text_concatenation = ', '.join(element_list)
        self.mainFrame_ui.targetformat_lineedit.setText(text_concatenation)

        element_list = [fmt.strip() for fmt in keyword["error_keyword"]]
        text_concatenation = ', '.join(element_list)
        self.mainFrame_ui.error_lineedit.setText(text_concatenation)

        # element_list = [fmt.strip() for fmt in keyword["op_exe_cmd"]]
        # text_concatenation = ', '.join(element_list)
        # self.mainFrame_ui.command_lineedit.setText(text_concatenation)

        self.setWindowTitle(Version)

        # image history
        history = os.path.join(BASE_DIR, "source", "history", "release_history.json")
        _, history_data = json_load_f(file_path=history)
        for cnt, data_ in enumerate(history_data):
            for item_num, (key, value) in enumerate(data_.items()):
                label = QtWidgets.QLabel(str(value))  # QLabel에 텍스트 추가
                font = label.font()
                font.setBold(False)  # 폰트를 기본 스타일로 설정 (굵기 제거)
                label.setFont(font)
                label.setAlignment(QtCore.Qt.AlignCenter)  # 가운데 정렬
                self.mainFrame_ui.history_table.addWidget(label, cnt + 1, item_num)

        self.mainFrame_ui.cmdlabel.hide()
        self.mainFrame_ui.command_lineedit.hide()

    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?\nAll data will be lost.",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if answer == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def normalOutputWritten(self, text):
        cursor = self.mainFrame_ui.logtextbrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)

        # 기본 글자 색상 설정
        color_format = cursor.charFormat()

        keywords = ["enntools init", "enntools conversion", "enntools compile", "enntools estimation",
                    "enntools analysis",
                    "enntools profiling"]
        color_format.setForeground(
            QtCore.Qt.red if any(keyword in text.lower() for keyword in keywords) else QtCore.Qt.black)

        cursor.setCharFormat(color_format)
        cursor.insertText(text)

        # 커서를 최신 위치로 업데이트
        self.mainFrame_ui.logtextbrowser.setTextCursor(cursor)
        self.mainFrame_ui.logtextbrowser.ensureCursorVisible()

    def cleanLogBrowser(self):
        self.mainFrame_ui.logtextbrowser.clear()

    def connectSlotSignal(self):
        """ sys.stdout redirection """
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.mainFrame_ui.log_clear_pushButton.clicked.connect(self.cleanLogBrowser)

        # evaluation tab
        self.mainFrame_ui.open_pushButton.clicked.connect(self.open_directory)
        self.mainFrame_ui.analyze_pushButton.clicked.connect(self.start_analyze)
        self.mainFrame_ui.all_check_scenario.clicked.connect(self.check_all_scenario)
        self.mainFrame_ui.all_uncheck_scenario.clicked.connect(self.check_all_scenario)
        self.mainFrame_ui.save_pushButton.clicked.connect(self.save_result)

        self.mainFrame_ui.actionOn.triggered.connect(self.log_browser_ctrl)
        self.mainFrame_ui.actionOff.triggered.connect(self.log_browser_ctrl)

        self.mainFrame_ui.imageregpushButton.clicked.connect(self.image_registration)

        self.mainFrame_ui.configpushButton.clicked.connect(self.open_model_config)

        self.single_op_ctrl = Model_Verify_Class(parent=self, grand_parent=self.mainFrame_ui)

    def save_changes(self, full_file, content, dialog):
        # YAML 문자열을 파싱하여 데이터로 변환
        try:
            # StringIO를 사용해 문자열을 YAML 객체로 로드
            yaml_data = self.yaml.load(StringIO(content))

            # 수정된 내용을 파일에 저장
            with open(full_file, 'w') as file:
                self.yaml.dump(yaml_data, file)

            # 성공 메시지 (필요시)
            print("파일이 성공적으로 저장되었습니다.")
            dialog.accept()  # 저장 후 다이얼로그 닫기

        except Exception as e:
            # 에러 메시지 (필요시)
            print(f"저장 중 오류 발생: {e}")

    def open_model_config(self):
        repo_tag = self.mainFrame_ui.dockerimagecomboBox.currentText()
        tag = int(repo_tag.split(":")[1].split(".")[0])
        if tag >= 7:
            full_file = os.path.join(BASE_DIR, "model_configuration", "Ver2.0_model_config_new.yaml")
        else:
            full_file = os.path.join(BASE_DIR, "model_configuration", "Ver1.0_model_config_new.yaml")

        with open(full_file, 'r') as model_config_file:
            model_config_data = self.yaml.load(model_config_file)

        # YAML 데이터를 문자열로 변환
        model_config_str = StringIO()
        self.yaml.dump(model_config_data, model_config_str)
        model_config_str = model_config_str.getvalue()  # 문자열로 변환

        rt = load_module_func(module_name="source.ui_designer.fileedit")

        self.dialog = QtWidgets.QDialog()
        ui = rt.Ui_Dialog()
        ui.setupUi(self.dialog)
        # 시그널 슬롯 연결 람다 사용해서 직접 인자를 넘기자...........
        ui.fileedit_save.clicked.connect(lambda: self.save_changes(full_file, ui.textEdit.toPlainText(), self.dialog))
        ui.fileedit_cancel.clicked.connect(self.dialog.close)

        # QSyntaxHighlighter를 통해 ':'로 끝나는 줄을 파란색으로 강조 표시
        highlighter = ColonLineHighlighter(ui.textEdit.document())

        ui.textEdit.setPlainText(model_config_str)  # 변환된 문자열 설정
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.show()

    @staticmethod
    def is_image_loaded():
        # docker image ls 명령어로 로드된 이미지 목록 가져오기
        list_cmd = 'docker image ls --format "{{.Repository}}:{{.Tag}}"'
        loaded_images, _, _ = user_subprocess(cmd=list_cmd, run_time=False, log=False)

        return loaded_images

    def image_registration(self):
        load_file = easygui.fileopenbox(
            msg="Select AI Studio Image",
            title="Image Load",
            default="*.image",
            filetypes=["*.image", "*.tar"]
        )

        if load_file is None:
            self.update_docker_imageList()
            return

        repo_tag = get_image_info_from_dockerImg(load_file.replace("\\", "/"))
        repo_tag_in_drive = self.is_image_loaded()

        if not repo_tag or not repo_tag_in_drive:
            result = False
        else:
            result = any(element_a in element_b for element_a in repo_tag for element_b in repo_tag_in_drive)

        if result:
            print(f"Image '{repo_tag}' is already loaded. Skipping load.")
            self.imageLoadResult(success=True, elapsed_time=1)
            return

        message = f"Want to load '{repo_tag}'?."
        yes_ = Open_QMessageBox(message=message)

        if yes_:
            if self.mainFrame_ui.popctrl_radioButton.isChecked():
                self.load_progress = ModalLess_ProgressDialog(message="Image Loading", parent=self)
            else:
                self.load_progress = Modal_ProgressDialog(message="Image Loading", parent=self)

            # 드라이브 문자 추출 및 변환
            drive, path = os.path.splitdrive(load_file)
            mnt_path = "/mnt/" + drive.lower().replace(":", "") + path.replace("\\", "/")

            load_command = f"docker load -i {mnt_path}"
            self.load_thread = ImageLoadThread(parent=self, arg1=load_command, arg2=True)
            self.load_thread.send_finish_ui_sig.connect(self.imageLoadResult)
            self.load_thread.start()

            self.load_progress.showModal_less()

    def update_docker_imageList(self):
        self.mainFrame_ui.dockerimagecomboBox.clear()
        self.mainFrame_ui.dockerimagecomboBox.addItems(self.is_image_loaded())

    def imageLoadResult(self, success, elapsed_time):
        if self.load_progress is not None:
            self.load_progress.close()

        if success:
            message = f"[Info] 이미지가 성공적으로 로드되었습니다.\nelapsed_time: {elapsed_time}"
        else:
            message = "[Warning] 이미지 로딩 실패 입니다."

        yes_ = Open_QMessageBox(message=message, no_b=False)

        self.update_docker_imageList()
        # self.update_containerList()

    def select_docker_image(self):
        load_file = easygui.fileopenbox(
            msg="Select Test Set Scenario",
            title="Test Set Selection",
            default="*.image",
            filetypes=["*.image"]
        )

        if load_file is None:
            return

        image_name = self.mainFrame_ui.replineEdit.text().strip()
        image_tag = self.mainFrame_ui.taglineEdit.text().strip()

        check_command = f"docker image ls --format '{{{{.Repository}}}}:{{{{.Tag}}}}'"
        out, error, _ = user_subprocess(cmd=check_command, run_time=False)

        if f"{image_name}:{image_tag}" in out:
            print(f"{image_name}:{image_tag} 이미 로드되어 있습니다.")
        else:
            message = "이미지가 로드 되어 있지 않습니다. 로드 하시겠습니까?\n로딩하는데 최소 몇 분 이상이 소요가 됩니다"
            yes_ = Open_QMessageBox(message=message)

            if yes_:
                load_command = f"docker load -i {load_file}"
                load_result, _, _ = user_subprocess(cmd=load_command, run_time=False)
                if any("Loaded image:" in line for line in load_result):
                    print(f"{image_name}:{image_tag} 이미지가 성공적으로 로드되었습니다.")
                    self.mainFrame_ui.aistudiolineEdit.setText(os.path.basename(load_file))
                else:
                    print(f"{image_name}:{image_tag} 로딩 Fail 입니다.")

    def log_browser_ctrl(self):
        sender = self.sender()
        if sender:
            if sender.objectName() == "actionOff":
                self.mainFrame_ui.logtextbrowser.hide()
            else:
                self.mainFrame_ui.logtextbrowser.show()

    def open_directory(self):
        _directory = easygui.diropenbox()

        if _directory is None:
            return

        self.directory = _directory.replace("\\", "/")
        if self.single_op_ctrl is not None:
            self.single_op_ctrl.open_file()

    def start_analyze(self):
        if self.single_op_ctrl is not None:
            self.single_op_ctrl.analyze()

    def check_all_scenario(self):
        sender = self.sender()
        check = False

        if sender:
            if sender.objectName() == "all_check_scenario":
                check = True
            elif sender.objectName() == "all_uncheck_scenario":
                check = False

        self.single_op_ctrl.select_all_scenario(check=check)

    def save_result(self):
        if self.single_op_ctrl is not None:
            self.single_op_ctrl.save()


if __name__ == "__main__":
    start_docker_desktop()

    import sys

    app = QtWidgets.QApplication(sys.argv)  # QApplication 생성 (필수)

    app.setStyle("Fusion")
    ui = Project_MainWindow()
    ui.showMaximized()
    ui.connectSlotSignal()
    ui.update_docker_imageList()

    sys.exit(app.exec_())
