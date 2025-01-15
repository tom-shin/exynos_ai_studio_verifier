import sys
import os
import subprocess
import json
import chardet
from collections import OrderedDict
from datetime import datetime
import shutil
import re
import itertools
import platform

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QThread
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QSpacerItem, \
    QSizePolicy, QRadioButton, QWidget, QMessageBox

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter

Version = "AI Studio Analyzer ver.3.2.1_20250115 (made by tom.shin)"

# "enntools profiling"
keyword = {
    "element_1": ["onnx"],  # , "tflite", "caffemodel"],
    "error_keyword": ["Error Code:", "Error code:", "Error msg:"],
    "op_exe_cmd": ["enntools init", "enntools conversion", "enntools compile", "enntools estimation",
                   "enntools analysis", "enntest execute"],
    "exclusive_dir": ["DATA", "recipe", "yolox_darknet", "etc"]
    # yolox_darknet  --> timeout(12시간)
}

# ANSI 코드 정규식 패턴 (터미널 컬러 코드)
ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


class ProgressDialog(QDialog):  # This class will handle both modal and non-modal dialogs
    send_user_close_event = pyqtSignal(bool)

    def __init__(self, message, modal=True, show=False, parent=None):
        super().__init__(parent)

        self.setWindowTitle(message)

        # Set the dialog as modal or non-modal based on the 'modal' argument
        if modal:
            self.setModal(True)
        else:
            self.setWindowModality(QtCore.Qt.NonModal)

        self.resize(700, 100)  # Resize to desired dimensions

        self.progress_bar = QProgressBar(self)
        self.label = QLabel("", self)
        self.close_button = QPushButton("Close", self)
        self.radio_button = QRadioButton("", self)

        # Create a horizontal layout for the close button and spacer
        h_layout = QHBoxLayout()
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_layout.addSpacerItem(spacer)
        h_layout.addWidget(self.close_button)

        # Create the main layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.label)
        layout.addWidget(self.radio_button)
        layout.addLayout(h_layout)
        self.setLayout(layout)

        # Close button click event
        self.close_button.clicked.connect(self.close)

        # Show or hide the close button based on 'show'
        if show:
            self.close_button.show()
        else:
            self.close_button.hide()

        # Timer to toggle radio button every 500ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.toggle_radio_button)
        self.timer.start(500)  # 500ms interval

        self.radio_state = False  # Initial blink state

    def setProgressBarMaximum(self, max_value):
        self.progress_bar.setMaximum(int(max_value))

    def onCountChanged(self, value):
        self.progress_bar.setValue(int(value))

    def onProgressTextChanged(self, text):
        self.label.setText(text)

    def show_progress(self):
        if self.isModal():
            super().exec_()  # Execute as modal
        else:
            self.show()  # Show as non-modal

    def closeEvent(self, event):
        self.send_user_close_event.emit(True)
        event.accept()

    def toggle_radio_button(self):
        if self.radio_state:
            self.radio_button.setStyleSheet("""
                        QRadioButton::indicator {
                            width: 12px;
                            height: 12px;
                            background-color: red;
                            border-radius: 5px;
                        }
                    """)
        else:
            self.radio_button.setStyleSheet("""
                        QRadioButton::indicator {
                            width: 12px;
                            height: 12px;
                            background-color: blue;
                            border-radius: 5px;
                        }
                    """)
        self.radio_state = not self.radio_state


# class ModalLess_ProgressDialog(QWidget):  # popup 메뉴가 있어도 뒤 main gui의 제어가 가능 함
#     send_user_close_event = pyqtSignal(bool)
#
#     def __init__(self, message, show=False, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle(message)
#
#         self.resize(700, 100)  # 원하는 크기로 조절
#
#         self.progress_bar = QProgressBar(self)
#         self.label = QLabel("", self)
#         self.close_button = QPushButton("Close", self)
#         self.radio_button = QRadioButton("", self)
#
#         # Create a horizontal layout for the close button and spacer
#         h_layout = QHBoxLayout()
#         spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         h_layout.addSpacerItem(spacer)
#         h_layout.addWidget(self.close_button)
#
#         # Create the main layout
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.progress_bar)
#         layout.addWidget(self.label)
#         layout.addWidget(self.radio_button)
#         layout.addLayout(h_layout)
#         self.setLayout(layout)
#
#         # Close 버튼 클릭 시 다이얼로그를 닫음
#         self.close_button.clicked.connect(self.close)
#
#         if show:
#             self.close_button.show()
#         else:
#             self.close_button.hide()
#
#         # Timer 설정
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.toggle_radio_button)
#         self.timer.start(500)  # 500ms 간격으로 토글
#
#         self.radio_state = False  # 깜빡임 상태 초기화
#
#     def setProgressBarMaximum(self, max_value):
#         self.progress_bar.setMaximum(int(max_value))
#
#     def onCountChanged(self, value):
#         self.progress_bar.setValue(int(value))
#
#     def onProgressTextChanged(self, text):
#         self.label.setText(text)
#
#     def showModal_less(self):
#         self.showModal()
#
#     def showModal(self):
#         self.show()
#
#     def closeEvent(self, event):
#         # subprocess.run("taskkill /f /im cmd.exe /t", shell=True)
#
#         self.send_user_close_event.emit(True)
#         event.accept()
#
#     def toggle_radio_button(self):
#         if self.radio_state:
#             self.radio_button.setStyleSheet("""
#                         QRadioButton::indicator {
#                             width: 12px;
#                             height: 12px;
#                             background-color: red;
#                             border-radius: 5px;
#                         }
#                     """)
#         else:
#             self.radio_button.setStyleSheet("""
#                         QRadioButton::indicator {
#                             width: 12px;
#                             height: 12px;
#                             background-color: blue;
#                             border-radius: 5px;
#                         }
#                     """)
#         self.radio_state = not self.radio_state
#
#
# class Modal_ProgressDialog(QDialog):  # popup 메뉴가 있으면 뒤 main gui의 제어 불 가능 -> modal
#     send_user_close_event = pyqtSignal(bool)
#
#     def __init__(self, message, show=False, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle(message)
#         self.setModal(True)
#
#         self.resize(700, 100)  # 원하는 크기로 조절
#
#         self.progress_bar = QProgressBar(self)
#         self.label = QLabel("", self)
#         self.close_button = QPushButton("Close", self)
#         self.radio_button = QRadioButton("", self)
#
#         # Create a horizontal layout for the close button and spacer
#         h_layout = QHBoxLayout()
#         spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         h_layout.addSpacerItem(spacer)
#         h_layout.addWidget(self.close_button)
#
#         # Create the main layout
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.progress_bar)
#         layout.addWidget(self.label)
#         layout.addWidget(self.radio_button)
#         layout.addLayout(h_layout)
#         self.setLayout(layout)
#
#         # Close 버튼 클릭 시 다이얼로그를 닫음
#         self.close_button.clicked.connect(self.close)
#
#         if show:
#             self.close_button.show()
#         else:
#             self.close_button.hide()
#
#         # Timer 설정
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.toggle_radio_button)
#         self.timer.start(500)  # 500ms 간격으로 토글
#
#         self.radio_state = False  # 깜빡임 상태 초기화
#
#     def setProgressBarMaximum(self, max_value):
#         self.progress_bar.setMaximum(int(max_value))
#
#     def onCountChanged(self, value):
#         self.progress_bar.setValue(int(value))
#
#     def onProgressTextChanged(self, text):
#         self.label.setText(text)
#
#     def showModal(self):
#         super().exec_()
#
#     def showModal_less(self):
#         super().show()
#
#     def closeEvent(self, event):
#         # subprocess.run("taskkill /f /im cmd.exe /t", shell=True)
#
#         self.send_user_close_event.emit(True)
#         event.accept()
#
#     def toggle_radio_button(self):
#         if self.radio_state:
#             self.radio_button.setStyleSheet("""
#                         QRadioButton::indicator {
#                             width: 12px;
#                             height: 12px;
#                             background-color: red;
#                             border-radius: 5px;
#                         }
#                     """)
#         else:
#             self.radio_button.setStyleSheet("""
#                         QRadioButton::indicator {
#                             width: 12px;
#                             height: 12px;
#                             background-color: blue;
#                             border-radius: 5px;
#                         }
#                     """)
#         self.radio_state = not self.radio_state


def json_dump_f(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".json"):
        file_path += ".json"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False)

    return True


def json_load_f(file_path, use_encoding=False):
    if file_path is None:
        return False, False

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "r", encoding=encoding) as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    return True, json_data


def load_markdown(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    with open(data_path, 'r', encoding=encoding) as file:
        print(data_path)
        data_string = file.read()
        documents = markdown_splitter.split_text(data_string)

        # 파일명을 metadata에 추가
        domain = data_path  # os.path.basename(data_path)
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가

        return documents

    # with open(data_path, 'r') as file:
    #     data_string = file.read()
    #     return markdown_splitter.split_text(data_string)


def load_txt(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    text_splitter = CharacterTextSplitter(
        separator="\n",
        length_function=len,
        is_separator_regex=False,
    )

    with open(data_path, 'r', encoding=encoding) as file:
        data_string = file.read().split("\n")
        domain = data_path  # os.path.basename(data_path)
        documents = text_splitter.create_documents(data_string)

        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["domain"] = domain  # Document 객체의 metadata 속성에 파일명 추가

        return documents
    # with open(data_path, 'r') as file:
    #     data_string = file.read().split("\n")
    #     domain = os.path.splitext(os.path.basename(data_path))[0]
    #     metadata = [{"domain": domain} for _ in data_string]
    #     return text_splitter.create_documents(
    #         data_string,
    #         metadata
    #     )


def load_general(base_dir):
    data = []
    cnt = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:
                    cnt += 1
                    data += load_txt(file_path)

    print(f"the number of txt files is : {cnt}")
    return data


def load_document(base_dir):
    data = []
    cnt = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:
                    cnt += 1
                    data += load_markdown(file_path)

    print(f"the number of md files is : {cnt}")
    return data


def get_markdown_files(source_dir):
    dir_ = source_dir
    loader = DirectoryLoader(dir_, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    print("number of doc: ", len(documents))
    return documents


def get_directory(base_dir, user_defined_fmt=None, file_full_path=False):
    data = set()

    for root, dirs, files in os.walk(base_dir):
        for exclusive in keyword["exclusive_dir"]:
            if exclusive in dirs:
                dirs.remove(exclusive)

        for file in files:
            if user_defined_fmt is None:
                if any(file.endswith(ext) for ext in keyword["element_1"]):
                    if file_full_path:
                        data.add(os.path.join(root, file))
                    else:
                        data.add(root)
            else:
                if any(file.endswith(ext) for ext in user_defined_fmt):
                    if file_full_path:
                        data.add(os.path.join(root, file))
                    else:
                        data.add(root)

    unique_paths = sorted(data)
    return unique_paths


def CheckDir(dir_):
    if os.path.exists(dir_):
        shutil.rmtree(dir_)  # 기존 디렉터리 삭제

    os.makedirs(dir_)  # 새로 생성


def GetCurrentDate():
    current_date = datetime.now()

    # 날짜 형식 지정 (예: YYYYMMDD)
    formatted_date = current_date.strftime("%Y%m%d")

    return formatted_date


def save2html(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".html"):
        file_path += ".html"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        f.write(data)

    # print("Saved as HTML text.")


def save2txt(file_path, data, use_encoding=False):
    if file_path is None:
        return False

    if not file_path.endswith(".txt"):
        file_path += ".txt"

    if use_encoding:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
    else:
        encoding = "utf-8"

    with open(file_path, "w", encoding=encoding) as f:
        f.write(data)

    # print("Saved as TEXT.")


def check_environment():
    env = ''
    system = platform.system()
    if system == "Windows":
        # Windows인지 확인, WSL 포함
        if "microsoft" in platform.version().lower() or "microsoft" in platform.release().lower():
            env = "WSL"  # Windows Subsystem for Linux
        env = "Windows"  # 순수 Windows
    elif system == "Linux":
        # Linux에서 WSL인지 확인
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
            if "microsoft" in version_info:
                env = "WSL"  # WSL 환경
        except FileNotFoundError:
            pass
        env = "Linux"  # 순수 Linux
    else:
        env = "Other"  # macOS 또는 기타 운영체제

    print(env)
    return env


def user_subprocess(cmd=None, run_time=False, timeout=None, log=True, shell=True):
    line_output = []
    error_output = []
    timeout_expired = False

    if sys.platform == "win32":
        # WSL 명령으로 변환
        if not shell:
            cmd.insert(0, "wsl")
        else:
            cmd = rf"wsl {cmd}"

    if run_time:
        try:
            with subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) as process:
                while True:
                    # stdout, stderr에서 비동기적으로 읽기
                    line = process.stdout.readline()
                    if line:
                        line_output.append(line.strip())
                        cleaned_sentence = ANSI_ESCAPE.sub('', line)

                        if log:
                            print(cleaned_sentence.strip())  # 실시간 출력
                        if "OPTYPE : DROPOUT" in line:
                            if log:
                                print("operror")

                    err_line = process.stderr.readline()
                    if err_line:
                        error_output.append(err_line.strip())
                        cleaned_sentence = ANSI_ESCAPE.sub('', err_line)
                        if log:
                            print("ERROR:", cleaned_sentence.strip())

                    # 프로세스가 종료되었는지 확인
                    if process.poll() is not None and not line and not err_line:
                        break

                # 프로세스 종료 코드 체크
                process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            process.kill()
            if log:
                print("Timeout occurred, process killed.")
            error_output.append("Process terminated due to timeout.")
            timeout_expired = True

    else:
        try:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=False, timeout=timeout)

            # Decode stdout and stderr, handling encoding issues
            encoding = "utf-8"
            errors = "replace"
            if result.stdout:
                line_output.extend(result.stdout.decode(encoding, errors).splitlines())
            if result.stderr:
                error_output.extend(result.stderr.decode(encoding, errors).splitlines())

            if log:
                for line in line_output:
                    cleaned_sentence = ANSI_ESCAPE.sub('', line)
                    print(cleaned_sentence)  # 디버깅을 위해 주석 해제

                for err_line in error_output:
                    cleaned_sentence = ANSI_ESCAPE.sub('', err_line)
                    print("ERROR:", cleaned_sentence)  # 에러 메시지 구분을 위해 prefix 추가

        except subprocess.TimeoutExpired:
            if log:
                print("Timeout occurred, command terminated.")
            error_output.append("Command terminated due to timeout.")
            timeout_expired = True
        except Exception as e:
            # 기타 예외 처리 추가
            if log:
                print(f"Error occurred: {str(e)}")
            error_output.append(f"Command failed: {str(e)}")

    return line_output, error_output, timeout_expired


def Open_QMessageBox(message="", yes_b=True, no_b=True):
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Information")
    msg_box.setText(message)

    if yes_b and no_b:
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    elif yes_b and not no_b:
        msg_box.setStandardButtons(QMessageBox.Yes)
    elif not yes_b and no_b:
        msg_box.setStandardButtons(QMessageBox.No)
    else:
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

    # Always show the message box on top
    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)

    # 메시지 박스를 최상단에 표시
    answer = msg_box.exec_()

    if answer == QMessageBox.Yes:
        return True
    else:
        return False


def check_for_specific_string_in_files(directory, check_keywords):
    check_files = []  # 에러가 발견된 파일을 저장할 리스트
    context_data = {}

    # 디렉터리 내의 모든 파일 검사
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 파일인지 확인
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 파일을 줄 단위로 읽으면서 키워드 확인
                    for line in file:
                        if any(re.search(keyword, line) for keyword in check_keywords):
                            check_files.append(filename)  # 에러가 발견된 파일 추가
                            break  # 한 번 발견되면 해당 파일에 대한 검사는 종료
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return check_files, context_data


# 에러가 발생하면 해당 에러 내용까지 추출하는 구조
def upgrade_check_for_specific_string_in_files(directory, check_keywords):
    check_files = []  # 에러가 발견된 파일 목록을 저장할 리스트
    context_data = {}  # 파일별로 키워드 발견 시 해당 줄 주변 내용을 저장할 딕셔너리

    # 디렉터리 내의 모든 파일 검사
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 파일인지 확인
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()  # 파일 내용을 줄 단위로 모두 읽음

                for i, line in enumerate(lines):
                    # 키워드가 현재 줄에 포함되어 있는지 확인
                    if any(re.search(keyword, line) for keyword in check_keywords):
                        check_files.append(filename)  # 에러가 발견된 파일 추가

                        # "Traceback (most recent call last)" 라인을 위쪽으로 탐색
                        start_index = 0
                        for j in range(i, -1, -1):  # 발견된 줄부터 역방향 탐색
                            if "Traceback (most recent call last)" in lines[j]:
                                start_index = j
                                break

                        # 아래쪽으로는 발견된 줄의 다음 2줄까지 포함
                        # end_index = min(len(lines), i + 2)
                        # 아래 방향으로 "Command:"가 나오기 전까지 포함
                        end_index = i
                        for j in range(i + 1, len(lines)):
                            if "Command:" in lines[j]:
                                end_index = j
                                break
                        else:
                            end_index = len(lines)  # "Command:"가 없으면 파일 끝까지

                        # 각 라인의 끝에 줄바꿈 추가
                        context = [line + "\n" if not line.endswith("\n") else line for line in
                                   lines[start_index:end_index]]

                        # 파일 이름을 키로 사용하여 해당 내용 저장
                        if filename not in context_data:
                            context_data[filename] = []
                        context_data[filename].append(''.join(context))
                        break  # 한 번 발견되면 해당 파일에 대한 검사는 종료

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return check_files, context_data


# def X_upgrade_check_for_specific_string_in_files(directory, check_keywords):
#     check_files = []  # 에러가 발견된 파일 목록을 저장할 리스트
#     context_data = {}  # 파일별로 키워드 발견 시 해당 줄 주변 내용을 저장할 딕셔너리

#     # 디렉터리 내의 모든 파일 검사
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)

#         # 파일인지 확인
#         if os.path.isfile(file_path):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     lines = file.readlines()  # 파일 내용을 줄 단위로 모두 읽음

#                 for i, line in enumerate(lines):
#                     # 키워드가 현재 줄에 포함되어 있는지 확인
#                     if any(re.search(keyword, line) for keyword in check_keywords):
#                         check_files.append(filename)  # 에러가 발견된 파일 추가

#                         # 주변 줄(위로 4줄, 아래로 3줄) 가져오기
#                         # start_index = max(0, i - 4)
#                         # end_index = min(len(lines), i + 4)
#                         # context = lines[start_index:end_index]  # 주변 줄 추출

#                         # # 주변 줄(위로 4줄, 아래로 3줄) 가져오기
#                         start_index = max(0, i - 4)
#                         end_index = min(len(lines), i + 2)
#                         # end_index = min(len(lines), i + 4)

#                         # 각 라인의 끝에 줄바꿈 추가
#                         context = [line + "\n" if not line.endswith("\n") else line for line in
#                                    lines[start_index:end_index]]

#                         # 파일 이름을 키로 사용하여 해당 내용 저장
#                         if filename not in context_data:
#                             context_data[filename] = []
#                         context_data[filename].append(''.join(context))
#                         break  # 한 번 발견되면 해당 파일에 대한 검사는 종료

#             except Exception as e:
#                 print(f"Error reading file {file_path}: {e}")

#     return check_files, context_data


def remove_dir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def remove_alldata_files_except_specific_extension(directory, extension):
    # 주어진 디렉토리 내 모든 파일과 서브디렉토리 확인
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)

            if name.endswith('.caffemodel') or name.endswith('.prototxt'):  # or name.endswith('.protobin'):
                continue

            elif name.endswith(f'.{extension}'):
                continue

            try:
                os.remove(file_path)
                print(f"삭제됨: {file_path}")
            except Exception as e:
                print(f"파일 삭제 실패: {file_path}, 이유: {e}")

                # if not name.endswith(f'.{extension}'):
            #     file_path = os.path.join(root, name)
            #     os.remove(file_path)  # 파일 삭제

        for name in dirs:
            # 서브디렉토리는 파일이 모두 삭제된 후에 삭제
            dir_path = os.path.join(root, name)
            # print(f"Deleting directory: {dir_path}")
            shutil.rmtree(dir_path)  # 디렉토리 삭제


class ColonLineHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 파란색 형식 (':'로 끝나는 줄)
        self.colon_format = QTextCharFormat()
        self.colon_format.setForeground(QColor("blue"))

        # 빨간색 형식 (특정 키워드)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("red"))

        # 녹색 형식 (특정 키워드와 ':'로 끝나는 줄)
        self.green_format = QTextCharFormat()
        self.green_format.setForeground(QColor("#00CED1"))

        # 강조할 키워드 설정
        self.keywords = re.compile(r'\b(global_config|model_config)\b')

        # tflite, onnx, caffemodel 키워드 설정
        self.special_keywords = re.compile(r'\b(.tflite|.onnx|.caffemodel)\b')

    def highlightBlock(self, text):
        # 'tflite', 'onnx', 'caffemodel'이 포함되고 ':'로 끝나는 경우 녹색으로 강조
        if self.special_keywords.search(text) and text.strip().endswith(':'):
            self.setFormat(0, len(text), self.green_format)

        # 키워드가 포함된 경우 빨간색 강조
        for match in self.keywords.finditer(text):
            start, end = match.span()
            self.setFormat(start, end - start, self.keyword_format)

        # ':'로 끝나는 줄에 파란색 적용 (키워드가 포함되지 않은 경우)
        if text.strip().endswith(':') and not self.keywords.search(text) and not self.special_keywords.search(text):
            self.setFormat(0, len(text), self.colon_format)


# def find_paired_files_1(directory):
#     """
#     주어진 디렉토리 하위의 'input' 폴더에서 파일 쌍을 검색하여 경로를 튜플로 저장합니다.

#     Args:
#         directory (str): 검색할 디렉토리 경로.

#     Returns:
#         list: 파일 경로 튜플의 리스트.
#               동일 이름의 파일 쌍을 찾아 튜플로 저장
#     """
#     paired_files = []

#     # 'input' 폴더를 찾기
#     for root, dirs, files in os.walk(directory):
#         if 'inout' in dirs:
#             input_folder = os.path.join(root, 'inout').replace("\\", "/")
#             input_files = os.listdir(input_folder)

#             # 파일 필터링 및 쌍(pair) 확인
#             for file in input_files:
#                 if "input_data" in file and file.endswith(".bin"):
#                     # golden 파일 이름 생성
#                     golden_file = file.replace("input_data", "golden_data")
#                     golden_path = os.path.join(input_folder, golden_file).replace("\\", "/")

#                     # 두 파일이 모두 존재하면 튜플로 저장
#                     if golden_file in input_files:
#                         input_path = os.path.join(input_folder, file).replace("\\", "/")
#                         paired_files.append((input_path, golden_path))
#             break  # 첫 번째 'input' 폴더만 처리 후 종료

#     return paired_files


# def find_paired_files_2(directory):
#     """
#     주어진 디렉토리 하위의 'inout' 폴더에서 파일 쌍을 검색하여 경로를 리스트에 저장하고,
#     가능한 모든 조합을 paired_files에 추가합니다.

#     Args:
#         directory (str): 검색할 디렉토리 경로.

#     Returns:
#         list: 가능한 모든 조합의 파일 쌍 (input_binary, golden_binary).
#     """
#     input_binary = []
#     golden_binary = []
#     paired_files = []

#     # 'inout' 폴더를 찾기
#     for root, dirs, files in os.walk(directory):
#         if 'inout' in dirs:
#             input_folder = os.path.join(root, 'inout').replace("\\", "/")
#             input_files = os.listdir(input_folder)

#             # 파일 필터링 및 경로 저장
#             for file in input_files:
#                 full_path = os.path.join(input_folder, file).replace("\\", "/")
#                 if "input_data" in file and file.endswith(".bin"):
#                     input_binary.append(full_path)
#                 elif "golden_data" in file and file.endswith(".bin"):
#                     golden_binary.append(full_path)

#             break  # 첫 번째 'inout' 폴더만 처리 후 종료

#     # 두 리스트에 원소가 모두 1개 이상 있는 경우에만 조합 생성
#     if input_binary and golden_binary:
#         paired_files = list(itertools.product(input_binary, golden_binary))

#     return paired_files

def find_paired_files(directory, mode=2):
    input_binary = []
    golden_binary = []
    paired_files = []

    for root, dirs, files in os.walk(directory):
        if 'inout' in dirs:
            input_folder = os.path.join(root, 'inout').replace("\\", "/")
            input_files = os.listdir(input_folder)

            for file in input_files:
                full_path = os.path.join(input_folder, file).replace("\\", "/")
                if "input_data" in file and file.endswith(".bin"):
                    input_binary.append(full_path)
                elif "golden_data" in file and file.endswith(".bin"):
                    golden_binary.append(full_path)

            break

    if mode == 1:
        for input_file in input_binary:
            golden_file = input_file.replace("input_data", "golden_data")
            if golden_file in golden_binary:
                paired_files.append((input_file, golden_file))
    else:
        paired_files = list(itertools.product(input_binary, golden_binary))

    return paired_files


def upgrade_find_paired_files(directory):
    paired_files = {}
    inout_dir = None

    # `_fp` 여부에 따라 파일 분리 및 `input`과 `golden` 분리
    def categorize_files(file_list):
        input_files = [f for f in file_list if f.startswith("input_data")]
        golden_files = [f for f in file_list if f.startswith("golden_data")]
        return input_files[0], golden_files

    for root, dirs, files in os.walk(directory):
        if 'inout' in dirs:
            inout_dir = os.path.join(root, 'inout').replace("\\", "/")

            fp_files = []
            non_fp_files = []

            for filename in os.listdir(inout_dir):
                if filename.endswith(".bin"):  # `.bin` 확장자만 처리
                    (fp_files if "_fp" in filename else non_fp_files).append(filename)

            non_fp_input_files, non_fp_golden_files = categorize_files(non_fp_files)
            fp_input_files, fp_golden_files = categorize_files(fp_files)

            paired_files = {non_fp_input_files: sorted(non_fp_golden_files), fp_input_files: sorted(fp_golden_files)}
            break

    return paired_files, inout_dir


def separate_folders_and_files(directory_path):
    directory, file_name = os.path.split(directory_path)

    return directory, file_name


def separate_filename_and_extension(filename):
    name, extension = os.path.splitext(filename)

    return name, extension.replace(".", "")
