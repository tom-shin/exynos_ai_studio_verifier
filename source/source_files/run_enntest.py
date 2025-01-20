############################################################################################################

import os
import time
import paramiko
import re
from tqdm import tqdm
import logging
import socket
import stat
import multiprocessing
import shutil
from colorama import Fore, Back, Style, init
import subprocess

############################################################################################################

ANSI_Escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')  # ANSI 이스케이프 코드 제거용 정규식


class remote_ssh_server:
    def __init__(self, deviceID):
        self.remote_host = '1.220.53.154'
        self.remote_port = 63522
        self.remote_user = 'sam'
        self.remote_password = 'Thunder$@88'
        self.remote_device = deviceID  # '0000100d8e38c0e0'
        self.remote_temp_dir = '/home/sam/tom/temp'
        self.android_device_path = '/data/vendor/enn'
        self.remote_adb_path = '/home/sam/platform-tools/adb'

        self.ProfileCMD = "EnnTest_v2_lib"
        self.ProfileOption = "--profile summary --monitor_iter 1 --iter 1 --useSNR"

        self.ssh = None
        self.error_log = None

    def user_ssh_exec_command(self, command, print_log=True):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        stdout.channel.recv_exit_status()  # Wait for the command to complete

        # 결과 출력
        output = stdout.read().decode()
        error = stderr.read().decode()

        if print_log:
            if output:
                print(f"Output: {output}")
            if error:
                print(f"Error: {error}")

        return output, error

    def check_enn_directory_exist(self):
        # 디바이스 경로 확인 및 처리
        check_path_cmd = f"{self.remote_adb_path} -s {self.remote_device} shell ls {self.android_device_path}"
        output, error = self.user_ssh_exec_command(command=check_path_cmd)

        if "No such file or directory" in error:
            print(f"Path {self.android_device_path} does not exist. Creating directory...")

            # 경로 생성
            create_dir_cmd = f"{self.remote_adb_path} -s {self.remote_device} shell mkdir -p {self.android_device_path}"
            _, _ = self.user_ssh_exec_command(command=create_dir_cmd)
        else:
            print(f"Path {self.android_device_path} exists. Clearing contents...")

            # 디렉토리 내용 삭제
            clear_dir_cmd = f"{self.remote_adb_path} -s {self.remote_device} shell rm -rf {self.android_device_path}/*"
            _, _ = self.user_ssh_exec_command(command=clear_dir_cmd)

    def check_ssh_connection(self):
        if self.ssh is not None:
            self.ssh_close()

        try:
            # SSH 연결 시도
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(self.remote_host, username=self.remote_user, password=self.remote_password,
                             port=self.remote_port)

            commands = [rf"{self.remote_adb_path} -s {self.remote_device} root",
                        f"{self.remote_adb_path} -s {self.remote_device} remount"]
            for command in commands:
                _, _ = self.user_ssh_exec_command(command=command)

            self.check_enn_directory_exist()

            return True, self.error_log  # 연결 성공 시 True 반환

        except paramiko.AuthenticationException:
            # 인증 실패 시
            self.error_log = "Authentication failed, please check your credentials."
            return False, self.error_log  # 연결 실패 시 False 반환

        except paramiko.SSHException as ssh_error:
            # SSH 연결 실패 시
            self.error_log = f"SSH connection failed: {ssh_error}"
            return False, self.error_log  # 연결 실패 시 False 반환

        except Exception as e:
            # 기타 오류 처리
            self.error_log = f"Error: {e}"
            return False, self.error_log  # 연결 실패 시 False 반환

    def push_file_to_android_on_remote_server(self, f_local_file):
        remote_temp_file = os.path.join(self.remote_temp_dir, os.path.basename(f_local_file)).replace("\\", "/")

        # SFTP를 사용해 로컬 파일을 리모트 서버로 전송
        sftp = self.ssh.open_sftp()

        file_size = os.path.getsize(f_local_file)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Uploading') as pbar:
            def callback(transferred, total):
                pbar.update(transferred - pbar.n)

            sftp.put(f_local_file, remote_temp_file.replace("\\", "/"), callback=callback)

        sftp.close()

        # 리모트 서버에서 adb push 명령어 실행
        command = f"{self.remote_adb_path} -s {self.remote_device} push {remote_temp_file} {self.android_device_path}"
        _, _ = self.user_ssh_exec_command(command=command)

    def clear_remote_temp_dir(self):
        """remote_temp_dir에 있는 모든 파일 삭제"""
        try:
            print(f"Clearing files in remote temp directory: {self.remote_temp_dir}")
            # 파일 삭제 명령
            delete_command = f"rm -rf {self.remote_temp_dir}/*"
            output, error = self.user_ssh_exec_command(command=delete_command)

            if error:
                print(f"Error clearing remote temp directory: {error}")
            else:
                print(f"Remote temp directory cleared successfully.")

        except Exception as e:
            print(f"Error while clearing remote temp directory: {e}")

    def ssh_close(self):
        self.clear_remote_temp_dir()
        self.check_enn_directory_exist()
        self.ssh.close()
        self.ssh = None


def upgrade_remote_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board, deviceID):
    failed_pairs = []
    CHECK_ENNTEST = []

    instance = remote_ssh_server(deviceID=deviceID)
    ret, error = instance.check_ssh_connection()

    if not ret:
        return False, failed_pairs.append(error)

    NNC_Model = nnc_files[0]
    arg_nnc_model = os.path.join(instance.android_device_path, os.path.basename(NNC_Model)).replace("\\", "/")
    instance.push_file_to_android_on_remote_server(f_local_file=NNC_Model)

    for cnt, (input_b, golden_bins) in enumerate(input_golden_pairs.items()):
        full_input_b = os.path.join(current_binary_pos, input_b).replace("\\", "/")
        arg_input_b = os.path.join(instance.android_device_path, input_b).replace("\\", "/")
        instance.push_file_to_android_on_remote_server(f_local_file=full_input_b)

        arg_golden_bin = ''
        for golden_b in golden_bins:
            full_golden_bin = os.path.join(current_binary_pos, golden_b).replace("\\", "/")
            golden_path = os.path.join(instance.android_device_path, golden_b).replace("\\", "/")
            arg_golden_bin += f'{golden_path} '

            instance.push_file_to_android_on_remote_server(f_local_file=full_golden_bin)

        execute_cmd = [
            f"{instance.remote_adb_path}", *(["-s", instance.remote_device] if instance.remote_device else []), "shell",
            instance.ProfileCMD,
            "--model", arg_nnc_model.rstrip(),
            "--input", arg_input_b.rstrip(),
            "--golden", arg_golden_bin.rstrip(),
            instance.ProfileOption
        ]
        command_str = " ".join(execute_cmd)
        result, error = instance.user_ssh_exec_command(command=command_str, print_log=False)

        # 출력값을 파일로 저장
        filename = f"result_enntest_{cnt}.txt"
        SaveOutput = os.path.join(out_dir, filename).replace('\\', '/')

        with open(SaveOutput, "w", encoding="utf-8") as f:
            cleaned_result = ANSI_Escape.sub('', result)  # ANSI 이스케이프 코드 제거
            f.write(cleaned_result)
        print(cleaned_result)

        # 결과 확인
        if "PASSED" in cleaned_result.split("\n")[-2]:
            CHECK_ENNTEST.append(True)
            success_logs = cleaned_result.split("\n")
            for logs in success_logs:
                if "total execution time" in logs.lower() or "measured snr" in logs.lower():
                    failed_pairs.append(logs+"\n")
        else:
            CHECK_ENNTEST.append(False)
            failed_pairs.append(cleaned_result)
            break

    instance.ssh_close()

    if all(CHECK_ENNTEST):  # 모든 값이 True인 경우
        return True, failed_pairs  # 실패한 쌍도 반환
    else:
        return False, failed_pairs  # 실패한 쌍도 반환


def upgrade_local_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board, deviceID):
    def check_enn_directory_exist(DeviceId, android_device_path):
        # Android device path 존재 여부 확인
        check_path_cmd = f"adb {'-s ' + DeviceId if DeviceId else ''} shell ls {android_device_path}"

        try:
            result = subprocess.run(
                check_path_cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            # 경로가 존재하는 경우
            if "No such file or directory" not in result.stderr:
                print(f"Path {android_device_path} exists. Clearing contents...")

                # 해당 경로의 모든 파일 및 폴더 삭제
                clear_cmd = f"adb {'-s ' + DeviceId if DeviceId else ''} shell rm -rf {android_device_path}/*"
                subprocess.run(clear_cmd, shell=True)
            else:
                print(f"Path {android_device_path} does not exist. Creating directory...")

                # 경로 생성
                create_dir_cmd = f"adb {'-s ' + DeviceId if DeviceId else ''} shell mkdir -p {android_device_path}"
                subprocess.run(create_dir_cmd, shell=True)

        except Exception as e:
            print(f"Error during path check or modification: {e}")

    instance = remote_ssh_server(deviceID=deviceID)

    DeviceTargetDir = instance.android_device_path
    ProfileCMD = instance.ProfileCMD
    ProfileOption = instance.ProfileOption
    DeviceId = instance.remote_device  # "0000100d0f246013"

    # Device 권한 설정
    if DeviceId is None:
        auth = [f"adb root", f"adb remount"]
    else:
        auth = [f"adb -s {DeviceId} root", f"adb -s {DeviceId} remount"]

    for cmd in auth:
        subprocess.run(cmd, shell=True)

    check_enn_directory_exist(DeviceId=DeviceId, android_device_path=DeviceTargetDir)

    # Model Binary push
    NNC_Model = nnc_files[0]
    arg_nnc_model = os.path.join(DeviceTargetDir, os.path.basename(NNC_Model)).replace("\\", "/")

    if DeviceId is None:
        subprocess.run(rf"adb push {NNC_Model} .{DeviceTargetDir}", shell=True)
    else:
        subprocess.run(rf"adb -s {DeviceId} push {NNC_Model} .{DeviceTargetDir}", shell=True)

    # print(input_golden_pairs)
    CHECK_ENNTEST = []
    failed_pairs = []  # 실패한 파일 쌍을 저장할 리스트

    for cnt, (input_b, golden_bins) in enumerate(input_golden_pairs.items()):
        full_input_b = os.path.join(current_binary_pos, input_b).replace("\\", "/")
        arg_input_b = os.path.join(DeviceTargetDir, input_b).replace("\\", "/")

        # tqdm으로 input_b 파일 전송 진행 표시
        input_file_size = os.path.getsize(full_input_b)
        with tqdm(total=input_file_size, desc=f"Pushing {input_b}", unit="B", unit_scale=True) as pbar:
            if DeviceId is None:
                process = subprocess.Popen(rf"adb push {full_input_b} .{DeviceTargetDir}", shell=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                process = subprocess.Popen(rf"adb -s {DeviceId} push {full_input_b} .{DeviceTargetDir}", shell=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 진행 상황 업데이트
            for line in process.stdout:
                pbar.update(len(line))

        arg_golden_bin = ''

        for golden_b in tqdm(golden_bins, desc=f"Pushing golden files for {input_b}", unit="file"):
            full_golden_bin = os.path.join(current_binary_pos, golden_b).replace("\\", "/")
            golden_path = os.path.join(DeviceTargetDir, golden_b).replace("\\", "/")
            arg_golden_bin += f'{golden_path} '

            # tqdm으로 각 golden_b 파일 전송 진행 표시
            golden_file_size = os.path.getsize(full_golden_bin)
            with tqdm(total=golden_file_size, desc=f"Pushing {golden_b}", unit="B", unit_scale=True) as pbar:
                if DeviceId is None:
                    process = subprocess.Popen(rf"adb push {full_golden_bin} .{DeviceTargetDir}", shell=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    process = subprocess.Popen(rf"adb -s {DeviceId} push {full_golden_bin} .{DeviceTargetDir}",
                                               shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # 진행 상황 업데이트
                for line in process.stdout:
                    pbar.update(len(line))

        execute_cmd = [
            "adb", *(["-s", DeviceId] if DeviceId else []), "shell",
            ProfileCMD,
            "--model", arg_nnc_model.rstrip(),
            "--input", arg_input_b.rstrip(),
            "--golden", arg_golden_bin.rstrip(),
            ProfileOption
        ]
        result = subprocess.run(execute_cmd, capture_output=True, text=True, shell=False)

        # 출력값을 파일로 저장
        filename = f"result_enntest_{cnt}.txt"
        SaveOutput = os.path.join(out_dir, filename).replace('\\', '/')

        with open(SaveOutput, "w", encoding="utf-8") as f:
            cleaned_result = ANSI_Escape.sub('', result.stdout)  # ANSI 이스케이프 코드 제거
            f.write(cleaned_result)
        print(cleaned_result)

        # 결과 확인
        if "PASSED" in cleaned_result.split("\n")[-2]:
            CHECK_ENNTEST.append(True)
            success_logs = cleaned_result.split("\n")
            for logs in success_logs:
                if "total execution time" in logs.lower() or "measured snr" in logs.lower():
                    failed_pairs.append(logs+"\n")
        else:
            CHECK_ENNTEST.append(False)
            failed_pairs.append(cleaned_result)
            break

    check_enn_directory_exist(DeviceId=DeviceId, android_device_path=DeviceTargetDir)

    if all(CHECK_ENNTEST):  # 모든 값이 True인 경우
        return True, failed_pairs  # 실패한 쌍도 반환
    else:
        return False, failed_pairs  # 실패한 쌍도 반환


# 함수 실행 예시
if __name__ == "__main__":
    target_board = ''

    out_dir = os.getcwd()
    current_binary_pos = rf'C:\Work\tom\python_project\AI_MODEL_Rep\Test_Result\Result_v2_20250108_webportalResult\zero_dce_lite_160x160_iter8_30_dynamic2static\Converter_result\NPU_zero_dce_lite_160x160_iter8_30_dynamic2static\testvector\inout'

    nnc_files = [
        rf"C:\Work\tom\python_project\AI_MODEL_Rep\Test_Result\Result_v2_20250108_webportalResult\zero_dce_lite_160x160_iter8_30_dynamic2static\Compiler_result\zero_dce_lite_160x160_iter8_30_dynamic2static_simplify_O2_SingleCore.nnc"
    ]

    input_golden_pairs = {
        "input_data_float32.bin": ['golden_data_float320.bin', 'golden_data_float321.bin'],
        "input_data_float32_fp32.bin": ['golden_data_float320_fp32.bin', 'golden_data_float321_fp32.bin']

    }

    # upgrade_local_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board)
    upgrade_remote_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board)
