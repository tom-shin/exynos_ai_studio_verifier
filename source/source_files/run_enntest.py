############################################################################################################

import os
import time
import paramiko
import re

from PyQt5.QtCore import QThread, pyqtSignal
from tqdm import tqdm
import logging
import socket
import stat
import multiprocessing
import shutil
from colorama import Fore, Back, Style, init
import subprocess
import numpy as np
from source.__init__ import check_environment

############################################################################################################

ANSI_Escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')  # ANSI 이스케이프 코드 제거용 정규식


class MemoryTracing(QThread):
    interval = 0.01
    encoding = "utf-8"
    errors = "replace"
    send_memory_profile_sig = pyqtSignal(list)

    def __init__(self, use_local_device=False, ssh_instance=None, deviceID=None):
        super().__init__()
        self.running = True
        self.memory_profile = []
        self.use_local_device = use_local_device
        self.cmd = None
        self.ssh_instance = ssh_instance
        self.deviceID = deviceID

        self.set_airplane_mode(enable=True)
        self.memory_initialization()

        # /home/sam/platform-tools/adb -s 000011344eac6013 shell dumpsys meminfo EnnTest_v2_lib | grep "TOTAL PSS" | awk '{print $3}'
        env = check_environment()
        if not self.use_local_device:
            self.cmd = f"{self.ssh_instance.remote_adb_path} {'-s ' + self.deviceID if self.deviceID else ''} shell dumpsys meminfo {self.ssh_instance.ProfileCMD}"
        else:
            self.cmd = f"adb {'-s ' + self.deviceID if self.deviceID else ''} shell dumpsys meminfo {self.ssh_instance.ProfileCMD}"

    def memory_initialization(self):
        device_id = self.deviceID
        app_package = self.ssh_instance.ProfileCMD

        # -s 옵션에 device_id를 추가할지 여부 결정
        device_option = f"-s {device_id}" if device_id else ""  # device_id가 있으면 -s 옵션 추가, 없으면 생략

        if self.use_local_device:
            adb_path = "adb"

            try:
                # 모든 background 앱 종료
                subprocess.run([adb_path, device_option, "shell", "am", "kill-all"], check=True)

                # 앱 캐시 초기화
                try:
                    subprocess.run([adb_path, device_option, "shell", "pm", "clear", app_package], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error while clearing app cache: {e}")

                # 캐시 초기화 (루트 권한 필요 시)
                subprocess.run(
                    [adb_path, device_option, "shell", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                    check=True)

                print(
                    f"Memory and cache cleared on device {device_id} for app {app_package}" if device_id else f"Memory and cache cleared for app {app_package} (device not specified)")

            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e}")

        else:
            adb_path = self.ssh_instance.remote_adb_path

            # 명령어 리스트
            cmds = [
                f'{adb_path} {device_option} shell am kill-all',  # device_option이 비어 있으면 -s 생략
                f'{adb_path} {device_option} shell pm clear {app_package}',  # 동일하게 적용
                f'{adb_path} {device_option} shell "sync; echo 3 > /proc/sys/vm/drop_caches"'
            ]

            # 각 명령어를 반복문으로 실행
            for cmd in cmds:
                try:
                    stdout, stderr = self.ssh_instance.user_ssh_exec_command(command=cmd, print_log=False)
                    # if stderr:
                    #     print(f"SSH Command : {stderr}")
                    # else:
                    #     print(
                    #         f"Memory and cache cleared on remote device {device_id} for app {app_package}" if device_id else f"Memory and cache cleared for app {app_package} (device not specified)")

                except Exception as e:
                    print(f"SSH Error: {e}")

        print(f"Executed Memory Initialization.")

    def set_airplane_mode(self, enable=True):
        try:
            state = "1" if enable else "0"
            broadcast_state = "true" if enable else "false"

            if self.use_local_device:
                # 로컬 디바이스에서 비행기 모드 설정
                subprocess.run(
                    ["adb", "shell", "settings", "put", "global", "airplane_mode_on", state],
                    check=True
                )

                # 브로드캐스트 전송
                subprocess.run(
                    ["adb", "shell", "am", "broadcast", "-a", "android.intent.action.AIRPLANE_MODE", "--ez", "state",
                     broadcast_state],
                    check=True
                )
            else:
                # 원격 디바이스에서 비행기 모드 설정
                cmd = (
                    f"{self.ssh_instance.remote_adb_path} {'-s ' + self.deviceID if self.deviceID else ''} shell "
                    f"settings put global airplane_mode_on {state} && "
                    f"{self.ssh_instance.remote_adb_path} {'-s ' + self.deviceID if self.deviceID else ''} shell "
                    f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {broadcast_state}"
                )

                # SSH 명령 실행
                _, _ = self.ssh_instance.user_ssh_exec_command(command=cmd, print_log=False)

            print(f"Airplane mode {'enabled' if enable else 'disabled'} successfully.")

        except subprocess.CalledProcessError as e:
            print(f"Error setting airplane mode on local device: {e}")
        except Exception as e:
            print(f"Error setting airplane mode on remote device: {e}")

    def run(self):
        while self.running:
            if self.use_local_device:
                try:
                    result = subprocess.run(self.cmd, shell=True, capture_output=True, text=True, timeout=10)

                    if result.returncode == 0:
                        # Extract available memory and store in memory_profile
                        line = result.stdout.strip()
                        if line:
                            try:
                                value = int(re.search(r"TOTAL PSS:\s+(\d+)", line).group(1)) / 1024
                                self.memory_profile.append(value)
                            except:
                                self.memory_profile.append(0)

                    else:
                        print(f"Command failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print("Command timed out.")
                except Exception as e:
                    print(f"Unexpected error: {e}")

            else:
                result, error = self.ssh_instance.user_ssh_exec_command(command=self.cmd, print_log=False)
                if "No proces" in result:
                    self.memory_profile.append(0)
                    # print("No Process")
                else:
                    try:
                        value = int(re.search(r"TOTAL PSS:\s+(\d+)", result).group(1)) / 1024
                        self.memory_profile.append(value)
                        # print(value)
                    except Exception as e:
                        pass
                        # print("exception", e)

                # if result:
                #     mem_available = result.split(":")[1].strip().replace("kB", "").strip()
                #     self.memory_profile.append(int(mem_available))

            # Sleep to control frequency of sampling
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.set_airplane_mode(enable=False)
        self.send_memory_profile_sig.emit(self.memory_profile)
        self.wait(3000)


def PrintMemoryProfile(memory_profile):
    print("[Profile] Recorded memory values:")
    with open(os.path.join(os.getcwd(), "memory_trace.log"), "w") as file:
        for mem_val in memory_profile:
            file.write(f"{mem_val}\n")
            # print(f"{mem}")


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
        # self.ProfileOption = "--profile summary --monitor_iter 1 --iter 10000 --useSNR"
        self.ProfileOption = "--monitor_iter 1 --iter 10000 --useSNR"

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


def upgrade_remote_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board, deviceID=None,
                               wait_time=5):
    failed_pairs = []
    CHECK_ENNTEST = []

    instance = remote_ssh_server(deviceID=deviceID)
    ret, error = instance.check_ssh_connection()

    if not ret:
        return False, failed_pairs.append(error)

    memory_profile_instance = MemoryTracing(use_local_device=False, ssh_instance=instance, deviceID=deviceID)
    memory_profile_instance.send_memory_profile_sig.connect(PrintMemoryProfile)
    memory_profile_instance.start()
    time.sleep(wait_time)

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

        memory_profile_instance.memory_profile.append("Start")
        result, error = instance.user_ssh_exec_command(command=command_str, print_log=False)
        memory_profile_instance.memory_profile.append("End")

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
            # success_logs = cleaned_result.split("\n")
            # for logs in success_logs:
            #     if "total execution time" in logs.lower() or "measured snr" in logs.lower():
            #         failed_pairs.append(logs + "\n")
            failed_pairs = list(set(
                log.strip() + "\n"
                for log in cleaned_result.split("\n")
                if "# monitoriter:" not in log.lower() and
                any(keyword in log.lower() for keyword in
                    ["execution performance", "total execution time", "measured snr"])
            ))
        else:
            CHECK_ENNTEST.append(False)
            failed_pairs.append(cleaned_result)
            break

    time.sleep(wait_time)
    memory_profile_instance.stop()
    instance.ssh_close()

    # 출력값을 파일로 저장
    filename = f"memory_trace.log"
    SaveOutput = os.path.join(out_dir, filename).replace('\\', '/')

    with open(SaveOutput, "w", encoding="utf-8") as f:
        for mem_val in memory_profile_instance.memory_profile:
            f.write(str(f"{mem_val}\n"))

    if all(CHECK_ENNTEST):  # 모든 값이 True인 경우
        return True, failed_pairs, memory_profile_instance.memory_profile
    else:
        return False, failed_pairs, memory_profile_instance.memory_profile  # 실패한 쌍도 반환


def upgrade_local_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board, deviceID,
                              wait_time=5):
    instance = remote_ssh_server(deviceID=deviceID)

    DeviceTargetDir = instance.android_device_path
    ProfileCMD = instance.ProfileCMD
    ProfileOption = instance.ProfileOption
    DeviceId = instance.remote_device  # "0000100d0f246013"

    memory_profile_instance = MemoryTracing(use_local_device=True, ssh_instance=instance, deviceID=deviceID)
    memory_profile_instance.send_memory_profile_sig.connect(PrintMemoryProfile)
    memory_profile_instance.start()
    time.sleep(wait_time)

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

        memory_profile_instance.memory_profile.append("Start")
        result = subprocess.run(execute_cmd, capture_output=True, text=True, shell=False)
        memory_profile_instance.memory_profile.append("End")

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
            # success_logs = cleaned_result.split("\n")
            # for logs in success_logs:
            #     if "execution performance" in logs.lower() or "total execution time" in logs.lower() or "measured snr" in logs.lower():
            #         failed_pairs.append(logs + "\n")

            failed_pairs = list(set(
                log.strip() + "\n"
                for log in cleaned_result.split("\n")
                if "# monitoriter:" not in log.lower() and
                any(keyword in log.lower() for keyword in
                    ["execution performance", "total execution time", "measured snr"])
            ))
        else:
            CHECK_ENNTEST.append(False)
            failed_pairs.append(cleaned_result)
            break

    time.sleep(wait_time)
    memory_profile_instance.stop()

    check_enn_directory_exist(DeviceId=DeviceId, android_device_path=DeviceTargetDir)

    # 출력값을 파일로 저장
    filename = f"memory_trace.log"
    SaveOutput = os.path.join(out_dir, filename).replace('\\', '/')

    with open(SaveOutput, "w", encoding="utf-8") as f:
        for mem_val in memory_profile_instance.memory_profile:
            f.write(str(f"{mem_val}\n"))

    if all(CHECK_ENNTEST):  # 모든 값이 True인 경우
        return True, failed_pairs, memory_profile_instance.memory_profile
    else:
        return False, failed_pairs, memory_profile_instance.memory_profile  # 실패한 쌍도 반환


# 함수 실행 예시
if __name__ == "__main__":
    target_board = ''
    out_dir = os.getcwd()

    Test_use_remote_device = True

    if Test_use_remote_device:
        current_binary_pos = rf'C:\Work\tom\python_project\AI_MODEL_Rep\Test_Result\Result_v2_20250108_webportalResult\zero_dce_lite_160x160_iter8_30_dynamic2static\Converter_result\NPU_zero_dce_lite_160x160_iter8_30_dynamic2static\testvector\inout'

        nnc_files = [
            rf"C:\Work\tom\python_project\AI_MODEL_Rep\Test_Result\Result_v2_20250108_webportalResult\zero_dce_lite_160x160_iter8_30_dynamic2static\Compiler_result\zero_dce_lite_160x160_iter8_30_dynamic2static_simplify_O2_SingleCore.nnc"
        ]

        input_golden_pairs = {
            "input_data_float32.bin": ['golden_data_float320.bin', 'golden_data_float321.bin']
        }

        upgrade_remote_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board,
                                   deviceID="000011344eac6013")

    else:
        current_binary_pos = rf"/home/tom/work/linux_enntools/mobilenetv2-7/Converter_result/NPU_mobilenetv2-7/testvector/inout"

        nnc_files = [
            rf"/home/tom/work/linux_enntools/mobilenetv2-7/Compiler_result/mobilenetv2-7_simplify_O2_SingleCore.nnc"
        ]

        input_golden_pairs = {
            "input_data_float32.bin": ['golden_data_float32.bin']
        }

        upgrade_local_run_enntest(nnc_files, input_golden_pairs, current_binary_pos, out_dir, target_board,
                                  deviceID="0000100d0f246013")
