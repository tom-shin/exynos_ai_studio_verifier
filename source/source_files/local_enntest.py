import subprocess
import re
import paramiko
import os

DeviceTargetDir = "/data/vendor/enn"
ProfileCMD = "EnnTest_v2_lib"
ProfileOption = "--profile summary --monitor_iter 1 --iter 1 --useSNR"
ANSI_Escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


def remote_server():
    def push_file_to_android_on_remote_server(
            remote_host, remote_port, remote_user, remote_password,
            local_file, remote_temp_file, android_device_path, remote_device
    ):
        try:
            # 리모트 서버에 SSH 연결
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(remote_host, username=remote_user, password=remote_password, port=remote_port)

            # SFTP를 사용해 로컬 파일을 리모트 서버로 전송
            sftp = ssh.open_sftp()
            sftp.put(local_file, remote_temp_file)
            sftp.close()

            print(f"File uploaded to remote server: {remote_temp_file}")

            # 리모트 서버에서 adb push 명령어 실행
            command = f"/home/sam/platform-tools/adb -s {remote_device} push {remote_temp_file} {android_device_path}"
            stdin, stdout, stderr = ssh.exec_command(command)

            # 결과 출력
            output = stdout.read().decode()
            error = stderr.read().decode()

            if output:
                print(f"Output: {output}")
            if error:
                print(f"Error: {error}")

            # remote_temp_file에서 디렉토리 경로만 추출
            remote_temp_dir = os.path.dirname(remote_temp_file)

            # 디렉토리 내 모든 파일 삭제 명령어
            remove_command = f"rm -rf {remote_temp_dir}/*"  # temp 디렉토리 내 모든 파일 삭제
            stdin, stdout, stderr = ssh.exec_command(remove_command)

            # 삭제 결과 출력
            remove_output = stdout.read().decode()
            remove_error = stderr.read().decode()

            if remove_output:
                print(f"Removed files in temp folder: {remove_output}")
            if remove_error:
                print(f"Error removing files: {remove_error}")

            ssh.close()

        except Exception as e:
            print(f"Error: {e}")

    # 예시 사용
    remote_host = '1.220.53.154'
    remote_port = 63522
    remote_user = 'sam'
    remote_password = 'Thunder$@88'
    remote_device = '0000100d8e38c0e0'
    local_file = r"C:\Work\tom\python_project\gpt_key.txt"
    remote_temp_file = "/home/sam/tom/temp/gpt_key.txt"  # 리모트 서버의 임시 경로
    android_device_path = 'data/vendor/enn'

    push_file_to_android_on_remote_server(
        remote_host, remote_port, remote_user, remote_password,
        local_file, remote_temp_file, android_device_path, remote_device
    )


def main(DeviceId=None, NNC_Model=None, InputBinary=None, GoldenBinary=None, SaveOutput=None):

    # 혹시 윈도우 스타일로 경로되어 있다면 리눅스 스타일로 경로 변경
    Binary = [NNC_Model.replace("\\", "/"), InputBinary.replace("\\", "/"), GoldenBinary.replace("\\", "/")]

    # Device 권한 설정
    authority = [f"adb -s {DeviceId} root", f"adb -s {DeviceId} remount"]
    for cmd in authority:
        subprocess.run(cmd)

    # Binary push
    for b_file in Binary:
        subprocess.run(rf"adb -s {DeviceId} push {b_file} .{DeviceTargetDir}")

    # Profile 시작
    # 윈도우 경로와 리눅스에서 실행하는 경로 호환성
    model = os.path.join(f'{DeviceTargetDir}', os.path.basename(NNC_Model)).replace("\\", "/")
    input_binary = os.path.join(f'{DeviceTargetDir}', os.path.basename(InputBinary)).replace("\\", "/")
    golden_binary = os.path.join(f'{DeviceTargetDir}', os.path.basename(GoldenBinary)).replace("\\", "/")
    
    execute_cmd = [
        "adb", "-s", DeviceId, "shell",
        f"{ProfileCMD}",
        "--model", model,
        "--input", input_binary,
        "--golden", golden_binary,
        f"{ProfileOption}"
    ]

    result = subprocess.run(execute_cmd, capture_output=True, text=True)

    # 출력값을 파일로 저장
    SaveOutput = SaveOutput.replace("\\", "/")
    with open(f"{SaveOutput}", "w", encoding="utf-8") as f:
        cleaned_result = ANSI_Escape.sub('', result.stdout)
        f.write(cleaned_result)

    print(f"Saved: {SaveOutput}")


if __name__ == "__main__":
    # User 설정 값
    DeviceId = "0000100d0f246013"
    NNC_Model = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\webportalResult\midas_v2\Compiler_result\midas_v2_simplify_O2_SingleCore.nnc"
    InputBinary = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\webportalResult\midas_v2\Converter_result\NPU_midas_v2\testvector\inout\input_data_float32_fp32.bin"
    GoldenBinary = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\webportalResult\midas_v2\Converter_result\NPU_midas_v2\testvector\inout\golden_data_float32_fp32.bin"
    SaveOutput = os.path.join(r"C:\Work\tom\python_project\exynos_ai_studio_verifier\webportalResult\midas_v2\Compiler_result", "result_enntest.txt")

    main(DeviceId=DeviceId, NNC_Model=NNC_Model, InputBinary=InputBinary, GoldenBinary=GoldenBinary,
         SaveOutput=SaveOutput)
