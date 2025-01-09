"""
    # 참고
    # cmd = "docker rm -f $(docker ps -aq)" # delete all container
    # user_subprocess(cmd=f"{cmd}", run_time=False, log=False)
     
    # user_subprocess(cmd=f"docker stop {ContainerName}", run_time=False, log=False)
    # user_subprocess(cmd=f"docker rm {ContainerName}", run_time=False, log=False)

    WSL에서 직접 실행 시
    docker rm -f $(docker ps -aq)

    docker run -d --name enntools_rel_7_11_22_21_container_1 --security-opt seccomp:unconfined --cap-add=ALL --privileged --net host --ipc host -v /mnt/c/Work/tom/python_project/ai_studio_verifier/ai_studio_2_x/test_model_repo/mobilenetv2-7:/workspace -v /etc/timezone:/etc/timezone -w /workspace/ ubuntu-22.04/enntools-rel:7.11.22.21 /bin/bash -c "tail -f /dev/null"

    docker exec -it enntools_rel_7_11_22_21_container_1 /bin/bash -c "cd /workspace/mobilenetv2-7 && enntools init"

    docker exec -it enntools_rel_7_11_22_21_container_1 /bin/bash -c "cd /workspace/mobilenetv2-7 && enntools conversion"

    docker exec -it enntools_rel_7_11_22_21_container_1 /bin/bash -c "cd /workspace/mobilenetv2-7 && enntools compile"
"""

import os
import time
import subprocess
import re
import uuid

ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

ERR_KEYWORD= ["Error Code:", "Error code:", "Error msg:"]

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

                        # 에러 위치 기준(위로 4줄, 아래로 3줄) 가져오기
                        start_index = max(0, i - 4)
                        end_index = min(len(lines), i + 2)                        

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

def user_subprocess(cmd='', shell=False, capture_output=True, text=False, timeout=1000, log_print=True):
    line_output = []
    error_output = []
    timeout_expired = False

    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=capture_output, text=text, timeout=timeout
        )

        # 실시간으로 stdout과 stderr 처리
        encoding = "utf-8"
        errors = "replace"

        if result.stdout:
            # stdout을 한 줄씩 처리
            for line in result.stdout.decode(encoding, errors).splitlines():
                line_output.append(line)
                if log_print:
                    print(line)

        if result.stderr:
            # stderr을 한 줄씩 처리
            for line in result.stderr.decode(encoding, errors).splitlines():
                error_output.append(line)
                if log_print:
                    print("ERROR:", line)

    except subprocess.TimeoutExpired:        
        error_output.append("Command terminated due to timeout.")
        timeout_expired = True
    except Exception as e:
        if log_print:
            print(f"Error occurred: {str(e)}")
        error_output.append(f"Command failed: {str(e)}")

    return line_output, error_output, timeout_expired


if __name__ == "__main__":

    ContainerName = f"container_{uuid.uuid4().hex}"  # 고유값 필요요
    
    DockerImg = "ubuntu-22.04/enntools-rel:7.11.22.21"
    Shared_Volume = f"/mnt/c/Work/tom/python_project/ai_studio_verifier/ai_studio_2_x/test_model_repo/test_model"
    Model = "yolov5s"    
    # Model = "efficientnet-lite4-11-int8"

    CMD = [
        "docker",
        "run",
        "-d",
        "--name",
        ContainerName,
        "--security-opt", "seccomp:unconfined",
        "--cap-add=ALL",
        "--privileged",
        "--net", "host",
        "--ipc", "host",
        "-v", f"{Shared_Volume}:/workspace",
        "-v", "/etc/timezone:/etc/timezone",
        "-w", "/workspace",
        DockerImg,
        "/bin/bash",
        "-c",
        "tail -f /dev/null"
    ]

    out, error, timeout = user_subprocess(cmd=CMD, shell=False, capture_output=True, text=False, timeout=1000, log_print=True)    
    
    # test mobilenetv2-7가 있다고 했을 때때
    C_List = ["enntools init", "enntools conversion", "enntools compile"]

    TestResult = {
        "enntools init": "SKIP",
        "enntools conversion": "SKIP",
        "enntools compile": "SKIP",
        "timeout": ""
    }

    for enntools_cmd in C_List:
        CMD = [
            "docker",
            "exec",
            "-it",
            ContainerName, 
            "/bin/bash",
            "-c", 
            f"cd /workspace/{Model} && {enntools_cmd}"
            ]
        
        out, error, timeout = user_subprocess(cmd=CMD, shell=False, capture_output=True, text=False, timeout=1000, log_print=True)

        if timeout:
            TestResult["timeout"] = "TimeoutExpired"
            break

        if enntools_cmd == "enntools init":
            # init이 성공적으로 되었는지 확인하는 코드 추가
            CheckDataDir = os.path.join(Shared_Volume, Model, "DATA")
            Generated_Yaml_File = os.path.join(Shared_Volume, Model, f"{Model}.yaml")

            if os.path.isdir(CheckDataDir) and os.path.isfile(Generated_Yaml_File):
                TestResult[enntools_cmd] = "Success"
            else:
                TestResult[enntools_cmd] = "Fail"
                break

        elif enntools_cmd == "enntools conversion":
            # conversion이 성공적으로 되었는지 확인하는 코드 추가
            conversion_log_path = os.path.join(Shared_Volume, Model, "Converter_result", ".log")
            ret, error_contents_dict = upgrade_check_for_specific_string_in_files(conversion_log_path, check_keywords=ERR_KEYWORD)

            if len(ret) == 0:
                TestResult[enntools_cmd] = "Success"
            else:
                TestResult[enntools_cmd] = "Fail"
                break            

        if enntools_cmd == "enntools compile":
            # compile이 성공적으로 되었는지 확인하는 코드 추가가
            conpile_log_path = os.path.join(Shared_Volume, Model, "Compiler_result", ".log")
            ret, error_contents_dict = upgrade_check_for_specific_string_in_files(conpile_log_path, check_keywords=ERR_KEYWORD)

            if len(ret) == 0:
                TestResult[enntools_cmd] = "Success"
            else:
                TestResult[enntools_cmd] = "Fail"
                break
    
    print("===============================================================================================================\n")
    for cmd, result in TestResult.items():
        print(f"{cmd}: {result}")


