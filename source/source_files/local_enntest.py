import subprocess
import re
import os

DeviceTargetDir = "/data/vendor/enn"
ProfileCMD = "EnnTest_v2_lib"
ProfileOption = "--profile summary --monitor_iter 1 --iter 1 --useSNR"
ANSI_Escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


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
