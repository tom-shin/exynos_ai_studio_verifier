import subprocess
import re
import os

DeviceTargetDir = "/data/vendor/enn"
ProfileCMD = "EnnTest_v2_lib"
ProfileOption = "--profile summary --monitor_iter 1 --iter 1 --useSNR"
ANSI_Escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


def main(DeviceId=None, NNC_Model=None, InputBinary=None, GoldenBinary=None, SaveOutput=None):
    Binary = [NNC_Model.replace("\\", "/"), InputBinary.replace("\\", "/"), GoldenBinary.replace("\\", "/")]

    # Device 권한 설정
    authority = [f"adb -s {DeviceId} root", f"adb -s {DeviceId} remount"]
    for cmd in authority:
        subprocess.run(cmd)

    # Binary push
    for b_file in Binary:
        subprocess.run(rf"adb -s {DeviceId} push {b_file} .{DeviceTargetDir}")

    # Profile 시작
    execute_cmd = [
        "adb", "-s", DeviceId, "shell",
        f"{ProfileCMD}",
        "--model", f"{os.path.join(f'{DeviceTargetDir}', os.path.basename(NNC_Model))}",
        "--input", f"{os.path.join(f'{DeviceTargetDir}', os.path.basename(InputBinary))}",
        "--golden", f"{os.path.join(f'{DeviceTargetDir}', os.path.basename(GoldenBinary))}",
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
    NNC_Model = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\v2_license_Result\squeezenet1.0-12-dropout_softmax_removed\Compiler_result\squeezenet1.0-12-dropout_softmax_removed_simplify_O2_SingleCore.nnc"
    InputBinary = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\v2_license_Result\squeezenet1.0-12-dropout_softmax_removed\Converter_result\NPU_squeezenet1.0-12-dropout_softmax_removed\testvector\inout\input_data_float32.bin"
    GoldenBinary = rf"C:\Work\tom\python_project\exynos_ai_studio_verifier\v2_license_Result\squeezenet1.0-12-dropout_softmax_removed\Converter_result\NPU_squeezenet1.0-12-dropout_softmax_removed\testvector\inout\golden_data_float32.bin"
    SaveOutput = os.path.join(os.getcwd(), "result_enntest.txt")

    main(DeviceId=DeviceId, NNC_Model=NNC_Model, InputBinary=InputBinary, GoldenBinary=GoldenBinary,
         SaveOutput=SaveOutput)
