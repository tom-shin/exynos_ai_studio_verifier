#!/usr/bin/env python3
import os
import onnx


class ONNXINFO:

    def __init__(self, p_model_path=None, r_op=None):
        self.p_model = p_model_path
        self.r_op = r_op

        self.model = onnx.load(p_model_path)

    def remove_op(self):
        print(f"\n[Remove {self.r_op}]")

        nodes = self.model.graph.node
        new_nodes = []

        # 현재 그래프의 출력 이름 추출
        output_names = {output.name for output in self.model.graph.output}

        for node in nodes:
            if node.op_type == self.r_op:
                input_name = node.input[0]
                output_name = node.output[0]

                # Softmax가 그래프 출력에 포함된 경우 출력 이름 대체
                if output_name in output_names:
                    for output in self.model.graph.output:
                        if output.name == output_name:
                            output.name = input_name  # 이전 노드의 출력을 최종 출력으로 설정

                # Softmax 노드의 출력을 사용하는 노드 업데이트
                for next_node in nodes:
                    next_node.input[:] = [
                        input_name if inp == output_name else inp for inp in next_node.input
                    ]
            else:
                new_nodes.append(node)

        # 수정된 노드로 모델 그래프 갱신
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)

        # 수정된 모델 저장
        dir_path, filename = os.path.split(self.p_model)
        name, ext = os.path.splitext(filename)

        new_name = rf"{name}_{self.r_op}_removed{ext}"
        new_path = os.path.join(dir_path, new_name)
        onnx.save(self.model, new_path)

        print(f"{new_path} saved")


if __name__ == "__main__":
    m_path = rf"C:\Work\tom\python_project\AI_MODEL_Rep\test_model_repo\ml_group\dropout\squeezenet1.0-12.onnx"
    r_op = "Dropout"

    model_instance = ONNXINFO(p_model_path=m_path, r_op=r_op)
    model_instance.remove_op()
