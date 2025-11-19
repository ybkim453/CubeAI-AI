# blocks/ModelDesign/test_code.py
# 모델 테스트 코드

def generate_test_code_snippet():
    """
    모델 테스트용 메인 블록 생성
    """
    lines = []
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    # 테스트용 코드")
    lines.append("    model = build_model()")
    lines.append("    ")
    lines.append("    # 더미 입력으로 테스트")
    lines.append("    dummy_input = torch.randn(1, 1, 28, 28)")
    lines.append("    output = model(dummy_input)")
    lines.append("    print(f\"Input shape: {dummy_input.shape}\")")
    lines.append("    print(f\"Output shape: {output.shape}\")")
    lines.append("    ")
    lines.append("    # 모델 구조 출력")
    lines.append("    print(\"\\nModel structure:\")")
    lines.append("    print(model)")
    lines.append("")
    return "\n".join(lines)

