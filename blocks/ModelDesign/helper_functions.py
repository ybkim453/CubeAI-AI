# blocks/ModelDesign/helper_functions.py
# 모델 관련 헬퍼 함수들

def generate_helper_functions_snippet():
    """
    모델 생성, 로드, 저장 등의 헬퍼 함수들 생성
    """
    lines = []
    lines.append("")
    lines.append("def build_model():")
    lines.append("    \"\"\"모델 생성 함수\"\"\"")
    lines.append("    model = CNN()")
    lines.append("    print(f\"Model created with {model.get_num_parameters():,} parameters\")")
    lines.append("    return model")
    lines.append("")
    lines.append("")
    lines.append("def load_model(model, checkpoint_path, map_location='cpu'):")
    lines.append("    \"\"\"모델 로드 함수\"\"\"")
    lines.append("    import os")
    lines.append("    if os.path.exists(checkpoint_path):")
    lines.append("        state_dict = torch.load(checkpoint_path, map_location=map_location)")
    lines.append("        model.load_state_dict(state_dict)")
    lines.append("        print(f\"Model loaded from {checkpoint_path}\")")
    lines.append("    else:")
    lines.append("        print(f\"Checkpoint not found at {checkpoint_path}\")")
    lines.append("    return model")
    lines.append("")
    lines.append("")
    lines.append("def save_model(model, checkpoint_path):")
    lines.append("    \"\"\"모델 저장 함수\"\"\"")
    lines.append("    torch.save(model.state_dict(), checkpoint_path)")
    lines.append("    print(f\"Model saved to {checkpoint_path}\")")
    lines.append("")
    return "\n".join(lines)

