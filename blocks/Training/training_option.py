# blocks/Training/training_option.py
from dataclasses import dataclass

@dataclass
class TrainingOptionBlock:
    # 학습 옵션 블록 구조체
    epochs: int  # 반복횟수
    batch_size: int  # 배치 크기
    patience: int  # 조기 종료 대기 에폭

def generate_training_option_snippet(block: TrainingOptionBlock) -> str:
    """
    ⑦ Training Option 블록: 학습 옵션 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# -------[학습 옵션 블록]--------")
    lines.append("# -----------------------------")
    lines.append(f"# epochs={block.epochs}, batch_size={block.batch_size}, patience={block.patience}")
    lines.append(f"num_epochs = {block.epochs}        # 전체 학습 반복 횟수")
    lines.append(f"batch_size = {block.batch_size}     # 한 배치 크기")
    lines.append(f"patience = {block.patience}         # 조기 종료 전 대기 에폭 수")
    lines.append("")
    return "\n".join(lines)
