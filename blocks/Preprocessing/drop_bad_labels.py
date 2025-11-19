# blocks/Preprocessing/drop_bad_labels.py
from dataclasses import dataclass

@dataclass
class DropBadLabelsBlock:
    # 잘못된 라벨 제거 블록 구조체
    min_label: int  # 최소 라벨 값
    max_label: int  # 최대 라벨 값

def generate_drop_bad_labels_snippet(block: DropBadLabelsBlock) -> str:
    """
    train_df/test_df 에서 라벨이 지정한 범위(min_label~max_label) 밖에 있는 행을 제거하는 코드 블록 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# --[잘못된 라벨 삭제 블록]--")
    lines.append("# -----------------------------")
    lines.append(f"# 라벨값 허용 범위: {block.min_label} ~ {block.max_label}")
    lines.append(f"train_df = train_df[train_df['label'].between({block.min_label}, {block.max_label})]  # 학습 데이터 필터링")
    lines.append(f"test_df  = test_df[test_df['label'].between({block.min_label}, {block.max_label})]   # 테스트 데이터 필터링")
    lines.append("")
    return "\n".join(lines)
