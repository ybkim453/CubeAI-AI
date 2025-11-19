# blocks/Preprocessing/data_selection.py
from dataclasses import dataclass

@dataclass
class DataSelectionBlock:
    # 데이터 선택 블록 구조체
    dataset: str  # CSV 파일명
    is_test: str  # 테스트 데이터 분할 여부 ('true' 또는 '')
    testdataset: str = ""  # 테스트 데이터셋 파일명
    a: int = 100  # 샘플링 비율 (%)

def generate_data_selection_snippet(block: DataSelectionBlock) -> str:
    """
    train_df / test_df 로드 및 분할 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[데이터 선택 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"train_df = pd.read_csv('{block.dataset}')  # '{block.dataset}' 파일에서 학습용 데이터로드")
    lines.append("")
    if block.is_test == 'true':
        lines.append(f"test_df  = pd.read_csv('{block.testdataset}')  # 사용자가 지정한 테스트 데이터로드")
    else:
        lines.append(f"# 테스트 미지정 → 학습 데이터를 {block.a}% 사용, 나머지 {(100-block.a)}%를 테스트로 분할")
        lines.append(f"test_df  = train_df.sample(frac={(100-block.a)/100.0}, random_state=42)")
        lines.append("train_df = train_df.drop(test_df.index)")
    lines.append("")
    return "\n".join(lines)
