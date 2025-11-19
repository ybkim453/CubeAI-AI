# blocks/Training/optimizer.py
from dataclasses import dataclass

@dataclass
class OptimizerBlock:
    # 옵티마이저 블록 구조체
    method: str  # 'Adam', 'SGD', 'RMSprop', 등
    lr: float  # learning rate

def generate_optimizer_snippet(block: OptimizerBlock) -> str:
    """
    ⑥ Optimizer 블록: 옵티마이저 설정 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[옵티마이저 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"# 선택된 옵티마이저: {block.method}, Learning Rate={block.lr}")
    if block.method == 'Adam':
        lines.append(f"optimizer = optim.Adam(model.parameters(), lr={block.lr})")
    elif block.method == 'SGD':
        lines.append(f"optimizer = optim.SGD(model.parameters(), lr={block.lr})")
    elif block.method == 'RMSprop':
        lines.append(f"optimizer = optim.RMSprop(model.parameters(), lr={block.lr})")
    lines.append("")
    return "\n".join(lines)
