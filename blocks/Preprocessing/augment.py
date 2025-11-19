# blocks/Preprocessing/augment.py
from dataclasses import dataclass

@dataclass
class AugmentBlock:
    # 이미지 증강 블록 구조체
    method: str  # 'rotate', 'hflip', 'vflip', 'translate'
    param: int  # 회전 각도(°) 또는 이동 픽셀 수

def generate_augment_snippet(block: AugmentBlock) -> str:
    """
    X_train/X_test와 y_train/y_test를 증강하고 합치는 스니펫 생성
    method: 'rotate','hflip','vflip','translate'
    param: 회전 각도(°) 또는 이동 픽셀 수
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---[이미지 증강 블록]---")
    lines.append("# -----------------------------")
    lines.append(f"# 방법: {block.method}, 파라미터: {block.param}")
    lines.append("from torchvision import transforms")
    # transform 정의
    if block.method == 'rotate':
        lines.append(f"transform_aug = transforms.RandomRotation(({block.param}, {block.param}))  # 회전 증강")
    elif block.method == 'hflip':
        lines.append("transform_aug = transforms.RandomHorizontalFlip(p=1.0)  # 수평 뒤집기")
    elif block.method == 'vflip':
        lines.append("transform_aug = transforms.RandomVerticalFlip(p=1.0)    # 수직 뒤집기")
    elif block.method == 'translate':
        lines.append("_, _, H, W = X_train.shape")
        lines.append(f"tx = {block.param} / W; ty = {block.param} / H")
        lines.append("transform_aug = transforms.RandomAffine(degrees=0, translate=(tx, ty))  # 이동 증강")
    lines.append("")
    # 학습 데이터 증강
    lines.append("# 학습 데이터 증강 및 라벨 복제")
    lines.append("aug_train = torch.stack([transform_aug(x) for x in X_train], dim=0)  # 증강된 이미지")
    lines.append("X_train = torch.cat([X_train, aug_train], dim=0)  # 원본+증강 이미지 합치기")
    lines.append("y_train = torch.cat([y_train, y_train], dim=0)   # 라벨도 원본 복제하여 합치기")
    lines.append("")
    # 테스트 데이터 증강
    lines.append("# 테스트 데이터 증강 및 라벨 복제")
    lines.append("aug_test  = torch.stack([transform_aug(x) for x in X_test], dim=0)   # 증강된 이미지")
    lines.append("X_test   = torch.cat([X_test, aug_test], dim=0)   # 원본+증강 이미지 합치기")
    lines.append("y_test   = torch.cat([y_test, y_test], dim=0)     # 테스트 라벨 복제하여 합치기")
    lines.append("")
    return "\n".join(lines)
