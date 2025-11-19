# blocks/Training/start.py

from .loss            import generate_loss_snippet, LossBlock
from .optimizer       import generate_optimizer_snippet, OptimizerBlock
from .training_option import generate_training_option_snippet, TrainingOptionBlock
from .data_loader     import generate_data_loader_snippet
from .training_functions import generate_training_functions_snippet

def generate_training_snippet(form):
    """
    form: request.form 딕셔너리
    → ⑤~⑦ 학습하기 단계 전체 스니펫 생성
    """
    # 1) form에서 파라미터 추출
    loss_method      = form.get('loss_method', '')
    optimizer_method = form.get('optimizer_method', '')
    learning_rate    = form.get('learning_rate', '')
    epochs           = form.get('epochs', '')
    batch_size       = form.get('batch_size', '')
    patience         = form.get('patience', '')

    lines = [
        "# 자동 생성된 training.py",
        "# AI 블록코딩 - 학습 파이프라인",
        "",
        "import os",
        "import torch",
        "import torch.nn as nn",
        "import torch.optim as optim",
        "from torch.utils.data import TensorDataset, DataLoader",
        "from tqdm import tqdm",
        "import time",
        "import datetime",
        "",
        "# 로깅 유틸",
        "def log(msg):",
        "    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "    print(f\"[train][{timestamp}] {msg}\")",
        "",
    ]
    # 학습 모듈 import
    lines.append("# -----------------------------")
    lines.append("# --------- [학습하기 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("")

    # ⑤ 손실함수
    if loss_method:
        loss_block = LossBlock(loss_method=loss_method)
        lines.append(generate_loss_snippet(loss_block))

    # ⑥ 옵티마이저
    if optimizer_method and learning_rate:
        optimizer_block = OptimizerBlock(
            method=optimizer_method,
            lr=float(learning_rate)
        )
        lines.append(generate_optimizer_snippet(optimizer_block))

    # ⑦ 학습 옵션
    if epochs and batch_size and patience:
        training_option_block = TrainingOptionBlock(
            epochs=int(epochs),
            batch_size=int(batch_size),
            patience=int(patience)
        )
        lines.append(generate_training_option_snippet(training_option_block))

    # 설정 및 디바이스 추가
    lines.append("# 시드 설정")
    lines.append("torch.manual_seed(42)")
    lines.append("if torch.cuda.is_available():")
    lines.append("    torch.cuda.manual_seed(42)")
    lines.append("")
    lines.append("# 디바이스 설정")
    lines.append("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    lines.append("log(f'Using device: {device}')")
    lines.append("")

    # 데이터 로더 함수 추가
    lines.append(generate_data_loader_snippet())

    # 학습 및 검증 함수들 추가
    lines.append(generate_training_functions_snippet())

    # 전체 학습 프로세스 추가
    lines.append("# 전체 학습 프로세스")
    lines.append("def train_model(model, train_loader, test_loader, criterion, optimizer):")
    lines.append("    \"\"\"전체 학습 프로세스\"\"\"")
    lines.append("    WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("    checkpoint_dir = os.path.join(WORKDIR, 'artifacts')")
    lines.append("    os.makedirs(checkpoint_dir, exist_ok=True)")
    lines.append("    ")
    lines.append("    best_val_loss = float('inf')")
    lines.append("    patience_counter = 0")
    lines.append("    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}")
    lines.append("    ")
    lines.append("    log('Starting training...')")
    lines.append("    ")
    lines.append("    for epoch in range(1, num_epochs + 1):")
    lines.append("        log(f'\\nEpoch {epoch}/{num_epochs}')")
    lines.append("        ")
    lines.append("        # 학습")
    lines.append("        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)")
    lines.append("        ")
    lines.append("        # 검증")
    lines.append("        val_loss, val_acc = validate(model, test_loader, criterion)")
    lines.append("        ")
    lines.append("        # 결과 출력")
    lines.append("        log(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')")
    lines.append("        log(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')")
    lines.append("        ")
    lines.append("        # 히스토리 저장")
    lines.append("        history['train_loss'].append(train_loss)")
    lines.append("        history['train_acc'].append(train_acc)")
    lines.append("        history['val_loss'].append(val_loss)")
    lines.append("        history['val_acc'].append(val_acc)")
    lines.append("        ")
    lines.append("        # 체크포인트 저장")
    lines.append("        if val_loss < best_val_loss:")
    lines.append("            best_val_loss = val_loss")
    lines.append("            patience_counter = 0")
    lines.append("            ")
    lines.append("            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')")
    lines.append("            torch.save(model.state_dict(), checkpoint_path)")
    lines.append("            log(f'✓ Best model saved (val_loss: {val_loss:.4f})')")
    lines.append("        else:")
    lines.append("            patience_counter += 1")
    lines.append("            log(f'No improvement ({patience_counter}/{patience})')")
    lines.append("        ")
    lines.append("        # Early stopping")
    lines.append("        if patience_counter >= patience:")
    lines.append("            log('Early stopping triggered!')")
    lines.append("            break")
    lines.append("    ")
    lines.append("    # 최종 체크포인트 저장")
    lines.append("    final_checkpoint = os.path.join(checkpoint_dir, 'final_model.pth')")
    lines.append("    torch.save(model.state_dict(), final_checkpoint)")
    lines.append("    ")
    lines.append("    # 학습 히스토리 저장")
    lines.append("    import json")
    lines.append("    history_path = os.path.join(checkpoint_dir, 'training_history.json')")
    lines.append("    with open(history_path, 'w') as f:")
    lines.append("        json.dump(history, f)")
    lines.append("    ")
    lines.append("    log(f'Training completed! Best val_loss: {best_val_loss:.4f}')")
    lines.append("    log(f'Checkpoints saved to {checkpoint_dir}')")
    lines.append("    ")
    lines.append("    return history")
    lines.append("")

    # 메인 실행 블록 추가
    lines.append("if __name__ == '__main__':")
    lines.append("    log('=== TRAINING START ===')")
    lines.append("    ")
    lines.append("    try:")
    lines.append("        # 데이터 로드")
    lines.append("        train_loader, test_loader = load_data()")
    lines.append("        ")
    lines.append("        # 모델 설정")
    lines.append("        from model import build_model")
    lines.append("        model = build_model().to(device)")
    lines.append("        ")
    lines.append("        # 학습 실행")
    lines.append("        history = train_model(")
    lines.append("            model, ")
    lines.append("            train_loader, ")
    lines.append("            test_loader, ")
    lines.append("            criterion, ")
    lines.append("            optimizer")
    lines.append("        )")
    lines.append("        ")
    lines.append("        log('=== TRAINING DONE ===')")
    lines.append("        ")
    lines.append("    except Exception as e:")
    lines.append("        log(f'Error during training: {e}')")
    lines.append("        raise")

    return "\n".join(lines)
