# blocks/Preprocessing/save_data.py
# 데이터 저장 블록

def generate_save_data_snippet():
    """
    전처리된 데이터를 .pt 파일로 저장하는 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# -------[데이터 저장 블록]-------")
    lines.append("# -----------------------------")
    lines.append("t0_save = time.perf_counter()")
    lines.append("log('START: 전처리 결과 저장(dataset.pt)')")
    lines.append("")
    lines.append("WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("os.makedirs(os.path.join(WORKDIR, 'data'), exist_ok=True)")
    lines.append("save_path = os.path.join(WORKDIR, 'data', 'dataset.pt')")
    lines.append("")
    lines.append("torch.save({")
    lines.append("    'X_train': torch.as_tensor(X_train, dtype=torch.float32),")
    lines.append("    'y_train': torch.as_tensor(y_train, dtype=torch.long),")
    lines.append("    'X_test':  torch.as_tensor(X_test,  dtype=torch.float32),")
    lines.append("    'y_test':  torch.as_tensor(y_test,  dtype=torch.long),")
    lines.append("}, save_path)")
    lines.append("")
    lines.append("log(f'Saved to: {save_path}')")
    lines.append("log(f'END  : 전처리 결과 저장 (elapsed={time.perf_counter()-t0_save:.3f}s)')")
    lines.append("log('=== PREPROCESSING DONE ===')")
    lines.append("")
    return "\n".join(lines)

