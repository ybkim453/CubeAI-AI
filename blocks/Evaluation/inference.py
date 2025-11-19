# blocks/Evaluation/inference.py
# 모델 추론 및 예측

def generate_inference_snippet():
    """
    모델 추론 함수 생성
    """
    lines = []
    lines.append("# 모델 추론")
    lines.append("def run_inference(model, test_loader):")
    lines.append("    \"\"\"테스트 데이터에 대한 추론 실행\"\"\"")
    lines.append("    model.eval()")
    lines.append("    all_preds = []")
    lines.append("    all_labels = []")
    lines.append("    all_probs = []")
    lines.append("    ")
    lines.append("    log('Running inference on test data...')")
    lines.append("    ")
    lines.append("    with torch.no_grad():")
    lines.append("        progress_bar = tqdm(test_loader, desc='Inference', ncols=100)")
    lines.append("        for data, target in progress_bar:")
    lines.append("            data, target = data.to(device), target.to(device)")
    lines.append("            ")
    lines.append("            # 순전파")
    lines.append("            output = model(data)")
    lines.append("            probs = torch.softmax(output, dim=1)")
    lines.append("            preds = output.argmax(dim=1)")
    lines.append("            ")
    lines.append("            # 결과 저장")
    lines.append("            all_preds.extend(preds.cpu().numpy())")
    lines.append("            all_labels.extend(target.cpu().numpy())")
    lines.append("            all_probs.extend(probs.cpu().numpy())")
    lines.append("    ")
    lines.append("    return np.array(all_labels), np.array(all_preds), np.array(all_probs)")
    lines.append("")
    return "\n".join(lines)
