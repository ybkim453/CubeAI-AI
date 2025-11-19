# blocks/Evaluation/metrics.py
# 평가 메트릭 계산

def generate_metrics_snippet():
    """
    평가 메트릭 계산 함수들 생성
    """
    lines = []
    lines.append("# 평가 메트릭 계산")
    lines.append("def calculate_metrics(y_true, y_pred, target_names=None):")
    lines.append("    \"\"\"분류 메트릭 계산\"\"\"")
    lines.append("    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report")
    lines.append("    ")
    lines.append("    accuracy = accuracy_score(y_true, y_pred)")
    lines.append("    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')")
    lines.append("    ")
    lines.append("    log(f'Accuracy: {accuracy:.4f}')")
    lines.append("    log(f'Precision: {precision:.4f}')")
    lines.append("    log(f'Recall: {recall:.4f}')")
    lines.append("    log(f'F1-Score: {f1:.4f}')")
    lines.append("    ")
    lines.append("    # 분류 리포트")
    lines.append("    if target_names:")
    lines.append("        report = classification_report(y_true, y_pred, target_names=target_names)")
    lines.append("    else:")
    lines.append("        report = classification_report(y_true, y_pred)")
    lines.append("    ")
    lines.append("    log('Classification Report:')")
    lines.append("    for line in report.split('\\n'):")
    lines.append("        if line.strip():")
    lines.append("            log(line)")
    lines.append("    ")
    lines.append("    return {")
    lines.append("        'accuracy': accuracy,")
    lines.append("        'precision': precision,")
    lines.append("        'recall': recall,")
    lines.append("        'f1_score': f1,")
    lines.append("        'classification_report': report")
    lines.append("    }")
    lines.append("")
    return "\n".join(lines)
