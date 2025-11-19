# blocks/Evaluation/start.py

from .confusion_matrix import generate_confusion_matrix_code, ConfusionMatrix
from .evaluation import generate_performance_evaluation_code, PerformanceEvaluation
from .predict_image import generate_prediction_images_code, PredictionImages
from .misclassification_image import generate_misclassified_images_code, MisclassifiedImages
from .metrics import generate_metrics_snippet
from .inference import generate_inference_snippet

def generate_evaluation_snippet(form):
    """
    form: request.form 딕셔너리
    → 평가하기 단계 전체 스니펫 생성
    """
    # form에서 파라미터 추출
    metrics = form.get('metrics', [])  # 리스트 형태로 받을 것으로 가정
    model_path = form.get('model_path', 'best_model.pth')
    
    # Confusion Matrix 관련
    show_confusion_matrix = 'confusion_matrix' in form
    cm_figure_size = form.get('cm_figure_size', '(10, 8)')
    cm_color_map = form.get('cm_color_map', 'Blues')
    cm_show_numbers = form.get('cm_show_numbers', 'true').lower() == 'true'
    
    # Prediction Images 관련
    show_predictions = 'show_predictions' in form
    pred_num_samples = form.get('pred_num_samples', '10')
    pred_figure_size = form.get('pred_figure_size', '(12, 5)')
    
    # Misclassified Images 관련
    show_misclassified = 'show_misclassified' in form
    mis_max_show = form.get('mis_max_show', '5')
    
    lines = [
        "# 자동 생성된 evaluation.py",
        "# AI 블록코딩 - 평가 파이프라인",
        "",
        "import os",
        "import torch",
        "import numpy as np",
        "from torch.utils.data import TensorDataset, DataLoader",
        "from tqdm import tqdm",
        "import json",
        "import datetime",
        "",
        "# 로깅 유틸",
        "def log(msg):",
        "    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "    print(f\"[eval][{timestamp}] {msg}\")",
        "",
        "log('=== EVALUATION START ===')",
        "",
    ]
    lines.append("# -----------------------------")
    lines.append("# --------- [평가하기 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("")

    # 설정 및 디바이스 추가
    lines.append("# 평가 설정")
    lines.append("CONFIG = {")
    lines.append("    'batch_size': 128,")
    lines.append("    'num_classes': 10,")
    lines.append("    'force_cpu': False")
    lines.append("}")
    lines.append("")
    lines.append("# 디바이스 설정")
    lines.append("if CONFIG['force_cpu']:")
    lines.append("    device = torch.device('cpu')")
    lines.append("else:")
    lines.append("    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    lines.append("log(f'Using device: {device}')")
    lines.append("")

    # 데이터 로딩 함수 추가
    lines.append("# 데이터 로드")
    lines.append("def load_data():")
    lines.append("    \"\"\"전처리된 데이터 로드\"\"\"")
    lines.append("    WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("    data_path = os.path.join(WORKDIR, 'data', 'dataset.pt')")
    lines.append("    ")
    lines.append("    if not os.path.exists(data_path):")
    lines.append("        raise FileNotFoundError(f'Dataset not found at {data_path}. Run preprocessing first.')")
    lines.append("    ")
    lines.append("    data = torch.load(data_path, map_location='cpu')")
    lines.append("    ")
    lines.append("    X_test = data['X_test']")
    lines.append("    y_test = data['y_test']")
    lines.append("    ")
    lines.append("    log(f'Test data loaded: {len(X_test)} samples')")
    lines.append("    ")
    lines.append("    # DataLoader 생성")
    lines.append("    test_dataset = TensorDataset(X_test, y_test)")
    lines.append("    test_loader = DataLoader(")
    lines.append("        test_dataset,")
    lines.append("        batch_size=CONFIG['batch_size'],")
    lines.append("        shuffle=False,")
    lines.append("        num_workers=0")
    lines.append("    )")
    lines.append("    ")
    lines.append("    return test_loader, X_test, y_test")
    lines.append("")

    # 모델 로딩 함수 추가
    lines.append("# 모델 로드")
    lines.append("def load_model():")
    lines.append("    \"\"\"학습된 모델 로드\"\"\"")
    lines.append("    from model import build_model, load_model")
    lines.append("    ")
    lines.append("    WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("    checkpoint_path = os.path.join(WORKDIR, 'artifacts', 'best_model.pth')")
    lines.append("    ")
    lines.append("    # 모델 생성 및 로드")
    lines.append("    model = build_model()")
    lines.append("    model = load_model(model, checkpoint_path, map_location=device)")
    lines.append("    model = model.to(device)")
    lines.append("    model.eval()")
    lines.append("    ")
    lines.append("    return model")
    lines.append("")

    # 추론 실행 함수 추가
    lines.append("# 추론 실행")
    lines.append("def run_inference(model, loader):")
    lines.append("    \"\"\"모델 추론 실행\"\"\"")
    lines.append("    all_preds = []")
    lines.append("    all_labels = []")
    lines.append("    all_probs = []")
    lines.append("    ")
    lines.append("    model.eval()")
    lines.append("    with torch.no_grad():")
    lines.append("        for data, target in tqdm(loader, desc='Evaluating'):")
    lines.append("            data = data.to(device)")
    lines.append("            ")
    lines.append("            # 추론")
    lines.append("            output = model(data)")
    lines.append("            probs = torch.softmax(output, dim=1)")
    lines.append("            preds = output.argmax(dim=1)")
    lines.append("            ")
    lines.append("            # 결과 저장")
    lines.append("            all_preds.extend(preds.cpu().numpy())")
    lines.append("            all_labels.extend(target.numpy())")
    lines.append("            all_probs.extend(probs.cpu().numpy())")
    lines.append("    ")
    lines.append("    return np.array(all_labels), np.array(all_preds), np.array(all_probs)")
    lines.append("")

    # 메트릭 계산 함수 추가
    lines.append("# 메트릭 계산")
    lines.append("def calculate_metrics(y_true, y_pred, y_proba=None):")
    lines.append("    \"\"\"평가 메트릭 계산\"\"\"")
    lines.append("    from sklearn.metrics import (")
    lines.append("        accuracy_score, precision_score, recall_score, f1_score,")
    lines.append("        classification_report, confusion_matrix")
    lines.append("    )")
    lines.append("    ")
    lines.append("    results = {}")
    lines.append("    ")
    lines.append("    # Accuracy")
    lines.append("    results['accuracy'] = accuracy_score(y_true, y_pred)")
    lines.append("    log(f'Accuracy: {results[\"accuracy\"]:.4f}')")
    lines.append("    ")
    lines.append("    # Precision")
    lines.append("    results['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)")
    lines.append("    log(f'Precision (macro): {results[\"precision\"]:.4f}')")
    lines.append("    ")
    lines.append("    # Recall")
    lines.append("    results['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)")
    lines.append("    log(f'Recall (macro): {results[\"recall\"]:.4f}')")
    lines.append("    ")
    lines.append("    # F1-score")
    lines.append("    results['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)")
    lines.append("    log(f'F1-score (macro): {results[\"f1\"]:.4f}')")
    lines.append("    ")
    lines.append("    # Classification Report")
    lines.append("    log('\\nClassification Report:')")
    lines.append("    report = classification_report(y_true, y_pred, zero_division=0)")
    lines.append("    print(report)")
    lines.append("    results['classification_report'] = report")
    lines.append("    ")
    lines.append("    # Confusion Matrix")
    lines.append("    cm = confusion_matrix(y_true, y_pred)")
    lines.append("    results['confusion_matrix'] = cm.tolist()")
    lines.append("    log('\\nConfusion Matrix:')")
    lines.append("    print(cm)")
    lines.append("    ")
    lines.append("    return results")
    lines.append("")

    # 시각화 함수 추가
    lines.append("# 시각화 함수")
    lines.append("def visualize_results(y_true, y_pred, X_test, results):")
    lines.append("    \"\"\"결과 시각화\"\"\"")
    lines.append("    import matplotlib.pyplot as plt")
    lines.append("    import matplotlib")
    lines.append("    matplotlib.use('Agg')  # GUI 없는 환경 지원")
    lines.append("    ")
    lines.append("    WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("    artifacts_dir = os.path.join(WORKDIR, 'artifacts')")
    lines.append("    os.makedirs(artifacts_dir, exist_ok=True)")
    lines.append("    ")
    lines.append("    # Confusion Matrix 히트맵")
    lines.append("    if 'confusion_matrix' in results:")
    lines.append("        cm = np.array(results['confusion_matrix'])")
    lines.append("        ")
    lines.append("        fig, ax = plt.subplots(figsize=(10, 8))")
    lines.append("        ")
    lines.append("        # 히트맵 그리기")
    lines.append("        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')")
    lines.append("        ax.figure.colorbar(im, ax=ax)")
    lines.append("        ")
    lines.append("        # 라벨 설정")
    lines.append("        ax.set(xticks=np.arange(cm.shape[1]),")
    lines.append("               yticks=np.arange(cm.shape[0]),")
    lines.append("               xlabel='Predicted',")
    lines.append("               ylabel='True',")
    lines.append("               title='Confusion Matrix')")
    lines.append("        ")
    lines.append("        # 각 셀에 값 표시")
    lines.append("        thresh = cm.max() / 2.")
    lines.append("        for i in range(cm.shape[0]):")
    lines.append("            for j in range(cm.shape[1]):")
    lines.append("                ax.text(j, i, format(cm[i, j], 'd'),")
    lines.append("                       ha='center', va='center',")
    lines.append("                       color='white' if cm[i, j] > thresh else 'black')")
    lines.append("        ")
    lines.append("        plt.tight_layout()")
    lines.append("        cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')")
    lines.append("        plt.savefig(cm_path, dpi=100, bbox_inches='tight')")
    lines.append("        plt.close()")
    lines.append("        log(f'Confusion matrix saved to {cm_path}')")
    lines.append("    ")
    lines.append("    # 예측 샘플 시각화")
    lines.append("    n_samples = min(10, len(X_test))")
    lines.append("    if n_samples > 0:")
    lines.append("        fig = plt.figure(figsize=(15, 3 * ((n_samples - 1) // 5 + 1)))")
    lines.append("        ")
    lines.append("        for i in range(n_samples):")
    lines.append("            ax = fig.add_subplot((n_samples - 1) // 5 + 1, 5, i + 1)")
    lines.append("            ")
    lines.append("            # 이미지 표시")
    lines.append("            img = X_test[i].cpu().numpy()")
    lines.append("            if img.ndim == 3:")
    lines.append("                if img.shape[0] == 1:")
    lines.append("                    img = img[0]")
    lines.append("                elif img.shape[0] == 3:")
    lines.append("                    img = np.transpose(img, (1, 2, 0))")
    lines.append("            ")
    lines.append("            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)")
    lines.append("            ax.set_title(f'True: {y_true[i]}\\nPred: {y_pred[i]}', fontsize=9)")
    lines.append("            ax.axis('off')")
    lines.append("        ")
    lines.append("        plt.tight_layout()")
    lines.append("        samples_path = os.path.join(artifacts_dir, 'prediction_samples.png')")
    lines.append("        plt.savefig(samples_path, dpi=100, bbox_inches='tight')")
    lines.append("        plt.close()")
    lines.append("        log(f'Prediction samples saved to {samples_path}')")
    lines.append("    ")
    lines.append("    # 오분류 샘플 시각화")
    lines.append("    misclassified_idx = np.where(y_true != y_pred)[0]")
    lines.append("    n_mis = min(10, len(misclassified_idx))")
    lines.append("    ")
    lines.append("    if n_mis > 0:")
    lines.append("        fig = plt.figure(figsize=(15, 3 * ((n_mis - 1) // 5 + 1)))")
    lines.append("        ")
    lines.append("        for i in range(n_mis):")
    lines.append("            idx = misclassified_idx[i]")
    lines.append("            ax = fig.add_subplot((n_mis - 1) // 5 + 1, 5, i + 1)")
    lines.append("            ")
    lines.append("            # 이미지 표시")
    lines.append("            img = X_test[idx].cpu().numpy()")
    lines.append("            if img.ndim == 3:")
    lines.append("                if img.shape[0] == 1:")
    lines.append("                    img = img[0]")
    lines.append("                elif img.shape[0] == 3:")
    lines.append("                    img = np.transpose(img, (1, 2, 0))")
    lines.append("            ")
    lines.append("            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)")
    lines.append("            ax.set_title(f'True: {y_true[idx]}\\nPred: {y_pred[idx]}', fontsize=9, color='red')")
    lines.append("            ax.axis('off')")
    lines.append("        ")
    lines.append("        plt.tight_layout()")
    lines.append("        mis_path = os.path.join(artifacts_dir, 'misclassified_samples.png')")
    lines.append("        plt.savefig(mis_path, dpi=100, bbox_inches='tight')")
    lines.append("        plt.close()")
    lines.append("        log(f'Misclassified samples saved to {mis_path}')")
    lines.append("    else:")
    lines.append("        log('No misclassified samples found!')")
    lines.append("")

    # Performance Evaluation (metrics가 있으면 실행)
    if metrics:
        if isinstance(metrics, str):
            metrics = [metrics]  # 단일 값을 리스트로 변환
        eval_block = PerformanceEvaluation(
            metrics=metrics,
            model_path=model_path
        )
        lines.append(generate_performance_evaluation_code(eval_block))
        lines.append("")

    # Confusion Matrix
    if show_confusion_matrix:
        # figure_size 파싱 (문자열 "(10, 8)" -> 튜플 (10, 8))
        try:
            fig_size = eval(cm_figure_size) if isinstance(cm_figure_size, str) else cm_figure_size
        except:
            fig_size = (10, 8)
        
        cm_block = ConfusionMatrix(
            figure_size=fig_size,
            color_map=cm_color_map,
            show_numbers=cm_show_numbers
        )
        lines.append(generate_confusion_matrix_code(cm_block))
        lines.append("")

    # Prediction Images
    if show_predictions:
        try:
            pred_fig_size = eval(pred_figure_size) if isinstance(pred_figure_size, str) else pred_figure_size
        except:
            pred_fig_size = (12, 5)
        
        pred_block = PredictionImages(
            num_samples=int(pred_num_samples),
            figure_size=pred_fig_size
        )
        lines.append(generate_prediction_images_code(pred_block))
        lines.append("")

    # Misclassified Images
    if show_misclassified:
        mis_block = MisclassifiedImages(
            max_show=int(mis_max_show)
        )
        lines.append(generate_misclassified_images_code(mis_block))
        lines.append("")

    # 메인 실행 블록 추가
    lines.append("# 메인 실행")
    lines.append("if __name__ == '__main__':")
    lines.append("    try:")
    lines.append("        # 데이터 로드")
    lines.append("        test_loader, X_test, y_test = load_data()")
    lines.append("        ")
    lines.append("        # 모델 로드")
    lines.append("        model = load_model()")
    lines.append("        ")
    lines.append("        # 추론 실행")
    lines.append("        log('Running inference...')")
    lines.append("        y_true, y_pred, y_proba = run_inference(model, test_loader)")
    lines.append("        ")
    lines.append("        # 메트릭 계산")
    lines.append("        log('\\nCalculating metrics...')")
    lines.append("        results = calculate_metrics(y_true, y_pred, y_proba)")
    lines.append("        ")
    lines.append("        # 시각화")
    lines.append("        visualize_results(y_true, y_pred, X_test, results)")
    lines.append("        ")
    lines.append("        # 결과 저장")
    lines.append("        WORKDIR = os.environ.get('AIB_WORKDIR', '.')")
    lines.append("        results_path = os.path.join(WORKDIR, 'artifacts', 'evaluation_results.json')")
    lines.append("        ")
    lines.append("        # numpy 배열을 리스트로 변환 (JSON 직렬화를 위해)")
    lines.append("        save_results = {}")
    lines.append("        for key, value in results.items():")
    lines.append("            if isinstance(value, np.ndarray):")
    lines.append("                save_results[key] = value.tolist()")
    lines.append("            elif isinstance(value, (int, float, str, list, dict)):")
    lines.append("                save_results[key] = value")
    lines.append("        ")
    lines.append("        with open(results_path, 'w') as f:")
    lines.append("            json.dump(save_results, f, indent=2)")
    lines.append("        ")
    lines.append("        log(f'\\nResults saved to {results_path}')")
    lines.append("        log('=== EVALUATION DONE ===')")
    lines.append("        ")
    lines.append("    except Exception as e:")
    lines.append("        log(f'Error during evaluation: {e}')")
    lines.append("        raise")

    return "\n".join(lines)
