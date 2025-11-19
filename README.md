# CubeAI - AI 블록코딩 플랫폼 🧊

## 🛠️ 기술 스택

### Backend
- **FastAPI** - 고성능 웹 프레임워크
- **Uvicorn** - ASGI 서버
- **Server-Sent Events** - 실시간 로그 스트리밍

### AI/ML
- **PyTorch 2.1.0** - 딥러닝 프레임워크
- **torchvision 0.16.0** - Computer Vision 라이브러리
- **scikit-learn 1.3.2** - 머신러닝 유틸리티
- **NumPy 1.24.3** - 수치 연산
- **Pandas 2.0.3** - 데이터 처리

### Visualization
- **Matplotlib 3.8.2** - 그래프 생성
- **Seaborn 0.13.2** - 통계 시각화
- **Pillow 10.0.1** - 이미지 처리

## 📁 프로젝트 구조

```
CubeAI-AI/
│
├── blocks/                    # 블록 코드 생성 모듈
│   ├── Preprocessing/        # 데이터 전처리 블록
│   │   ├── data_selection.py
│   │   ├── drop_na.py
│   │   ├── normalize.py
│   │   └── ...
│   ├── ModelDesign/          # 모델 설계 블록
│   │   ├── conv2d.py
│   │   ├── pooling.py
│   │   ├── fc_layer.py
│   │   └── ...
│   ├── Training/             # 학습 블록
│   │   ├── optimizer.py
│   │   ├── loss_function.py
│   │   └── ...
│   └── Evaluation/           # 평가 블록
│       ├── metrics.py
│       ├── confusion_matrix.py
│       └── ...
│
├── core/                     # 핵심 기능 모듈
│   ├── config.py            # 설정 관리
│   ├── workspace.py         # 워크스페이스 관리
│   ├── process.py           # 프로세스 실행 관리
│   └── dataset.py           # 데이터셋 관리
│
├── dataset/                  # 데이터셋 저장소
│   └── *.csv                # MNIST 등 CSV 데이터
│
├── workspace/                # 사용자별 작업 공간
│   └── {user_id}/           # UUID 기반 격리
│       ├── generated_code.py
│       ├── model.pth
│       └── logs/
│
├── logs/                     # 실행 로그
│
├── main.py                   # FastAPI 메인 애플리케이션
├── requirements.txt          # Python 패키지 의존성
└── README.md                # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 요구사항
- Python 3.8 이상
- CUDA 11.8 (GPU 사용시, 선택사항)

### 2. 설치

```bash
# 레포지토리 클론
git clone https://github.com/your-username/CubeAI-AI.git
cd CubeAI-AI

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
# 개발 서버 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 http://localhost:8000 에서 API 문서를 확인할 수 있습니다.

## 📡 API 엔드포인트

### 주요 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | 루트 페이지 - `/app`으로 리다이렉트 |
| GET | `/app` | 메인 애플리케이션 페이지 (워크스페이스 초기화) |
| GET | `/app/{user_id}` | 사용자 ID가 포함된 URL 지원 |
| POST | `/convert` | 블록 설정을 Python 코드로 변환 |
| POST | `/run` | 생성된 코드 실행 (subprocess) |
| GET | `/logs/stream` | 실시간 로그 스트리밍 (SSE) |
| GET | `/download/{stage}` | 생성된 코드 파일 다운로드 |
| GET | `/result` | 실행 결과 조회 (모델 파일, 메트릭 등) |
| GET | `/result/status` | 실행 상태 확인 |
| GET | `/data-info` | CSV 데이터셋 정보 조회 (shape/structure/sample/images) |
| GET | `/debug/dataset` | 데이터셋 디버깅 정보 | 

### 블록 파라미터 예시

```json
{
  "stage": "all",
  "dataset": "mnist_train.csv",
  "resize_n": 28,
  "normalize": "0-1",
  "conv1_filters": 32,
  "conv1_kernel": 3,
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "epochs": 10,
  "batch_size": 64
}
```

## 🔄 워크플로우

1. **데이터 선택**: CSV 데이터셋 업로드 또는 선택
2. **전처리 설정**: 정규화, 리사이즈, 증강 등 설정
3. **모델 설계**: CNN 레이어 구성 (Conv2D, Pooling, Dense)
4. **학습 설정**: 옵티마이저, 손실함수, 에폭 설정
5. **코드 생성**: 설정된 블록을 Python 코드로 자동 변환
6. **실행 및 모니터링**: 코드 실행 및 실시간 로그 확인
7. **평가**: 정확도, 혼동행렬, 샘플 예측 확인

## 📊 지원 기능

### 전처리 블록
- ✅ 데이터 로드 및 분할
- ✅ 결측치 처리
- ✅ 라벨 필터링
- ✅ 이미지 리사이즈
- ✅ 데이터 증강 (회전, 뒤집기, 이동)
- ✅ 정규화 (Min-Max, Z-score)

### 모델 설계 블록
- ✅ Convolutional Layer (Conv2D)
- ✅ Pooling Layer (Max/Average)
- ✅ Dropout Layer
- ✅ Fully Connected Layer
- ✅ Activation Functions (ReLU, Sigmoid, Tanh)

### 학습 블록
- ✅ Optimizers (Adam, SGD, RMSprop)
- ✅ Loss Functions (CrossEntropy, MSE)
- ✅ Learning Rate Scheduler
- ✅ Early Stopping
- ✅ Checkpoint 저장

### 평가 블록
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Sample Predictions
- ✅ Misclassified Samples