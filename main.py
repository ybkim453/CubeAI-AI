# ==========================================
# CubeAI - AI 블록코딩 플랫폼 메인 서버
# FastAPI 기반 RESTful API 서버
# ==========================================

# 표준 라이브러리
import os
import io
import base64
import json
import time
import re
from pathlib import Path
from typing import Optional

# FastAPI 관련 임포트
from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 데이터 처리 라이브러리
import pandas as pd  # CSV 데이터 처리
import numpy as np   # 배열 연산
from PIL import Image  # 이미지 처리

# 코어 모듈 임포트
from core.config import settings  # 설정 관리
from core.workspace import WorkspaceManager  # 워크스페이스 관리
from core.process import ProcessManager  # 프로세스 실행 관리
from core.dataset import DatasetManager  # 데이터셋 관리

# 블록별 코드 생성기 임포트
from blocks.Preprocessing.start import generate_preprocessing_snippet  # 전처리 코드 생성
from blocks.ModelDesign.start import generate_modeldesign_snippet  # 모델 설계 코드 생성
from blocks.Training.start import generate_training_snippet  # 학습 코드 생성
from blocks.Evaluation.start import generate_evaluation_snippet  # 평가 코드 생성

# ==========================================
# FastAPI 앱 초기화
# ==========================================
app = FastAPI(
    title="CubeAI AI Sever API",
    version="2.0.0"
)

# CORS 미들웨어 설정 - 크로스 오리진 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 허용할 오리진 목록
    allow_credentials=True,  # 쿠키 포함 요청 허용
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

# 스테이지별 코드 생성기 매핑
code_generators = {
    "pre": generate_preprocessing_snippet,  # 전처리
    "model": generate_modeldesign_snippet,  # 모델 설계
    "train": generate_training_snippet,  # 학습
    "eval": generate_evaluation_snippet  # 평가
}

# ==========================================
# 라우트 핸들러 정의
# ==========================================

@app.get("/")
async def root():
    """루트 경로 - /app으로 리다이렉트"""
    return RedirectResponse(url="/app")

@app.get("/app")
async def main_app(user_id: Optional[str] = Query("anonymous")):
    """메인 애플리케이션 페이지 - 워크스페이스 초기화 및 컨텍스트 반환"""
    # FastAPI에서는 HTML 템플릿 대신 JSON으로 데이터 반환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]  # 사용자 ID를 안전한 파일명으로 변환
    
    from core.config import WORKSPACE_DIR
    workspace_path = WORKSPACE_DIR / uid  # 사용자별 워크스페이스 경로 생성
    workspace_path.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성
    
    # 클라이언트에 전달할 컨텍스트 데이터 준비
    context = {
        "options": DatasetManager.list_datasets(),  # 사용 가능한 CSV 파일 목록
        "form_state": WorkspaceManager.load_inputs(uid),  # 저장된 블록 설정값들
        "current_user_id": user_id,  # 현재 사용자 ID
        **WorkspaceManager.load_snippets(uid)  # 생성된 코드 스니펫들 (snippet_pre, snippet_model 등)
    }
    
    return context  # JSON 형태로 반환

@app.get("/app/{user_id}")
async def main_app_with_user(user_id: str):
    """사용자 ID가 포함된 URL 지원"""
    return RedirectResponse(url=f"/app?user_id={user_id}")

@app.post("/convert", response_class=PlainTextResponse)
async def convert_code(
    stage: str = Form(...),  # 변환할 스테이지 (pre/model/train/eval/all)
    user_id: Optional[str] = Form(None),  # 사용자 ID
    **form_data  # 블록 설정 데이터
):
    """
    블록 설정을 Python 코드로 변환하는 핵심 API
    - 각 스테이지별 블록 설정을 받아 실행 가능한 Python 코드 생성
    - 생성된 코드는 워크스페이스에 저장
    """
    try:
        uid, _ = WorkspaceManager.get_or_create_uid(user_id)  # 사용자 ID 생성 또는 가져오기
        
        # 익명 사용자는 허용하지 않음
        if uid == "anonymous" and not user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다.")
        
        print(f"[INFO] /convert 요청: stage={stage}, user_id={uid}")  # 디버깅용 로그
        
        # Form 데이터를 딕셔너리로 변환
        form_dict = dict(form_data)  # **form_data를 일반 딕셔너리로 변환
        if user_id:
            form_dict["user_id"] = user_id  # 사용자 ID를 form 데이터에 추가
        
        # 모든 스테이지를 한번에 처리하는 경우
        if stage == "all":
            all_codes = {}  # 각 스테이지별 생성된 코드를 저장할 딕셔너리
            # 스테이지별 한글 이름 매핑 (주석용)
            stage_names = {
                "pre": "전처리 (preprocessing.py)",  # 데이터 로드, 정규화 등
                "model": "모델 설계 (model.py)",  # CNN 아키텍처 정의
                "train": "학습 (training.py)",  # 모델 학습 루프
                "eval": "평가 (evaluation.py)"  # 성능 평가 및 시각화
            }
            
            # 각 스테이지별로 순차적으로 코드 생성
            for s in ["pre", "model", "train", "eval"]:
                if s in code_generators:  # 해당 스테이지의 코드 생성기가 존재하는지 확인
                    code = code_generators[s](form_dict)  # 블록 설정을 바탕으로 Python 코드 생성
                    WorkspaceManager.save_code(uid, s, code)  # 생성된 코드를 파일로 저장
                    WorkspaceManager.save_inputs(uid, s, form_dict)  # 입력값도 JSON으로 저장 (재사용 위해)
                    all_codes[s] = code  # 메모리에도 저장 (통합 코드 생성용)
            
            # 모든 스테이지의 코드를 하나로 통합
            combined_code = ""  # 통합된 코드를 저장할 문자열
            for s in ["pre", "model", "train", "eval"]:
                if s in all_codes:  # 해당 스테이지의 코드가 생성되었는지 확인
                    # 각 스테이지별로 구분선과 함께 코드 추가
                    combined_code += f"# ========== {stage_names[s]} ==========\n\n"
                    combined_code += all_codes[s]  # 실제 Python 코드 추가
                    combined_code += "\n\n"  # 스테이지 간 구분을 위한 빈 줄
            
            return combined_code  # 통합된 전체 코드 반환
            
        # 단일 스테이지만 처리하는 경우
        else:
            # 요청된 스테이지가 유효한지 확인
            if stage not in code_generators:
                raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
            
            # 해당 스테이지의 코드 생성기 호출
            code = code_generators[stage](form_dict)  # 블록 설정 → Python 코드 변환
            WorkspaceManager.save_code(uid, stage, code)  # 생성된 코드를 파일로 저장
            WorkspaceManager.save_inputs(uid, stage, form_dict)  # 입력값도 저장
            
            return code  # 생성된 코드 반환
    
    except Exception as e:
        print(f"[ERROR] /convert: {e}")
        raise HTTPException(status_code=500, detail=f"코드 생성 중 오류 발생: {str(e)}")

@app.options("/convert")
async def convert_options():
    from fastapi import Response
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.post("/run/{stage}")
async def run_stage(
    stage: str,
    user_id: Optional[str] = Form(None)
):
    try:
        if not user_id or user_id.strip() == "":
            raise HTTPException(status_code=400, detail="user_id가 필요합니다.")
        
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id.strip())[:50]
        
        result = ProcessManager.run_script(uid, stage)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/stream")
async def logs_stream(
    user_id: str = Query(...),
    stage: str = Query(default="train")
):
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id가 필요합니다.")
        
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
        
        def generate_logs():
            last_size = 0
            while True:
                try:
                    content, new_size = ProcessManager.get_log_content(uid, stage, last_size)
                    if content:
                        last_size = new_size
                        for line in content.splitlines():
                            yield f"data: {line}\n\n"
                    
                    import time
                    time.sleep(0.3)
                except GeneratorExit:
                    break
                except Exception as e:
                    yield f"data: [error] {e}\n\n"
                    import time
                    time.sleep(1)
        
        return StreamingResponse(
            generate_logs(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{stage}")
async def download_file(
    stage: str,
    user_id: str = Query(...)
):
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id가 필요합니다.")
        
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
        
        from core.config import WORKSPACE_DIR
        workspace_path = WORKSPACE_DIR / uid
        
        if stage == "all":
            import zipfile
            import tempfile
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"workspace_{uid}_{timestamp}.zip"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                    file_map = {
                        "pre": "preprocessing.py",
                        "model": "model.py",
                        "train": "training.py",
                        "eval": "evaluation.py"
                    }
                    
                    for filename in file_map.values():
                        file_path = workspace_path / filename
                        if file_path.exists():
                            zipf.write(file_path, arcname=filename)
                    
                    # inputs 파일들도 포함
                    for s in ["pre", "model", "train", "eval"]:
                        input_file = workspace_path / f"inputs_{s}.json"
                        if input_file.exists():
                            zipf.write(input_file, arcname=input_file.name)
                
                return FileResponse(
                    tmp_file.name,
                    filename=zip_filename,
                    media_type='application/zip'
                )
        else:
            file_map = {
                "pre": "preprocessing.py",
                "model": "model.py",
                "train": "training.py",
                "eval": "evaluation.py"
            }
            
            if stage not in file_map:
                raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
            
            file_path = workspace_path / file_map[stage]
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"{file_map[stage]} 파일이 없습니다.")
            
            return FileResponse(
                file_path,
                filename=file_map[stage],
                media_type='text/plain'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result")
async def get_result(user_id: str = Query(...)):
    """평가 결과 조회 - accuracy, 혼동행렬, 오분류, 예측 샘플 이미지를 base64로 반환"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id가 필요합니다.")
        
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
        
        from core.config import WORKSPACE_DIR
        workspace_path = WORKSPACE_DIR / uid
        artifacts_dir = workspace_path / "artifacts"
        
        print(f"[INFO] /result 요청: user_id={uid}")
        print(f"[INFO] artifacts 경로: {artifacts_dir}")
        
        # 1. accuracy 값 읽기
        results_path = artifacts_dir / "evaluation_results.json"
        accuracy = None
        
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
                accuracy = eval_results.get('accuracy', None)
                print(f"[INFO] Accuracy 값: {accuracy}")
        else:
            print(f"[WARNING] 평가 결과 파일이 없습니다: {results_path}")
        
        # 2. 이미지 파일들을 base64로 인코딩
        def image_to_base64(image_path):
            if not image_path.exists():
                print(f"[WARNING] 이미지 파일이 없습니다: {image_path}")
                return None
            
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    base64_str = base64.b64encode(image_data).decode('utf-8')
                    print(f"[INFO] 이미지 인코딩 완료: {image_path.name} ({len(base64_str)} chars)")
                    return base64_str
            except Exception as e:
                print(f"[ERROR] 이미지 인코딩 실패 {image_path}: {e}")
                return None
        
        # 혼동행렬 이미지
        confusion_matrix_path = artifacts_dir / "confusion_matrix.png"
        confusion_matrix_b64 = image_to_base64(confusion_matrix_path)
        
        # 오분류 샘플 이미지
        misclassified_path = artifacts_dir / "misclassified_samples.png"
        misclassified_b64 = image_to_base64(misclassified_path)
        
        # 예측 샘플 이미지
        prediction_samples_path = artifacts_dir / "prediction_samples.png"
        prediction_samples_b64 = image_to_base64(prediction_samples_path)
        
        # 3. 응답 JSON 구성
        response_data = {
            "ok": True,
            "user_id": uid,
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix_b64,
            "misclassified_samples": misclassified_b64,
            "prediction_samples": prediction_samples_b64,
            "message": "평가 결과를 성공적으로 가져왔습니다."
        }
        
        # 4. 누락된 데이터 체크 및 경고
        missing_items = []
        if accuracy is None:
            missing_items.append("accuracy")
        if not confusion_matrix_b64:
            missing_items.append("confusion_matrix")
        if not misclassified_b64:
            missing_items.append("misclassified_samples")
        if not prediction_samples_b64:
            missing_items.append("prediction_samples")
        
        if missing_items:
            response_data["warning"] = f"일부 데이터가 누락되었습니다: {', '.join(missing_items)}"
            response_data["message"] = "일부 결과가 누락된 상태로 반환되었습니다. 평가를 먼저 실행해주세요."
        
        print(f"[INFO] 응답 데이터 준비 완료. 누락된 항목: {missing_items}")
        
        return response_data
        
    except Exception as e:
        print(f"[ERROR] /result 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"결과 처리 중 오류 발생: {str(e)}")

@app.options("/result")
async def result_options():
    from fastapi import Response
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.get("/result/status")
async def get_result_status(user_id: str = Query(...)):
    """평가 결과 파일 존재 여부 확인"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id가 필요합니다.")
        
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
        
        from core.config import WORKSPACE_DIR
        workspace_path = WORKSPACE_DIR / uid
        artifacts_dir = workspace_path / "artifacts"
        
        # 필요한 파일들 체크
        files_status = {
            "evaluation_results.json": (artifacts_dir / "evaluation_results.json").exists(),
            "confusion_matrix.png": (artifacts_dir / "confusion_matrix.png").exists(),
            "misclassified_samples.png": (artifacts_dir / "misclassified_samples.png").exists(),
            "prediction_samples.png": (artifacts_dir / "prediction_samples.png").exists()
        }
        
        all_ready = all(files_status.values())
        
        return {
            "user_id": uid,
            "ready": all_ready,
            "files": files_status,
            "message": "모든 결과 파일이 준비되었습니다." if all_ready else "일부 결과 파일이 누락되었습니다. 평가를 실행해주세요."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/result/status")
async def result_status_options():
    from fastapi import Response
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.get("/data-info")
async def get_data_info(
    file: str = Query(...),
    type: str = Query(default="shape"),
    n: int = Query(default=5)
):
    """데이터셋 정보 조회"""
    try:
        result = DatasetManager.get_dataset_info(file, type, n)
        if result is None:
            raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/data-info")
async def data_info_options():
    from fastapi import Response
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.get("/debug/dataset")
async def debug_dataset():
    """데이터셋 폴더 상태 디버그"""
    try:
        dataset_files = DatasetManager.list_datasets()
        
        from core.config import DATASET_DIR
        debug_info = {
            "dataset_dir": str(DATASET_DIR),
            "dataset_dir_exists": DATASET_DIR.exists(),
            "csv_files": dataset_files,
            "file_details": []
        }
        
        for filename in dataset_files:
            file_path = DATASET_DIR / filename
            try:
                df = pd.read_csv(file_path)
                debug_info["file_details"].append({
                    "filename": filename,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "first_columns": list(df.columns[:5])
                })
            except Exception as e:
                debug_info["file_details"].append({
                    "filename": filename,
                    "error": str(e)
                })
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    settings.initialize_directories()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )