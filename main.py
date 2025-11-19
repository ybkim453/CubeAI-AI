import os
import io
import base64
import json
import time
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import pandas as pd
import numpy as np
from PIL import Image

from core.config import settings
from core.workspace import WorkspaceManager
from core.process import ProcessManager
from core.dataset import DatasetManager

from blocks.Preprocessing.start import generate_preprocessing_snippet
from blocks.ModelDesign.start import generate_modeldesign_snippet
from blocks.Training.start import generate_training_snippet
from blocks.Evaluation.start import generate_evaluation_snippet

app = FastAPI(
    title="CubeAI AI Sever API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

code_generators = {
    "pre": generate_preprocessing_snippet,
    "model": generate_modeldesign_snippet, 
    "train": generate_training_snippet,
    "eval": generate_evaluation_snippet
}

@app.get("/")
async def root():
    return RedirectResponse(url="/app")

@app.get("/app")
async def main_app(user_id: Optional[str] = Query("anonymous")):
    """메인 애플리케이션 페이지 - 원본에서는 HTML 템플릿 렌더링"""
    # FastAPI에서는 HTML 템플릿 대신 JSON으로 데이터 반환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
    
    from core.config import WORKSPACE_DIR
    workspace_path = WORKSPACE_DIR / uid
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    context = {
        "options": DatasetManager.list_datasets(),
        "form_state": WorkspaceManager.load_inputs(uid),
        "current_user_id": user_id,
        **WorkspaceManager.load_snippets(uid)
    }
    
    return context

@app.get("/app/{user_id}")
async def main_app_with_user(user_id: str):
    """사용자 ID가 포함된 URL 지원"""
    return RedirectResponse(url=f"/app?user_id={user_id}")

@app.post("/convert", response_class=PlainTextResponse)
async def convert_code(
    stage: str = Form(...),
    user_id: Optional[str] = Form(None),
    **form_data
):
    try:
        uid, _ = WorkspaceManager.get_or_create_uid(user_id)
        
        if uid == "anonymous" and not user_id:
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다.")
        
        print(f"[INFO] /convert 요청: stage={stage}, user_id={uid}")
        
        form_dict = dict(form_data)
        if user_id:
            form_dict["user_id"] = user_id
        
        if stage == "all":
            all_codes = {}
            stage_names = {
                "pre": "전처리 (preprocessing.py)",
                "model": "모델 설계 (model.py)", 
                "train": "학습 (training.py)",
                "eval": "평가 (evaluation.py)"
            }
            
            for s in ["pre", "model", "train", "eval"]:
                if s in code_generators:
                    code = code_generators[s](form_dict)
                    WorkspaceManager.save_code(uid, s, code)
                    WorkspaceManager.save_inputs(uid, s, form_dict)
                    all_codes[s] = code
            
            combined_code = ""
            for s in ["pre", "model", "train", "eval"]:
                if s in all_codes:
                    combined_code += f"# ========== {stage_names[s]} ==========\n\n"
                    combined_code += all_codes[s]
                    combined_code += "\n\n"
            
            return combined_code
            
        else:
            if stage not in code_generators:
                raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
            
            code = code_generators[stage](form_dict)
            WorkspaceManager.save_code(uid, stage, code)
            WorkspaceManager.save_inputs(uid, stage, form_dict)
            
            return code
    
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