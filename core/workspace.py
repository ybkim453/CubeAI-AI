import os
import json
import time
import re
from pathlib import Path
from .config import WORKSPACE_DIR

class WorkspaceManager:
    
    @staticmethod
    def get_or_create_uid(user_id: str = None) -> tuple[str, Path]:
        """UID 가져오기 또는 생성"""
        created = False
        
        if not user_id:
            user_id = "anonymous"
            created = True
        
        # 안전한 파일명으로 변환 (특수문자 제거)
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]  # 최대 50자, 안전한 문자만
        
        workspace_path = WORKSPACE_DIR / uid
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        if created:
            readme_path = workspace_path / "README.txt"
            readme_path.write_text(
                "AI 블록코딩 워크스페이스\n"
                f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"User ID: {uid}\n",
                encoding="utf-8"
            )
            
            (workspace_path / "data").mkdir(exist_ok=True)
            (workspace_path / "artifacts").mkdir(exist_ok=True)
            (workspace_path / "logs").mkdir(exist_ok=True)
        
        return uid, workspace_path
    
    @staticmethod
    def save_code(uid: str, stage: str, code: str):
        """생성된 코드 저장"""
        workspace_path = WORKSPACE_DIR / uid
        
        filename_map = {
            "pre": "preprocessing.py",
            "model": "model.py", 
            "train": "training.py",
            "eval": "evaluation.py"
        }
        
        if stage in filename_map:
            file_path = workspace_path / filename_map[stage]
            file_path.write_text(code, encoding="utf-8")
    
    @staticmethod
    def save_inputs(uid: str, stage: str, form_data: dict):
        """폼 입력값 저장"""
        workspace_path = WORKSPACE_DIR / uid
        
        inputs = {}
        for key, value in form_data.items():
            if isinstance(value, list):
                inputs[key] = value if len(value) > 1 else value[0] if value else ""
            else:
                inputs[key] = value
        
        json_path = workspace_path / f"inputs_{stage}.json"
        json_path.write_text(
            json.dumps(inputs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    @staticmethod
    def load_inputs(uid: str, stage: str = None) -> dict:
        """저장된 입력값 로드"""
        workspace_path = WORKSPACE_DIR / uid
        
        if stage:
            json_path = workspace_path / f"inputs_{stage}.json"
            if json_path.exists():
                try:
                    return json.loads(json_path.read_text(encoding="utf-8"))
                except:
                    return {}
        else:
            merged = {}
            for s in ["pre", "model", "train", "eval"]:
                json_path = workspace_path / f"inputs_{s}.json"
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        merged.update(data)
                    except:
                        pass
            return merged
    
    @staticmethod
    def load_snippets(uid: str) -> dict:
        """모든 코드 스니펫 로드"""
        workspace_path = WORKSPACE_DIR / uid
        
        snippets = {}
        file_map = {
            "snippet_pre": "preprocessing.py",
            "snippet_model": "model.py",
            "snippet_train": "training.py", 
            "snippet_eval": "evaluation.py"
        }
        
        for key, filename in file_map.items():
            file_path = workspace_path / filename
            snippets[key] = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        
        return snippets
