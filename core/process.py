# core/process.py
# 프로세스 실행 관리

import os
import subprocess
import threading
import time
from pathlib import Path
from .config import BASE_DIR, WORKSPACE_DIR, LOGS_DIR

class ProcessManager:
    """프로세스 실행 관리"""
    
    @staticmethod
    def run_script(uid: str, stage: str) -> dict:
        """스크립트 실행"""
        workspace_path = WORKSPACE_DIR / uid
        
        script_map = {
            "pre": "preprocessing.py",
            "model": "model.py",
            "train": "training.py", 
            "eval": "evaluation.py"
        }
        
        if stage not in script_map:
            return {"error": "Unknown stage"}
        
        script_path = workspace_path / script_map[stage]
        if not script_path.exists():
            return {"error": f"{script_map[stage]} not found"}
        
        log_path = LOGS_DIR / f"{uid}_{stage}.log"
        log_path.unlink(missing_ok=True)
        
        env = os.environ.copy()
        env["AIB_WORKDIR"] = str(workspace_path)
        
        proc = subprocess.Popen(
            ["python", "-u", str(script_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1
        )
        
        def stream_logs():
            with open(log_path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
            proc.wait()
        
        threading.Thread(target=stream_logs, daemon=True).start()
        
        return {"ok": True, "pid": proc.pid}
    
    @staticmethod
    def get_log_content(uid: str, stage: str, last_size: int = 0) -> tuple[str, int]:
        """로그 내용 가져오기"""
        log_path = LOGS_DIR / f"{uid}_{stage}.log"
        
        if not log_path.exists():
            return "", 0
        
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(last_size)
                content = f.read()
                new_size = f.tell()
                return content, new_size
        except Exception as e:
            return f"[error] {e}", last_size
