# ==========================================
# Process Manager
# 프로세스 실행 및 로그 관리
# ==========================================

import os
import subprocess
import threading
import time
from pathlib import Path
from .config import BASE_DIR, WORKSPACE_DIR, LOGS_DIR

class ProcessManager:
    """
    프로세스 실행 관리 클래스
    - subprocess를 통한 격리된 Python 스크립트 실행
    - 실행 로그를 파일로 저장 및 실시간 스트리밍
    - 비동기 로그 처리
    """
    
    @staticmethod
    def run_script(uid: str, stage: str) -> dict:
        """
        생성된 Python 스크립트 실행
        
        Args:
            uid: 사용자 ID
            stage: 실행할 스테이지 (pre/model/train/eval)
        
        Returns:
            dict: 실행 결과 {"ok": True, "pid": 프로세스ID} 또는 {"error": 에러메시지}
        """
        workspace_path = WORKSPACE_DIR / uid
        
        # 스테이지별 스크립트 파일 매핑
        script_map = {
            "pre": "preprocessing.py",  # 전처리
            "model": "model.py",  # 모델 정의
            "train": "training.py",  # 학습
            "eval": "evaluation.py"  # 평가
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
        
        # subprocess로 Python 스크립트 실행
        proc = subprocess.Popen(
            ["python", "-u", str(script_path)],  # -u: unbuffered output
            cwd=str(BASE_DIR),  # 작업 디렉토리
            stdout=subprocess.PIPE,  # stdout 캐처
            stderr=subprocess.STDOUT,  # stderr를 stdout으로 리다이렉트
            env=env,  # 환경 변수 전달
            text=True,  # 텍스트 모드
            bufsize=1  # 라인 버퍼링
        )
        
        def stream_logs():
            """비동기로 로그를 파일에 저장하는 함수"""
            with open(log_path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line)  # 로그 파일에 쓰기
                    f.flush()  # 버퍼 비우기 (실시간 저장)
            proc.wait()  # 프로세스 종료 대기
        
        # 별도 스레드에서 로그 스트리밍 실행
        threading.Thread(target=stream_logs, daemon=True).start()
        
        return {"ok": True, "pid": proc.pid}
    
    @staticmethod
    def get_log_content(uid: str, stage: str, last_size: int = 0) -> tuple[str, int]:
        """
        로그 파일에서 새로운 내용만 읽기 (SSE용)
        
        Args:
            uid: 사용자 ID
            stage: 로그 스테이지
            last_size: 마지막으로 읽은 파일 위치
        
        Returns:
            tuple: (새로운 로그 내용, 현재 파일 위치)
        """
        log_path = LOGS_DIR / f"{uid}_{stage}.log"
        
        if not log_path.exists():
            return "", 0
        
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(last_size)  # 마지막 위치로 이동
                content = f.read()  # 새로운 내용 읽기
                new_size = f.tell()  # 현재 파일 위치 저장
                return content, new_size
        except Exception as e:
            # 에러 발생시 에러 메시지 반환
            return f"[error] {e}", last_size
