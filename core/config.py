from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
WORKSPACE_DIR = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"

class Settings:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    CORS_ORIGINS = [
        "http://localhost:5173",
        "https://4th-security-cube-ai-fe.vercel.app", 
        "http://localhost:5174",
        "http://localhost:9022",
        "http://localhost:9000",
        "http://cubeai.kro.kr/"
    ]
    
    HOST = "127.0.0.1"
    PORT = 9000
    DEBUG = True
    
    @staticmethod
    def initialize_directories():
        for directory in [DATASET_DIR, WORKSPACE_DIR, LOGS_DIR]:
            directory.mkdir(exist_ok=True)

settings = Settings()