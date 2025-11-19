# blocks/Preprocessing/logging_utils.py
# 전처리 로깅 유틸리티

def generate_logging_utils_snippet():
    """
    로깅 유틸리티 함수들 생성
    """
    lines = []
    lines.append("# 로깅 유틸리티")
    lines.append("import sys")
    lines.append("import time")
    lines.append("import datetime")
    lines.append("")
    lines.append("def _ts():")
    lines.append("    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')")
    lines.append("")
    lines.append("def log(msg):")
    lines.append("    sys.stdout.write(f\"[pre][{_ts()}] {msg}\\n\")")
    lines.append("    sys.stdout.flush()")
    lines.append("")
    lines.append("log('=== PREPROCESSING START ===')")
    lines.append("")
    return "\n".join(lines)

