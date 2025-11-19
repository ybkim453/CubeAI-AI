# core/dataset.py
# 데이터셋 관리

import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from .config import DATASET_DIR

class DatasetManager:
    """데이터셋 관리"""
    
    @staticmethod
    def list_datasets() -> list[str]:
        """사용 가능한 데이터셋 목록"""
        DATASET_DIR.mkdir(exist_ok=True)
        return sorted([f.name for f in DATASET_DIR.glob("*.csv")])
    
    @staticmethod
    def get_dataset_info(filename: str, info_type: str = "shape", n: int = 5) -> dict:
        """데이터셋 정보 조회"""
        try:
            file_path = DATASET_DIR / filename
            if not file_path.exists():
                print(f"[ERROR] 파일이 존재하지 않습니다: {file_path}")
                return None
            
            print(f"[INFO] CSV 파일 읽는 중: {file_path}")
            df = pd.read_csv(file_path)
            print(f"[INFO] CSV 로드 완료: shape={df.shape}")
            
            if info_type == "shape":
                return {"rows": df.shape[0], "cols": df.shape[1]}
            
            elif info_type == "structure":
                columns = []
                for col in df.columns:
                    try:
                        dtype_str = str(df[col].dtype)
                        columns.append({"name": col, "dtype": dtype_str})
                    except Exception as e:
                        print(f"[WARNING] 컬럼 {col} 타입 확인 실패: {e}")
                        columns.append({"name": col, "dtype": "unknown"})
                return {"columns": columns}
            
            elif info_type == "sample":
                try:
                    sample_data = df.head(n).values.tolist()
                    return {
                        "columns": list(df.columns), 
                        "sample": sample_data
                    }
                except Exception as e:
                    print(f"[ERROR] 샘플 데이터 생성 실패: {e}")
                    return {
                        "columns": list(df.columns), 
                        "sample": []
                    }
            
            elif info_type == "images":
                try:
                    images = []
                    
                    # MNIST 형태 가정: 첫 번째 컬럼이 label, 나머지가 픽셀값
                    if df.shape[1] < 785:  # 784 픽셀 + 1 라벨
                        print(f"[WARNING] 이미지 형태가 아닌 것 같습니다. 컬럼 수: {df.shape[1]}")
                        return {"images": [], "error": "이미지 형태의 데이터가 아닙니다."}
                    
                    for i in range(min(n, len(df))):
                        try:
                            # 첫 번째 컬럼(label) 제외하고 픽셀 데이터 추출
                            pixel_data = df.iloc[i, 1:].values
                            
                            # 28x28로 reshape (MNIST 기준)
                            img_size = int(np.sqrt(len(pixel_data)))
                            if img_size * img_size != len(pixel_data):
                                print(f"[WARNING] 이미지 크기 계산 실패: 픽셀 수={len(pixel_data)}")
                                continue
                            
                            pixels = pixel_data.astype(np.uint8).reshape(img_size, img_size)
                            
                            # PIL Image로 변환
                            img = Image.fromarray(pixels, mode='L')  # 'L'은 grayscale
                            
                            # PNG로 base64 인코딩
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            images.append(img_base64)
                            
                        except Exception as e:
                            print(f"[WARNING] 이미지 {i} 처리 실패: {e}")
                            continue
                    
                    print(f"[INFO] 이미지 생성 완료: {len(images)}개")
                    return {"images": images}
                    
                except Exception as e:
                    print(f"[ERROR] 이미지 처리 전체 실패: {e}")
                    return {"images": [], "error": str(e)}
            
            else:
                return {"error": f"지원하지 않는 info_type: {info_type}"}
                
        except Exception as e:
            print(f"[ERROR] get_dataset_info 전체 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
