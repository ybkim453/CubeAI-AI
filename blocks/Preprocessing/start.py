# blocks/Preprocessing/start.py

from .data_selection     import generate_data_selection_snippet, DataSelectionBlock
from .drop_na           import generate_drop_na_snippet
from .drop_bad_labels   import generate_drop_bad_labels_snippet, DropBadLabelsBlock
from .split_xy          import generate_split_xy_snippet
from .resize            import generate_resize_snippet, ResizeBlock
from .augment           import generate_augment_snippet, AugmentBlock
from .normalize         import generate_normalize_snippet, NormalizeBlock
from .logging_utils     import generate_logging_utils_snippet
from .save_data         import generate_save_data_snippet

def generate_preprocessing_snippet(form):
    """
    form: request.form 딕셔너리
    → preprocessing 단계(1~7) 전체 코드를 조립하여 반환
    """
    # 1) form에서 파라미터 꺼내기
    dataset      = form['dataset']
    is_test      = form.get('is_test', '')
    testdataset  = form.get('testdataset','')
    a            = int(form.get('a','100'))
    drop_na_flag = 'drop_na' in form
    drop_bad_flag= 'drop_bad' in form
    min_label    = int(form.get('min_label','0'))
    max_label    = int(form.get('max_label','9'))
    split_xy_flag= 'split_xy' in form
    resize_n     = form.get('resize_n','')
    augment_m    = form.get('augment_method','')
    augment_p    = form.get('augment_param','')
    normalize_m  = form.get('normalize','')

    # 2) 스니펫 조립
    lines = [
        "# 자동 생성된 preprocessing.py",
        "# AI 블록코딩 - 전처리 파이프라인",
        "",
        "import pandas as pd",
        "import torch",
        "import numpy as np",
        "from PIL import Image",
        "from torchvision import transforms",
        "import os",
        "",
    ]
    
    # 로깅 유틸리티 추가
    lines.append(generate_logging_utils_snippet())
    lines.append("")
    
    # 1) 데이터 불러오기
    data_block = DataSelectionBlock(
        dataset=dataset,
        is_test=is_test,
        testdataset=testdataset,
        a=a
    )
    lines.append(generate_data_selection_snippet(data_block))
    lines.append("")

    # 2)–7) 블록별 조건적 삽입
    if drop_na_flag:
        lines.append(generate_drop_na_snippet())
        lines.append("")
        
    if drop_bad_flag:
        bad_labels_block = DropBadLabelsBlock(min_label=min_label, max_label=max_label)
        lines.append(generate_drop_bad_labels_snippet(bad_labels_block))
        lines.append("")
        
    if split_xy_flag:
        lines.append(generate_split_xy_snippet())
        lines.append("")
        
    if resize_n:
        resize_block = ResizeBlock(resize_n=int(resize_n))
        lines.append(generate_resize_snippet(resize_block))
        lines.append("")
        
    if augment_m and augment_p:
        augment_block = AugmentBlock(method=augment_m, param=int(augment_p))
        lines.append(generate_augment_snippet(augment_block))
        lines.append("")
        
    if normalize_m:
        normalize_block = NormalizeBlock(method=normalize_m)
        lines.append(generate_normalize_snippet(normalize_block))
        lines.append("")

    # 데이터 저장 블록 추가
    lines.append(generate_save_data_snippet())
    lines.append("")
    # 최종 문자열 반환
    return "\n".join(lines)
