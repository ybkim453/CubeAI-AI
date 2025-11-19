# blocks/ModelDesign/start.py

from .activation      import generate_activation_code, ActivationLayer
from .cnn_create      import generate_cnn_class_code
from .conv2d          import generate_conv2d_code, Conv2DLayer
from .dropout         import generate_dropout_code, DropoutLayer
from .fc_final        import generate_fc_final_code, OutputLayer
from .fc_layer        import generate_fc_layer_code, FCLayer
from .pooling         import generate_pooling_code, PoolingLayer
from .model_methods   import generate_model_methods_snippet
from .helper_functions import generate_helper_functions_snippet
from .test_code       import generate_test_code_snippet

def generate_modeldesign_snippet(form):
    """
    form: request.form 딕셔너리
    → 모델 설계 단계 전체 스니펫 생성
    """
    # form에서 파라미터 추출
    in_channels = form.get('in_channels', '')
    out_channels = form.get('out_channels', '')
    kernel_size = form.get('kernel_size', '')
    padding = form.get('padding', '')
    p = form.get('p', '')
    activation_type = form.get('activation_type', '')
    pool_type = form.get('pool_type', '')
    size = form.get('size', '')
    dense_input_size = form.get('dense_input_size', '')
    dense_output_size = form.get('dense_output_size', '')
    num_classes = form.get('num_classes', '')

    lines = [
        "# 자동 생성된 model.py",
        "# AI 블록코딩 - CNN 모델 정의",
        "",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        ""
    ]
    # 모델 설계 블록
    lines.append("# -----------------------------")
    lines.append("# --------- [모델설계 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("")

    # CNN 클래스 시작
    lines.append(generate_cnn_class_code())
    lines.append("")

    # Conv2D 레이어
    if in_channels and out_channels and kernel_size and padding:
        conv_block = Conv2DLayer(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(padding)
        )
        lines.append(generate_conv2d_code(conv_block))
        lines.append("")

    # Activation
    if activation_type:
        act_block = ActivationLayer(activation_type=activation_type)
        lines.append(generate_activation_code(act_block))
        lines.append("")

    # Pooling
    if pool_type and size:
        pool_block = PoolingLayer(pool_type=pool_type, size=int(size))
        lines.append(generate_pooling_code(pool_block))
        lines.append("")

    # Dropout (선택적)
    if p:
        dropout_block = DropoutLayer(p=float(p))
        lines.append(generate_dropout_code(dropout_block))
        lines.append("")

    # FC Layer
    if dense_input_size and dense_output_size:
        fc_block = FCLayer(
            dense_input_size=int(dense_input_size),
            dense_output_size=int(dense_output_size)
        )
        lines.append(generate_fc_layer_code(fc_block))
        lines.append("")

    # Output Layer (Final FC)
    if num_classes and dense_output_size:
        output_block = OutputLayer(
            num_classes=int(num_classes),
            dense_output_size=int(dense_output_size)
        )
        lines.append(generate_fc_final_code(output_block))
        lines.append("")

    # Forward 메서드 추가
    lines.append(generate_model_methods_snippet())
    
    # 헬퍼 함수들 추가
    lines.append(generate_helper_functions_snippet())
    
    # 테스트 코드 추가
    lines.append(generate_test_code_snippet())

    return "\n".join(lines)
