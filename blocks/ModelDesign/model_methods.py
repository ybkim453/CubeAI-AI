# blocks/ModelDesign/model_methods.py
# CNN 모델의 메서드들

def generate_model_methods_snippet():
    """
    CNN 클래스의 forward 메서드와 기타 메서드들 생성
    """
    lines = []
    lines.append("")
    lines.append("\t" + "def forward(self, x):")
    lines.append("\t\t" + "\"\"\"순전파\"\"\"")
    lines.append("\t\t" + "x = self.conv_layers(x)")
    lines.append("\t\t" + "x = self.fc_layers(x)")
    lines.append("\t\t" + "return x")
    lines.append("")
    lines.append("\t" + "def get_num_parameters(self):")
    lines.append("\t\t" + "\"\"\"파라미터 수 계산\"\"\"")
    lines.append("\t\t" + "return sum(p.numel() for p in self.parameters())")
    lines.append("")
    return "\n".join(lines)

