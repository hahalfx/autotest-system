# 车机大模型视觉验证系统

## 简介

车机大模型视觉验证系统是一个专为车载信息娱乐系统(IVI)设计的自动化视觉验证工具。该系统能够通过摄像头实时捕获车机屏幕，分析UI元素、识别文本、检测状态变化，并验证车机系统是否按照预期响应用户指令。

## 主要功能

- **实时视频处理**：通过本地摄像头捕获车机屏幕
- **OCR文本识别**：识别屏幕上的文本内容
- **目标检测**：检测UI元素（按钮、图标等）
- **状态变化分析**：检测UI元素状态变化
- **验证规则引擎**：支持多种验证规则和复合逻辑
- **结果报告**：生成详细的验证报告

## 系统架构

```
视觉验证系统
├── 配置管理
├── 视频处理
│   ├── 摄像头控制
│   ├── 帧缓冲队列
│   └── 预处理流水线
├── 分析引擎
│   ├── OCR引擎
│   ├── 目标检测器
│   └── 状态分析器
└── 验证逻辑
    ├── 规则引擎
    └── 结果聚合
```

## 安装

### 依赖项

- Python 3.8+
- OpenCV
- EasyOCR
- ONNX Runtime
- PyYAML
- NumPy
- Loguru

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/autotest-system.git
cd autotest-system
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备模型文件（可选）：

如果需要使用目标检测功能，请下载YOLOv8模型文件并放置在`models`目录中：

```bash
mkdir -p models
# 下载YOLOv8s.onnx模型文件到models目录
```

## 使用方法

### 基本用法

```python
from src.vision import VisionValidator, ROI, OCRValidationRule, ValidationOperator

# 创建验证器实例
validator = VisionValidator()

# 初始化系统
validator.initialize()

# 添加验证规则
ocr_rule = OCRValidationRule(
    name="temperature_test",
    roi=ROI(0, 0, 640, 100, name="top_area", relative=True),
    operator=ValidationOperator.CONTAINS,
    expected_text="温度",
    timeout=10.0,
    description="检测屏幕顶部是否包含'温度'文本"
)
validator.add_validation_rule(ocr_rule)

# 启动系统
validator.start()

# 等待验证完成
# ...

# 获取验证结果
results = validator.get_all_validation_results()
summary = validator.get_validation_summary()

# 导出结果
validator.export_validation_results("validation_results.json")

# 释放资源
validator.release()
```

### 运行示例

项目包含一个基本示例，展示如何使用视觉验证系统：

```bash
python examples/basic_validation.py
```

## 配置

系统使用YAML配置文件，默认配置位于`src/vision/config/default_config.yaml`。您可以创建自定义配置文件并在初始化时指定：

```python
validator = VisionValidator(config_path="path/to/your/config.yaml")
```

### 配置示例

```yaml
camera:
  device_index: 0
  resolution: [1280, 720]
  fps: 30

analysis:
  ocr:
    languages: ["ch_sim", "en"]
    gpu: true
  detector:
    model_path: "./models/yolov8s.onnx"
    confidence: 0.6
```

## 验证规则

系统支持以下类型的验证规则：

1. **OCR验证规则**：验证屏幕上的文本内容
2. **目标检测规则**：验证UI元素的存在与否
3. **状态变化规则**：验证UI元素状态变化
4. **复合规则**：组合多个规则进行逻辑验证（AND/OR）

### 验证操作符

- `EQUALS`：等于
- `CONTAINS`：包含
- `STARTS_WITH`：以...开头
- `ENDS_WITH`：以...结尾
- `REGEX`：正则表达式匹配
- `PRESENT`：存在
- `NOT_PRESENT`：不存在
- `CHANGED`：已变化
- `NOT_CHANGED`：未变化
- `AND`：逻辑与
- `OR`：逻辑或

## 自定义扩展

### 添加自定义分析器

您可以通过继承`BaseAnalyzer`类来创建自定义分析器：

```python
from src.vision.analysis.base import BaseAnalyzer, AnalysisResult

class MyCustomAnalyzer(BaseAnalyzer):
    def initialize(self) -> bool:
        # 初始化代码
        return True
        
    def analyze(self, frame, roi=None) -> AnalysisResult:
        # 分析代码
        return AnalysisResult(success=True, data={"result": "custom_data"})
```

### 添加自定义验证规则

您可以通过继承`ValidationRule`类来创建自定义验证规则：

```python
from src.vision.validation.validation_logic import ValidationRule, ValidationType

class MyCustomRule(ValidationRule):
    def __init__(self, name, **kwargs):
        super().__init__(type=ValidationType.CUSTOM, name=name, **kwargs)
        
    def validate(self, result) -> bool:
        # 验证逻辑
        return True
```

## 注意事项

- 确保摄像头能够清晰捕获车机屏幕，避免反光和遮挡
- 对于OCR识别，建议使用高分辨率摄像头以提高文本识别准确率
- 目标检测功能需要预先训练的模型，默认使用YOLOv8s

## 故障排除

- **摄像头无法访问**：检查设备索引是否正确，以及摄像头是否被其他应用程序占用
- **OCR识别不准确**：调整摄像头位置，确保文本清晰可见，或调整OCR配置参数
- **目标检测失败**：检查模型文件路径是否正确，以及是否支持目标类别

## 许可证

[MIT License](LICENSE)