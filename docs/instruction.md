# 车机语音操作llm自动化测试开发指导文档

将指令语料（文本）输入给车机（通过语音）车机在识别到相应指令后作出相应的语音回复，该自动化测试软件将车机的语音回复转换为文本，将该文本与发出的指令文本一起喂给大模型，让大模型通过发出的指令与车机的回复判断该语音操作流程是否成功

eg：“打开空调”指令输入给车机，车机在识别到该指令后回复：“空调已打开”，该自动化测试应用将输入的“打开空调”与车机的回复“空调已打开”一起给到大模型，大模型识别并给出该语音操作成功的回复

## 模块分解与开发指南

### 1. 测试用例管理模块
#### 功能描述
- 支持CSV/JSON/YAML格式的测试用例批量导入
- 提供测试用例的增删改查功能
- 数据有效性验证（指令格式校验）

#### 开发步骤
1. 文件解析器实现
```python
import pandas as pd
import yaml
import json

class TestCaseLoader:
    def load_csv(self, file_path):
        return pd.read_csv(file_path).to_dict('records')
  
    def load_json(self, file_path):
        with open(file_path) as f:
            return json.load(f)
  
    def load_yaml(self, file_path):
        with open(file_path) as f:
            return yaml.safe_load(f)
```

2. 数据结构验证
```python
from pydantic import BaseModel

class TestCase(BaseModel):
    instruction: str
    expected_response: str
    environment: dict = None
```

3. 异常处理
```python
try:
    cases = loader.load_csv("test_cases.csv")
except Exception as e:
    logging.error(f"文件加载失败: {str(e)}")
    raise InvalidFormatError("不支持的文件格式")
```

### 2. 语音模拟输入模块
#### 功能实现要点
- TTS引擎接口抽象化
- 多平台播放支持（ADB命令/蓝牙协议）
- 音频格式转换（16kHz/16bit PCM）

#### 代码示例
```python
from gtts import gTTS
import sounddevice as sd

class VoiceSimulator:
    def __init__(self, tts_service="google"):
        self.tts_service = tts_service
      
    def generate_audio(self, text):
        tts = gTTS(text=text, lang='zh-CN')
        tts.save("output.mp3")
        return self.convert_to_pcm("output.mp3")
  
    def play_via_adb(self, file):
        subprocess.run(["adb", "push", file, "/sdcard/"])
        subprocess.run(["adb", "shell", "am", "start", "-a", 
                       "android.intent.action.VIEW", "-t", "audio/wav", 
                       "-d", f"file:///sdcard/{file}"])
```

### 3. 车机响应捕获模块
#### 实现方案
1. 音频采集参数配置
```python
RECORD_CONFIG = {
    "format": "int16",
    "channels": 1,
    "rate": 16000,
    "frames_per_buffer": 1024
}
```

2. 多接口支持
```python
import serial
import sounddevice as sd

class ResponseRecorder:
    def record_via_serial(self, port):
        ser = serial.Serial(port, 115200)
        return ser.read_all()
  
    def record_audio(self, duration=10):
        return sd.rec(int(duration * RECORD_CONFIG["rate"]), 
                     samplerate=RECORD_CONFIG["rate"],
                     channels=RECORD_CONFIG["channels"])
```

### 4. 语音转文本模块
#### ASR接口适配
```python
class ASRProcessor:
    def __init__(self, engine="whisper"):
        self.engine = load_model(engine)
      
    def transcribe(self, audio):
        if self.engine == "whisper":
            return whisper.transcribe(audio)
        elif self.engine == "iflytek":
            return iflytek_asr(audio)
```

### 5. LLM评估引擎
#### 评估逻辑实现
1. Prompt工程
```python
EVALUATION_PROMPT = """
作为汽车语音系统测试专家，请严格评估以下交互：
指令：[{instruction}]
响应：[{response}]

评估维度：
1. 语义正确性（是否准确理解指令意图）
2. 操作有效性（是否触发正确系统行为）
3. 响应规范性（是否符合交互设计规范）

返回JSON格式：{
    "valid": bool,
    "reason": str,
    "confidence": 0-1
}
"""
```

2. 结果解析
```python
def parse_llm_response(response):
    try:
        result = json.loads(response)
        return result["valid"], result["confidence"]
    except:
        logging.warning("LLM响应解析失败")
        return False, 0.0
```

### 6. 异常处理模块
#### 错误类型处理
```python
ERROR_HANDLERS = {
    "TimeoutError": lambda: retry(max_attempts=3),
    "ASRError": lambda: switch_engine(),
    "LLMError": lambda: cache_and_retry(),
    "DeviceNotResponding": lambda: reboot_device()
}
```

### 7. 多环境模拟模块
#### 噪声生成实现
```python
import numpy as np

def add_noise(audio, snr=15):
    noise = np.random.normal(0, 10**(-snr/20), len(audio))
    return audio + noise
```

## 系统集成测试
### 端到端测试流程
```bash
# 运行测试套件示例
python main.py \
    --test-case test_cases/导航.yaml \
    --environment noise=20db \
    --asr-engine whisper \
    --llm gpt-4
```

## 性能优化建议
1. 并行处理架构
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_test_case, case) 
              for case in test_cases]
```

2. 结果缓存机制
```python
@lru_cache(maxsize=100)
def llm_evaluation(instruction, response):
    return call_llm_api(instruction, response)
```

## 质量保障措施
1. 标准测试集构建
```markdown
测试集应包含：
- 基础功能指令（导航/音乐/空调）200条
- 边界测试指令（超长语句/特殊字符）50条 
- 多方言测试指令（粤语/四川话）100条
- 噪声环境测试（信噪比<15dB）50条
```

2. 持续集成配置
```yaml
# .github/workflows/test.yml
jobs:
  automation-test:
    steps:
      - name: Run Core Tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Performance Check
        run: python benchmark.py --threshold 5s
```

## 部署说明
### 环境要求
```markdown
- Python 3.10+
- FFmpeg（音频处理）
- ADB工具（Android设备连接）
- CUDA 11.8（GPU加速ASR）
```

### 设备配置示例
```yaml
# device_config.yaml
connection:
  type: usb
  params:
    vid: 0x1234
    pid: 0x5678
audio:
  input_device: 3
  output_device: 2
```

