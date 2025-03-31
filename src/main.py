from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .core.evaluation import LLMEvaluator, EvaluationResult
from .audio.asr import ASRModule
from typing import Optional
import sys
from pathlib import Path
import asyncio

app = FastAPI()

# ASR连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.asr = ASRModule()
        self.running = False
        self.current_websocket = None

    async def handle_asr_result(self, result: dict):
        if not self.running:
            print("INFO: ASR result received but ASR is marked as not running, ignoring")
            return
            
        if self.current_websocket:
            try:
                # 从识别结果中提取text字段
                text = result.get('text', '')
                if text:
                    await self.current_websocket.send_text(f"识别结果: {text}")
            except RuntimeError as e:
                if "websocket connection is closed" in str(e).lower():
                    print("INFO: WebSocket connection closed, stopping ASR")
                    self.asr.stop()
                    self.running = False
                    self.current_websocket = None
                else:
                    print(f"WebSocket发送错误: {e}")
            except Exception as e:
                print(f"WebSocket发送错误: {e}")
                # If we can't send to the websocket, it might be disconnected
                if self.current_websocket not in self.active_connections:
                    print("INFO: WebSocket no longer in active connections, stopping ASR")
                    self.asr.stop()
                    self.running = False
                    self.current_websocket = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # If the disconnected websocket was the one using ASR, stop it
        if websocket == self.current_websocket:
            print("INFO: Stopping ASR module as the controlling WebSocket disconnected")
            self.asr.stop()
            self.running = False
            self.current_websocket = None
        # Also stop if no connections remain
        elif not self.active_connections and self.running:
            print("INFO: Stopping ASR module as no active connections remain")
            self.asr.stop()
            self.running = False

    async def start_asr(self, websocket: WebSocket):
        if not self.running:
            self.running = True
            self.current_websocket = websocket
            self.asr.start(result_callback=self.handle_asr_result)
            await self.stream_audio(websocket)

    async def stream_audio(self, websocket: WebSocket):
        try:
            while self.running and self.asr.stream:
                # Check if websocket is still connected
                if websocket not in self.active_connections:
                    print("INFO: WebSocket no longer in active connections during streaming, stopping ASR")
                    self.asr.stop()
                    self.running = False
                    self.current_websocket = None
                    break
                    
                data = self.asr.stream.read(3200, exception_on_overflow=False)
                self.asr.recognition.send_audio_frame(data)
                await asyncio.sleep(0.1)
        except RuntimeError as e:
            if "websocket connection is closed" in str(e).lower():
                print("INFO: WebSocket connection closed during streaming, stopping ASR")
            else:
                print(f"ASR streaming error: {e}")
            self.asr.stop()
            self.running = False
            self.current_websocket = None
        except Exception as e:
            print(f"ASR streaming error: {e}")
            try:
                # Only try to send error if websocket is still in active connections
                if websocket in self.active_connections:
                    await websocket.send_text(f"ASR error: {str(e)}")
            except Exception:
                # Ignore errors when trying to send to potentially closed websocket
                pass
            self.asr.stop()
            self.running = False
            self.current_websocket = None

manager = ConnectionManager()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

class AnalyzeRequest(BaseModel):
    sample: str
    machineResponse: str

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    print(f"INFO: New WebSocket connection established")
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "start":
                print(f"INFO: Received 'start' command from WebSocket")
                await manager.start_asr(websocket)
            elif data == "stop":
                print(f"INFO: Received 'stop' command from WebSocket")
                manager.asr.stop()
                manager.running = False
                await websocket.send_text("ASR stopped")
            else:
                print(f"INFO: Received unknown command from WebSocket: {data}")
    except WebSocketDisconnect:
        print(f"INFO: WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"ERROR: WebSocket error: {e}")
        manager.disconnect(websocket)
        raise
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest) -> EvaluationResult:
    """评估车机系统响应"""
    try:
        evaluator = LLMEvaluator("openrouter")
        result = evaluator.evaluate(request.sample, request.machineResponse)
        print(result)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"评估过程中发生错误: {str(e)}"
        )

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    sys.path.append(str(Path(__file__).parent.parent))
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
