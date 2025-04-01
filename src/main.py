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
