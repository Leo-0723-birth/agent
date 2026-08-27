"""FastAPI 入口 —— 监管问询扫雷预警系统 API。

启动方式：
    cd D:\\competition_agent
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

或直接：
    python api/main.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import AnalyzeRequest, AnalyzeResponse, ProgressMessage, ScanRequest, ScanResponse
from .pipeline import (
    create_task,
    get_task,
    list_available_companies,
    offline_to_response,
    run_pipeline,
    run_scan_task,
)

app = FastAPI(title="监管问询扫雷预警系统 API", version="2.0")

# CORS（开发期全开）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze", response_model=list[AnalyzeResponse])
async def analyze(req: AnalyzeRequest):
    """核心接口：输入公司代码列表，返回风险分析结果（离线数据）。"""
    return await run_pipeline(req.codes, req.window, req.use_llm, req.use_bge)


@app.get("/api/company/{code}", response_model=AnalyzeResponse)
async def get_company(code: str):
    """单公司查询（离线）。"""
    results = await run_pipeline([code], 60, False, True)
    if not results:
        raise HTTPException(status_code=404, detail=f"未找到 {code} 的离线报告")
    return results[0]


@app.get("/api/companies")
async def companies():
    """返回所有有离线报告的公司列表。"""
    return list_available_companies()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agents": 7,
        "online": True,
        "mode": "hybrid",           # 支持离线 + 实时
        "realtime_ready": True,
    }


# ==================== 方案 C：实时扫雷 ====================

@app.post("/api/scan", response_model=ScanResponse)
async def scan(req: ScanRequest):
    """创建实时扫雷任务，返回 task_id。前端用 task_id 连接 WebSocket。"""
    state = create_task(req.code)
    # 启动后台任务（不 await，立即返回 task_id）
    import asyncio
    asyncio.create_task(run_scan_task(state, req))
    return ScanResponse(
        task_id=state.task_id,
        code=req.code,
        status=state.status,
        message=f"任务已创建，请连接 ws://host/ws/pipeline/{state.task_id}",
    )


@app.get("/api/scan/{task_id}")
async def scan_status(task_id: str):
    """查询任务当前状态（WebSocket 断线重连时可用）。"""
    state = get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "task_id": state.task_id,
        "code": state.code,
        "status": state.status,
        "error": state.error,
        "progress": [p.model_dump(exclude_none=True) for p in state.progress],
    }


@app.websocket("/ws/pipeline/{task_id}")
async def pipeline_websocket(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送扫雷进度。"""
    await websocket.accept()
    state = get_task(task_id)
    if not state:
        await websocket.send_json({
            "type": "error",
            "message": f"任务 {task_id} 不存在",
        })
        await websocket.close(code=1008)
        return

    # 先把已有进度补发（断线重连场景）
    for p in state.progress:
        await websocket.send_json(p.model_dump(exclude_none=True))

    # 如果任务已经完成/失败，直接发送结果后关闭
    if state.status == "completed" and state.result:
        await websocket.send_json(ProgressMessage(
            type="complete",
            step=7,
            total=7,
            agent="Reporter",
            status="done",
            message="报告生成完成",
            elapsed_ms=0,
            result=state.result,
        ).model_dump(exclude_none=True))
        await websocket.close(code=1000)
        return

    if state.status == "failed":
        await websocket.send_json(ProgressMessage(
            type="error",
            step=0,
            total=7,
            agent="SweepingOrchestrator",
            status="error",
            message=state.error or "任务失败",
            elapsed_ms=0,
            error=state.error,
        ).model_dump(exclude_none=True))
        await websocket.close(code=1000)
        return

    # 实时从 queue 读取新进度
    try:
        while state.status in ("pending", "running"):
            try:
                msg = await asyncio.wait_for(state.queue.get(), timeout=1.0)
                state.progress.append(msg)
                await websocket.send_json(msg.model_dump(exclude_none=True))
                if msg.type in ("complete", "error"):
                    # 再读取一次，确保没有遗漏；随后优雅关闭
                    await asyncio.sleep(0.3)
                    break
            except asyncio.TimeoutError:
                # 发送心跳保持连接
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close(code=1000)
        except Exception:
            pass


# 挂载静态文件（放在最后，避免覆盖 API 路由）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
