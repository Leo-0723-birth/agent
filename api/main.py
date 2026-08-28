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

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import AnalyzeRequest, AnalyzeResponse, ProgressMessage, ScanRequest, ScanResponse
from .pipeline import (
    AGENT_TOTAL,
    TERMINAL_STATUSES,
    active_tasks,
    agent_metadata,
    bind_task_handle,
    cancel_task,
    create_task,
    get_cached_result,
    get_model_metrics,
    get_offline_result,
    get_report_download_path,
    get_task,
    list_available_companies,
    offline_to_response,
    run_pipeline,
    run_scan_task,
    subscribe_task,
    unsubscribe_task,
)
from backend.skills.evaluator import (
    get_alerts,
    get_confusion_matrix,
    get_dashboard_metrics,
    get_review_queue,
    get_trend,
    submit_feedback,
)
from backend.skills.stock_code import StockCodeError, normalize_stock_code

app = FastAPI(title="监管问询扫雷预警系统 API", version="2.0")

# CORS（开发期全开）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 可选 API 鉴权：设置环境变量 API_KEY 后，/api/* 需携带 X-API-Key 头；
# 未配置时默认关闭，不影响本地演示。
API_KEY = os.getenv("API_KEY", "").strip()


@app.middleware("http")
async def api_key_middleware(request, call_next):
    if API_KEY and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if request.headers.get("x-api-key", "") != API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "invalid or missing api key"})
    return await call_next(request)


# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze", response_model=list[AnalyzeResponse])
def analyze(req: AnalyzeRequest):
    """核心接口：输入公司代码列表，返回风险分析结果（离线数据）。"""
    return run_pipeline(req.codes, req.window, req.use_llm, req.use_bge)


@app.get("/api/company/{code}", response_model=AnalyzeResponse)
def get_company(code: str):
    """单公司查询（离线）。"""
    results = run_pipeline([code], 60, False, True)
    if not results:
        raise HTTPException(status_code=404, detail=f"未找到 {code} 的离线报告")
    return results[0]


@app.get("/api/reports/{code}/download")
def download_report(code: str, format: str = "md"):
    """下载某公司最新报告的 Markdown 或 JSON 文件。"""
    try:
        normalized = normalize_stock_code(code)
    except StockCodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    fmt = "json" if format.lower() == "json" else "md"
    path = get_report_download_path(normalized, fmt)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"未找到 {normalized} 的报告文件")
    media_type = "application/json" if fmt == "json" else "text/markdown; charset=utf-8"
    filename = f"{normalized}_risk_report.{fmt}"
    return FileResponse(str(path), media_type=media_type, filename=filename)


@app.get("/api/companies")
def companies():
    """返回所有有离线报告的公司列表。"""
    return list_available_companies()


@app.get("/api/agents")
def agents():
    """返回 7 个 Agent 的元数据（前端流水线清单的单一来源）。"""
    return agent_metadata()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agents": AGENT_TOTAL,
        "online": True,
        "mode": "hybrid",           # 支持离线 + 实时
        "realtime_ready": True,
    }


@app.get("/api/model-metrics")
def model_metrics():
    """返回当前训练好的模型在测试集上的官方指标（AUC / F1 / Top10%Recall）。"""
    metrics = get_model_metrics()
    if not metrics or all(metrics.get(h) is None for h in ("30d", "60d", "90d")):
        raise HTTPException(status_code=404, detail="未找到模型指标，请先运行训练脚本")
    return metrics


# ==================== 方案 C：实时扫雷 ====================

@app.post("/api/scan", response_model=ScanResponse)
async def scan(req: ScanRequest):
    """默认秒返离线快照；realtime=true 时创建实时 Agent 任务。"""
    try:
        code = normalize_stock_code(req.code)
    except StockCodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    req = req.model_copy(update={"code": code})

    if not req.realtime:
        result = get_offline_result(code, req.window, req.as_of)
        if result is None:
            raise HTTPException(status_code=404, detail=f"未找到 {code} 的离线报告")
        return ScanResponse(
            task_id="cached", code=code, status="completed", cached=True,
            message="已返回离线快照", result=result,
        )

    running = active_tasks()
    if running and not req.force:
        current = running[0]
        raise HTTPException(
            status_code=409,
            detail={
                "message": "当前有实时扫雷任务正在执行，请先取消或使用 force=true 切换。",
                "task_id": current.task_id,
                "code": current.code,
            },
        )
    if running and req.force:
        for current in running:
            await cancel_task(current.task_id)

    state = create_task(code)
    handle = asyncio.create_task(run_scan_task(state, req), name=state.task_id)
    bind_task_handle(state.task_id, handle)
    return ScanResponse(
        task_id=state.task_id,
        code=code,
        status=state.status,
        message=f"任务已创建，请连接 /ws/pipeline/{state.task_id}",
    )


@app.delete("/api/scan/{task_id}")
async def delete_scan(task_id: str):
    state = await cancel_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"task_id": state.task_id, "code": state.code, "status": state.status}


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
        "result": state.result.model_dump() if state.result else None,
    }


@app.get("/api/mock/{code}", response_model=AnalyzeResponse)
async def mock_company(code: str):
    """前端联调用固定结构数据；有离线快照时优先返回真实快照。"""
    try:
        normalized = normalize_stock_code(code)
    except StockCodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = get_offline_result(normalized, 60)
    if result is None:
        companies = list_available_companies()
        result = get_offline_result(companies[0]["code"], 60) if companies else None
    if result is None:
        raise HTTPException(status_code=404, detail="仓库中没有可用的 mock/离线报告")
    return result.model_copy(update={"code": normalized})


@app.websocket("/ws/pipeline/{task_id}")
async def pipeline_websocket(websocket: WebSocket, task_id: str):
    """WebSocket 实时推送扫雷进度。"""
    # 可选鉴权：API_KEY 非空时，握手需带 ?token=<API_KEY>（浏览器 WS 无法自定义 header）
    if API_KEY and websocket.query_params.get("token", "") != API_KEY:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state = get_task(task_id)
    if not state:
        await websocket.send_json({
            "type": "error",
            "message": f"任务 {task_id} 不存在",
        })
        await websocket.close(code=1008)
        return

    # 先把生产端已记录的进度补发（断线重连场景）
    for p in state.progress:
        await websocket.send_json(p.model_dump(exclude_none=True))
    queue = subscribe_task(state)
    terminal_since = time.monotonic() if state.status in TERMINAL_STATUSES else None

    # 结束后保持 30 秒，给断线重连留窗口。
    try:
        while True:
            if state.status in TERMINAL_STATUSES and terminal_since is None:
                terminal_since = time.monotonic()
            if terminal_since is not None and time.monotonic() - terminal_since >= 30:
                break
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(msg.model_dump(exclude_none=True))
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe_task(state, queue)
        try:
            await websocket.close(code=1000)
        except Exception:
            pass


# ==================== 评价中心接口 ====================

@app.get("/api/metrics/dashboard")
def metrics_dashboard(range: str = "7"):
    """三项核心指标当前值：准确率 / 证据召回率 / Top-5 命中率。"""
    return get_dashboard_metrics(range)


@app.get("/api/metrics/trend")
def metrics_trend(range: str = "7"):
    """三项指标随时间变化趋势（按 publish_date 聚合）。"""
    return get_trend(range)


@app.get("/api/metrics/confusion-matrix")
def metrics_confusion_matrix(range: str = "7"):
    """45 类二级主题的 TP/FP/FN 混淆矩阵统计。"""
    return get_confusion_matrix(range)


@app.get("/api/metrics/alerts")
def metrics_alerts(range: str = "7"):
    """指标未达标与主题召回异常告警。"""
    return get_alerts(range)


@app.get("/api/evaluate/review-queue")
def evaluate_review_queue(limit: int = 20):
    """待复核样本：任一指标未达标或案例未命中。"""
    return get_review_queue(limit)


@app.post("/api/evaluate/feedback")
def evaluate_feedback(payload: dict = Body(...)):
    """人工标注回流入口。"""
    return submit_feedback(payload)


# 挂载静态文件（放在最后，避免覆盖 API 路由）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    # 端口/监听统一从环境变量读取，避免多入口脚本硬编码导致端口冲突。
    # 默认端口 8000，与 `start_api.bat`「前端错误提示」保持一致。
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
