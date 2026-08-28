#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零额外 Web 依赖的上市公司扫雷 API。"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents import SweepingOrchestrator
from backend.config import PREDICTOR_HORIZONS

DELIVERY_ROOT = Path(__file__).resolve().parents[2] / "官方要求交付物"
TRACE_DIR = DELIVERY_ROOT / "02_Agent推理日志"


def run_one(company: str, window: int, as_of: str | None, use_llm: bool = False) -> dict:
    ctx = SweepingOrchestrator(use_llm=use_llm, use_finbert=False).sweep_one(
        company, window=window, as_of=as_of or str(date.today())
    )
    return ctx.to_dict()


def write_trace(request_id: str, company: str, ctx: dict) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"trace_{datetime.now():%Y%m%d}.jsonl"
    for item in ctx.get("trace_log", []):
        record = {"request_id": request_id, "company": company, **item}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "RiskSweepAPI/1.0"

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send({"status": "ok", "service": "上市公司扫雷预警 API"})
            return
        self._send({"service": "上市公司扫雷预警 API", "endpoints": ["GET /health", "POST /predict"]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send({"error": "仅支持 POST /predict"}, 404)
            return
        request_id = uuid.uuid4().hex[:12]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            companies = payload.get("companies") or [payload.get("company")]
            if isinstance(companies, str):
                companies = [companies]
            companies = [str(item).strip() for item in companies if str(item).strip()]
            if not companies:
                raise ValueError("company 或 companies 不能为空")
            window = int(payload.get("window", 60))
            valid_windows = tuple(int(h.replace("d", "")) for h in PREDICTOR_HORIZONS)
            if window not in valid_windows:
                raise ValueError(f"window 只能是 {valid_windows}")
            results = []
            for company in companies:
                ctx = run_one(company, window, payload.get("as_of"), bool(payload.get("use_llm", False)))
                write_trace(request_id, company, ctx)
                results.append(ctx)
            self._send({"request_id": request_id, "results": results})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send({"request_id": request_id, "error": str(exc)}, 400)
        except Exception as exc:  # API 边界统一返回，不泄露完整内部堆栈
            self._send({"request_id": request_id, "error": f"分析失败: {type(exc).__name__}: {exc}"}, 500)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"扫雷 API 已启动: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
