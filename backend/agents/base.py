#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 基类 + 推理链路追踪
=========================
统一所有 Agent 的接口，保证"每个 Agent 都能独立运行、链路可追踪"（方案 5.4）。

子类只需实现 execute(company, ctx)：
    - company: 公司代码
    - ctx: 共享 Context（读需要的字段、写自己的字段）
    - 返回 ctx（写回）
run() 自动包装：记录 run_id / 耗时 / 输入输出摘要，并追加进 ctx.trace_log。
"""
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime


class AgentBase(ABC):
    """所有 Agent 的基类。子类只需实现 execute()，追踪日志自动生成。"""

    name = "AgentBase"

    def __init__(self, progress_callback=None):
        self.run_id = None
        self.trace = None
        self.progress_callback = progress_callback

    def _report_progress(self, percent, message):
        """向上层流水线报告 Agent 内部进度；未传回调时不影响原逻辑。"""
        if self.progress_callback is not None:
            self.progress_callback({
                "event": "agent_progress",
                "percent": max(0, min(100, int(percent))),
                "message": str(message),
            })

    @staticmethod
    def _check_cancel(ctx, message="任务已取消"):
        """让支持取消的 Agent 在耗时步骤之间尽快退出。"""
        cancel = getattr(ctx, "cancel_event", None)
        if cancel is not None and cancel.is_set():
            raise RuntimeError(message)

    @abstractmethod
    def execute(self, company, ctx):
        """Agent 主逻辑：读 ctx 需要的数据，写回 ctx 对应字段。子类必须实现。"""
        raise NotImplementedError

    def run(self, company, ctx):
        """
        统一入口：包装 execute()，记录链路追踪并追加进 ctx.trace_log。
        返回 (ctx, trace_log)。
        """
        self.run_id = str(uuid.uuid4())[:8]
        start = time.time()
        self._report_progress(5, f"{self.name} 已启动")

        ctx = self.execute(company, ctx)

        self._report_progress(100, f"{self.name} 处理完成")

        latency_ms = int((time.time() - start) * 1000)
        self.trace = {
            "run_id": self.run_id,
            "agent": self.name,
            "company": company,
            "timestamp": datetime.now().isoformat(),
            "output_summary": self._summarize(getattr(ctx, "report", None) or {}),
            "latency_ms": latency_ms,
            "trace_complete": True,
        }
        if ctx is not None:
            ctx.trace_log.append(self.trace)
        return ctx, self.trace

    @staticmethod
    def _summarize(result):
        """生成输出摘要（避免把整个结果塞进日志）。"""
        if isinstance(result, dict):
            return {k: str(v)[:80] for k, v in list(result.items())[:10]}
        return str(result)[:200]


class TraceLogger:
    """把多条 trace 追加写入 JSONL，用于 100% 可追溯。"""

    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, trace):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
