#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告生成 Agent (ReporterAgent) —— 任务4 的对外交付
==================================================
职责：聚合 ctx 全量结果（预测/财务/公告/案例/归因/trace）→ 渲染报告。
输出（写回 ctx.report）：
    ctx.report["json"]     结构化报告 dict
    ctx.report["markdown"] 人类可读 Markdown 报告

依赖：skills/risk_report_render.py。
"""
from ..skills import risk_report_render
from .base import AgentBase


class ReporterAgent(AgentBase):
    name = "Reporter"

    def __init__(self):
        super().__init__()

    def render_markdown(self, ctx):
        """渲染 Markdown 报告（供导出/展示）。"""
        return risk_report_render.render_markdown(ctx)

    def render_json(self, ctx):
        """渲染结构化 JSON 报告。"""
        return risk_report_render.render_json(ctx)

    # ============ 主入口 ============
    def execute(self, company, ctx):
        ctx.report = {
            "json": risk_report_render.render_json(ctx),
            "markdown": risk_report_render.render_markdown(ctx),
        }
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.reporter）
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.agents import SweepingOrchestrator

    orch = SweepingOrchestrator(use_llm=False, use_finbert=False)
    ctx = orch.sweep_one("000004.SZ", window=60, as_of="2025-12-02")
    print("报告 JSON 顶层字段:", list(ctx.report["json"].keys()))
    print("\n===== Markdown 报告（前 30 行） =====")
    print("\n".join(ctx.report["markdown"].splitlines()[:30]))
