#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告生成 Agent (ReporterAgent) —— 任务4 的对外交付
==================================================
职责：聚合 ctx 全量结果（预测/财务/公告/案例/归因/trace）→ 渲染八章风控函件式报告。
输出（写回 ctx.report）：
    ctx.report["json"]     结构化报告 dict
    ctx.report["markdown"] 人类可读 Markdown 报告
落盘（每次生成自动存档，供主控界面浏览/审计）：
    backend/data/output/reports/{报告编号}.md / .json + manifest.json 索引

执行摘要：
    - ctx.use_llm_summary=True 时用 DeepSeek（deepseek-v4-flash，backend.llm.chat）生成叙事，
      失败自动回退规则拼装（skills/risk_report_render.rules_summary），不阻断流水线；
    - 默认 False（演示时勾选）。

依赖：skills/risk_report_render.py。
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from ..config import OUTPUT_DIR
from ..llm import chat
from ..skills import risk_report_render
from .base import AgentBase

_logger = logging.getLogger(__name__)

REPORTS_DIR = Path(OUTPUT_DIR) / "reports"
MANIFEST_PATH = REPORTS_DIR / "manifest.json"


class ReporterAgent(AgentBase):
    name = "Reporter"

    def __init__(self):
        super().__init__()

    # ================= 执行摘要 =================
    def _executive_summary(self, ctx) -> str:
        """执行摘要：LLM 叙事（deepseek-v4-flash）优先，失败/关闭回退规则拼装。"""
        if getattr(ctx, "use_llm_summary", False):
            try:
                facts = risk_report_render.build_summary_facts(ctx)
                summary = chat(
                    system=risk_report_render.SUMMARY_SYSTEM,
                    prompt=f"请基于以下事实生成执行摘要：\n{facts}",
                    temperature=0.3,
                    max_tokens=400,
                )
                summary = (summary or "").strip().replace("\n", " ")
                if summary:
                    return summary[:300]
                print("[reporter] 摘要为空，回退规则拼装")
            except Exception as e:
                print(f"[reporter] LLM 摘要失败，回退规则拼装: {type(e).__name__}: {e}")
        return risk_report_render.rules_summary(ctx)

    # ================= 落盘（报告存档） =================
    def _save_report(self, ctx, report_json: dict, markdown: str):
        """把报告写入 output/reports/ 并更新 manifest 索引。"""
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            rid = report_json["report_id"]
            md_path = REPORTS_DIR / f"{rid}.md"
            json_path = REPORTS_DIR / f"{rid}.json"
            md_path.write_text(markdown, encoding="utf-8")
            json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

            # 索引：读旧 -> 追加 -> 写回（保留最近 200 条）
            manifest = []
            if MANIFEST_PATH.exists():
                try:
                    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                    if not isinstance(manifest, list):
                        manifest = []
                except Exception:
                    manifest = []
            manifest = [
                item for item in manifest
                if isinstance(item, dict)
                and (REPORTS_DIR / str(item.get("md_file", ""))).is_file()
                and (REPORTS_DIR / str(item.get("json_file", ""))).is_file()
            ]
            pred = ctx.prediction or {}
            manifest.insert(0, {
                "report_id": rid,
                "company": ctx.company,
                "name": ctx.name,
                "as_of": ctx.as_of,
                "window": ctx.window,
                "probability_60d": pred.get("probability_60d"),
                "risk_level": pred.get("risk_level"),
                "generated_at": report_json["generated_at"],
                "md_file": md_path.name,
                "json_file": json_path.name,
            })
            MANIFEST_PATH.write_text(
                json.dumps(manifest[:200], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _logger.warning("报告落盘失败（不影响报告返回）: %s: %s", type(e).__name__, e)

    # ================= 渲染 =================
    def render_markdown(self, ctx, executive_summary=""):
        """渲染 Markdown 报告（供导出/展示）。"""
        return risk_report_render.render_markdown(ctx, executive_summary=executive_summary)

    def render_json(self, ctx, executive_summary=""):
        """渲染结构化 JSON 报告。"""
        return risk_report_render.render_json(ctx, executive_summary=executive_summary)

    # ============ 主入口 ============
    def execute(self, company, ctx):
        summary = self._executive_summary(ctx)
        report_json = risk_report_render.render_json(ctx, executive_summary=summary)
        markdown = risk_report_render.render_markdown(ctx, executive_summary=summary)
        ctx.report = {"json": report_json, "markdown": markdown}
        self._save_report(ctx, report_json, markdown)
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.reporter）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.agents import SweepingOrchestrator

    orch = SweepingOrchestrator(use_llm=False, use_finbert=False)
    ctx = orch.sweep_one("000004.SZ", window=60, as_of="2025-12-02")
    print("报告 JSON 顶层字段:", list(ctx.report["json"].keys()))
    print("执行摘要:", ctx.report["json"]["executive_summary"][:120])
    print("\n===== Markdown 报告（前 20 行） =====")
    print("\n".join(ctx.report["markdown"].splitlines()[:20]))
