#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫雷预警系统 —— 一键演示
========================
运行: python run_demo.py
流程: 公告研读(内置索引) → 财务检测(东财联网) → (预测占位) → 案例检索(内置库) → 归因解释
说明: 默认关闭 LLM 与 FinBERT（离线可用）；想启用风险要素抽取：
      SweepingOrchestrator(use_llm=True, use_finbert=True) 并配置 .env 的 DEEPSEEK_API_KEY
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agents import SweepingOrchestrator


def main():
    orch = SweepingOrchestrator(use_llm=True,use_finbert=True)
    ctx = orch.sweep_one("000004.SZ", window=60, as_of="2025-12-02")

    print("=" * 64)
    print("扫雷流水线执行摘要（000004.SZ 国华网安）")
    print("=" * 64)
    print(f"公司    : {ctx.company} {ctx.name}")
    print(f"公告研读: {ctx.semantic.stats.get('announcement_count')} 份公告 | "
          f"风险要素 {len(ctx.semantic.risk_factors)} 条 | "
          f"F1特征 {len(ctx.semantic.f1_features)} 项")
    print(
        f"风险标签: {len(ctx.semantic.risk_labels)} 条"
    )

    for r in ctx.semantic.risk_labels[:5]:
        print(
            " - ",
            r["taxonomy_labels"],
            r["label_names"]
        )
    print(ctx.semantic.stats)
    print(f"财务检测: 风险等级 {ctx.financial.risk_level} | 异常 {len(ctx.financial.anomaly_list)} 条")
    for a in ctx.financial.anomaly_list:
        print(f"          - [{a['type']}/{a['severity']}] {a['evidence']}")
    print(f"预测建模: {ctx.prediction.get('probability_60d', '未填充')}（待填充）")
    print(f"案例检索: {len(ctx.cases)} 个相似案例")
    for c in ctx.cases[:3]:
        print(f"          - [{c['similarity']}] {c['company']} | {c['inquiry_type']} | {c['publish_date']}")
    print(f"归因解释: {len(ctx.attribution.get('top_risk_factors', []))} 个诱因 | "
          f"证据 {len(ctx.attribution.get('evidence_citations', []))} 条 | "
          f"案例链接 {len(ctx.attribution.get('case_links', []))} 个")
    print(f"链路追踪: {len(ctx.trace_log)} 步")
    print("\n完整共享 context 结构见 data/sample_context_000004.json")


if __name__ == "__main__":
    main()
