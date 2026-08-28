# -*- coding: utf-8 -*-
"""预跑 000858.SZ（五粮液）完整流水线，生成报告入 manifest 供「切换公司」缓存使用。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents import SweepingOrchestrator

orch = SweepingOrchestrator(use_llm=False, use_finbert=True)
ctx = orch.sweep_one("000858.SZ", window=60)
print("=" * 60)
print("000858.SZ 预跑完成")
print("概率:", ctx.prediction.get("probability_60d"))
print("风险等级:", ctx.prediction.get("risk_level"))
print("公告数:", ctx.semantic.stats.get("announcement_count"))
print("风险要素:", len(ctx.semantic.risk_factors))
print("财务异常:", len(ctx.financial.anomaly_list))
print("案例:", len(ctx.cases))
print("报告已落盘:", bool(ctx.report))
