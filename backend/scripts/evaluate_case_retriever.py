#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测：案例检索 Top-5 命中率（对齐官方 GT 关注点的自洽评测）
============================================================
思路：用官方 evaluation_ground_truth_normalized.csv 的事件（公司+日期+官方关注点）作为
"目标公司画像"的替代（画像文本 = 官方关注点），走真实 case_retriever 检索 Top-5
（含时间穿越控制 + 排除自身），判断命中率：
  - 相关案例定义：检索结果中存在 与查询关注点 共享 ≥MIN_SHARED 个 2-gram 的案例（且非自身）
  - Hit@5 = 有相关案例命中的事件占比

注意：这是"自洽代理评测"（用官方关注点既作查询又作相关性依据），非官方人工标注的
相似案例参考答案；官方正式评测需用其标注数据。用于监控检索质量趋势与调参。

用法：python -m backend.scripts.evaluate_case_retriever [--max_events 200]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import EVAL_GT_NORMALIZED_CSV
from backend.context import Context
from backend.agents.case_retriever import CaseRetrieverAgent

MIN_SHARED_GRAMS = 3   # 相关判定：共享 2-gram 数阈值
MAX_EVENTS = 200       # 默认评测事件数（时间原因限制）


def _load_events(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    events = []
    for _, row in df.iterrows():
        try:
            fps = json.loads(row.get("regulatory_focus_points") or "[]")
        except (json.JSONDecodeError, TypeError):
            try:
                fps = json.loads(row.get("regulatory_focus_points_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                fps = []
        if isinstance(fps, list) and fps:
            events.append({
                "code": str(row.get("stock_code") or row.get("secucode") or ""),
                "date": str(row.get("publish_date") or "")[:10],
                "focus_points": [str(x) for x in fps],
            })
    return events


def _bigrams(text):
    s = (text or "").replace(" ", "").replace("\n", "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _is_relevant(query_grams, case, query_code):
    if str(case.get("company")) == str(query_code):
        return False
    fps_text = " ".join(str(t) for t in case.get("topics", []))
    shared = len(query_grams & _bigrams(fps_text))
    return shared >= MIN_SHARED_GRAMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_events", type=int, default=MAX_EVENTS)
    args = parser.parse_args()

    events = _load_events(EVAL_GT_NORMALIZED_CSV)
    print(f"官方 GT 事件: {len(events)}，本次评测前 {args.max_events} 条")

    agent = CaseRetrieverAgent()
    hits, total = 0, 0
    examples = []
    for ev in events[:args.max_events]:
        # 构造目标公司画像（官方关注点模拟风险标签）
        ctx = Context(company=ev["code"], window=60, as_of=ev["date"])
        ctx.semantic.risk_factors = [
            {"category": "其他", "description": fp, "severity": 3}
            for fp in ev["focus_points"][:5]
        ]
        agent.execute(ev["code"], ctx)

        query_grams = set()
        for fp in ev["focus_points"]:
            query_grams |= _bigrams(fp)

        hit = any(_is_relevant(query_grams, c, ev["code"]) for c in ctx.cases)
        hits += int(hit)
        total += 1
        if hit and len(examples) < 3:
            top = ctx.cases[0]
            examples.append(f"  ✓ {ev['code']} {ev['date']} → Top1 {top['company']} {top['inquiry_type']}")

    hit_rate = hits / max(total, 1)
    print(f"\n=== 评测结果（{total} 个事件，相关判定阈值={MIN_SHARED_GRAMS} 个共享 2-gram） ===")
    print(f"Top-5 命中率: {hit_rate:.1%}（{hits}/{total}）")
    print("\n命中示例：")
    for e in examples:
        print(e)
    print(f"\n说明：自洽代理评测（官方关注点即查询又作相关性依据）；"
          f"目标指标 ≥70% 需按官方相似案例参考答案正式评测。")


if __name__ == "__main__":
    main()
