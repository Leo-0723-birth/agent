#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量运行扫雷并输出评估友好的 CSV 与完整 JSONL。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents import SweepingOrchestrator

DELIVERY_ROOT = Path(__file__).resolve().parents[2] / "官方要求交付物"
OUTPUT_DIR = DELIVERY_ROOT / "03_测试集结果"


def main() -> int:
    parser = argparse.ArgumentParser(description="批量生成上市公司扫雷预测结果")
    parser.add_argument("--companies", nargs="+", required=True, help="公司代码列表")
    parser.add_argument("--window", type=int, choices=(30, 60, 90), default=60)
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    orch = SweepingOrchestrator(use_llm=args.use_llm, use_finbert=False)
    results = []
    for company in args.companies:
        ctx = orch.sweep_one(company.strip(), window=args.window, as_of=args.as_of)
        results.append(ctx.to_dict())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"prediction_results_{stamp}.jsonl"
    csv_path = OUTPUT_DIR / f"prediction_results_{stamp}.csv"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    rows = []
    for result in results:
        prediction = result.get("prediction", {})
        rows.append({
            "company": result.get("company"),
            "name": result.get("name"),
            "as_of": result.get("as_of"),
            "window": result.get("window"),
            "probability_30d": prediction.get("probability_30d"),
            "probability_60d": prediction.get("probability_60d"),
            "probability_90d": prediction.get("probability_90d"),
            "risk_level": prediction.get("risk_level"),
            "confidence": prediction.get("confidence"),
            "data_source": prediction.get("data_source"),
            "trace_steps": len(result.get("trace_log", [])),
            "evidence_count": len((result.get("attribution") or {}).get("evidence_citations", [])),
            "case_count": len(result.get("cases", [])),
        })
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["company"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
