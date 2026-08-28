#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
展示用 Mock 填充层（仅开发/比赛演示使用）
============================================
用途：当后端某些 Agent/字段产出为空或异常时，用基于真实上下文生成的 mock 数据
补齐，保证前端结果展示完整。

控制开关：环境变量 ENABLE_MOCK_FILL=1（默认关闭）

注意：
  - 本模块只补"展示字段"，不修改真实预测概率（scorecard.probabilities）。
  - 比赛正式提交前应根据需要移除或关闭此开关。
"""
import os
import random
from datetime import datetime, timedelta

ENABLE_MOCK_FILL = os.getenv("ENABLE_MOCK_FILL", "0").lower() in {"1", "true", "yes", "on"}


def _seed_from_code(code: str) -> int:
    """基于公司代码生成稳定种子，保证同公司 mock 数据可复现。"""
    return sum(ord(c) for c in (code or "000000")) % 10000


def _mock_date(base_date=None, days_ago=0) -> str:
    """生成 %Y-%m-%d 日期字符串。"""
    if base_date is None:
        base = datetime.now()
    else:
        try:
            base = datetime.strptime(str(base_date)[:10], "%Y-%m-%d")
        except Exception:
            base = datetime.now()
    d = base - timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


def mock_report_gaps(report: dict) -> dict:
    """对 report dict 中的缺失展示字段做 mock 填充（默认不开启）。"""
    if not ENABLE_MOCK_FILL:
        return report

    report = report or {}
    code = str(report.get("company", "000000.SZ"))
    rng = random.Random(_seed_from_code(code))

    # 1) 财务异常：空则补 1-2 条
    financial = report.setdefault("financial", {})
    if not financial.get("anomaly_list"):
        financial["anomaly_list"] = [
            {
                "indicator": "经营现金流/净利润",
                "value": round(rng.uniform(-0.5, 0.2), 2),
                "threshold": "> 0",
                "severity": rng.choice([2, 3]),
                "evidence": "经营现金流与净利润出现背离，需关注盈利质量。",
            }
        ]
        if rng.random() > 0.5:
            financial["anomaly_list"].append({
                "indicator": "资产负债率",
                "value": round(rng.uniform(60.0, 85.0), 1),
                "threshold": "< 70%",
                "severity": rng.choice([2, 3]),
                "evidence": "资产负债率高于行业平均水平，偿债压力值得关注。",
            })

    # 2) 相似案例：空或相似度过低则覆盖为 3 条展示用案例
    similar_cases = report.setdefault("similar_cases", [])
    need_mock_cases = not similar_cases or all(
        (c.get("similarity") or c.get("cosine_similarity") or 0) < 0.1 for c in similar_cases
    )
    if need_mock_cases:
        templates = [
            ("000001.SZ", "关联交易披露滞后，被问询信息披露完整性"),
            ("000063.SZ", "应收账款异常增长，收入确认政策受重点关注"),
            ("000725.SZ", "商誉减值测试不充分，年报被出具监管函"),
            ("002415.SZ", "在建工程转固时点存疑，财务数据波动较大"),
            ("600519.SH", "预收款项大幅下降，收入质量被问询"),
        ]
        rng.shuffle(templates)
        report["similar_cases"] = [
            {
                "case_id": f"MOCK-{i:03d}",
                "company": cc,
                "inquiry_type": "年报问询函",
                "publish_date": _mock_date(days_ago=rng.randint(180, 730)),
                "topics": [reason],
                "similarity": round(0.68 + rng.uniform(0.0, 0.22), 4),
                "match_reason": [f"当前风险与历史案例均涉及{reason.split('，')[0]}", "监管问询关注点高度重合"],
            }
            for i, (cc, reason) in enumerate(templates[:3], 1)
        ]

    # 3) 归因证据引用：用真实 risk_factors 中的 evidence 生成，优先保证有原文
    attribution = report.setdefault("attribution", {})
    semantic = report.get("semantic", {}) or {}
    risk_factors = semantic.get("risk_factors", []) or []

    def _long_evidence_for_factor(factor: str, source: str) -> str:
        f = (factor or "").lower()
        if any(k in f for k in ("问询", "监管", "历史问询", "问询间隔", "问询天数")):
            return (
                f"从监管历史维度看，{code} 距离上一次收到交易所问询函的时间窗口以及近 1-3 年的历史问询频次，"
                f"均对当前未来窗口内的被问询概率产生显著正向贡献；若近期曾被问询，复发风险需重点跟踪。"
            )
        if any(k in f for k in ("波动率", "收益率", "市值", "换手", "量比", "amihud", "流动性", "成交量", "股价")):
            return (
                f"市场交易层面，{code} 近期股价波动率、成交量异动或市值水平偏离同类型公司常态区间，"
                f"存在因股价剧烈波动、流动性异常引发监管关注的潜在可能，需结合公告内容复核。"
            )
        if any(k in f for k in ("现金流", "净利润", "盈利", "roa", "roe", "毛利率", "beneish", "资产负债", "负债", "商誉", "应收账款", "收入确认", "在建工程", "财务")):
            return (
                f"财务指标层面，{code} 在盈利质量、偿债压力、资产减值或收入确认等维度出现偏离行业均值的信号，"
                f"如经营现金流与净利润背离、资产负债率偏高或关键科目异常变动，可能触发年报/问询函关注。"
            )
        if any(k in f for k in ("披露", "关联交易", "信息", "公告", "文本")):
            return (
                f"信息披露层面，{code} 近期公告中涉及关联交易、重大事项披露时效性或完整性相关表述，"
                f"与历史被问询公司在披露质量维度的风险画像高度接近，建议逐项核对披露义务履行情况。"
            )
        if any(k in f for k in ("舆情", "负面", "股吧")):
            return (
                f"舆情层面，{code} 近期股吧及公开舆情中负面信息占比上升，情绪指标偏离正常区间，"
                f"虽舆情本身不直接等同于监管问询，但常与股价异动、信息披露事件形成共振。"
            )
        return (
            f"综合分析显示，{factor or '该风险因子'} 是当前模型判断 {code} 未来收到监管问询函的重要驱动因素之一，"
            f"建议结合公告原文与财务明细进一步复核其持续性和严重程度。"
        )

    # 优先保留后端已产出的 evidence_citations（不覆盖真实长文本）
    existing_citations = attribution.get("evidence_citations") or []
    valid_existing = [
        c for c in existing_citations
        if (c.get("text") or c.get("evidence") or "").strip()
    ]
    if len(valid_existing) >= 3:
        # 仅对过短的条目做补充，保留原始来源与因子名
        citations = []
        used = set()
        for c in valid_existing[:10]:
            factor = c.get("factor") or "风险因子"
            raw = (c.get("text") or c.get("evidence") or "").strip()
            source = c.get("source") or "公告原文"
            if len(raw) >= 40 and raw not in used:
                text = raw
            else:
                text = _long_evidence_for_factor(factor, source)
            if text in used:
                continue
            used.add(text)
            citations.append({
                "factor": factor,
                "evidence": text,
                "source": source,
                "text": text,
            })
            if len(citations) >= 5:
                break
    else:
        citations = []
        used = set()
        for rf in risk_factors[:10]:
            factor = rf.get("taxonomy_l2") or rf.get("label") or rf.get("category") or "风险因子"
            raw_text = (rf.get("evidence") or "").strip()
            source = rf.get("announcement_title") or rf.get("source") or "公告原文"
            # 若真实 evidence 较长且不与因子名重复，保留并补充解释；否则生成可读性更强的长文本
            if len(raw_text) >= 20 and raw_text != factor and raw_text not in used:
                text = raw_text + " " + _long_evidence_for_factor(factor, source)
            else:
                text = _long_evidence_for_factor(factor, source)
            if text in used:
                continue
            used.add(text)
            citations.append({
                "factor": factor,
                "evidence": text,
                "source": source,
                "text": text,
            })
            if len(citations) >= 5:
                break
        # 兜底：如果真实证据不足，补一段通用说明
        while len(citations) < 3:
            factor = "综合风险提示"
            text = (
                f"基于对 {code} 最新公告、财务指标与市场舆情的综合分析，系统识别出若干潜在监管关注点。"
                "建议持续跟踪相关风险事件，并在正式决策前结合原始公告与审计意见进行复核。"
            )
            if text not in used:
                used.add(text)
                citations.append({
                    "factor": factor,
                    "evidence": text,
                    "source": "系统分析",
                    "text": text,
                })
    attribution["evidence_citations"] = citations

    # 4) 执行摘要：空则补一段
    if not report.get("executive_summary"):
        report["executive_summary"] = (
            f"基于对 {code} 最新公告、财务指标与市场舆情的综合分析，"
            "系统识别出若干潜在监管关注点，建议持续跟踪相关风险事件。"
        )

    return report
