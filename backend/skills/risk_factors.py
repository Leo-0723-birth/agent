#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险因素生成 skill（可复用）—— 财务异常 agent 的输出②
======================================================
把 F2-F6 特征里「触发的异常信号」翻译成结构化风险因素 JSON，供下游 agent 做归因/解释。

设计：
  - RULES 是唯一规则来源，每行一个 dict，加规则改这里即可；
  - 每条命中规则产出一条风险因素，带 evidence（特征名 + 条件 + 实际值），100% 可追溯；
  - risk_score = 命中规则严重度权重之和（高/中/低 = 3/2/1），risk_level 由分数判定。

输出结构（与 agent README 一致）：
  {
    "company_code", "report_period", "risk_level", "risk_score", "n_risk_factors",
    "risk_factors": [{"factor_id","family","name","severity","value","description","evidence"}],
    "summary"
  }
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

_SEVERITY_WEIGHT = {"高": 3, "中": 2, "低": 1}

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}

# ============================================================
# 规则表（小白友好：加一行 dict 即可）
# ============================================================

RULES = [
    # ---- F2 财务异常 ----
    dict(feature="f2_loss_flag", family="F2_财务异常", name="本期亏损",
         severity="高", op="==", threshold=1, desc="公司本期净利润为负，出现亏损"),
    dict(feature="f2_high_debt_flag", family="F2_财务异常", name="高负债率",
         severity="中", op="==", threshold=1, desc="资产负债率超过 70%，偿债压力大"),
    dict(feature="f2_roe", family="F2_财务异常", name="净资产收益率为负",
         severity="高", op="<", threshold=0, desc="ROE 为负，盈利能力恶化"),
    dict(feature="f2_profit_ocf_diverge", family="F2_财务异常", name="利润现金流背离",
         severity="高", op="==", threshold=1, desc="净利润与经营现金流方向背离，盈利质量存疑"),
    dict(feature="f2_neg_accruals_flag", family="F2_财务异常", name="应计利润为负",
         severity="中", op="==", threshold=1, desc="应计利润为负，盈余质量存疑"),
    dict(feature="f2_beneish_m", family="F2_财务异常", name="Beneish 可疑盈余管理",
         severity="高", op=">", threshold=-2.22, desc="Beneish M-Score 高于 -2.22，存在盈余操纵迹象"),
    dict(feature="f2_benford_flag", family="F2_财务异常", name="财务数字偏离 Benford",
         severity="中", op="==", threshold=1, desc="财务数字首位数分布偏离 Benford 定律"),
    dict(feature="f2_p_score", family="F2_财务异常", name="基本面偏弱",
         severity="中", op="<=", threshold=2, desc="Piotroski F-Score 偏低，基本面较弱"),

    # ---- F3 市场异动 ----
    dict(feature="mkt_risk_warning_flag_30d", family="F3_市场异动", name="风险警示公告",
         severity="高", op="==", threshold=1, desc="近 30 日有风险警示公告"),
    dict(feature="mkt_risk_warning_count_30d", family="F3_市场异动", name="风险警示频发",
         severity="中", op=">", threshold=0, desc="近 30 日出现风险警示公告"),
    dict(feature="mkt_return_20d", family="F3_市场异动", name="短期大幅下跌",
         severity="中", op="<", threshold=-0.2, desc="近 20 日累计跌幅超过 20%"),

    # ---- F4 舆情情绪 ----
    dict(feature="sent_negative_ratio_30d", family="F4_舆情情绪", name="负面新闻占比高",
         severity="中", op=">", threshold=0.5, desc="近 30 日负面新闻占比超过 50%"),
    dict(feature="sent_sentiment_mean_30d", family="F4_舆情情绪", name="新闻情绪偏负面",
         severity="中", op="<", threshold=0, desc="近 30 日新闻情绪整体偏负面"),
    dict(feature="sent_negative_news_count_30d", family="F4_舆情情绪", name="负面新闻集中",
         severity="中", op=">=", threshold=3, desc="近 30 日负面新闻不少于 3 条"),
    dict(feature="sent_guba_negative_ratio_30d", family="F4_舆情情绪", name="股吧负面情绪",
         severity="低", op=">", threshold=0.5, desc="近 30 日股吧负面帖子占比超过 50%"),

    # ---- F5 公司治理 ----
    dict(feature="gov_nonstandard_audit_opinion", family="F5_公司治理", name="非标审计意见",
         severity="高", op="==", threshold=1, desc="审计意见为非标准无保留，高风险"),
    dict(feature="gov_audit_firm_change", family="F5_公司治理", name="更换会计师事务所",
         severity="中", op="==", threshold=1, desc="本期更换会计师事务所，需关注审计独立性"),
    dict(feature="gov_auditor_change", family="F5_公司治理", name="更换签字会计师",
         severity="中", op="==", threshold=1, desc="本期更换签字注册会计师"),
    dict(feature="gov_top1_holder_ratio", family="F5_公司治理", name="股权高度集中",
         severity="低", op=">", threshold=0.5, desc="第一大股东持股超过 50%，一股独大"),

    # ---- F6 问询历史 ----
    dict(feature="f6_inquiry_count_12m", family="F6_问询历史", name="近期被问询",
         severity="高", op=">", threshold=0, desc="近 12 个月收到监管问询函"),
    dict(feature="f6_unreplied_count", family="F6_问询历史", name="问询未回复",
         severity="高", op=">", threshold=0, desc="存在未回复的监管问询函"),
    dict(feature="f6_attention_letter_count", family="F6_问询历史", name="收到关注函",
         severity="中", op=">", threshold=0, desc="历史上收到过监管关注函"),
    dict(feature="f6_severity_score", family="F6_问询历史", name="问询严重度高",
         severity="中", op=">=", threshold=0.5, desc="问询严重度评分不低于 0.5"),
]


def _is_missing(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def generate_risk_factors(features, company_code, report_period=None):
    """把 F2-F6 特征转成风险因素 JSON（输出②）。

    参数
    ----
    features : dict   F2-F6 特征（{feature_name: value}）
    company_code : str  公司代码
    report_period : str 报告期（可选）

    返回
    ----
    dict  结构化风险因素（risk_level/risk_score/risk_factors/summary）
    """
    risk_factors = []
    risk_score = 0

    for rule in RULES:
        feat_name = rule["feature"]
        raw = features.get(feat_name)
        if _is_missing(raw):
            continue
        value = _as_float(raw)
        if value is None:
            continue

        threshold = rule["threshold"]
        try:
            hit = _OPS[rule["op"]](value, threshold)
        except Exception:
            hit = False
        if not hit:
            continue

        risk_factors.append({
            "factor_id": feat_name,
            "family": rule["family"],
            "name": rule["name"],
            "severity": rule["severity"],
            "value": value,
            "description": rule["desc"],
            "evidence": f"{feat_name} {rule['op']} {threshold}（实际值={value}）",
        })
        risk_score += _SEVERITY_WEIGHT.get(rule["severity"], 1)

    n = len(risk_factors)
    if risk_score >= 12:
        risk_level = "高"
    elif risk_score >= 6:
        risk_level = "中"
    else:
        risk_level = "低"

    # 按严重度排序（高在前），再按家族
    order = {"高": 0, "中": 1, "低": 2}
    risk_factors.sort(key=lambda x: (order.get(x["severity"], 3), x["family"]))

    # 家族分布汇总
    fam_count: dict = {}
    for rf in risk_factors:
        fam_count[rf["family"]] = fam_count.get(rf["family"], 0) + 1
    top_fams = ", ".join(f"{k}({v})" for k, v in sorted(fam_count.items(), key=lambda x: -x[1]))

    if n:
        summary = f"共触发 {n} 项风险因素（风险等级 {risk_level}），主要分布在 {top_fams}。"
    else:
        summary = f"未触发显著风险因素（风险等级 {risk_level}），量化层面未发现明显异常。"

    return {
        "company_code": company_code,
        "report_period": report_period,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "n_risk_factors": n,
        "risk_factors": risk_factors,
        "summary": summary,
    }


if __name__ == "__main__":
    import io
    import json
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # 自测：用一份带明显异常的示例特征
    sample = {
        "f2_loss_flag": 1, "f2_roe": -97.68, "f2_high_debt_flag": 1,
        "f2_profit_ocf_diverge": 1, "f2_beneish_m": -1.5,
        "mkt_risk_warning_flag_30d": 1,
        "sent_negative_ratio_30d": 0.62, "sent_sentiment_mean_30d": -0.3,
        "gov_nonstandard_audit_opinion": 1, "gov_audit_firm_change": 1,
        "f6_inquiry_count_12m": 2, "f6_unreplied_count": 1, "f6_severity_score": 0.6,
    }
    out = generate_risk_factors(sample, "000004.SZ", "20241231")
    print(json.dumps(out, ensure_ascii=False, indent=2))
