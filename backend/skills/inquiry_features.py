#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: inquiry_features —— 监管问询函 F6 特征（纯函数，由公告元数据计算）
=================================================================
由公告研读 Agent 的公告列表（巨潮官方源）识别「问询函/关注函」事件，
计算 12 维 f6_ 历史问询特征——口径与队友 crawl_inquiry.py / 离线表
F6_inquiry_history.csv 完全一致（公式同源）。

数据流：
  公告研读 ctx.semantic.announcements（title + date）
      → extract_inquiry_events() → compute_inquiry_features() → ctx.semantic.f6_features

⚠️ 口径说明（与离线表的差异）：
  - 离线表用 2007 年至今全部历史；本模块只用公告研读采集到的公告（默认近一年）。
    → 12m 窗口计数准确；24m/60m 与「历史总次数」对较早期问询会低估。
  - 从未被问询（窗口内无问询公告）→ 12 维全 0（与离线「无问询 → 全 0」一致），
    由预测侧的填充字典以训练集中位数兜底。
"""
from datetime import date, timedelta

# 12 维 f6_ 特征（顺序与离线表 F6_inquiry_history.csv 完全一致）
F6_FEATURE_ORDER = [
    'f6_inquiry_count_12m',
    'f6_inquiry_count_24m',
    'f6_inquiry_count_60m',
    'f6_annual_report_inquiry_count',
    'f6_attention_letter_count',
    'f6_restructuring_inquiry_count',
    'f6_first_inquiry_interval_days',
    'f6_last_inquiry_interval_days',
    'f6_avg_inquiry_interval_days',
    'f6_inquiry_interval_cv',
    'f6_unreplied_count',
    'f6_severity_score',
]

# 问询类型 -> (标准类型, 严重度)。与队友 classify_inquiry_type 一致。
_INQUIRY_TYPE_RULES = [
    ("重组问询函", 3, lambda t: "重组" in t and "问询" in t),
    ("年报问询函", 3, lambda t: "年报" in t and "问询" in t),
    ("半年报问询函", 2, lambda t: "半年报" in t and "问询" in t),
    ("关注函", 2, lambda t: "关注函" in t),
    ("普通问询函", 1, lambda t: "问询" in t),
]


def is_reply(title) -> bool:
    """标题是否为「回复/答复」类公告（问询函的回复）。"""
    t = title or ""
    return ("回复" in t) or ("答复" in t)


def classify_inquiry(title):
    """把公告标题映射到 (标准类型, 严重度)。顺序重要：先匹配具体类型。"""
    t = title or ""
    for name, severity, pred in _INQUIRY_TYPE_RULES:
        if pred(t):
            return name, severity
    return "普通问询函", 1


def _is_inquiry_title(title) -> bool:
    """标题是否属于问询/关注类公告（不含回复）。"""
    t = title or ""
    return not is_reply(t) and (("问询" in t) or ("关注函" in t))


def extract_inquiry_events(announcements, as_of: date):
    """从公告研读的公告列表提取问询/回复事件。

    参数
    ----
    announcements : list[dict]  每条含 title 与日期（date / announcement_date）
    as_of : date                计算时点 T

    返回 (inquiries, replies)
        inquiries : [{date, type, severity}]   问询事件（按日期升序）
        replies   : [{date}]                   回复事件
    """
    inquiries, replies = [], []
    for ann in announcements or []:
        title = ann.get("title") or ann.get("announcement_title") or ""
        d = ann.get("date") or ann.get("announcement_date")
        if not title or not d:
            continue
        try:
            d = date.fromisoformat(str(d)[:10])
        except ValueError:
            continue
        if is_reply(title):
            replies.append({"date": d})
        elif _is_inquiry_title(title):
            typ, severity = classify_inquiry(title)
            inquiries.append({"date": d, "type": typ, "severity": severity})
    return inquiries, replies


def compute_inquiry_features(inquiries, replies, as_of: date) -> dict:
    """由问询/回复事件表计算 12 维 f6_ 特征（公式与离线表同源）。

    从未被问询 → 12 维全 0（与离线「无问询 → 全 0」一致）。
    """
    if not inquiries:
        return {k: 0.0 for k in F6_FEATURE_ORDER}

    inquiries = sorted(inquiries, key=lambda e: e["date"])
    dates = [e["date"] for e in inquiries]

    def months_ago(n):
        y, m = as_of.year, as_of.month - n
        while m <= 0:
            y -= 1
            m += 12
        return as_of.replace(year=y, month=m, day=min(as_of.day, 28))

    c12 = sum(1 for d in dates if months_ago(12) <= d < as_of)
    c24 = sum(1 for d in dates if months_ago(24) <= d < as_of)
    c60 = sum(1 for d in dates if months_ago(60) <= d < as_of)

    n_annual = sum(1 for e in inquiries if e["type"] == "年报问询函")
    n_attention = sum(1 for e in inquiries if e["type"] == "关注函")
    n_restruct = sum(1 for e in inquiries if e["type"] == "重组问询函")

    first_days = float((as_of - dates[0]).days)
    last_days = float((as_of - dates[-1]).days)

    if len(dates) >= 2:
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_gap = float(sum(gaps) / len(gaps))
        var = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
        cv = float(var ** 0.5 / avg_gap) if avg_gap > 0 else 0.0
    else:
        avg_gap, cv = 0.0, 0.0

    rdates = sorted(r["date"] for r in replies)
    unreplied = 0
    for d in dates:
        has_reply = any(d < rd <= d + timedelta(days=365) for rd in rdates)
        if not has_reply:
            unreplied += 1

    total_sev = sum(e["severity"] for e in inquiries)
    severity_score = min(float(total_sev) / 30.0, 1.0)

    return {
        "f6_inquiry_count_12m": float(c12),
        "f6_inquiry_count_24m": float(c24),
        "f6_inquiry_count_60m": float(c60),
        "f6_annual_report_inquiry_count": float(n_annual),
        "f6_attention_letter_count": float(n_attention),
        "f6_restructuring_inquiry_count": float(n_restruct),
        "f6_first_inquiry_interval_days": first_days,
        "f6_last_inquiry_interval_days": last_days,
        "f6_avg_inquiry_interval_days": avg_gap,
        "f6_inquiry_interval_cv": cv,
        "f6_unreplied_count": float(unreplied),
        "f6_severity_score": severity_score,
    }


def compute_f6_from_announcements(announcements, as_of: date) -> dict:
    """一键入口：公告列表 → 12 维 f6_ 特征（供公告研读 Agent 调用）。"""
    inquiries, replies = extract_inquiry_events(announcements, as_of)
    return compute_inquiry_features(inquiries, replies, as_of)
