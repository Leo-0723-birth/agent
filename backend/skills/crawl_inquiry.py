#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F6 监管问询历史 · PDF 离线解析 skill（可复用）
============================================
数据源：监管问询函及回复数据集（10056 份 PDF）。问询函首页提取函件类型
（年报问询函/关注函/重组问询函/其他），末页提取下发日期（落款「XXXX年X月X日」）。

两层实现：
  1) 优先读缓存的问询事件表 inquiry_events.csv（已解析好日期+类型，秒级）；
  2) 缓存缺失时，按公司逐个解析 PDF（pdfplumber），只解析该公司的问询函/回复函。

特征（12 维，与离线 F6_inquiry_history.csv 一致）：
  问询频次：f6_inquiry_count_12m/24m/60m
  问询类型：f6_annual_report_inquiry_count / f6_attention_letter_count / f6_restructuring_inquiry_count
  时间间隔：f6_first_inquiry_interval_days / f6_last_inquiry_interval_days
            f6_avg_inquiry_interval_days / f6_inquiry_interval_cv
  回复严重度：f6_unreplied_count / f6_severity_score
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import INQUIRY_DATA_DIR as INQUIRY_DIR, INQUIRY_EVENTS_CSV

# 函件类型关键字（与 build_f6_inquiry_history.py 一致）
_TYPE_KEYWORDS = [
    ("annual_report", ["年报问询", "年报"]),
    ("restructuring", ["重组问询", "重大资产重组", "重组"]),
    ("attention", ["关注函"]),
    ("semi_report", ["半年报问询", "半年度报告问询"]),
    ("quarterly", ["三季报问询", "一季报问询", "季报问询"]),
]

_DATE_RE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

F6_FEATURE_ORDER = [
    "f6_inquiry_count_12m", "f6_inquiry_count_24m", "f6_inquiry_count_60m",
    "f6_annual_report_inquiry_count", "f6_attention_letter_count", "f6_restructuring_inquiry_count",
    "f6_first_inquiry_interval_days", "f6_last_inquiry_interval_days",
    "f6_avg_inquiry_interval_days", "f6_inquiry_interval_cv",
    "f6_unreplied_count", "f6_severity_score",
]


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _extract_date(text: str):
    """从文本提取最后一个落款日期，返回 pd.Timestamp 或 None。"""
    dates = _DATE_RE.findall(_clean(text))
    if not dates:
        return None
    y, m, d = dates[-1]
    try:
        return pd.Timestamp(int(y), int(m), int(d))
    except ValueError:
        return None


def _extract_type(text: str) -> str:
    t = _clean(text)
    for name, kws in _TYPE_KEYWORDS:
        for kw in kws:
            if kw in t:
                return name
    return "other"


# ============================================================
# 1. 事件加载（缓存优先，缺失回退按公司解析 PDF）
# ============================================================

def load_events() -> pd.DataFrame:
    """读缓存的问询事件表（date 解析为 datetime）。"""
    if not INQUIRY_EVENTS_CSV.exists():
        raise FileNotFoundError(f"问询事件缓存不存在: {INQUIRY_EVENTS_CSV}")
    return pd.read_csv(str(INQUIRY_EVENTS_CSV), parse_dates=["date"])


def _parse_company_events(company_code: str) -> pd.DataFrame:
    """按公司解析问询函/回复函 PDF，返回该公司的问询事件表（date/type/kind）。"""
    base = Path(INQUIRY_DIR) / company_code
    if not base.exists():
        return pd.DataFrame(columns=["secucode", "year", "kind", "date", "type"])

    import pdfplumber
    rows = []
    for kind, subdir in (("letter", "inquiry_letter"), ("reply", "inquiry_reply")):
        for p in sorted(base.glob(f"*/{subdir}/*.pdf")):
            year = int(p.parts[-3])
            date, typ = None, ""
            try:
                with pdfplumber.open(str(p)) as pdf:
                    first = pdf.pages[0].extract_text() or ""
                    last = pdf.pages[-1].extract_text() or ""
                    date = _extract_date(last) or _extract_date(first)
                    typ = _extract_type(first) if kind == "letter" else ""
            except Exception:
                pass
            rows.append({"secucode": company_code, "year": year, "kind": kind,
                         "date": date, "type": typ})

    ev = pd.DataFrame(rows)
    if len(ev):
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
        ev.loc[ev["date"].isna(), "date"] = pd.to_datetime(
            ev.loc[ev["date"].isna(), "year"].astype(str) + "0630", format="%Y%m%d")
    return ev


def _company_events(company_code: str) -> pd.DataFrame:
    """返回该公司的问询事件表：优先缓存，缺失则解析 PDF。"""
    try:
        ev = load_events()
        return ev[ev["secucode"] == company_code].copy()
    except FileNotFoundError:
        return _parse_company_events(company_code)


# ============================================================
# 2. 特征计算（与 build_f6_inquiry_history 的 compute_features 对齐）
# ============================================================

def compute_inquiry_features(company_code: str, as_of=None, events=None) -> dict:
    """计算单公司截至 as_of 的 12 维问询历史特征。

    as_of : 报告期(YYYYMMDD) / 日期字符串 / None（默认今天）
    """
    ev = events if events is not None else _company_events(company_code)

    out = {k: np.nan for k in F6_FEATURE_ORDER}
    letters = ev[ev["kind"] == "letter"].sort_values("date")
    if len(letters) == 0:
        # 无问询记录：计数类填 0，间隔类填 0（与离线口径一致）
        for k in F6_FEATURE_ORDER:
            out[k] = 0.0
        out["f6_severity_score"] = 0.0
        return out

    replies = ev[ev["kind"] == "reply"].sort_values("date")
    letters["d"] = letters["date"].values.astype("datetime64[D]")
    replies["d"] = replies["date"].values.astype("datetime64[D]")

    asof = np.datetime64(pd.to_datetime(str(as_of) if as_of is not None else "today").date(), "D")
    Ldates = letters["d"].to_numpy()
    Ltypes = letters["type"].to_numpy()
    idx = int(np.searchsorted(Ldates, asof, side="right"))

    if idx == 0:
        for k in F6_FEATURE_ORDER:
            out[k] = 0.0
        return out

    sub = Ldates[:idx]
    one_day = np.timedelta64(1, "D")
    out["f6_inquiry_count_12m"] = float(int((sub >= (asof - np.timedelta64(365, "D"))).sum()))
    out["f6_inquiry_count_24m"] = float(int((sub >= (asof - np.timedelta64(730, "D"))).sum()))
    out["f6_inquiry_count_60m"] = float(int((sub >= (asof - np.timedelta64(1825, "D"))).sum()))
    out["f6_annual_report_inquiry_count"] = float(int(np.cumsum(Ltypes[:idx] == "annual_report")[-1]))
    out["f6_attention_letter_count"] = float(int(np.cumsum(Ltypes[:idx] == "attention")[-1]))
    out["f6_restructuring_inquiry_count"] = float(int(np.cumsum(Ltypes[:idx] == "restructuring")[-1]))
    out["f6_first_inquiry_interval_days"] = float(int((asof - sub[0]) / one_day))
    out["f6_last_inquiry_interval_days"] = float(int((asof - sub[-1]) / one_day))

    gaps = np.diff(sub.astype("int64")) / 86400
    out["f6_avg_inquiry_interval_days"] = float(gaps.mean()) if len(gaps) else 0.0
    out["f6_inquiry_interval_cv"] = float(gaps.std() / gaps.mean()) if (len(gaps) and gaps.mean() != 0) else 0.0

    Rdates = replies["d"].to_numpy()
    n_replied = int(np.searchsorted(Rdates, asof, side="right"))
    out["f6_unreplied_count"] = float(max(0, idx - n_replied))
    out["f6_severity_score"] = float(min(1.0, round(idx * 0.1, 4)))
    return out


def crawl_inquiry_features(company_code: str, as_of=None) -> dict:
    """F6 统一入口：返回 12 维特征 dict（按标准顺序）。"""
    feat = compute_inquiry_features(company_code, as_of=as_of)
    return {k: feat.get(k, np.nan) for k in F6_FEATURE_ORDER}


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    for code in ("000004.SZ", "600519.SH"):
        f = crawl_inquiry_features(code, as_of="20241231")
        print(f"\n{code} F6 问询历史特征 (截至 20241231):")
        for k, v in f.items():
            print(f"  {k} = {v}")
