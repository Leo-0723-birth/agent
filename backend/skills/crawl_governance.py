#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F5 公司治理 · 在线爬取 skill（可复用）
====================================
数据源：
  - C1 股权结构（6 维）：东方财富 F10 股东接口（datacenter-web.eastmoney.com，
    reportName=RPT_F10_EH_HOLDERS，已实测可用），取最新一期前十大股东算集中度。
  - C3 审计与内控（4 维）：年报 PDF（上市公司公告与定期报告数据集）离线解析，
    从「非标准审计意见提示」勾选框 + 「XX会计师事务所」文本提取审计意见类型、
    四大审计、事务所变更。
  - C2 董事会治理（3 维）：暂无稳定免费在线接口（依赖 CNRDS 董监高库），返回 NaN。

输出 13 维（与离线 f5c1/c2/c3 列顺序一致）：
  C1: gov_top1_holder_ratio, gov_top10_holder_ratio, gov_top10_holder_count,
      gov_top10_hhi, gov_top1_top10_ratio, gov_top1_top2_gap
  C2: gov_board_size, gov_independent_director_count, gov_independent_director_ratio
  C3: gov_big4_auditor, gov_auditor_change, gov_audit_firm_change, gov_nonstandard_audit_opinion
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import ANNUAL_REPORT_DIR

try:
    from .financial_data_fetch import _to_6digit, _market_of
except ImportError:
    from crawl_financial import _to_6digit, _market_of

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://data.eastmoney.com/",
}

# 四大会计师事务所关键词
_BIG4_KEYWORDS = ("普华永道", "德勤", "安永", "毕马威")

# 输出顺序（离线口径）
F5_FEATURE_ORDER = [
    "gov_top1_holder_ratio", "gov_top10_holder_ratio", "gov_top10_holder_count",
    "gov_top10_hhi", "gov_top1_top10_ratio", "gov_top1_top2_gap",
    "gov_board_size", "gov_independent_director_count", "gov_independent_director_ratio",
    "gov_big4_auditor", "gov_auditor_change", "gov_audit_firm_change", "gov_nonstandard_audit_opinion",
]


def _secucode(company_code: str) -> str:
    """'000004' / 'SZ000004' / '000004.SZ' -> '000004.SZ'"""
    return f"{_to_6digit(company_code)}.{_market_of(company_code)}"


# ============================================================
# 1. C1 股权结构（东财 F10 股东）
# ============================================================

def _get_top10_holders(company_code: str) -> list:
    """东财 F10 前十大股东 -> [{rank, name, ratio(0~1)}]，无数据返回 []。"""
    code = _secucode(company_code)
    try:
        r = requests.get(
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            params={
                "sortColumns": "END_DATE,HOLDER_RANK", "sortTypes": "-1,1",
                "pageSize": 50, "pageNumber": 1,
                "reportName": "RPT_F10_EH_HOLDERS", "columns": "ALL",
                "filter": f'(SECUCODE="{code}")',
            },
            headers=_HEADERS, timeout=15,
        )
        rows = (r.json().get("result") or {}).get("data") or []
    except Exception:
        return []

    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["END_DATE"] = pd.to_datetime(df["END_DATE"], errors="coerce")
    df["HOLDER_RANK"] = pd.to_numeric(df["HOLDER_RANK"], errors="coerce")
    df["HOLD_NUM_RATIO"] = pd.to_numeric(df["HOLD_NUM_RATIO"], errors="coerce")

    latest = df["END_DATE"].max()
    df = df[(df["END_DATE"] == latest) & (df["HOLDER_RANK"].between(1, 10))]
    df = df.sort_values("HOLDER_RANK").drop_duplicates("HOLDER_RANK", keep="last")
    return [
        {"rank": int(r["HOLDER_RANK"]), "name": str(r.get("HOLDER_NAME", "")),
         "ratio": float(r["HOLD_NUM_RATIO"]) / 100.0}
        for _, r in df.iterrows() if pd.notna(r["HOLD_NUM_RATIO"])
    ]


def _compute_c1(holders: list) -> dict:
    """C1 股权结构与集中度（6 维），公式与离线 build_f5c1 一致。"""
    feat: dict = {}
    if not holders:
        return feat

    g = sorted(holders, key=lambda x: x["rank"])
    ratios = [h["ratio"] for h in g]

    top1 = ratios[0] if ratios else np.nan
    top10 = float(np.sum(ratios)) if ratios else np.nan
    hhi = float(np.sum(np.square(ratios))) if ratios else np.nan

    feat["gov_top1_holder_ratio"] = float(top1) if pd.notna(top1) else np.nan
    feat["gov_top10_holder_ratio"] = top10
    feat["gov_top10_hhi"] = hhi
    feat["gov_top1_top10_ratio"] = float(top1 / top10) if (pd.notna(top1) and top10 and top10 > 0) else np.nan

    if len(ratios) >= 2 and pd.notna(top1):
        feat["gov_top1_top2_gap"] = float(top1 - ratios[1])
    else:
        feat["gov_top1_top2_gap"] = np.nan

    feat["gov_top10_holder_count"] = int(len(g))
    return feat


# ============================================================
# 2. C2 董事会治理（暂无免费接口，返回 NaN）
# ============================================================

def _compute_c2() -> dict:
    """C2 董事会规模/独董（3 维）—— CNRDS 董监高库，暂无稳定免费在线接口。"""
    return {
        "gov_board_size": np.nan,
        "gov_independent_director_count": np.nan,
        "gov_independent_director_ratio": np.nan,
    }


# ============================================================
# 3. C3 审计与内控（年报 PDF 离线解析）
# ============================================================

def _annual_report_pdfs(company_code: str) -> dict:
    """按年份分组返回公司年报 PDF 列表：{year(int): [Path, ...]}（降序）。"""
    code = _secucode(company_code)
    base = Path(ANNUAL_REPORT_DIR) / code
    if not base.exists():
        return {}
    out: dict = {}
    for pdf in sorted(base.glob("*/annual_report/*.pdf")):
        try:
            year = int(pdf.parts[-3])
        except (ValueError, IndexError):
            continue
        out.setdefault(year, []).append(pdf)
    return dict(sorted(out.items(), reverse=True))


def _read_pdf_text(path: Path, max_pages: int = 12) -> str:
    """读 PDF 前 max_pages 页文本（审计意见/事务所通常在前面）。"""
    import pdfplumber
    try:
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages[:max_pages])
    except Exception:
        return ""


def _extract_audit_opinion(text: str):
    """从年报文本提取审计意见：0=标准无保留，1=非标准，None=未识别。

    优先看「非标准审计意见提示」勾选框（□适用 ☑不适用 / ☑适用 □不适用）。
    """
    # 勾选框：☑(U+2611) / 私有区  / ■ / √ 等
    m = re.search(r"非标准审计意见提示", text)
    if m:
        seg = text[m.end(): m.end() + 40]
        if re.search(r"[☑■√✓]\s*适用", seg):
            return 1
        if re.search(r"[☑■√✓]\s*不适用", seg):
            return 0
        # 无勾选标记时看「适用/不适用」出现位置兜底
        if "适用" in seg and "不适用" in seg:
            if seg.find("适用") < seg.find("不适用"):
                return 1
            return 0

    # 兜底：显式意见措辞
    if re.search(r"标准无保留意见", text):
        return 0
    if re.search(r"保留意见|无法表示意见|否定意见", text):
        return 1
    return None


def _extract_audit_firm(text: str) -> str:
    """从年报文本提取会计师事务所名称（首个匹配），失败返回 ''。"""
    m = re.search(r"([一-龥]{2,10})会计师事务所", text)
    return m.group(1) if m else ""


def _best_report_text(pdfs: list) -> str:
    """从多个年报 PDF 里挑出最合适的一份文本（含审计意见勾选框 > 含事务所 > 中文）。"""
    best = ""
    best_score = -1
    for p in pdfs:
        t = _read_pdf_text(p, max_pages=25)
        if not t:
            continue
        score = 0
        if "非标准审计意见提示" in t:      # 年报摘要的勾选框，最优
            score += 100
        if "会计师事务所" in t:
            score += 10
        if re.search(r"[一-鿿]", t):   # 中文文本优先于英文年报
            score += 1
        if score > best_score:
            best, best_score = t, score
    return best


def _compute_c3(company_code: str) -> dict:
    """C3 审计与内控（4 维），用年报 PDF 解析。"""
    feat: dict = {}
    pdfs_by_year = _annual_report_pdfs(company_code)
    if not pdfs_by_year:
        return feat

    latest_year = max(pdfs_by_year)
    text = _best_report_text(pdfs_by_year[latest_year])

    if text:
        opinion = _extract_audit_opinion(text)
        if opinion is not None:
            feat["gov_nonstandard_audit_opinion"] = float(opinion)

        firm = _extract_audit_firm(text)
        if firm:
            feat["gov_big4_auditor"] = float(any(k in firm for k in _BIG4_KEYWORDS))

    # 事务所变更：对比上一自然年的年报里的会计师事务所
    prev_years = [y for y in pdfs_by_year if y < latest_year]
    if text and prev_years:
        prev_text = _best_report_text(pdfs_by_year[prev_years[0]])
        cur_firm = _extract_audit_firm(text)
        prev_firm = _extract_audit_firm(prev_text)
        if cur_firm and prev_firm:
            feat["gov_audit_firm_change"] = float(cur_firm != prev_firm)

    # 签字注册会计师变更：年报文本里一般无「签字注册会计师」字样（需专门公告），返回 NaN
    feat.setdefault("gov_auditor_change", np.nan)
    return feat


# ============================================================
# 4. 统一入口
# ============================================================

def compute_governance_features(company_code: str) -> dict:
    """计算单公司 F5 全量 13 维治理特征。"""
    feat: dict = {}
    try:
        holders = _get_top10_holders(company_code)
        feat.update(_compute_c1(holders))
    except Exception as e:
        print(f"[F5] 股东抓取失败: {e}")
    feat.update(_compute_c2())
    try:
        feat.update(_compute_c3(company_code))
    except Exception as e:
        print(f"[F5] 年报审计解析失败: {e}")
    return feat


def crawl_governance_features(company_code: str) -> dict:
    """F5 统一入口：返回 13 维特征 dict（按标准顺序，缺失补 NaN）。"""
    feat = compute_governance_features(company_code)
    return {k: feat.get(k, np.nan) for k in F5_FEATURE_ORDER}


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    for code in ("600519.SH", "000004.SZ"):
        f = crawl_governance_features(code)
        non_nan = [k for k, v in f.items() if v is not None and not (isinstance(v, float) and v != v)]
        print(f"\n{code} F5 在线特征: {len(f)} 维, 非 NaN {len(non_nan)} 维")
        for k, v in f.items():
            if v is not None and not (isinstance(v, float) and v != v):
                print(f"  {k} = {v}")
