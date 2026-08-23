#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理特征表加载 skill（F3/F4/F5/F6 通用）
==========================================
F3/F4/F5/F6 的最终特征已在离线预处理 CSV 里算好（团队用 Wind/CNRDS/舆情/问询 PDF 等
原始数据批量算出的结果）。本 skill 负责：给定公司代码，从对应 CSV 读出该公司
「最新一期」的特征。

与旧版差异（关键改造）：
  1) 各特征文件的键不一致 —— F3/F4/F5 用 stock_code + (report_period|T_date)，
     F6 用 company_code + report_period，且 F5-C2/C3 的 stock_code 无交易所后缀；
     这里统一归一为 (company_code "000004.SZ", report_period int 20200331)。
  2) F5 由 C1/C2/C3 三个文件组成，需在同键上外连接合并。
  3) 元数据列（industry / matched_stat_date / governance_year / audit_year 等）
     一律不算特征。

在线爬取（crawl_sentiment/crawl_governance/crawl_inquiry）失败时，也会回退到本 skill
读离线表，保证不中断整体流程。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import FEATURE_TABLE_CONFIG, META_COLS


# ============================================================
# 键归一化
# ============================================================

def normalize_company_code(code) -> str:
    """把任意公司代码归一为 '000004.SZ' 形式。

    - "000004.SZ" -> "000004.SZ"
    - "000004"     -> "000004.SZ"（按首位推断交易所）
    - "4" / 4      -> "000004.SZ"（补零 + 推断）
    """
    s = str(code).strip()
    if "." in s:
        return s
    digits = "".join(ch for ch in s if ch.isdigit()) or "000000"
    digits = digits.zfill(6)[:6]
    first = digits[0]
    if first in ("6", "9"):
        return f"{digits}.SH"
    if first in ("4", "8"):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def normalize_report_period(p) -> int:
    """把任意报告期归一为 int YYYYMMDD。

    - 20200331          -> 20200331
    - "2020-03-31"      -> 20200331
    - "20200331"        -> 20200331
    - pandas Timestamp  -> 20200331
    """
    s = str(p).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return int(digits[:8])
    return 0


def _clean(v):
    """把 numpy 标量 / NaN 转成 Python 原生类型，便于 JSON 序列化。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):       # numpy 标量 -> Python 原生
        v = v.item()
    return v


# ============================================================
# 加载
# ============================================================

def _resolve_spec(agent_id_or_spec):
    """兼容两种入参：'F3' 字符串（查 config）或 config 里的 spec dict 本身。"""
    if isinstance(agent_id_or_spec, str):
        return FEATURE_TABLE_CONFIG[agent_id_or_spec]
    return agent_id_or_spec


def _read_family(agent_id_or_spec) -> pd.DataFrame:
    """按 config 里的口径读取某族特征，归一键后返回 DataFrame。

    返回列：__code / __period + 该族全部特征列（已转数值）。
    多文件（F5）在 (__code, __period) 上外连接合并。
    """
    spec = _resolve_spec(agent_id_or_spec)
    key, period = spec["key"], spec["period"]

    parts = []
    for f in spec["files"]:
        # dtype=str 读入，避免 000004 被读成整数 4（丢前导零）
        df = pd.read_csv(str(f), encoding="utf-8-sig", dtype=str, low_memory=False)
        df["__code"] = df[key].map(normalize_company_code)
        df["__period"] = df[period].map(normalize_report_period)
        # 元数据列（含 key/period 本身）不算特征
        feature_cols = [c for c in df.columns
                        if c not in META_COLS and c not in (key, period, "__code", "__period")]
        df = df[["__code", "__period"] + feature_cols]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # 同 (code, period) 去重取最后一条
        df = df.drop_duplicates(subset=["__code", "__period"], keep="last")
        parts.append(df)

    out = parts[0]
    for other in parts[1:]:
        out = out.merge(other, on=["__code", "__period"], how="outer")
    return out.reset_index(drop=True)


def load_latest_features(company_code: str, agent_id_or_spec) -> dict:
    """读取某族（F3/F4/F5/F6）预处理特征，返回该公司「最新一期」的特征 dict。

    参数
    ----
    company_code : str  公司代码，如 '000004.SZ'
    agent_id_or_spec : str 或 dict  特征族键 'F3'/'F4'/'F5'/'F6'，
                      或直接传 FEATURE_TABLE_CONFIG 里的 spec dict（两者皆可）

    返回
    ----
    dict  该族全部特征名 -> 最新一期的值（找不到该公司时返回空 dict）
    """
    df = _read_family(agent_id_or_spec)
    code = normalize_company_code(company_code)
    comp = df[df["__code"] == code]
    if len(comp) == 0:
        return {}

    latest = comp.sort_values("__period").iloc[-1]
    feats = {}
    for k, v in latest.items():
        if k.startswith("__"):
            continue
        feats[k] = _clean(v)
    return feats


def get_latest_report_period(company_code: str, agent_id_or_spec):
    """返回该公司在某族特征表里的最新报告期（int YYYYMMDD），找不到返回 None。"""
    df = _read_family(agent_id_or_spec)
    code = normalize_company_code(company_code)
    comp = df[df["__code"] == code]
    if len(comp) == 0:
        return None
    return int(comp["__period"].max())


if __name__ == "__main__":
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    for fam in ("F3", "F4", "F5", "F6"):
        feats = load_latest_features("000004.SZ", fam)
        rp = get_latest_report_period("000004.SZ", fam)
        print(f"{fam}: 最新报告期={rp}, 特征 {len(feats)} 维")
        print("   示例:", list(feats.items())[:4])
