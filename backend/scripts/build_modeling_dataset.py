#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理：构建建模数据集（F1 前50语义 + F2-F6 + 30/60/90d 标签）
=================================================================
流程：
  1. 加载 F1 语义特征 parquet（300 维 PCA 主成分，announcement_semantic_*）
  2. 按"训练集 Spearman |corr| 与 target_60d"筛选 Top-50
  3. 加载 F2/F3/F4/F5/F6（键：company_code + report_period）
  4. 由 inquiry_events.csv（kind=='letter'）构建 30/60/90 天未来问询标签
  5. 合并保存 processed_dataset.csv + F1_top50_features.csv

输出：backend/data/modeling/processed_dataset.csv
说明：与上游版本相比仅更换 F1 数据源（问询函语义 → 官方公告语义，
      20% 年报解析；全量数据替换 raw/F1_announcement_semantic_features.parquet 后重跑即可）。
      标签口径、F2 骨架、合并逻辑保持不变。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 训练原料从项目内读取（backend/data/modeling/raw/），输出到建模数据根目录
_MODELING = Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "modeling"
RAW = _MODELING / "raw"
PROC = _MODELING
SEL = RAW / "f1_selection"
PROC.mkdir(parents=True, exist_ok=True)
SEL.mkdir(parents=True, exist_ok=True)

WINDOWS = [30, 60, 90]
N_KEEP = 50

# ============================================================
# 1) F1 语义特征 Top-50 选取（训练集 Spearman |corr| vs target_60d）
#    数据源：官方公告语义特征（announcement_semantic_*，20% 年报解析）
# ============================================================
f1 = pd.read_parquet(RAW / "F1_announcement_semantic_features.parquet")
f1 = f1.rename(columns={"stock_code": "company_code", "T_date": "report_period"})
f1["company_code"] = f1["company_code"].astype(str)
f1["report_period"] = pd.to_datetime(f1["report_period"], errors="coerce").dt.strftime("%Y%m%d")
semantic_cols = [c for c in f1.columns if c.startswith("announcement_semantic_")]
print(f"F1 公告语义特征: {len(semantic_cols)} 维（announcement_semantic_*）")

# 选取用骨架：公司级 split（F2）+ 每个 (公司, 报告期) 的 target_60d
_f2_sel = pd.read_csv(RAW / "F2_financial_anomaly.csv", encoding="utf-8-sig")
_f2_sel["company_code"] = _f2_sel["company_code"].astype(str)
_f2_sel["report_period"] = _f2_sel["report_period"].astype(str).str.replace(".0", "", regex=False)
_events = pd.read_csv(RAW / "inquiry_events.csv", encoding="utf-8-sig")
_events = _events[_events["kind"] == "letter"].copy()
_events["secucode"] = _events["secucode"].astype(str)
_events["date"] = pd.to_datetime(_events["date"], errors="coerce")
_emap = {c: sorted(g["date"].dropna().tolist()) for c, g in _events.groupby("secucode")}


def _build_target(code, t, days):
    for d in _emap.get(code, []):
        if t < d <= t + pd.Timedelta(days=days):
            return 1
    return 0


_sel = _f2_sel[["company_code", "report_period", "split"]].copy()
_t0 = pd.to_datetime(_sel["report_period"], format="%Y%m%d", errors="coerce")
_sel["target_60d"] = [_build_target(c, t, 60) for c, t in zip(_sel["company_code"], _t0)]
f1_sel = f1[["company_code", "report_period"] + semantic_cols].merge(
    _sel[["company_code", "report_period", "split", "target_60d"]],
    on=["company_code", "report_period"], how="inner")
_tr = (f1_sel["split"] == "Train") & (f1_sel["target_60d"] >= 0)
_y = f1_sel.loc[_tr, "target_60d"]

corr_scores = {}
for c in semantic_cols:
    col = f1_sel.loc[_tr, c]
    if col.nunique() > 1:
        try:
            r, _ = spearmanr(col, _y)
            corr_scores[c] = abs(r) if not np.isnan(r) else 0.0
        except Exception:
            corr_scores[c] = 0.0
    else:
        corr_scores[c] = 0.0

top50 = [c for c, _ in sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)[:N_KEEP]]
print(f"F1 Top-{len(top50)} 特征已确定（|corr| 区间 "
      f"{corr_scores[top50[0]]:.4f} ~ {corr_scores[top50[-1]]:.4f}）")
pd.DataFrame({"rank": range(1, len(top50) + 1), "feature": top50}).to_csv(
    SEL / "F1_top50_features.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 2) 加载 F1 并归一化键
# ============================================================
keep_f1 = ["company_code", "report_period"] + [c for c in top50 if c in f1.columns]
f1 = f1[keep_f1]
print(f"F1: {f1.shape}（Top-50 公告语义特征）")

# ============================================================
# 3) 加载 F2-F6
# ============================================================
f2 = pd.read_csv(RAW / "F2_financial_anomaly.csv", encoding="utf-8-sig")
f3 = pd.read_csv(RAW / "F3_market_features.csv", encoding="utf-8-sig")
f4 = pd.read_csv(RAW / "F4_sentiment_features.csv", encoding="utf-8-sig")
f5 = pd.read_csv(RAW / "F5_ownership_governance.csv", encoding="utf-8-sig")
f6 = pd.read_csv(RAW / "F6_inquiry_history.csv", encoding="utf-8-sig")

def norm(df):
    df = df.copy()
    df["company_code"] = df["company_code"].astype(str)
    df["report_period"] = df["report_period"].astype(str).str.replace(".0", "", regex=False)
    return df

f2, f3, f4, f5, f6 = [norm(d) for d in (f2, f3, f4, f5, f6)]
# split 以 F2 为主表
split_df = f2[["company_code", "report_period", "split"]].copy()
features = [f2, f3, f4, f5, f6]
print("各家族维度:", [d.shape[1] for d in features])

# ============================================================
# 4) 合并（以 F2 键为骨架，left join 其余）
# ============================================================
df = f2.drop(columns=["split"], errors="ignore")
for name, d in [("F3", f3), ("F4", f4), ("F5", f5), ("F6", f6)]:
    cols = [c for c in d.columns if c not in ("company_code", "report_period", "split")]
    df = df.merge(d[["company_code", "report_period"] + cols], on=["company_code", "report_period"], how="left")
    print(f"  合并 {name}: {df.shape}")

# 合并 F1（保留 split 骨架的全部行）
df = split_df.merge(f1, on=["company_code", "report_period"], how="left")
for name, d in [("F2", f2.drop(columns=["split"])), ("F3", f3.drop(columns=["split"])),
                ("F4", f4.drop(columns=["split"])), ("F5", f5.drop(columns=["split"])),
                ("F6", f6.drop(columns=["split"]))]:
    df = df.merge(d, on=["company_code", "report_period"], how="left")
df = df.loc[:, ~df.columns.duplicated()]
print(f"合并后: {df.shape}")

# ============================================================
# 5) 构建 30/60/90 天标签（kind=='letter' 的未来问询事件）
# ============================================================
events = pd.read_csv(RAW / "inquiry_events.csv", encoding="utf-8-sig")
events = events[events["kind"] == "letter"].copy()
events["secucode"] = events["secucode"].astype(str)
events["date"] = pd.to_datetime(events["date"], errors="coerce")
print(f"问询函事件: {len(events)} 条（含年报/关注/重组/其他）")

event_map = {}
for code, grp in events.groupby("secucode"):
    event_map[code] = sorted(grp["date"].dropna().tolist())

def build_target(row_date, days):
    code = row_date[0]
    t = row_date[1]
    for d in event_map.get(code, []):
        if t < d <= t + pd.Timedelta(days=days):
            return 1
    return 0


def count_future(row_date, days):
    """未来 days 天窗口内问询函数量（sample_weight 用，非模型特征）。"""
    code = row_date[0]
    t = row_date[1]
    return sum(1 for d in event_map.get(code, []) if t < d <= t + pd.Timedelta(days=days))

t0 = pd.to_datetime(df["report_period"], format="%Y%m%d", errors="coerce")
for w in WINDOWS:
    df[f"target_{w}d"] = [build_target((c, t), w) for c, t in zip(df["company_code"], t0)]
# 60 天窗口问询函数量（仅训练加权用；train_models.py 特征筛选已排除 n_inq_* 前缀防泄漏）
df["n_inq_60d"] = [count_future((c, t), 60) for c, t in zip(df["company_code"], t0)]

# ============================================================
# 6) 保存
# ============================================================
df.to_csv(PROC / "processed_dataset.csv", index=False, encoding="utf-8-sig")
print(f"\n已保存: {PROC / 'processed_dataset.csv'}（{df.shape}）")
print("\n正样本率（Train 集合）:")
for w in WINDOWS:
    m = (df["split"] == "Train") & (df[f"target_{w}d"] >= 0)
    pos = df.loc[m, f"target_{w}d"].sum()
    n = m.sum()
    print(f"  {w}d: 正样本 {pos}/{n} = {pos / max(n, 1) * 100:.2f}%")
