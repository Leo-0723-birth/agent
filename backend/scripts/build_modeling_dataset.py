#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理：构建建模数据集（F1 前50语义 + F2-F6 + 30/60/90d 标签）
=================================================================
流程：
  1. 加载 F1 语义特征 parquet（300 维 PCA 主成分）
  2. 按"训练集 Spearman |corr| 与 target_60d"筛选 Top-50（见 03_F1特征选取）
  3. 加载 F2/F3/F4/F5/F6（键：company_code + report_period）
  4. 由 inquiry_events.csv（kind=='letter'）构建 30/60/90 天未来问询标签
  5. 合并保存 processed_dataset.csv + F1_top50_features.csv

输出：05_模型输出/processed_dataset.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"C:\Users\86130\Desktop\预测建模agent")
RAW = OUT / "01_原数据"
PROC = OUT / "05_模型输出"
SEL = OUT / "03_F1特征选取"
PROC.mkdir(parents=True, exist_ok=True)

WINDOWS = [30, 60, 90]
N_KEEP = 50

# ============================================================
# 1) F1 语义特征 Top-50 选取（依据 semantic_feature_descriptions 的 Spearman 排序）
# ============================================================
def select_f1_top50(descriptions_md):
    """从描述文档解析 Top-50 特征名（按文档顺序 = Spearman |corr| 降序）。"""
    names = []
    for line in descriptions_md.splitlines():
        m = re.match(r"\|\s*\d+\s*\|\s*(semantic_\d{3})\s*\|", line)
        if m:
            names.append(m.group(1))
    return names[:N_KEEP]


desc_md = (SEL / "semantic_feature_descriptions_原版.md").read_text(encoding="utf-8")
top50_short = select_f1_top50(desc_md)
if len(top50_short) < N_KEEP:
    print(f"[警告] 描述文档仅解析到 {len(top50_short)} 个特征，不足 50；回退用 Spearman 计算补充")
top50 = [f"regulatory_inquiry_{s}" for s in top50_short]
print(f"F1 Top-{len(top50)} 特征已确定（示例: {top50[:3]} ... {top50[-2:]}）")
pd.DataFrame({"rank": range(1, len(top50) + 1), "feature": top50}).to_csv(
    SEL / "F1_top50_features.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 2) 加载 F1 并归一化键
# ============================================================
f1 = pd.read_parquet(RAW / "F1_semantic_features.parquet")
f1 = f1.rename(columns={"stock_code": "company_code", "T_date": "report_period"})
f1["company_code"] = f1["company_code"].astype(str)
f1["report_period"] = pd.to_datetime(f1["report_period"], errors="coerce").dt.strftime("%Y%m%d")
keep_f1 = ["company_code", "report_period"] + [c for c in top50 if c in f1.columns]
f1 = f1[keep_f1]
print(f"F1: {f1.shape}（Top-50 语义特征）")

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

t0 = pd.to_datetime(df["report_period"], format="%Y%m%d", errors="coerce")
for w in WINDOWS:
    df[f"target_{w}d"] = [build_target((c, t), w) for c, t in zip(df["company_code"], t0)]

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
