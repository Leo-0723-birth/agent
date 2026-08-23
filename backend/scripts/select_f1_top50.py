#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F1 语义特征 Top-50 选取（复现脚本）
==================================
依据：训练集上 与 target_60d 的 Spearman 秩相关系数绝对值，取 Top-50。

为什么选 Top-50：
  1. 300 维 PCA 主成分中大量维度与标签相关性极低（|corr| 普遍 <0.01），
     属于"低效特征"，会稀释树模型的分裂收益并增加过拟合风险；
  2. 保留 Top-50 后维度从 300 → 50，训练更快、可解释性更好；
  3. 描述文档（semantic_feature_descriptions）给出了每维的业务含义
     （如 semantic_001=合规与诉讼风险—经营与市场风险语义对比），
     便于人工核对选取是否合理。

用法：python select_f1_top50.py（需 F1_base_financial_full.csv 含 split 与 target_60d）
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 加载含标签与 split 的 F1 全量表（本流程已由 build_dataset.py 内联完成同等逻辑）
f1 = pd.read_csv("F1_base_financial_full.csv")  # 路径按需调整
semantic_cols = [c for c in f1.columns if "semantic" in c]
print(f"Total semantic features: {len(semantic_cols)}")

train_mask = (f1["split"] == "Train") & (f1["target_60d"] >= 0)
y_train = f1.loc[train_mask, "target_60d"]

corr_scores = {}
for c in semantic_cols:
    col = f1.loc[train_mask, c]
    if col.nunique() > 1:
        try:
            corr, _ = spearmanr(col, y_train)
            corr_scores[c] = abs(corr) if not np.isnan(corr) else 0
        except Exception:
            corr_scores[c] = 0
    else:
        corr_scores[c] = 0

N_KEEP = 50
top = sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)[:N_KEEP]
print(f"Selected {len(top)} features, |corr| range {top[0][1]:.4f} ~ {top[-1][1]:.4f}")
pd.DataFrame({"rank": range(1, N_KEEP + 1), "feature": [c for c, _ in top],
              "abs_corr": [round(c, 4) for _, c in top]}).to_csv(
    "F1_top50_features.csv", index=False, encoding="utf-8-sig")
for i, (c, corr) in enumerate(top, 1):
    print(f"{i:3d}. {c}  |corr|={corr:.4f}")
