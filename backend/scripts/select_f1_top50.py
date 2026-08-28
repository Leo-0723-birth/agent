#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F1 语义特征 Top-K 选取（复现脚本）
==================================
依据：训练集上与 target_60d 的 Spearman 秩相关系数绝对值，取 Top-K。

输入：backend/data/modeling/processed_dataset.csv
输出：backend/data/modeling/raw/f1_selection/F1_top50_features.csv

说明：
  - 该脚本用于复现/审计 build_modeling_dataset.py 内的 F1 筛选逻辑；
  - 实际建模流程中 build_modeling_dataset.py 已内联完成同等筛选，无需单独运行；
  - N_KEEP 默认与 build_modeling_dataset.py 保持一致（100）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import FEATURE_FAMILY_PREFIXES, TRAIN_SPLIT_NAMES

BASE = Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "modeling"
DATA = BASE / "processed_dataset.csv"
OUT = BASE / "raw" / "f1_selection" / "F1_top50_features.csv"
N_KEEP = 100  # 与 build_modeling_dataset.py 保持一致


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"未找到建模数据集：{DATA}")

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    f1_prefix = next((p for p in FEATURE_FAMILY_PREFIXES if "semantic" in p or p.startswith("f1_")), None)
    if f1_prefix is None:
        raise ValueError(f"config.FEATURE_FAMILY_PREFIXES 中未找到 F1 前缀：{FEATURE_FAMILY_PREFIXES}")

    semantic_cols = [c for c in df.columns if c.startswith(f1_prefix)]
    print(f"F1 特征前缀: {f1_prefix}，共 {len(semantic_cols)} 维")

    train_mask = df["split"].isin(TRAIN_SPLIT_NAMES) & (df["target_60d"] >= 0)
    y_train = df.loc[train_mask, "target_60d"]
    print(f"训练集有效样本: {len(y_train)}，正样本 {y_train.sum()}")

    corr_scores = {}
    for c in semantic_cols:
        col = df.loc[train_mask, c]
        if col.nunique() > 1:
            try:
                corr, _ = spearmanr(col, y_train)
                corr_scores[c] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                corr_scores[c] = 0.0
        else:
            corr_scores[c] = 0.0

    top = sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)[:N_KEEP]
    print(f"Selected {len(top)} features, |corr| range {top[0][1]:.4f} ~ {top[-1][1]:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "rank": range(1, len(top) + 1),
        "feature": [c for c, _ in top],
        "abs_corr": [round(c, 4) for _, c in top],
    }).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已保存：{OUT}")

    for i, (c, corr) in enumerate(top, 1):
        print(f"{i:3d}. {c}  |corr|={corr:.4f}")


if __name__ == "__main__":
    main()
