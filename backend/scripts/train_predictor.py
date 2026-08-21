#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线任务：训练被问询概率预测模型
================================
特征工程（F1-F6，T 时点对齐）→ 训练 XGBoost/LightGBM/DeepFM → Stacking
→ 概率校准 → 评估（AUC/Top-10%召回/F1）→ 产物保存到 models/predictor/。

运行（从项目根目录）：
    python -m backend.scripts.train_predictor

TODO: 由后续开发填充实现（参考底层建模方案 3.1/3.2）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    # TODO: 构建样本 → 训练 → 校准 → 评估 → 保存模型
    raise NotImplementedError


if __name__ == "__main__":
    main()
