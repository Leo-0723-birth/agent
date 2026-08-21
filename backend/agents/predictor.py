#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测建模 Agent (PredictorAgent) —— 任务2 的模型推理
===================================================
职责：特征组装（F1-F6 时点对齐）→ 集成模型推理（XGB/LGBM/DeepFM → Stacking）
      → 概率校准 → 风险等级判定。纯计算（ML 模型），不调用 LLM。
输入：company（代码）、ctx（读取 semantic/financial 特征，组装为 ctx.features）
输出：写入 ctx.features（组装后向量）与 ctx.prediction
      {probability_60d, risk_level, confidence, model_version}

依赖：models/predictor/ 下训练产物（scripts/train_predictor.py 产出）。

TODO: 由后续开发填充实现。
"""
from .base import AgentBase


class PredictorAgent(AgentBase):
    name = "Predictor"

    def __init__(self, model_dir=None, window=60):
        # TODO: 加载训练好的模型与校准器
        super().__init__()

    def assemble_features(self, ctx):
        # TODO: Skill —— 特征组装与时点对齐（F1 语义 + F2-F6 量化）
        raise NotImplementedError

    def execute(self, company, ctx):
        # TODO: 组装 → 推理 → 校准 → 写 ctx.features 与 ctx.prediction
        raise NotImplementedError


if __name__ == "__main__":
    # TODO: 自测入口
    pass
