#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测建模 Agent (PredictorAgent) —— 任务2 的模型推理（已实现）
================================================================
职责：按 (company, as_of) 从建模数据集查"截至 T 的最新一行特征" → 三模型集成推理
      （RF + LightGBM + XGBoost，30/60/90 三窗口）→ 概率 + SHAP + 风险等级。

推理路径（查表模式，对齐比赛口径）：
  - 查 backend/data/modeling/processed_dataset.csv 中该股票 report_period ≤ as_of 的最新一行
  - 按 models_manifest.json 的每窗口特征清单截取 → 三模型分别预测 → 集成加权
  - SHAP：LightGBM pred_contrib（TreeSHAP 内置）作为集成贡献代理，Top-K 写入 shap_features
  - 公司不在建模表内 → 优雅降级为"未预测"（不打断流水线）

依赖：backend/scripts/train_models.py 先训练产出 models/predictor/（9 模型 + manifest）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (MODELING_DATASET, PREDICTOR_HORIZONS, PREDICTOR_MODEL_DIR,
                      PREDICTOR_TOP_SHAP, RISK_THRESHOLDS)
from .base import AgentBase


class PredictorAgent(AgentBase):
    name = "Predictor"

    def __init__(self, horizons=PREDICTOR_HORIZONS, model_dir=None, top_shap=PREDICTOR_TOP_SHAP):
        super().__init__()
        self.horizons = horizons
        self.top_shap = top_shap
        self.model_dir = Path(model_dir or PREDICTOR_MODEL_DIR)
        self._manifest = None
        self._df = None
        self._models = {}   # horizon -> {"rf","lgb","xgb"}

    # ================= 懒加载 =================
    def _load_manifest(self):
        if self._manifest is None:
            p = self.model_dir / "models_manifest.json"
            if p.exists():
                self._manifest = json.loads(p.read_text(encoding="utf-8"))
            else:
                self._manifest = {"windows": {}}
        return self._manifest

    def _load_df(self):
        if self._df is None:
            if Path(MODELING_DATASET).exists():
                self._df = pd.read_csv(MODELING_DATASET, encoding="utf-8-sig")
            else:
                self._df = pd.DataFrame()
        return self._df

    def _load_models(self, horizon):
        if horizon in self._models:
            return self._models[horizon]
        w = horizon.replace("d", "")
        loaded = {}
        try:
            import joblib
            loaded["rf"] = joblib.load(self.model_dir / f"model_rf_{w}d.pkl")
        except Exception:
            loaded["rf"] = None
        try:
            import lightgbm as lgb
            loaded["lgb"] = lgb.Booster(
                model_str=(self.model_dir / f"model_lgb_{w}d.txt").read_text(encoding="utf-8"))
        except Exception:
            loaded["lgb"] = None
        try:
            import xgboost as xgb
            loaded["xgb"] = xgb.Booster(
                model_file=str(self.model_dir / f"model_xgb_{w}d.json"))
        except Exception:
            loaded["xgb"] = None
        self._models[horizon] = loaded
        return loaded

    # ================= 查表（as-of） =================
    def _lookup(self, code, as_of):
        """查该股票 report_period ≤ as_of 的最新一行。返回 (行 Series) 或 None。"""
        df = self._load_df()
        if df.empty:
            return None
        as_of_yyyymmdd = str(as_of or "")[:10].replace("-", "")[:8]
        sub = df[df["company_code"].astype(str) == str(code)].copy()
        if sub.empty:
            return None
        if as_of_yyyymmdd:
            sub["_rp"] = sub["report_period"].astype(str).str.replace(".0", "", regex=False).str[:8]
            sub = sub[sub["_rp"] <= as_of_yyyymmdd]
        if sub.empty:
            return None
        return sub.sort_values("_rp" if as_of_yyyymmdd else "report_period").iloc[-1]

    # ================= 单窗口推理 =================
    def _predict_one(self, horizon, row, manifest):
        w = horizon.replace("d", "")
        cfg = manifest["windows"].get(w)
        if not cfg:
            return None, None, []
        feats = cfg["features"]
        models = self._load_models(horizon)
        if not any(models.values()):
            return None, None, []

        missing = [f for f in feats if f not in row.index]
        if missing:
            return None, None, []
        X = np.asarray(row[feats].values, dtype=np.float32).reshape(1, -1)

        probs = []
        if models.get("rf") is not None:
            probs.append(0.30 * models["rf"].predict_proba(X)[0][1])
        if models.get("lgb") is not None:
            probs.append(0.35 * models["lgb"].predict(X)[0])
        if models.get("xgb") is not None:
            probs.append(0.35 * models["xgb"].predict(xgb_dmatrix(X, feats))[0])
        if not probs:
            return None, None, []
        p = float(sum(probs))

        # SHAP：用 LightGBM pred_contrib 作集成贡献代理
        shap_features = []
        if models.get("lgb") is not None:
            contrib = models["lgb"].predict(X, pred_contrib=True)[0][:-1]
            order = np.argsort(-np.abs(contrib))[:self.top_shap]
            shap_features = [(feats[i], round(float(contrib[i]), 5))
                             for i in order if abs(contrib[i]) > 1e-6]
        return p, round(max(p, 1.0 - p), 4), shap_features

    # ================= 主入口 =================
    def execute(self, company, ctx):
        code = company
        as_of = ctx.as_of or None
        ctx.company = code

        manifest = self._load_manifest()
        row = self._lookup(code, as_of)
        if row is None:
            ctx.prediction = {"probability_30d": None, "probability_60d": None,
                              "probability_90d": None, "risk_level": "未预测",
                              "confidence": None, "shap_features": [],
                              "reason": "未找到该股票特征（不在建模数据集内）"}
            return ctx

        anchor = str(row["report_period"])
        pred = {"feature_anchor": anchor}
        p60, conf, shap = None, None, []
        for h in self.horizons:
            p, c, s = self._predict_one(h, row, manifest)
            key = f"probability_{h}"
            pred[key] = p
            if h == "60d":
                p60, conf, shap = p, c, s
        if p60 is None:
            # 主窗口不可用则取任一
            for h in self.horizons:
                if pred.get(f"probability_{h}") is not None:
                    p60 = pred[f"probability_{h}"]
                    conf = round(max(p60, 1 - p60), 4)
                    break

        pred["confidence"] = conf
        pred["risk_level"] = "未预测" if p60 is None else (
            "高" if p60 >= RISK_THRESHOLDS["high"] else
            "中" if p60 >= RISK_THRESHOLDS["medium"] else "低")
        pred["shap_features"] = shap
        ctx.prediction = pred
        return ctx


def xgb_dmatrix(X, feats):
    import xgboost as xgb
    return xgb.DMatrix(X, feature_names=feats)


# ============================================================
# 自测入口（python -m backend.agents.predictor）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context

    agent = PredictorAgent()
    ctx = Context(company="000004.SZ", window=60, as_of="2025-12-02")
    agent.execute("000004.SZ", ctx)
    p = ctx.prediction
    print(f"公司 {ctx.company} | 特征锚点 T={p.get('feature_anchor')}")
    print(f"30d={p.get('probability_30d')} | 60d={p.get('probability_60d')} | "
          f"90d={p.get('probability_90d')} | 等级={p.get('risk_level')} | 置信度={p.get('confidence')}")
    for name, v in p.get("shap_features", [])[:8]:
        print(f"  SHAP {v:+8.4f}  {name}")
