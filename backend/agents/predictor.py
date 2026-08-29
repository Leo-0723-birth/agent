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
import hashlib
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (MODELING_DATASET, PREDICTOR_HORIZONS, PREDICTOR_MODEL_DIR,
                      PREDICTOR_SURVIVAL_BASELINE, PREDICTOR_SURVIVAL_FEATURES,
                      PREDICTOR_SURVIVAL_XGB, PREDICTOR_TOP_SHAP, RISK_THRESHOLDS)
from ..skills.stock_code import normalize_stock_code
from .base import AgentBase


# 进程级单例缓存：多个 PredictorAgent 实例共享已加载的模型/数据集/清单，
# 避免 orchestrator pool 中每个实例重复读取全量 CSV 与 9 个模型（内存成倍）。
# 实时任务由 _scan_lock 串行，离线路径不实例化 Predictor，故无需额外加锁。
_shared_df = None
_shared_manifests: dict = {}  # absolute manifest path -> parsed manifest
_shared_models: dict = {}     # (absolute model dir, horizon) -> {"rf", "lgb", "xgb"}
_shared_calibrators: dict = {}  # (absolute model dir, horizon) -> calibrator or None


class PredictorAgent(AgentBase):
    name = "Predictor"

    def __init__(self, horizons=PREDICTOR_HORIZONS, model_dir=None, top_shap=PREDICTOR_TOP_SHAP,
                 run_config=None):
        super().__init__()
        self.run_config = run_config
        self.horizons = horizons
        self.top_shap = top_shap
        self.model_dir = Path(model_dir or PREDICTOR_MODEL_DIR)
        self._manifest = None
        self._df = None
        self._models = {}   # horizon -> {"rf","lgb","xgb"}
        self._survival = None   # 可选：XGBoost-Cox 生存模型 {"booster","baseline","features"}

    # ================= 懒加载 =================
    def _load_manifest(self):
        p = (self.model_dir / "models_manifest.json").resolve()
        key = str(p)
        if key not in _shared_manifests:
            if p.exists():
                _shared_manifests[key] = json.loads(p.read_text(encoding="utf-8"))
            else:
                _shared_manifests[key] = {"windows": {}}
        return _shared_manifests[key]

    def _load_df(self):
        global _shared_df
        if _shared_df is None:
            if Path(MODELING_DATASET).exists():
                _shared_df = pd.read_csv(MODELING_DATASET, encoding="utf-8-sig")
            else:
                _shared_df = pd.DataFrame()
        return _shared_df

    def _validate_manifest(self, manifest):
        """在推理前检查模型清单的基础结构，损坏时只降级而不终止流水线。"""
        if not isinstance(manifest, dict):
            return False, ["manifest 不是对象"]
        windows = manifest.get("windows")
        if not isinstance(windows, dict):
            return False, ["manifest 缺少 windows"]
        messages = []
        for horizon in self.horizons:
            cfg = windows.get(horizon.replace("d", ""))
            if not isinstance(cfg, dict) or not isinstance(cfg.get("features"), list):
                messages.append(f"{horizon} 缺少特征清单")
        return not messages, messages

    def _model_version(self) -> str:
        """以模型清单内容哈希作为可复核版本，清单缺失时明确降级。"""
        path = self.model_dir / "models_manifest.json"
        try:
            return "manifest-sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        except OSError:
            return "manifest-unavailable"

    # ================= XGBoost-Cox 生存模型接口（可选，预留） =================
    def _load_survival(self):
        """加载 XGBoost-Cox 生存模型（单模型产出 30/60/90d 概率）。

        需要三个文件（训练后放置，见 config.PREDICTOR_SURVIVAL_*）：
          - model_survival_xgb.json      Booster（objective=survival:cox）
          - survival_baseline_hazard.json Breslow 累积基线风险 {"30": H0(30), "60": H0(60), "90": H0(90)}
          - survival_features.json       训练特征清单（与 manifest 对齐）
        缺失任一 → 返回 None，调用方回退三模型集成（现有路径）。
        """
        if self._survival is not None:
            return self._survival or None
        if not (PREDICTOR_SURVIVAL_XGB.exists() and PREDICTOR_SURVIVAL_BASELINE.exists()
                and PREDICTOR_SURVIVAL_FEATURES.exists()):
            self._survival = {}
            return None
        try:
            import json
            import xgboost as xgb
            booster = xgb.Booster(model_file=str(PREDICTOR_SURVIVAL_XGB))
            baseline = json.loads(PREDICTOR_SURVIVAL_BASELINE.read_text(encoding="utf-8"))
            feats = json.loads(PREDICTOR_SURVIVAL_FEATURES.read_text(encoding="utf-8"))
            self._survival = {"booster": booster, "baseline": baseline, "features": feats}
            return self._survival
        except Exception:
            self._survival = {}
            return None

    def _infer_survival(self, X, feats, horizon):
        """生存模型推理：P(未来 w 天内被问询) = 1 - S(w) = 1 - exp(-H0(w)·exp(risk))。

        返回 (p, conf, shap_features) 与 _infer 同构。X 列顺序须与 survival_features.json 一致。
        """
        surv = self._load_survival()
        if surv is None:
            return None, None, []
        w = horizon.replace("d", "")
        h0 = surv["baseline"].get(w)
        if h0 is None:
            return None, None, []
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(X, feature_names=feats)
            risk = float(surv["booster"].predict(dm)[0])       # log 风险比
            p = 1.0 - float(__import__("math").exp(-h0 * __import__("math").exp(risk)))
            p = max(0.0, min(1.0, p))
            contrib = surv["booster"].predict(dm, pred_contribs=True)[0][:-1]
            order = np.argsort(-np.abs(contrib))[:self.top_shap]
            shap_features = [(feats[i], round(float(contrib[i]), 5))
                             for i in order if abs(contrib[i]) > 1e-6]
            return p, round(max(p, 1.0 - p), 4), shap_features
        except Exception:
            return None, None, []

    def _load_models(self, horizon):
        cache_key = (str(self.model_dir.resolve()), horizon)
        if cache_key in _shared_models:
            return _shared_models[cache_key]
        w = horizon.replace("d", "")
        loaded = {}
        manifest = self._load_manifest()
        feats = manifest.get("windows", {}).get(w, {}).get("features", [])
        try:
            import joblib
            loaded["rf"] = joblib.load(self.model_dir / f"model_rf_{w}d.pkl")
            # 单条样本推理固定单线程：避免 Windows 下 joblib 线程池/命名管道
            # 被安全软件或受限环境拒绝（WinError 5），同时更快、更稳。
            if loaded["rf"] is not None:
                loaded["rf"].n_jobs = 1
        except Exception:
            loaded["rf"] = None
        try:
            model_text = (self.model_dir / f"model_lgb_{w}d.txt").read_text(encoding="utf-8")
            header = {}
            for line in model_text.splitlines()[:20]:
                if "=" in line:
                    name, value = line.split("=", 1)
                    header[name] = value
            artifact_count = int(header.get("max_feature_idx", "-1")) + 1
            artifact_names = header.get("feature_names", "").split()
            if (not feats or artifact_count != len(feats)
                    or len(artifact_names) != len(feats) or artifact_names != feats):
                raise ValueError(
                    f"LightGBM 特征维度不一致: model={artifact_count}, "
                    f"names={len(artifact_names)}, manifest={len(feats)}, "
                    f"names_match={artifact_names == feats}"
                )
            import lightgbm as lgb
            loaded["lgb"] = lgb.Booster(model_str=model_text)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("跳过不可用的 LightGBM %s：%s", horizon, exc)
            loaded["lgb"] = None
        try:
            import xgboost as xgb
            loaded["xgb"] = xgb.Booster(
                model_file=str(self.model_dir / f"model_xgb_{w}d.json"))
            # 校验 booster feature_names 与 manifest 一致
            if feats:
                xgb_names = loaded["xgb"].feature_names or []
                if xgb_names and xgb_names != feats:
                    import logging
                    logging.getLogger(__name__).warning(
                        "XGBoost %s feature_names 与 manifest 不一致（booster=%d, manifest=%d）",
                        horizon, len(xgb_names), len(feats)
                    )
        except Exception:
            loaded["xgb"] = None
        _shared_models[cache_key] = loaded
        return loaded

    def _risk_thresholds(self, horizon):
        """获取风险阈值。优先使用 manifest 中该窗口的最优 F1 阈值；
        否则回退到 backend.config.RISK_THRESHOLDS。"""
        w = horizon.replace("d", "")
        manifest = self._load_manifest()
        thr = manifest.get("windows", {}).get(w, {}).get("metrics", {}).get("threshold")
        if thr is not None:
            return {"high": float(thr), "medium": float(thr) * 0.5}
        return RISK_THRESHOLDS

    def _risk_level_from_prob(self, prob, horizon):
        """基于概率与窗口阈值输出风险等级。"""
        if prob is None:
            return "未预测"
        thr = self._risk_thresholds(horizon)
        if prob >= thr["high"]:
            return "高"
        if prob >= thr["medium"]:
            return "中"
        return "低"

    def _ensemble_health(self):
        """返回实际可用的模型成员，避免把降级集成误报为完整三模型。"""
        health = {}
        for horizon in self.horizons:
            models = self._load_models(horizon)
            available = [name for name in ("rf", "lgb", "xgb") if models.get(name) is not None]
            health[horizon] = {
                "available": available,
                "missing": [name for name in ("rf", "lgb", "xgb") if name not in available],
            }
        return health

    def _load_calibrator(self, horizon):
        """加载概率校准器（训练时用验证集拟合的 isotonic 校准）。

        训练脚本对 ensemble 概率做校准后才定阈值、报指标，因此推理必须走同一校准，
        否则线上概率与训练报告指标口径不一致。文件缺失时返回 None，调用方降级为
        未校准概率（并记录一次警告日志）。
        """
        cache_key = (str(self.model_dir.resolve()), horizon)
        if cache_key in _shared_calibrators:
            return _shared_calibrators[cache_key]
        w = horizon.replace("d", "")
        cal = None
        p = self.model_dir / f"calibrator_{w}d.joblib"
        try:
            import joblib
            if p.exists():
                cal = joblib.load(p)
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "未找到校准器 %s，将返回未校准概率（与训练指标口径不一致）", p)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("加载校准器 %s 失败：%s", p, e)
        _shared_calibrators[cache_key] = cal
        return cal

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

    # ================= 模型推理（两条路径共享） =================
    def _infer(self, X, feats, horizon):
        """三模型集成推理 + 概率校准 + SHAP。X: (1, n) float32；feats: 与 X 列对应。"""
        models = self._load_models(horizon)
        if not any(models.values()):
            return None, None, []

        # 集成权重从 manifest 读取（与训练脚本 ENSEMBLE_W 对齐），缺省兜底为历史权重
        manifest = self._load_manifest()
        weights = manifest.get("ensemble_weights") or {"rf": 0.30, "lgb": 0.35, "xgb": 0.35}

        raw_probs = []
        used_weights = []
        # 每个模型单独 try/except：单个模型推理失败（如环境缺依赖）不打断整窗口
        if models.get("rf") is not None:
            try:
                raw_probs.append(models["rf"].predict_proba(X)[0][1])
                used_weights.append(float(weights.get("rf", 0.0)))
            except Exception:
                pass
        if models.get("lgb") is not None:
            try:
                raw_probs.append(models["lgb"].predict(X)[0])
                used_weights.append(float(weights.get("lgb", 0.0)))
            except Exception:
                pass
        if models.get("xgb") is not None:
            try:
                raw_probs.append(models["xgb"].predict(xgb_dmatrix(X, feats))[0])
                used_weights.append(float(weights.get("xgb", 0.0)))
            except Exception:
                pass
        if not raw_probs:
            return None, None, []

        # 权重归一化：当某个模型推理失败时，剩余模型权重按比例放大，保证概率仍为 [0,1] 加权平均
        wsum = sum(used_weights)
        if wsum <= 0:
            wsum = 1.0
            used_weights = [1.0 / len(raw_probs)] * len(raw_probs)
        p_raw = float(sum(p * w for p, w in zip(raw_probs, used_weights)) / wsum)

        # 概率校准：与训练脚本口径一致（isotonic，验证集拟合）
        p = p_raw
        # 校准器是在完整三模型集成分数上拟合的。任一成员缺失时不可继续套用，
        # 否则会把两模型降级分数伪装成训练口径下的校准概率。
        calibrator = self._load_calibrator(horizon) if len(raw_probs) == 3 else None
        if calibrator is not None:
            try:
                p = float(calibrator.predict([[p_raw]])[0])
                p = max(0.0, min(1.0, p))
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "校准器 %s 推理失败，回退未校准概率", horizon)

        # SHAP：用 LightGBM pred_contrib 作集成贡献代理
        shap_features = []
        if models.get("lgb") is not None:
            try:
                contrib = models["lgb"].predict(X, pred_contrib=True)[0][:-1]
                order = np.argsort(-np.abs(contrib))[:self.top_shap]
                shap_features = [(feats[i], round(float(contrib[i]), 5))
                                 for i in order if abs(contrib[i]) > 1e-6]
            except Exception:
                pass
        return p, round(max(p, 1.0 - p), 4), shap_features

    # ================= 单窗口推理（查表路径） =================
    def _predict_one(self, horizon, row, manifest):
        w = horizon.replace("d", "")
        cfg = manifest["windows"].get(w)
        if not cfg:
            return None, None, []
        feats = cfg["features"]
        missing = [f for f in feats if f not in row.index]
        if missing:
            return None, None, []
        X = np.asarray(row[feats].values, dtype=np.float32).reshape(1, -1)
        return self._infer(X, feats, horizon)

    # ================= 单窗口推理（实时路径） =================
    def _predict_realtime(self, horizon, ctx, manifest):
        """实时路径：ctx 实时特征（公告 F1 标量 + 财务 F2-F6）→ manifest 对齐向量 → 集成推理。"""
        from ..skills.feature_composer import compose_realtime_features, load_fill_dict
        w = horizon.replace("d", "")
        cfg = manifest["windows"].get(w)
        if not cfg:
            return None, None, []
        feats = cfg["features"]
        fill = load_fill_dict(w, expected_features=feats)
        vec = compose_realtime_features(ctx, feats, fill)
        X = np.asarray([vec[f] for f in feats], dtype=np.float32).reshape(1, -1)
        return self._infer(X, feats, horizon)

    # ================= 主入口 =================
    def execute(self, company, ctx):
        code = normalize_stock_code(company)
        as_of = ctx.as_of or None
        ctx.company = code

        manifest = self._load_manifest()
        ok, messages = self._validate_manifest(manifest)
        if not ok:
            import logging
            logging.getLogger(__name__).warning(
                "PredictorAgent manifest 校验未通过，仍尝试推理；问题：%s", "; ".join(messages)
            )
        # 可选：XGBoost-Cox 生存模型已部署 → 单模型产出 30/60/90d（输出契约不变）
        if self._load_survival():
            return self._execute_survival(ctx, code, as_of)
        from ..skills.feature_composer import realtime_f1_compatibility
        f1_ok, f1_audit = realtime_f1_compatibility(ctx, manifest)
        # 只有财务特征存在且训练所需 F1 已按完全相同口径生成，才允许标记实时推理。
        realtime_ok = bool(getattr(getattr(ctx, "financial", None), "features", None)) and f1_ok
        if realtime_ok:
            return self._execute_realtime(ctx, manifest)
        reasons = []
        if not f1_ok:
            reasons.append(
                f"实时F1与训练特征口径不一致（缺少 {f1_audit['missing']}/{f1_audit['required']} 个语义特征），已回退历史查表"
            )
        if not getattr(getattr(ctx, "financial", None), "features", None):
            reasons.append("实时财务特征不可用，已回退历史查表")
        ctx.meta["prediction_degraded_reasons"] = reasons
        ctx.meta["f1_compatibility"] = f1_audit
        return self._execute_lookup(ctx, manifest, code, as_of)

    def _execute_survival(self, ctx, code, as_of):
        """生存模型主逻辑：单模型按生存函数推 30/60/90d 概率，输出契约与集成路径一致。

        特征来源：有实时财务特征 → feature_composer 组装（缺失列用 fill 兜底）；
        否则 → 离线查表 + fill 兜底。
        """
        from ..skills.feature_composer import compose_realtime_features, load_fill_dict
        surv = self._load_survival()
        feats = list(surv["features"])
        fill = load_fill_dict("60", expected_features=feats)
        from ..skills.feature_composer import realtime_f1_compatibility
        f1_ok, f1_audit = realtime_f1_compatibility(
            ctx, {"windows": {"survival": {"features": feats}}}
        )
        realtime_ok = bool(getattr(getattr(ctx, "financial", None), "features", None)) and f1_ok
        if realtime_ok:
            vec = compose_realtime_features(ctx, feats, fill)
            missing = [f for f in feats if vec.get(f) is None]
        else:
            row = self._lookup(code, as_of)
            if row is None:
                ctx.prediction = {"probability_30d": None, "probability_60d": None,
                                  "probability_90d": None, "risk_level": "未预测",
                                  "confidence": None, "shap_features": [],
                                  "reason": "未找到该股票特征（不在建模数据集内）"}
                return ctx
            vec = {f: row[f] for f in feats if f in row.index}
            missing = [f for f in feats if f not in vec]
        for f in missing:
            vec[f] = fill.get(f, 0.0)
        X = np.asarray([vec[f] for f in feats], dtype=np.float32).reshape(1, -1)

        pred = {"data_source": "realtime" if realtime_ok else "offline_lookup",
                "model_version": self._model_version(),
                "confidence_meaning": "predicted_class_score"}
        if not realtime_ok:
            pred["degraded_reasons"] = ["生存模型实时特征口径不完整，已使用历史特征查表"]
            pred["f1_compatibility"] = f1_audit
        p60, conf, shap = None, None, []
        for h in self.horizons:
            p, c, s = self._infer_survival(X, feats, h)
            pred[f"probability_{h}"] = p
            if h == "60d":
                p60, conf, shap = p, c, s
        if p60 is None:
            for h in self.horizons:
                if pred.get(f"probability_{h}") is not None:
                    p60 = pred[f"probability_{h}"]
                    conf = round(max(p60, 1 - p60), 4)
                    break
        pred["confidence"] = conf
        pred["risk_level"] = self._risk_level_from_prob(p60, "60d")
        pred["shap_features"] = shap
        pred["feature_anchor"] = str(getattr(ctx, "as_of", "") or "")[:10]
        ctx.prediction = pred
        return ctx

    def _execute_realtime(self, ctx, manifest):
        """实时推理主逻辑：以公告研读 F1 + 财务异常 F2-F6 为数据源，概率由实时数据驱动。"""
        from ..skills.feature_composer import coverage_stats
        pred = {"data_source": "realtime", "model_version": self._model_version(),
                "confidence_meaning": "predicted_class_score"}
        p60, conf, shap = None, None, []
        for h in self.horizons:
            p, c, s = self._predict_realtime(h, ctx, manifest)
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

        pred["model_health"] = self._ensemble_health()
        pred["probability_calibration"] = (
            "isotonic_calibrated"
            if all(not item["missing"] for item in pred["model_health"].values())
            else "degraded_uncalibrated_partial_ensemble"
        )

        pred["confidence"] = conf
        pred["risk_level"] = self._risk_level_from_prob(p60, "60d")
        pred["shap_features"] = shap
        # 审计：特征锚点（财务最新报告期）+ 实时覆盖率
        fp = getattr(ctx.financial, "indicators", {}) or {}
        pred["feature_anchor"] = str(fp.get("report_period", ctx.as_of or ""))[:10]
        pred["coverage"] = coverage_stats(getattr(ctx, "features_origin", {}))
        ctx.prediction = pred
        return ctx

    def _execute_lookup(self, ctx, manifest, code, as_of):
        """兜底：离线建模数据查表推理（公司无实时财务数据时）。"""
        degraded_reasons = (getattr(ctx, "meta", {}) or {}).get("prediction_degraded_reasons", [])
        f1_audit = (getattr(ctx, "meta", {}) or {}).get("f1_compatibility", {})
        row = self._lookup(code, as_of)
        if row is None:
            ctx.prediction = {"probability_30d": None, "probability_60d": None,
                              "probability_90d": None, "risk_level": "未预测",
                              "confidence": None, "shap_features": [],
                              "reason": "未找到该股票特征（不在建模数据集内）",
                              "data_source": "unavailable",
                              "degraded_reasons": list(degraded_reasons or [])}
            return ctx

        anchor = str(row["report_period"])
        pred = {"feature_anchor": anchor, "data_source": "offline_lookup",
                "model_version": self._model_version(),
                "confidence_meaning": "predicted_class_score",
                "degraded_reasons": list(degraded_reasons or []),
                "f1_compatibility": f1_audit or {}}
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

        pred["model_health"] = self._ensemble_health()
        pred["probability_calibration"] = (
            "isotonic_calibrated"
            if all(not item["missing"] for item in pred["model_health"].values())
            else "degraded_uncalibrated_partial_ensemble"
        )

        pred["confidence"] = conf
        pred["risk_level"] = self._risk_level_from_prob(p60, "60d")
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
