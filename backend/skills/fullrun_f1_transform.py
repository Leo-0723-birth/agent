#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量公告流水线同口径的 208 维聚合与冻结 PCA50 变换。

输入必须是上游完整语义链路产生的 ``announcement_risk_features`` 行；普通
规则/LLM候选不能传入本模块冒充同口径结果。冻结模型来自
``run_20260826_001320_full``，加载前校验 SHA256。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "f1"
DEFAULT_MODEL = MODEL_DIR / "pca_model.pkl"
DEFAULT_MANIFEST = MODEL_DIR / "fullrun_f1_manifest.json"
WINDOWS = (30, 60, 90, 180)
L1_CODES = tuple("ABCDEFGH")
FINANCIAL_COLUMNS = (
    "market_cap", "pe_ratio", "pb_ratio", "total_revenue", "net_profit",
    "operating_cash_flow", "roe", "roa", "debt_to_assets_ratio",
    "revenue_yoy_growth", "net_profit_yoy_growth",
)
REQUIRED_ROW_COLUMNS = {
    "announcement_id", "publish_date", "risk_theme", "l1_code",
    "risk_strength_v2_top3", "rerank_score_max", "sent_neg_max",
    "strong_count", "rule_effective_hits",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FullRunF1Transformer:
    """复现全量运行的季度聚合与 PCA50，不重新 fit 任何预处理器。"""

    def __init__(self, model_path=None, manifest_path=None):
        self.model_path = Path(
            model_path or os.getenv("F1_PCA_MODEL_PATH") or DEFAULT_MODEL
        ).expanduser().resolve()
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST).expanduser().resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        actual_hash = _sha256(self.model_path)
        expected_hash = self.manifest["artifact_sha256"]
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f"F1 PCA 模型哈希不一致: expected={expected_hash}, actual={actual_hash}"
            )
        # pickle 只在哈希验证通过后加载；该文件是本仓库固定的受控训练产物。
        with self.model_path.open("rb") as handle:
            bundle = pickle.load(handle)
        required = {"imputer", "scaler", "pca", "feature_columns"}
        if not required.issubset(bundle):
            raise ValueError(f"F1 PCA bundle 缺少字段: {sorted(required - set(bundle))}")
        self.bundle = bundle
        self.feature_columns = list(bundle["feature_columns"])
        if len(self.feature_columns) != int(self.manifest["raw_feature_dim"]):
            raise ValueError("F1 PCA 原始维度与清单不一致")

    @property
    def themes(self):
        return sorted({
            name[3:-4] for name in self.feature_columns
            if name.startswith("th_") and name.endswith("_max")
        })

    def aggregate_raw(self, announcement_rows, as_of, financial_values=None) -> dict:
        """把公告主题级行聚合为冻结 PCA 所需的 208 维原始特征。"""
        frame = pd.DataFrame(announcement_rows).copy()
        missing = sorted(REQUIRED_ROW_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"公告风险行缺少字段: {missing}")
        target = pd.Timestamp(as_of).normalize()
        frame["publish_date"] = pd.to_datetime(frame["publish_date"], errors="coerce")
        frame = frame[frame.publish_date.notna() & (frame.publish_date <= target)].copy()
        score = "risk_strength_v2_top3"
        threshold = float(self.manifest["high_risk_threshold"])
        frame["is_high"] = pd.to_numeric(frame[score], errors="coerce").fillna(0) >= threshold
        for column in (
            score, "rerank_score_max", "sent_neg_max", "strong_count",
            "rule_effective_hits",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        record = {}
        for days in WINDOWS:
            prefix = f"w{days}"
            subset = frame[
                (frame.publish_date > target - timedelta(days=days))
                & (frame.publish_date <= target)
            ]
            record[f"{prefix}_n_ann"] = int(subset.announcement_id.nunique())
            record[f"{prefix}_n_rows"] = int(len(subset))
            record[f"{prefix}_n_high"] = int(subset.is_high.sum())
            record[f"{prefix}_n_themes"] = int(subset.risk_theme.nunique())
            record[f"{prefix}_score_mean"] = float(subset[score].mean()) if len(subset) else 0.0
            record[f"{prefix}_score_max"] = float(subset[score].max()) if len(subset) else 0.0
            record[f"{prefix}_score_sum"] = float(subset[score].sum()) if len(subset) else 0.0
            record[f"{prefix}_rerank_max"] = (
                float(subset.rerank_score_max.max()) if len(subset) else 0.0
            )
            record[f"{prefix}_sentneg_mean"] = (
                float(subset.sent_neg_max.mean()) if len(subset) else 0.0
            )
            record[f"{prefix}_rule_hits"] = int(subset.rule_effective_hits.sum())
            record[f"{prefix}_strong"] = int(subset.strong_count.sum())
            grouped = subset.groupby("l1_code")[score].agg(["max", "size"])
            for l1 in L1_CODES:
                record[f"{prefix}_L1_{l1}_max"] = (
                    float(grouped.loc[l1, "max"]) if l1 in grouped.index else 0.0
                )
                record[f"{prefix}_L1_{l1}_cnt"] = (
                    int(grouped.loc[l1, "size"]) if l1 in grouped.index else 0
                )

        subset180 = frame[frame.publish_date > target - timedelta(days=180)]
        theme_max = subset180.groupby("risk_theme")[score].max() if len(subset180) else {}
        theme_count = subset180.groupby("risk_theme").size() if len(subset180) else {}
        for theme in self.themes:
            record[f"th_{theme}_max"] = float(theme_max.get(theme, 0.0))
            record[f"th_{theme}_cnt"] = int(theme_count.get(theme, 0))

        high = frame[frame.is_high]
        record["days_since_high"] = (
            int((target - high.publish_date.max()).days) if len(high) else 9999
        )
        record["days_since_ann"] = (
            int((target - frame.publish_date.max()).days) if len(frame) else 9999
        )
        current90 = frame[frame.publish_date > target - timedelta(days=90)][score].sum()
        prior90 = frame[
            (frame.publish_date > target - timedelta(days=180))
            & (frame.publish_date <= target - timedelta(days=90))
        ][score].sum()
        record["trend_90_delta"] = float(current90 - prior90)

        financial_values = financial_values or {}
        for column in FINANCIAL_COLUMNS:
            record[column] = financial_values.get(column, np.nan)
        return {column: record.get(column, np.nan) for column in self.feature_columns}

    def transform_raw(self, raw_features: dict) -> dict:
        values = pd.DataFrame(
            [[raw_features.get(column, np.nan) for column in self.feature_columns]],
            columns=self.feature_columns,
            dtype="float64",
        )
        imputed = self.bundle["imputer"].transform(values)
        scaled = self.bundle["scaler"].transform(imputed)
        result = self.bundle["pca"].transform(scaled)[0]
        if len(result) != int(self.manifest["pca_components"]) or not np.isfinite(result).all():
            raise ValueError("F1 PCA50 输出维度错误或包含非有限值")
        return {
            f"announcement_semantic_{index:03d}": float(value)
            for index, value in enumerate(result)
        }

    def transform(self, announcement_rows, as_of, financial_values=None):
        raw = self.aggregate_raw(announcement_rows, as_of, financial_values)
        features = self.transform_raw(raw)
        return features, {
            "compatible": True,
            "source": "online_fullrun_announcement_risk_rows",
            "pipeline_version": self.manifest["pipeline_version"],
            "artifact_sha256": self.manifest["artifact_sha256"],
            "raw_feature_dim": len(raw),
            "pca_components": len(features),
            "as_of": str(as_of)[:10],
            "input_rows": len(announcement_rows),
        }
