#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: feature_composer —— 实时特征组装器（供 PredictorAgent 实时推理）
================================================================
把流水线实时产出的特征（公告研读 F1 标量 + 财务异常 F2-F6）按
models_manifest.json 的特征清单组装成模型输入向量；清单中实时拿不到的列
（如离线预计算的 F1 语义特征 announcement_semantic_*、governance_year）
用训练集(Train split)中位数填充——离线数据只做"初始建模/缺失兜底"，预测
主体由实时数据驱动。

数据流：
  ctx.semantic.f1_features（公告研读实时） ─┐
  ctx.financial.features（财务异常实时 F2-F6）─┼→ compose_realtime_features() → 模型向量
  fill_median_{w}d.csv（训练集中位数，兜底） ──┘

用法：
    from backend.skills.feature_composer import compose_realtime_features, load_fill_dict
    vec = compose_realtime_features(ctx, manifest_features, fill_dict)
"""
import math
from pathlib import Path

import pandas as pd

FILL_DIR = Path(__file__).resolve().parent.parent / "data" / "modeling" / "fill"

_fill_cache = {}


def load_fill_dict(window: str, fill_dir=FILL_DIR, expected_features: list = None) -> dict:
    """加载某窗口的填充字典（训练集中位数），懒加载 + 缓存。

    若传入 expected_features，会校验 fill 表是否包含全部期望特征，并打印警告。
    """
    if window not in _fill_cache:
        p = Path(fill_dir) / f"fill_median_{window}d.csv"
        if not p.exists():
            _fill_cache[window] = {}
        else:
            ser = pd.read_csv(p, index_col=0, encoding="utf-8-sig").iloc[:, 0]
            _fill_cache[window] = {str(k): v for k, v in ser.items()}

    fill = _fill_cache[window]
    if expected_features:
        import logging
        missing = [f for f in expected_features if f not in fill]
        extra = [f for f in fill if f not in expected_features]
        if missing:
            logging.getLogger(__name__).warning(
                "fill_median_%sd 缺少 %d 个 manifest 特征（示例：%s）",
                window, len(missing), missing[:5]
            )
        if extra:
            logging.getLogger(__name__).info(
                "fill_median_%sd 包含 %d 个不在 manifest 中的特征", window, len(extra)
            )
    return fill


def _clean(v):
    """把 numpy 标量/NaN/Inf 转成可入模型的值；无效返回 None。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compose_realtime_features(ctx, manifest_features, fill_dict=None) -> dict:
    """按 manifest 特征清单组装实时向量。

    优先级：实时值(财务 F2-F6 / 公告 F1 标量) → 训练集中位数 → 0.0。
    返回 dict{feature: float}，键与 manifest_features 一一对应。

    额外信息：ctx.features_real_time 写入 {feature: "realtime"|"filled"}，
    供审计页展示实时覆盖率。
    """
    real: dict = {}
    # ① 财务异常实时特征（F2/F3/F4/F5，列名已与训练表对齐；F6 由公告研读提供，见 ③）
    if getattr(ctx, "financial", None) and getattr(ctx.financial, "features", None):
        real.update(ctx.financial.features)
    # ② 与训练 manifest 精确同口径的 F1 特征。普通 scalar_features 不是
    # announcement_semantic_* 的等价替代，不能用补零/中位数伪装为实时值。
    if getattr(ctx, "semantic", None) and getattr(ctx.semantic, "f1_model_features", None):
        for k, v in ctx.semantic.f1_model_features.items():
            if k in manifest_features:
                real[k] = v
    # 兼容模型中确实直接使用公告标量的情况。
    if getattr(ctx, "semantic", None) and getattr(ctx.semantic, "f1_features", None):
        scalars = ctx.semantic.f1_features.get("scalar_features", {}) or {}
        for k, v in scalars.items():
            if k in manifest_features:
                real[k] = v
    # ③ 公告研读实时 F6 监管问询函特征（12 维 f6_*，覆盖财务侧同名键）
    if getattr(ctx, "semantic", None) and getattr(ctx.semantic, "f6_features", None):
        for k, v in ctx.semantic.f6_features.items():
            if k in manifest_features:
                real[k] = v

    fill = fill_dict or {}
    vec, origin = {}, {}
    for f in manifest_features:
        v = _clean(real.get(f))
        if v is None:
            # F1 语义特征（announcement_semantic_*）实时未生成时填 0.0，保证对模型无影响；
            # 其他特征仍优先用训练集中位数填充。
            if str(f).startswith("announcement_semantic_"):
                v = 0.0
                origin[f] = "filled_zero"
            else:
                v = _clean(fill.get(f))
                origin[f] = "filled" if v is not None else "filled_zero"
        else:
            origin[f] = "realtime"
        vec[f] = v if v is not None else 0.0

    try:
        ctx.features = dict(vec)                 # 实时数据文档：供建模/审计使用
        ctx.features_origin = origin
    except Exception:
        pass
    return vec


def coverage_stats(origin: dict) -> dict:
    """实时覆盖率统计：{realtime: n, filled: n, total: n, ratio: float}。"""
    n_real = sum(1 for v in origin.values() if v == "realtime")
    total = len(origin)
    return {
        "realtime": n_real,
        "filled": total - n_real,
        "total": total,
        "ratio": round(n_real / total, 4) if total else 0.0,
    }


def realtime_f1_compatibility(ctx, manifest: dict) -> tuple[bool, dict]:
    """检查训练模型需要的语义 F1 是否由实时流水线按同口径完整生成。"""
    required = set()
    for cfg in (manifest.get("windows", {}) or {}).values():
        for feature in cfg.get("features", []) or []:
            if str(feature).startswith("announcement_semantic_"):
                required.add(str(feature))
    provided_map = getattr(getattr(ctx, "semantic", None), "f1_model_features", {}) or {}
    provided = set(provided_map)
    missing = sorted(required - provided)
    audit = {
        "required": len(required),
        "provided": len(required & provided),
        "missing": len(missing),
        "missing_examples": missing[:10],
        "compatible": not missing,
    }
    return not missing, audit
