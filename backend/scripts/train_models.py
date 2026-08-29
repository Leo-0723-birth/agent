#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三窗口 × 三模型 预测建模：RandomForest + LightGBM + XGBoost（仓库内自包含版）
===================================================================
流程（每窗口 30/60/90d）：
  1. 有效样本（target>=0）+ Train/Validation/Test 切分（官方公司级 split）
  2. 特征筛选：零方差 → 高相关(>0.95) → LightGBM 重要性
  3. 样本平衡：SMOTE（正样本足够时）+ 类别权重
  4. 三模型训练（RF/LGB/XGB）+ 验证集 F1 最优阈值
  5. SHAP 贡献（xgb/lgb pred_contrib + rf TreeExplainer，集成加权）
  6. 输出到 backend/models/predictor/（模型 + models_manifest.json）
     + backend/data/modeling/fill/（fill_median_{w}d.csv）
     + backend/data/modeling/output/（预测概率/SHAP/风险排序/模型摘要）

运行（项目根目录）：python -m backend.scripts.train_models
数据：backend/data/modeling/processed_dataset.csv（build_modeling_dataset.py 产出）
"""
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    FEATURE_CORR_SAMPLE_SIZE,
    FEATURE_CORR_THRESHOLD,
    FEATURE_FAMILY_PREFIXES,
    FEATURE_FILTER_MIN_FEATURES,
    FEATURE_IMPORTANCE_THRESHOLD,
    FEATURE_VARIANCE_THRESHOLD,
    TARGET_INQUIRY_KIND,
    TEST_SPLIT_NAMES,
    TRAIN_SPLIT_NAMES,
    VALIDATION_SPLIT_NAMES,
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent.parent          # 项目根目录
DATA = BASE / "backend" / "data" / "modeling" / "processed_dataset.csv"
RAW_DIR = BASE / "backend" / "data" / "modeling" / "raw"
MODEL_DIR = BASE / "backend" / "models" / "predictor"
OUT_DIR = BASE / "backend" / "data" / "modeling" / "output"
FILL_DIR = BASE / "backend" / "data" / "modeling" / "fill"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILL_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [30, 60, 90]
ENSEMBLE_W = {"rf": 0.3, "lgb": 0.35, "xgb": 0.35}  # 集成权重

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, precision_recall_curve)

print("=" * 70)
print("三窗口 × 三模型 预测建模（RF/LGB/XGB）")
print(f"数据: {DATA}")
print(f"输出: {MODEL_DIR}")
print("=" * 70)

df = pd.read_csv(DATA, encoding="utf-8-sig")


def _normalize_split(s):
    """把 split 列归一化为 Train/Validation/Test，支持大小写混用。"""
    if pd.isna(s):
        return s
    s = str(s).strip()
    if s in TRAIN_SPLIT_NAMES:
        return "Train"
    if s in VALIDATION_SPLIT_NAMES:
        return "Validation"
    if s in TEST_SPLIT_NAMES:
        return "Test"
    return s


df["split"] = df["split"].apply(_normalize_split)

# 白名单特征：只保留 FEATURE_FAMILY_PREFIXES 前缀的数值列
feature_cols = [
    c for c in df.columns
    if any(c.startswith(p) for p in FEATURE_FAMILY_PREFIXES)
    and pd.api.types.is_numeric_dtype(df[c])
]
print(f"特征数（白名单）: {len(feature_cols)}")
print(f"  家族分布: {', '.join(f'{p}:{sum(c.startswith(p) for c in feature_cols)}' for p in FEATURE_FAMILY_PREFIXES)}")

summary = {}
manifest = {"ensemble_weights": ENSEMBLE_W, "windows": {}}
for w in WINDOWS:
    tcol = f"target_{w}d"
    print(f"\n{'=' * 60}\n 窗口: {w}d\n{'=' * 60}")
    t0 = time.time()

    # ---------- 1) 有效样本与切分 ----------
    valid = df[tcol] >= 0
    dfw = df[valid].copy()
    y = dfw[tcol].astype(int).values
    X = dfw[feature_cols].values.astype(np.float32)
    splits = dfw["split"].values

    tr, va, te = (splits == "Train"), (splits == "Validation"), (splits == "Test")
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    pos_rate = y_tr.mean()
    print(f"Train {len(y_tr)} (pos {y_tr.sum()} = {pos_rate * 100:.2f}%) | "
          f"Val {len(y_va)} | Test {len(y_te)}")

    # ---------- 2) 特征筛选 ----------
    # 2a) 零/近零方差
    var = np.var(X_tr, axis=0)
    m1 = var >= FEATURE_VARIANCE_THRESHOLD
    X_tr, X_va, X_te = X_tr[:, m1], X_va[:, m1], X_te[:, m1]
    feats = [feature_cols[i] for i in range(len(feature_cols)) if m1[i]]
    print(f"  方差筛选: {len(feature_cols)} -> {len(feats)}")

    # 2b) 高相关剔除
    if len(feats) > FEATURE_FILTER_MIN_FEATURES:
        sn = min(FEATURE_CORR_SAMPLE_SIZE, X_tr.shape[0])
        si = np.random.choice(X_tr.shape[0], sn, replace=False)
        cm = np.abs(np.corrcoef(X_tr[si], rowvar=False))
        ut = np.triu(np.ones(cm.shape), k=1).astype(bool)
        hc = np.where((cm > FEATURE_CORR_THRESHOLD) & ut)
        drop = set(hc[1])
        keep = [i for i in range(X_tr.shape[1]) if i not in drop]
        X_tr, X_va, X_te = X_tr[:, keep], X_va[:, keep], X_te[:, keep]
        feats = [feats[i] for i in keep]
        print(f"  相关筛选: 剔除高相关 {len(drop)} 维，剩余 {len(feats)}")

    # 2c) LightGBM 重要性筛选
    pw = (1 - pos_rate) / max(pos_rate, 1e-6)
    scr = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             scale_pos_weight=pw, random_state=SEED, verbose=-1, n_jobs=-1)
    scr.fit(X_tr, y_tr)
    m2 = scr.feature_importances_ > FEATURE_IMPORTANCE_THRESHOLD
    X_tr, X_va, X_te = X_tr[:, m2], X_va[:, m2], X_te[:, m2]
    feats = [feats[i] for i in range(len(feats)) if m2[i]]
    print(f"特征筛选后: {len(feats)} 维")

    # ---------- 3) 样本加权（无 SMOTE）：类别平衡 → scale_pos_weight；问询强度 → sample_weight ----------
    # 用户确认配置 B：w = 1 + ln(n_inq_60d)（正样本按强度，负样本=1.0）
    _n = dfw["n_inq_60d"].to_numpy(dtype=float)
    w_all = np.where(y == 1, 1.0 + np.log(np.maximum(_n, 1.0)), 1.0)
    X_tr_s, y_tr_s, w_tr_s = X_tr, y_tr, w_all[tr]
    lgb_pw = pw          # 类别平衡（LGB）
    xgb_pw = pw * 5.0    # 类别平衡（XGB，更强）
    w_mean = float(w_all[tr][y_tr == 1].mean())
    print(f"sample_weight: 正样本均值 {w_mean:.3f}（1+ln(n)）| 负样本恒 1.0 | 无 SMOTE")

    # ---------- 4) 三模型训练 ----------
    def threshold_from_val(pv, yv):
        pr, rc, th = precision_recall_curve(yv, pv)
        f1s = 2 * pr * rc / (pr + rc + 1e-12)
        return th[np.argmax(f1s)] if len(th) else 0.5

    models, probs_va, probs_te = {}, {}, {}

    rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_s, y_tr_s, sample_weight=w_tr_s)
    models["rf"] = rf
    probs_va["rf"] = rf.predict_proba(X_va)[:, 1]
    probs_te["rf"] = rf.predict_proba(X_te)[:, 1]
    print(f"  RF 完成 ({time.time() - t0:.0f}s)")

    lgbm = lgb.LGBMClassifier(n_estimators=800, max_depth=7, learning_rate=0.03, num_leaves=63,
                              scale_pos_weight=lgb_pw, subsample=0.75, colsample_bytree=0.75,
                              subsample_freq=1, reg_alpha=0.1, reg_lambda=0.1,
                              min_child_samples=20, random_state=SEED, verbose=-1, n_jobs=-1)
    # LightGBM 必须用带真实列名的 DataFrame 训练，否则模型文本只保存
    # Column_0...，与 manifest 的特征清单脱节，线上无法做严格口径校验。
    X_tr_lgb = pd.DataFrame(X_tr_s, columns=feats)
    X_va_lgb = pd.DataFrame(X_va, columns=feats)
    X_te_lgb = pd.DataFrame(X_te, columns=feats)
    lgbm.fit(X_tr_lgb, y_tr_s, sample_weight=w_tr_s,
             eval_set=[(X_va_lgb, y_va)], eval_metric="auc")
    models["lgb"] = lgbm
    probs_va["lgb"] = lgbm.predict_proba(X_va_lgb)[:, 1]
    probs_te["lgb"] = lgbm.predict_proba(X_te_lgb)[:, 1]
    print(f"  LightGBM 完成 ({time.time() - t0:.0f}s)")

    xgbm = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03,
                             scale_pos_weight=xgb_pw, subsample=0.75, colsample_bytree=0.75,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                             eval_metric="auc", verbosity=0)
    X_tr_xgb = pd.DataFrame(X_tr_s, columns=feats)
    X_va_xgb = pd.DataFrame(X_va, columns=feats)
    X_te_xgb = pd.DataFrame(X_te, columns=feats)
    xgbm.fit(X_tr_xgb, y_tr_s, sample_weight=w_tr_s,
             eval_set=[(X_va_xgb, y_va)], verbose=False)
    models["xgb"] = xgbm
    probs_va["xgb"] = xgbm.predict_proba(X_va_xgb)[:, 1]
    probs_te["xgb"] = xgbm.predict_proba(X_te_xgb)[:, 1]
    print(f"  XGBoost 完成 ({time.time() - t0:.0f}s)")

    ens_va = sum(ENSEMBLE_W[m] * probs_va[m] for m in ENSEMBLE_W)
    ens_te = sum(ENSEMBLE_W[m] * probs_te[m] for m in ENSEMBLE_W)

    # ---------- 4.5) 概率校准（isotonic，验证集拟合；ensemble 校准后重选阈值） ----------
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(ens_va, y_va)
    ens_va_c = cal.predict(ens_va)
    ens_te_c = cal.predict(ens_te)
    joblib.dump(cal, MODEL_DIR / ("calibrator_%dd.joblib" % w))
    bt = threshold_from_val(ens_va_c, y_va)
    top10_n = max(1, int(len(y_te) * 0.1))

    def metrics(p):
        yc = (p >= bt).astype(int)
        return {"AUC": roc_auc_score(y_te, p), "AP": average_precision_score(y_te, p),
                "F1": f1_score(y_te, yc, zero_division=0),
                "Precision": precision_score(y_te, yc, zero_division=0),
                "Recall": recall_score(y_te, yc, zero_division=0),
                "Top10%Recall": y_te[np.argsort(p)[-top10_n:]].sum() / max(y_te.sum(), 1)}

    results = {m: metrics(probs_te[m]) for m in ENSEMBLE_W}
    results["Ensemble_raw"] = metrics(ens_te)
    results["Ensemble_calibrated"] = metrics(ens_te_c)
    results["Ensemble_calibrated"]["threshold"] = float(bt)
    print("  calibration: raw F1=%.4f -> cal F1=%.4f (thr=%.3f)" % (
        results["Ensemble_raw"]["F1"], results["Ensemble_calibrated"]["F1"], bt))
    for m, r in results.items():
        print("  %-20s: AUC=%.4f AP=%.4f F1=%.4f Top10%%=%.4f" % (
            m, r["AUC"], r["AP"], r["F1"], r["Top10%Recall"]))

    # ---------- 5) SHAP（集成加权贡献） ----------
    print("  SHAP 计算中...")
    shap_map = {}
    shap_map["xgb"] = models["xgb"].get_booster().predict(
        xgb.DMatrix(X_te, feature_names=feats), pred_contribs=True)[:, :-1]
    shap_map["lgb"] = models["lgb"].booster_.predict(
        X_te_lgb, pred_contrib=True)[:, :-1]
    try:
        import shap as shap_lib
        rf_explainer = shap_lib.TreeExplainer(models["rf"])
        rf_sv = rf_explainer.shap_values(X_te)
        if isinstance(rf_sv, list):
            shap_map["rf"] = rf_sv[1] if len(rf_sv) == 2 else rf_sv[0]
        elif rf_sv.ndim == 3:
            shap_map["rf"] = rf_sv[..., 1]
        else:
            shap_map["rf"] = rf_sv
        shap_weights = ENSEMBLE_W
        shap_backend = "rf_treeexplainer+lgb_pred_contrib+xgb_pred_contrib"
    except ImportError:
        # SHAP Python 包是可选依赖；LGB/XGB 都能原生输出 TreeSHAP。
        # 缺包时只对可核验的两个原生贡献重新归一化，不能伪造 RF 贡献。
        shap_weights = {
            "lgb": ENSEMBLE_W["lgb"] / (ENSEMBLE_W["lgb"] + ENSEMBLE_W["xgb"]),
            "xgb": ENSEMBLE_W["xgb"] / (ENSEMBLE_W["lgb"] + ENSEMBLE_W["xgb"]),
        }
        shap_backend = "lgb_pred_contrib+xgb_pred_contrib (RF omitted: shap not installed)"
        print("  [SHAP降级] 未安装 shap，使用 LightGBM + XGBoost 原生贡献")

    shap_ens = sum(shap_weights[m] * shap_map[m] for m in shap_weights)
    top5 = np.argsort(-np.abs(shap_ens), axis=1)[:, :5]
    shap_rows = []
    for i in range(len(y_te)):
        row = {"company_code": dfw[te]["company_code"].iloc[i],
               "report_period": dfw[te]["report_period"].iloc[i],
               "actual": int(y_te[i]), "prob": float(ens_te[i])}
        for k in range(5):
            fi = top5[i, k]
            row[f"SHAP_top{k+1}_feature"] = feats[fi]
            row[f"SHAP_top{k+1}_direction"] = "positive_risk" if shap_ens[i, fi] > 0 else "negative_risk"
            row[f"SHAP_top{k+1}_value"] = round(float(shap_ens[i, fi]), 5)
        shap_rows.append(row)
    shap_df = pd.DataFrame(shap_rows)

    # ---------- 6) 保存（模型 / manifest / fill / 评估产物） ----------
    joblib.dump(models["rf"], MODEL_DIR / f"model_rf_{w}d.pkl")
    (MODEL_DIR / f"model_lgb_{w}d.txt").write_text(
        models["lgb"].booster_.model_to_string(), encoding="utf-8")
    (MODEL_DIR / f"model_xgb_{w}d.json").write_bytes(
        models["xgb"].get_booster().save_raw(raw_format="json"))
    manifest["windows"][str(w)] = {
        "features": feats,
        "shap_backend": shap_backend,
        "metrics": {k: round(v, 6) for k, v in results["Ensemble_calibrated"].items()
                    if isinstance(v, (int, float))},
    }

    pd.DataFrame({"feature": feats, "importance_ensemble": np.mean(
        [models["rf"].feature_importances_, models["lgb"].feature_importances_,
         models["xgb"].feature_importances_], axis=0)}
    ).sort_values("importance_ensemble", ascending=False).to_csv(
        MODEL_DIR / f"feature_importance_{w}d.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"company_code": dfw[te]["company_code"].values,
                  "report_period": dfw[te]["report_period"].values,
                  "actual": y_te,
                  **{f"prob_{m}": probs_te[m] for m in ENSEMBLE_W},
                  "prob_ensemble": ens_te,
                  "prob_ensemble_calibrated": ens_te_c}).to_csv(
        OUT_DIR / f"predictions_{w}d.csv", index=False, encoding="utf-8-sig")
    shap_df.to_csv(OUT_DIR / f"shap_{w}d.csv", index=False, encoding="utf-8-sig")
    risk = pd.DataFrame({"company_code": dfw[te]["company_code"].values,
                         "report_period": dfw[te]["report_period"].values,
                         "risk_probability": ens_te_c})
    risk["risk_rank"] = risk["risk_probability"].rank(ascending=False, method="first").astype(int)
    risk = risk.sort_values("risk_rank")
    risk["risk_level"] = pd.cut(risk["risk_rank"], bins=[0, len(risk) * 0.05, len(risk) * 0.15,
                              len(risk) * 0.3, len(risk) * 0.5, len(risk) * 0.7, len(risk)],
                              labels=["A-极高", "B-高", "C-中高", "D-中", "E-较低", "F-低"])
    risk.to_csv(OUT_DIR / f"risk_rank_{w}d.csv", index=False, encoding="utf-8-sig")

    # fill 字典：Train 有效样本的中位数（实时推理缺失列兜底）
    fill = dfw.loc[tr, feats].median(numeric_only=True)
    fill = fill.reindex(feats)
    fill.to_csv(FILL_DIR / f"fill_median_{w}d.csv", encoding="utf-8-sig", header=["0"])
    print(f"  [输出] 模型→{MODEL_DIR} | fill→{FILL_DIR} | 评估→{OUT_DIR}")

    summary[str(w)] = results
    del models, X_tr_s, y_tr_s
    gc.collect()

# ---------- 汇总 ----------
# 计算训练数据指纹（特征列 + 有效样本数）用于版本绑定
def _data_fingerprint(df, feature_cols, windows):
    import hashlib
    h = hashlib.sha256()
    h.update(",".join(sorted(feature_cols)).encode("utf-8"))
    for w in windows:
        tcol = f"target_{w}d"
        valid = df[tcol] >= 0
        h.update(f"{w}:{valid.sum()}:{df.loc[valid, tcol].sum()}".encode("utf-8"))
    return h.hexdigest()[:16]


generated_at = pd.Timestamp.now().isoformat()
data_hash = _data_fingerprint(df, feature_cols, WINDOWS)
f1_import_manifest_path = RAW_DIR / "f1_full_run_manifest.json"
try:
    f1_import_manifest = json.loads(f1_import_manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    f1_import_manifest = {}
manifest["metadata"] = {
    "generated_at": generated_at,
    "data_hash": data_hash,
    "feature_count": len(feature_cols),
    "feature_families": [p.rstrip("_") for p in FEATURE_FAMILY_PREFIXES],
    "f1_source": {
        "kind": "full_run_pca50",
        "artifact": "backend/data/modeling/raw/F1_announcement_semantic_features.parquet",
        "artifact_sha256": f1_import_manifest.get("output_sha256"),
        "source_sha256": f1_import_manifest.get("source_sha256"),
        "components": f1_import_manifest.get("pca", {}).get("components", 50),
    },
    "target_kind": TARGET_INQUIRY_KIND,
    "data_path": "backend/data/modeling/processed_dataset.csv",
}
(MODEL_DIR / "models_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
summary_json = {"generated_at": generated_at,
                "data_hash": data_hash,
                "models": ["RandomForest", "LightGBM", "XGBoost"],
                "ensemble_weights": ENSEMBLE_W,
                "windows": {str(w): {m: {k: round(v, 4) for k, v in r.items()}
                                     for m, r in summary[str(w)].items()}
                            for w in WINDOWS}}
(OUT_DIR / "model_summary.json").write_text(
    json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n" + "=" * 70)
print("全部完成")
print(f"  manifest: {MODEL_DIR / 'models_manifest.json'}")
print(f"  摘要:     {OUT_DIR / 'model_summary.json'}")
print("=" * 70)
