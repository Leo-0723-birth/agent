#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三窗口 × 三模型 预测建模：RandomForest + LightGBM + XGBoost
===========================================================
流程（每窗口 30/60/90d）：
  1. 有效样本（target>=0）+ Train/Validation/Test 切分
  2. 特征筛选：零方差 → 高相关(>0.95) → LightGBM 重要性
  3. 样本平衡：SMOTE（正样本足够时）+ 类别权重
  4. 三模型训练（RF/LGB/XGB）+ 验证集 F1 最优阈值
  5. SHAP 贡献（xgb/lgb pred_contrib + rf TreeExplainer，集成加权）
  6. 输出：模型文件 / 预测概率 / SHAP / 公司风险排序

输出：05_模型输出/{30d,60d,90d}/
"""
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

OUT = Path(r"C:\Users\86130\Desktop\预测建模agent\05_模型输出")
DATA = OUT / "processed_dataset.csv"
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
print("=" * 70)

df = pd.read_csv(DATA, encoding="utf-8-sig")
ID_COLS = ["company_code", "report_period", "industry", "split"]
feature_cols = [c for c in df.columns
                if c not in ID_COLS and not c.startswith("target_") and not c.startswith("future_")]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
print(f"特征数: {len(feature_cols)}")

summary = {}
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
    var = np.var(X_tr, axis=0)
    m1 = var >= 1e-12
    X_tr, X_va, X_te = X_tr[:, m1], X_va[:, m1], X_te[:, m1]
    feats = [feature_cols[i] for i in range(len(feature_cols)) if m1[i]]

    if X_tr.shape[1] > 50:
        sn = min(5000, X_tr.shape[0])
        si = np.random.choice(X_tr.shape[0], sn, replace=False)
        cm = np.abs(np.corrcoef(X_tr[si], rowvar=False))
        ut = np.triu(np.ones(cm.shape), k=1).astype(bool)
        hc = np.where((cm > 0.95) & ut)
        drop = set(hc[1])
        keep = [i for i in range(X_tr.shape[1]) if i not in drop]
        X_tr, X_va, X_te = X_tr[:, keep], X_va[:, keep], X_te[:, keep]
        feats = [feats[i] for i in keep]

    pw = (1 - pos_rate) / max(pos_rate, 1e-6)
    scr = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             scale_pos_weight=pw, random_state=SEED, verbose=-1, n_jobs=-1)
    scr.fit(X_tr, y_tr)
    m2 = scr.feature_importances_ > 0
    X_tr, X_va, X_te = X_tr[:, m2], X_va[:, m2], X_te[:, m2]
    feats = [feats[i] for i in range(len(feats)) if m2[i]]
    print(f"特征筛选后: {len(feats)} 维")

    # ---------- 3) 样本平衡（SMOTE + 权重） ----------
    from imblearn.over_sampling import SMOTE
    use_smote = False
    if y_tr.sum() >= 5:
        target_ratio = min(0.15, pos_rate * 3.0)
        if target_ratio > pos_rate * 1.5:
            sm = SMOTE(sampling_strategy=target_ratio, random_state=SEED, k_neighbors=min(5, y_tr.sum() - 1))
            X_tr_s, y_tr_s = sm.fit_resample(X_tr, y_tr)
            use_smote = True
            print(f"SMOTE: {len(y_tr)} → {len(y_tr_s)}（正样本占比 {y_tr_s.mean() * 100:.1f}%）")
        else:
            X_tr_s, y_tr_s = X_tr, y_tr
    else:
        X_tr_s, y_tr_s = X_tr, y_tr
    lgb_pw = None if use_smote else pw
    xgb_pw = None if use_smote else pw * 5.0

    # ---------- 4) 三模型训练 ----------
    def threshold_from_val(pv, yv):
        pr, rc, th = precision_recall_curve(yv, pv)
        f1s = 2 * pr * rc / (pr + rc + 1e-12)
        return th[np.argmax(f1s)] if len(th) else 0.5

    models, probs_va, probs_te = {}, {}, {}

    rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_s, y_tr_s)
    models["rf"] = rf
    probs_va["rf"] = rf.predict_proba(X_va)[:, 1]
    probs_te["rf"] = rf.predict_proba(X_te)[:, 1]
    print(f"  RF 完成 ({time.time() - t0:.0f}s)")

    lgbm = lgb.LGBMClassifier(n_estimators=800, max_depth=7, learning_rate=0.03, num_leaves=63,
                              scale_pos_weight=lgb_pw, subsample=0.75, colsample_bytree=0.75,
                              subsample_freq=1, reg_alpha=0.1, reg_lambda=0.1,
                              min_child_samples=20, random_state=SEED, verbose=-1, n_jobs=-1)
    lgbm.fit(X_tr_s, y_tr_s, eval_set=[(X_va, y_va)], eval_metric="auc")
    models["lgb"] = lgbm
    probs_va["lgb"] = lgbm.predict_proba(X_va)[:, 1]
    probs_te["lgb"] = lgbm.predict_proba(X_te)[:, 1]
    print(f"  LightGBM 完成 ({time.time() - t0:.0f}s)")

    xgbm = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03,
                             scale_pos_weight=xgb_pw, subsample=0.75, colsample_bytree=0.75,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                             eval_metric="auc", verbosity=0)
    xgbm.fit(X_tr_s, y_tr_s, eval_set=[(X_va, y_va)], verbose=False)
    models["xgb"] = xgbm
    probs_va["xgb"] = xgbm.predict_proba(X_va)[:, 1]
    probs_te["xgb"] = xgbm.predict_proba(X_te)[:, 1]
    print(f"  XGBoost 完成 ({time.time() - t0:.0f}s)")

    ens_va = sum(ENSEMBLE_W[m] * probs_va[m] for m in ENSEMBLE_W)
    ens_te = sum(ENSEMBLE_W[m] * probs_te[m] for m in ENSEMBLE_W)
    bt = threshold_from_val(ens_va, y_va)
    top10_n = max(1, int(len(y_te) * 0.1))

    def metrics(p):
        yc = (p >= bt).astype(int)
        return {"AUC": roc_auc_score(y_te, p), "AP": average_precision_score(y_te, p),
                "F1": f1_score(y_te, yc, zero_division=0),
                "Precision": precision_score(y_te, yc, zero_division=0),
                "Recall": recall_score(y_te, yc, zero_division=0),
                "Top10%Recall": y_te[np.argsort(p)[-top10_n:]].sum() / max(y_te.sum(), 1)}

    results = {m: metrics(probs_te[m]) for m in ENSEMBLE_W}
    results["Ensemble"] = metrics(ens_te)
    results["Ensemble"]["threshold"] = float(bt)
    for m, r in results.items():
        print(f"  {m:8s}: AUC={r['AUC']:.4f} AP={r['AP']:.4f} F1={r['F1']:.4f} "
              f"Top10%={r['Top10%Recall']:.4f}")

    # ---------- 5) SHAP（集成加权贡献） ----------
    print("  SHAP 计算中...")
    shap_map = {}
    shap_map["xgb"] = models["xgb"].get_booster().predict(
        xgb.DMatrix(X_te, feature_names=feats), pred_contribs=True)[:, :-1]
    shap_map["lgb"] = models["lgb"].booster_.predict(
        X_te, pred_contrib=True)[:, :-1]
    import shap as shap_lib
    rf_explainer = shap_lib.TreeExplainer(models["rf"])
    rf_sv = rf_explainer.shap_values(X_te)
    if isinstance(rf_sv, list):
        # 旧版 shap：binary 返回 [class0, class1]
        shap_map["rf"] = rf_sv[1] if len(rf_sv) == 2 else rf_sv[0]
    elif rf_sv.ndim == 3:
        # 新版 shap：返回 (n, features, n_classes)，取正类切片
        shap_map["rf"] = rf_sv[..., 1]
    else:
        shap_map["rf"] = rf_sv

    shap_ens = sum(ENSEMBLE_W[m] * shap_map[m] for m in ENSEMBLE_W)
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

    # ---------- 6) 保存 ----------
    wdir = OUT / f"{w}d"
    wdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["rf"], wdir / f"model_rf_{w}d.pkl")
    # 中文路径下 LightGBM/XGBoost 的 C++ 保存层不可写 → 用 Python 写字符串/字节
    (wdir / f"model_lgb_{w}d.txt").write_text(
        models["lgb"].booster_.model_to_string(), encoding="utf-8")
    (wdir / f"model_xgb_{w}d.json").write_bytes(
        models["xgb"].get_booster().save_raw(raw_format="json"))
    pd.DataFrame({"feature": feats, "importance_ensemble": np.mean(
        [models["rf"].feature_importances_, models["lgb"].feature_importances_,
         models["xgb"].feature_importances_], axis=0)}
    ).sort_values("importance_ensemble", ascending=False).to_csv(
        wdir / f"feature_importance_{w}d.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"company_code": dfw[te]["company_code"].values,
                  "report_period": dfw[te]["report_period"].values,
                  "actual": y_te,
                  **{f"prob_{m}": probs_te[m] for m in ENSEMBLE_W},
                  "prob_ensemble": ens_te}).to_csv(
        wdir / f"predictions_{w}d.csv", index=False, encoding="utf-8-sig")
    shap_df.to_csv(wdir / f"shap_{w}d.csv", index=False, encoding="utf-8-sig")
    risk = pd.DataFrame({"company_code": dfw[te]["company_code"].values,
                         "report_period": dfw[te]["report_period"].values,
                         "risk_probability": ens_te})
    risk["risk_rank"] = risk["risk_probability"].rank(ascending=False, method="first").astype(int)
    risk = risk.sort_values("risk_rank")
    risk["risk_level"] = pd.cut(risk["risk_rank"], bins=[0, len(risk) * 0.05, len(risk) * 0.15,
                              len(risk) * 0.3, len(risk) * 0.5, len(risk) * 0.7, len(risk)],
                              labels=["A-极高", "B-高", "C-中高", "D-中", "E-较低", "F-低"])
    risk.to_csv(wdir / f"risk_rank_{w}d.csv", index=False, encoding="utf-8-sig")
    print(f"  [输出] {wdir}/（模型×3 + 预测 + SHAP + 风险排序）")

    summary[str(w)] = results
    del models, X_tr_s, y_tr_s
    gc.collect()

# ---------- 汇总 ----------
summary_json = {"generated_at": pd.Timestamp.now().isoformat(),
                "models": ["RandomForest", "LightGBM", "XGBoost"],
                "ensemble_weights": ENSEMBLE_W,
                "n_features": len(feature_cols),
                "windows": {str(w): {m: {k: round(v, 4) for k, v in r.items()}
                                     for m, r in summary[str(w)].items()}
                            for w in WINDOWS}}
(OUT / "model_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
print("\n" + "=" * 70)
print("全部完成，输出目录:", OUT)
print("=" * 70)
