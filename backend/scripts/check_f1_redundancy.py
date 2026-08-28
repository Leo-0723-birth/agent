# -*- coding: utf-8 -*-
r"""
F1(197维公告语义) 与 F2-F6 的重复性检查

回答两个问题：
  1. F1 内部有没有互相冗余的维度（同一信息的多种写法）
  2. F1 与 F2-F6 之间有没有高度重复的特征（跨家族撞车）

方法：
  - 在 Train 集上算 Spearman 相关（对非正态、含大量 0 的特征更稳健）
  - |rho| >= 0.95 判为"接近重复"，0.85~0.95 判为"高度相似"
  - 输出去重建议清单，并给出可直接用于 build 脚本的排除列表

用法：
  python check_f1_redundancy.py --repo <agent-team 路径>
  python check_f1_redundancy.py --repo <路径> --threshold 0.92
"""
import io
import sys
import json
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

DUP_TH = 0.95      # 接近重复
SIM_TH = 0.85      # 高度相似


def log(s=""):
    print(s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--threshold", type=float, default=DUP_TH)
    ap.add_argument("--sample", type=int, default=8000, help="相关性计算抽样行数")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    M = Path(args.repo) / "backend" / "data" / "modeling"
    RAW = M / "raw"

    log("=" * 74)
    log("  F1(公告语义) 与 F2-F6 重复性检查")
    log(f"  接近重复阈值 |rho| >= {args.threshold}   高度相似 >= {SIM_TH}")
    log("=" * 74)

    # ---------- 载入 F1（全部 197 维，不是 Top-100） ----------
    f1 = pd.read_parquet(RAW / "F1_announcement_semantic_features.parquet")
    f1 = f1.rename(columns={"stock_code": "company_code", "T_date": "report_period"})
    f1["company_code"] = f1["company_code"].astype(str)
    f1["report_period"] = pd.to_datetime(f1["report_period"],
                                         errors="coerce").dt.strftime("%Y%m%d")
    f1_cols = [c for c in f1.columns if c.startswith("announcement_semantic_")]
    log(f"\nF1: {len(f1):,} 行 × {len(f1_cols)} 维")

    # 列名 -> 业务含义
    mp_path = RAW / "f1_selection" / "F1_semantic_column_mapping.csv"
    if mp_path.exists():
        mp = pd.read_csv(mp_path, encoding="utf-8-sig")
        name = dict(zip(mp.semantic_column, mp.source_feature))
        desc = dict(zip(mp.semantic_column, mp.description))
    else:
        name, desc = {}, {}
        log("  [警告] 找不到列名对照表，输出将只有 semantic 编号")

    def label(c):
        return f"{c}({name.get(c, '?')})" if c in name else c

    # ---------- 载入 F2-F6 ----------
    def norm(d):
        d = d.copy()
        d["company_code"] = d["company_code"].astype(str)
        d["report_period"] = d["report_period"].astype(str).str.replace(".0", "", regex=False)
        return d

    fam = {}
    for tag, fn in [("F2", "F2_financial_anomaly.csv"), ("F3", "F3_market_features.csv"),
                    ("F4", "F4_sentiment_features.csv"), ("F5", "F5_ownership_governance.csv"),
                    ("F6", "F6_inquiry_history.csv")]:
        d = norm(pd.read_csv(RAW / fn, encoding="utf-8-sig"))
        cols = [c for c in d.columns
                if c not in ("company_code", "report_period", "split")
                and pd.api.types.is_numeric_dtype(d[c])]
        fam[tag] = (d, cols)
        log(f"{tag}: {len(d):,} 行 × {len(cols)} 维")

    # ---------- 合并到同一张表 ----------
    split = fam["F2"][0][["company_code", "report_period", "split"]]
    big = split.merge(f1[["company_code", "report_period"] + f1_cols],
                      on=["company_code", "report_period"], how="inner")
    owner = {c: "F1" for c in f1_cols}
    for tag, (d, cols) in fam.items():
        big = big.merge(d[["company_code", "report_period"] + cols],
                        on=["company_code", "report_period"], how="left")
        for c in cols:
            owner[c] = tag

    tr = big[big.split == "Train"]
    if len(tr) > args.sample:
        tr = tr.sample(args.sample, random_state=42)
    all_cols = [c for c in big.columns if c in owner]
    log(f"\n合并后 {len(big):,} 行 × {len(all_cols)} 个特征   "
        f"相关性用 Train {len(tr):,} 行计算")

    # ---------- Spearman 相关 ----------
    log("\n计算 Spearman 相关矩阵 ...")
    X = tr[all_cols].astype(float)
    X = X.loc[:, X.nunique() > 1]          # 常量列无法算相关
    const = [c for c in all_cols if c not in X.columns]
    if const:
        log(f"  常量列 {len(const)} 个（无方差，建议直接剔除）")
        for c in const[:10]:
            log(f"    {owner[c]}  {label(c)}")
    R = X.rank().corr(method="pearson").abs().values      # rank+pearson == spearman，但快得多
    cols = list(X.columns)
    n = len(cols)

    # ---------- 找高相关对 ----------
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = R[i, j]
            if r >= SIM_TH:
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda x: -x[2])

    dup = [p for p in pairs if p[2] >= args.threshold]
    sim = [p for p in pairs if SIM_TH <= p[2] < args.threshold]

    def kind(a, b):
        oa, ob = owner[a], owner[b]
        return "F1内部" if oa == ob == "F1" else (f"F1↔{ob}" if oa == "F1" else
                                                 (f"F1↔{oa}" if ob == "F1" else f"{oa}↔{ob}"))

    log("\n" + "=" * 74)
    log(f"接近重复 (|rho| >= {args.threshold}): {len(dup)} 对")
    log("=" * 74)
    from collections import Counter
    log("按类型: " + str(dict(Counter(kind(a, b) for a, b, _ in dup))))
    log("")
    for a, b, r in dup[:40]:
        log(f"  {r:.4f}  [{kind(a,b):9s}]")
        log(f"          {owner[a]:3s} {label(a)}")
        log(f"          {owner[b]:3s} {label(b)}")
    if len(dup) > 40:
        log(f"  ... 另有 {len(dup)-40} 对，见 CSV")

    log("\n" + "=" * 74)
    log(f"高度相似 ({SIM_TH} <= |rho| < {args.threshold}): {len(sim)} 对")
    log("=" * 74)
    log("按类型: " + str(dict(Counter(kind(a, b) for a, b, _ in sim))))
    for a, b, r in sim[:15]:
        log(f"  {r:.4f}  [{kind(a,b):9s}]  {owner[a]} {label(a)}  <->  {owner[b]} {label(b)}")

    # ---------- 去重建议：跨家族冲突优先保留 F2-F6 ----------
    # 理由：F2-F6 是既有特征体系，F1 是新增，撞车时保留原有的更稳妥
    drop = set(const)
    keep_reason = {}
    for a, b, r in dup:
        if a in drop or b in drop:
            continue
        oa, ob = owner[a], owner[b]
        if oa == "F1" and ob != "F1":
            drop.add(a); keep_reason[a] = f"与 {ob} 的 {b} 重复 (rho={r:.3f})"
        elif ob == "F1" and oa != "F1":
            drop.add(b); keep_reason[b] = f"与 {oa} 的 {a} 重复 (rho={r:.3f})"
        elif oa == "F1" and ob == "F1":
            # F1 内部：丢掉编号靠后的
            x = b if b > a else a
            y = a if x == b else b
            drop.add(x); keep_reason[x] = f"与 F1 的 {y} 重复 (rho={r:.3f})"

    f1_drop = sorted(c for c in drop if owner.get(c) == "F1")
    log("\n" + "=" * 74)
    log("去重建议")
    log("=" * 74)
    log(f"  建议从 F1 剔除 {len(f1_drop)} 维，保留 {len(f1_cols) - len(f1_drop)} 维")
    log(f"  （跨家族冲突时保留 F2-F6 原有特征；F1 内部冲突保留编号靠前的）\n")
    for c in f1_drop[:25]:
        log(f"    {label(c)}")
        log(f"      -> {keep_reason.get(c, '常量列')}")
    if len(f1_drop) > 25:
        log(f"    ... 另有 {len(f1_drop)-25} 维")

    # ---------- 落盘 ----------
    out = RAW / "f1_selection"
    pd.DataFrame([{
        "rho": r, "type": kind(a, b),
        "feature_a": a, "family_a": owner[a],
        "source_a": name.get(a, ""), "desc_a": desc.get(a, ""),
        "feature_b": b, "family_b": owner[b],
        "source_b": name.get(b, ""), "desc_b": desc.get(b, ""),
    } for a, b, r in pairs]).to_csv(out / "F1_redundancy_pairs.csv",
                                    index=False, encoding="utf-8-sig")

    pd.DataFrame([{"feature": c, "source_feature": name.get(c, ""),
                   "description": desc.get(c, ""),
                   "reason": keep_reason.get(c, "常量列（无方差）")}
                  for c in f1_drop]).to_csv(out / "F1_drop_suggestion.csv",
                                            index=False, encoding="utf-8-sig")

    with open(out / "F1_redundancy_meta.json", "w", encoding="utf-8") as f:
        json.dump({"threshold_dup": args.threshold, "threshold_sim": SIM_TH,
                   "f1_dims": len(f1_cols), "total_features": len(all_cols),
                   "const_cols": len(const),
                   "pairs_dup": len(dup), "pairs_sim": len(sim),
                   "f1_drop_count": len(f1_drop),
                   "f1_keep_count": len(f1_cols) - len(f1_drop),
                   "drop_list": f1_drop,
                   "note": "跨家族冲突保留 F2-F6；F1 内部冲突保留编号靠前"},
                  f, ensure_ascii=False, indent=2)

    log(f"\n  明细: {out / 'F1_redundancy_pairs.csv'}")
    log(f"  建议: {out / 'F1_drop_suggestion.csv'}")
    log(f"  元信息: {out / 'F1_redundancy_meta.json'}")
    log("\n提示：train_models.py 已有 |corr|>0.95 自动去重，本检查用于")
    log("      建模前显式确认与出报告；如需在 build 阶段就排除，")
    log("      可把 drop_list 加入 build_modeling_dataset.py 的过滤。")


if __name__ == "__main__":
    main()
