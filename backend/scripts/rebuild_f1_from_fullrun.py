# -*- coding: utf-8 -*-
r"""
用本次全量重跑的产出替换建模管线的 F1 公告语义特征

数据来源（全部来自 81,208 份公告的全量重跑，非抽样）：
  announcement_risk_features.parquet   551,972 行 / 80,241 篇公告 / 45 主题体系

对齐目标（保持 F2-F6 与标签逻辑完全不变）：
  backend/data/modeling/raw/F1_announcement_semantic_features.parquet
  37,222 行 = 1,951 家 × 20 个季度末
  列: stock_code | T_date | announcement_semantic_000 ...

口径说明：
  - 他们的 T_date 是【报告期末】(03-31/06-30/09-30/12-31)，
    我们原先的季度锚点用【法定披露截止日】。为保证与 F2-F6 键对齐，
    本脚本按他们的 T_date 重新聚合，不套用原锚点。
  - 公告窗口严格 T - N < publish_date <= T，本身不引入未来信息。
  - 不写入 split、不写入任何标签列：标签由 build_modeling_dataset.py
    从 inquiry_events 动态生成，不能被特征表污染。

用法：
  python rebuild_f1_from_fullrun.py ^
    --run-dir  D:\fintech_nlp_output\run_20260826_001320_full ^
    --repo     C:\Users\papak\PycharmProjects\fintech_ai_competition\agent-team
"""
import io
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_VERSION = "v1"
WINDOWS = [30, 60, 90, 180]
L1_CODES = list("ABCDEFGH")
PREFIX = "announcement_semantic_"


def log(s=""):
    print(s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="全量重跑输出目录")
    ap.add_argument("--repo", required=True, help="agent-team 仓库根目录")
    ap.add_argument("--score-col", default="risk_strength_v2_top3")
    ap.add_argument("--dry-run", action="store_true", help="只生成不覆盖原文件")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    run_dir = Path(args.run_dir)
    raw = Path(args.repo) / "backend" / "data" / "modeling" / "raw"
    target = raw / "F1_announcement_semantic_features.parquet"

    log("=" * 72)
    log(f"  脚本版本   : {SCRIPT_VERSION}")
    log(f"  数据来源   : {run_dir / 'announcement_risk_features.parquet'}")
    log(f"  替换目标   : {target}")
    log(f"  时间窗口   : {WINDOWS} 天")
    log(f"  风险分列   : {args.score_col}")
    log("=" * 72)

    if not target.exists():
        log(f"[FATAL] 找不到 {target}"); sys.exit(1)

    # ---------- 1. 读原 F1，只取键网格 ----------
    log("\n[1] 读取原 F1 的键网格 ...")
    old = pd.read_parquet(target)
    grid = old[["stock_code", "T_date"]].copy()
    old_cols = [c for c in old.columns if c.startswith(PREFIX)]
    log(f"    {len(grid):,} 个键   原特征 {len(old_cols)} 维   公司 {grid.stock_code.nunique():,}")
    grid["T"] = pd.to_datetime(grid.T_date)

    # ---------- 2. 读全量重跑的公告级特征 ----------
    log("\n[2] 读取全量重跑产出 ...")
    SCORE = args.score_col
    af = pd.read_parquet(run_dir / "announcement_risk_features.parquet", columns=[
        "announcement_id", "company_code", "publish_date", "risk_theme", "l1_code",
        "rerank_score_max", "risk_strength_v2_max", SCORE, "risk_strength_v2_sum",
        "sent_neg_max", "strong_count", "rule_effective_hits"])
    af["publish_date"] = pd.to_datetime(af.publish_date, errors="coerce")
    af = af[af.publish_date.notna()]
    themes = sorted(af.risk_theme.unique())
    log(f"    {len(af):,} 行   {af.announcement_id.nunique():,} 篇公告   "
        f"{af.company_code.nunique():,} 家公司   {len(themes)} 个主题")

    # 高风险阈值只在 Train 公司上估计
    f2 = pd.read_csv(raw / "F2_financial_anomaly.csv",
                     usecols=["company_code", "split"], encoding="utf-8-sig")
    train_codes = set(f2[f2.split == "Train"].company_code.astype(str))
    sub = af[af.company_code.isin(train_codes)]
    thr = float(sub[SCORE].quantile(0.90)) if len(sub) else float(af[SCORE].quantile(0.90))
    af["is_high"] = af[SCORE] >= thr
    log(f"    高风险阈值 (Train p90): {thr:.4f}")

    cover = len(set(grid.stock_code.astype(str)) & set(af.company_code))
    log(f"    键网格公司与公告数据的交集: {cover:,} / {grid.stock_code.nunique():,}")

    # ---------- 3. 按他们的 T_date 重新聚合 ----------
    log("\n[3] 按 (stock_code, T_date) 重新聚合 ...")
    t0 = time.time()
    by_co = {c: g.sort_values("publish_date") for c, g in af.groupby("company_code")}

    rows = []
    for i, r in enumerate(grid.itertuples(), 1):
        rec = {}
        g = by_co.get(str(r.stock_code))
        T = r.T

        for w in WINDOWS:
            p = f"w{w}"
            s = g[(g.publish_date > T - timedelta(days=w)) & (g.publish_date <= T)] \
                if g is not None else None
            if s is None or len(s) == 0:
                for k in ["n_ann", "n_rows", "n_high", "n_themes", "rule_hits", "strong"]:
                    rec[f"{p}_{k}"] = 0
                for k in ["score_mean", "score_max", "score_sum", "rerank_max", "sentneg_mean"]:
                    rec[f"{p}_{k}"] = 0.0
                for L in L1_CODES:
                    rec[f"{p}_L1_{L}_max"] = 0.0
                    rec[f"{p}_L1_{L}_cnt"] = 0
                continue
            rec[f"{p}_n_ann"] = int(s.announcement_id.nunique())
            rec[f"{p}_n_rows"] = int(len(s))
            rec[f"{p}_n_high"] = int(s.is_high.sum())
            rec[f"{p}_n_themes"] = int(s.risk_theme.nunique())
            rec[f"{p}_rule_hits"] = int(s.rule_effective_hits.sum())
            rec[f"{p}_strong"] = int(s.strong_count.sum())
            rec[f"{p}_score_mean"] = float(s[SCORE].mean())
            rec[f"{p}_score_max"] = float(s[SCORE].max())
            rec[f"{p}_score_sum"] = float(s[SCORE].sum())
            rec[f"{p}_rerank_max"] = float(s.rerank_score_max.max())
            rec[f"{p}_sentneg_mean"] = float(s.sent_neg_max.mean())
            gl = s.groupby("l1_code")[SCORE].agg(["max", "size"])
            for L in L1_CODES:
                if L in gl.index:
                    rec[f"{p}_L1_{L}_max"] = float(gl.loc[L, "max"])
                    rec[f"{p}_L1_{L}_cnt"] = int(gl.loc[L, "size"])
                else:
                    rec[f"{p}_L1_{L}_max"] = 0.0
                    rec[f"{p}_L1_{L}_cnt"] = 0

        s180 = g[(g.publish_date > T - timedelta(days=180)) & (g.publish_date <= T)] \
            if g is not None else None
        if s180 is not None and len(s180):
            tm = s180.groupby("risk_theme")[SCORE].max().to_dict()
            tc = s180.groupby("risk_theme").size().to_dict()
        else:
            tm, tc = {}, {}
        for th in themes:
            rec[f"th_{th}_max"] = float(tm.get(th, 0.0))
            rec[f"th_{th}_cnt"] = int(tc.get(th, 0))

        if g is not None:
            hi = g[(g.publish_date <= T) & g.is_high]
            past = g[g.publish_date <= T]
            rec["days_since_high"] = int((T - hi.publish_date.max()).days) if len(hi) else 9999
            rec["days_since_ann"] = int((T - past.publish_date.max()).days) if len(past) else 9999
            a = g[(g.publish_date > T - timedelta(days=90)) & (g.publish_date <= T)][SCORE].sum()
            b = g[(g.publish_date > T - timedelta(days=180))
                  & (g.publish_date <= T - timedelta(days=90))][SCORE].sum()
            rec["trend_90_delta"] = float(a - b)
        else:
            rec["days_since_high"] = 9999
            rec["days_since_ann"] = 9999
            rec["trend_90_delta"] = 0.0

        rows.append(rec)
        if i % 5000 == 0:
            log(f"    {i:,}/{len(grid):,}   {time.time()-t0:.0f}s")

    feats = pd.DataFrame(rows)
    log(f"    完成 {len(feats):,} 行 × {feats.shape[1]} 维   {time.time()-t0:.0f}s")

    # ---------- 4. 列名映射 ----------
    log("\n[4] 映射为 announcement_semantic_* ...")
    orig = list(feats.columns)
    newname = {c: f"{PREFIX}{i:03d}" for i, c in enumerate(orig)}
    mapping = pd.DataFrame({
        "semantic_column": [newname[c] for c in orig],
        "source_feature": orig,
        "description": [describe(c) for c in orig],
    })
    feats = feats.rename(columns=newname)

    out = pd.concat([grid[["stock_code", "T_date"]].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
    log(f"    {len(out):,} 行 × {out.shape[1]} 列  (含 {len(orig)} 个特征)")

    # ---------- 5. 备份并写入 ----------
    bk = raw / "_backup_f1"
    bk.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        dst = bk / f"F1_announcement_semantic_features_{stamp}.parquet"
        shutil.copy2(target, dst)
        log(f"\n[5] 已备份原文件 -> {dst.name}")
        out.to_parquet(target, index=False)
        log(f"    已写入 {target}")
    else:
        alt = raw / "F1_announcement_semantic_features_NEW.parquet"
        out.to_parquet(alt, index=False)
        log(f"\n[5] DRY RUN：写入 {alt}，原文件未动")

    mapping.to_csv(raw / "f1_selection" / "F1_semantic_column_mapping.csv",
                   index=False, encoding="utf-8-sig")

    # ---------- 6. 质量报告 ----------
    nz = (feats.abs().sum(axis=1) > 0)
    yr = pd.to_datetime(out.T_date).dt.year
    log("\n" + "=" * 72)
    log("覆盖率报告")
    log("=" * 72)
    log(f"  非全零行: {nz.sum():,} / {len(out):,}  ({nz.mean()*100:.1f}%)")
    log("\n  按年份:")
    for y in sorted(yr.unique()):
        m = yr == y
        log(f"    {y}   {m.sum():>6,} 行   非全零 {nz[m].mean()*100:>5.1f}%")

    meta = {
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().isoformat(),
        "source": str(run_dir / "announcement_risk_features.parquet"),
        "source_rows": int(len(af)),
        "source_announcements": int(af.announcement_id.nunique()),
        "source_note": "81,208 份公告全量重跑，未抽样",
        "grid_rows": int(len(out)),
        "feature_dim": len(orig),
        "score_col": SCORE,
        "high_risk_threshold": thr,
        "threshold_fit_on": "Train companies only",
        "windows_days": WINDOWS,
        "themes": themes,
        "nonzero_ratio": float(nz.mean()),
        "t_date_convention": "报告期末（沿用原管线），非法定披露截止日",
        "contains_label_or_split": False,
    }
    with open(raw / "f1_selection" / "F1_rebuild_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"\n  列名对照表: {raw / 'f1_selection' / 'F1_semantic_column_mapping.csv'}")
    log(f"  元信息:     {raw / 'f1_selection' / 'F1_rebuild_meta.json'}")
    log("\n下一步:")
    log("  python -m backend.scripts.build_modeling_dataset")
    log("  python -m backend.scripts.train_models")


def describe(c: str) -> str:
    """把内部特征名翻译成人话，便于选出 Top-50 后解释每一维"""
    if c.startswith("th_"):
        p = c.split("_")
        return f"主题 {p[1]} 近180天{'最高风险分' if p[2]=='max' else '证据条数'}"
    if c.startswith("w"):
        w = c.split("_")[0][1:]
        if "_L1_" in c:
            L = c.split("_L1_")[1][0]
            return f"近{w}天 一级主题{L} {'最高分' if c.endswith('max') else '证据数'}"
        tail = c.split("_", 1)[1]
        m = {"n_ann": "公告数", "n_rows": "主题命中数", "n_high": "高风险条数",
             "n_themes": "覆盖主题数", "rule_hits": "规则命中数", "strong": "强风险信号数",
             "score_mean": "风险分均值", "score_max": "风险分最大值",
             "score_sum": "风险分总和", "rerank_max": "精排分最大值",
             "sentneg_mean": "负面情绪均值"}
        return f"近{w}天 {m.get(tail, tail)}"
    m2 = {"days_since_high": "距上次高风险公告天数", "days_since_ann": "距上次公告天数",
          "trend_90_delta": "风险分90天环比变化"}
    return m2.get(c, c)


if __name__ == "__main__":
    main()
