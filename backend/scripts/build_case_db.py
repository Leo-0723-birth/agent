#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建监管问询案例库（case_db.json + case_vectors.npy）
=====================================================
把 02_监管问询 的数据产物打包成案例检索 Agent 可用的案例库：

  输入：
    - inquiries.jsonl           （4785 份问询函结构化全文）
    - inquiry_rule_risks.jsonl  （规则引擎风险标签 + 证据）
    - evaluation_ground_truth_normalized.csv （官方关注点 regulatory_focus_points）

  输出（backend/data/）：
    - case_db.json     案例元数据：case_id / company / publish_date / inquiry_type
                       / title / focus_points（官方关注点优先，否则规则标签）/ letter_excerpt
    - case_vectors.npy 案例语义向量（embedding 后端与运行时一致）

关键设计（对齐框架 README 第六节）：
  - 关注点标签【优先用官方标准答案】regulatory_focus_points → 检索天然对齐官方口径
  - 无官方标签的问询函，退回规则引擎 category_name 作为 topic
  - evidence/excerpt 一律取原文（inquiries.jsonl 段落拼接），不存 LLM 转述

用法：
    python build_case_db.py                # 全量构建（4785 份问询函）
    python build_case_db.py --limit 200    # 先构建子集验证
    python build_case_db.py --backend bge  # 用 BGE 向量（需权重，质量更高）
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    INQUIRY_JSONL, RULE_RISKS_JSONL, EVAL_GT_NORMALIZED_CSV,
    CASE_DB_PATH, CASE_VEC_PATH, CASE_META_PATH, CASE_EXCERPT_CHARS, DATA_DIR,
    EMBEDDING_BACKEND, EMBEDDING_MODEL,
)


def _full_text(doc):
    paras = doc.get("paragraphs") or []
    return "\n".join(p.get("text", "") for p in paras if p.get("text"))


def _inquiry_type(title, text):
    """从标题/正文抽取问询函类型。"""
    t = (title or "") + "\n" + (text or "")[:400]
    if "重组问询" in t or ("重组" in t and "问询" in t):
        return "重组问询函"
    if "半年报" in t or "半年度报告" in t:
        return "半年报问询函"
    if "三季报" in t or "季报" in t or "第三季度" in t:
        return "季报问询函"
    if "年报" in t or "年度报告" in t:
        return "年报问询函"
    if "关注函" in t:
        return "关注函"
    if "许可类" in t:
        return "许可类问询函"
    return "问询函"


def _load_gt(gt_csv):
    """官方关注点：{(stock_code, publish_date): [focus_point, ...]}。"""
    import pandas as pd
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig")
    idx = {}
    for _, row in gt.iterrows():
        try:
            fps = json.loads(row["regulatory_focus_points"])
            if not isinstance(fps, list):
                fps = []
        except (json.JSONDecodeError, TypeError, KeyError):
            fps = []
        key = (str(row["stock_code"]), str(row["publish_date"]))
        idx.setdefault(key, [])
        idx[key].extend(fps)
    return idx


def _build_entries(limit=None):
    """从三个数据源组装案例条目列表。"""
    inquiries = []
    with open(INQUIRY_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("doc_type") == "inquiry_letter":
                inquiries.append(d)

    risks = {}
    with open(RULE_RISKS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            risks[r.get("doc_id")] = r

    gt = _load_gt(EVAL_GT_NORMALIZED_CSV)

    if limit:
        inquiries = inquiries[:limit]

    entries = []
    for d in inquiries:
        text = _full_text(d)
        title = d.get("title") or ""
        key = (d.get("stock_code"), d.get("publish_date"))

        # 官方关注点优先，否则用规则 category_name（去重）作 topic
        focus_points = gt.get(key, [])
        if not focus_points:
            rec = risks.get(d.get("doc_id"), {})
            cats = []
            for m in rec.get("risk_matches", []):
                c = m.get("category_name")
                if c and c not in cats:
                    cats.append(c)
            focus_points = cats

        entries.append({
            "case_id": d["doc_id"],
            "company": d.get("stock_code"),
            "publish_date": d.get("publish_date"),
            "inquiry_type": _inquiry_type(title, text),
            "title": title,
            "focus_points": focus_points,
            "letter_excerpt": text[:CASE_EXCERPT_CHARS],
        })

    return entries


def _embed_text(e):
    """案例的语义文本：类型 + 标题 + 关注点（与查询画像同源）。"""
    parts = [e["inquiry_type"], e["title"]]
    parts.extend(e.get("focus_points", []))
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="构建监管问询案例库")
    parser.add_argument("--limit", type=int, default=None, help="只构建前 N 份问询函（调试用）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的案例库")
    parser.add_argument("--backend", choices=["bge", "fallback"], default=None,
                        help="嵌入后端（默认读 config.EMBEDDING_BACKEND；bge=本地权重 1024 维，fallback=零依赖）")
    args = parser.parse_args()

    # 切后端：必须在 import embedding 之前设环境变量（embedding.embed 动态读取）
    if args.backend:
        os.environ["EMBEDDING_BACKEND"] = args.backend

    if CASE_DB_PATH.exists() and CASE_VEC_PATH.exists() and not args.force:
        print(f"案例库已存在，跳过（--force 覆盖）：{CASE_DB_PATH}")
        return 0

    print("[1/3] 组装案例条目（官方关注点优先 / 规则标签兜底）...")
    entries = _build_entries(limit=args.limit)
    print(f"      问询函数: {len(entries)}")

    print("[2/3] 语义向量化（backend 与运行时同后端）...")
    from backend.skills.embedding import embed
    import numpy as np
    texts = [_embed_text(e) for e in entries]
    vecs = embed(texts)   # (n, dim)，L2 归一化
    print(f"      向量 shape: {vecs.shape}")

    print("[3/3] 落盘 case_db.json + case_vectors.npy + case_meta.json ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DB_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(CASE_VEC_PATH, vecs.astype("float32"))
    backend = os.getenv("EMBEDDING_BACKEND", EMBEDDING_BACKEND)
    CASE_META_PATH.write_text(json.dumps({
        "embedding_backend": backend,
        "embedding_dim": int(vecs.shape[1]),
        "embedding_model": EMBEDDING_MODEL if backend == "bge" else "fallback_char_bigram",
        "n_cases": len(entries),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    gt_hit = sum(1 for e in entries if e["focus_points"] and len(e["focus_points"]) > 0)
    type_counter = Counter(e["inquiry_type"] for e in entries)
    print("\n== 构建完成 ==")
    print(f"案例库: {CASE_DB_PATH}")
    print(f"向量库: {CASE_VEC_PATH}（{backend} {vecs.shape[1]} 维）")
    print(f"案例数: {len(entries)} | 有关注点的: {gt_hit}")
    print("问询函类型分布:")
    for t, n in type_counter.most_common():
        print(f"  {t}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
