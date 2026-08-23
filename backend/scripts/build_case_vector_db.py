#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重建监管问询全量案例库：
- evaluation_ground_truth.csv：1,483 个问询事件
- llm_classify_results.jsonl：10,481 条关注点的 45 类分类结果

输出：
- backend/data/vector_db/case_db.json
- backend/data/vector_db/case_vectors.npy（除非 --metadata-only）

推荐从项目根目录运行：
python -m backend.scripts.build_case_vector_db \
    --ground-truth data/evaluation_ground_truth.csv \
    --classify-results data/llm_classify_results.jsonl

如果先只验证元数据：
python -m backend.scripts.build_case_vector_db \
    --ground-truth data/evaluation_ground_truth.csv \
    --classify-results data/llm_classify_results.jsonl \
    --metadata-only
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.agents.label_keywords_v2 import TAXONOMY_NAMES
from backend.skills import vector_store


def parse_focus_points(raw: Any) -> list[str]:
    """解析 regulatory_focus_points_json。源 CSV 实际为 Python list 字符串。"""
    if isinstance(raw, list):
        values = raw
    else:
        text = "" if raw is None else str(raw).strip()
        if not text:
            return []
        try:
            values = ast.literal_eval(text)
        except Exception:
            try:
                values = json.loads(text)
            except Exception as exc:
                raise ValueError(f"无法解析关注点列表: {text[:120]}") from exc
    if not isinstance(values, list):
        raise ValueError(f"关注点字段不是 list: {type(values).__name__}")
    return [str(x).strip() for x in values if str(x).strip()]


def normalize_date(value: Any) -> str:
    s = str(value).strip()
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
    if not m:
        return s
    y, mo, d = map(int, m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}"


def infer_inquiry_type(title: str) -> str:
    """尽量保留监管函件类型，不把所有非问询函都压成“其他”."""
    t = str(title or "")
    rules = [
        ("许可类重组问询函", "许可类重组问询函"),
        ("半年报问询函", "半年报问询函"),
        ("年报问询函", "年报问询函"),
        ("季报问询函", "季报问询函"),
        ("重组问询函", "重组问询函"),
        ("关注函", "关注函"),
        ("问询函", "问询函"),
        ("定期报告事后审核意见函", "定期报告事后审核意见函"),
        ("重大资产重组", "重大资产重组审核意见函"),
        ("监管工作函", "监管工作函"),
        ("工作函", "工作函"),
        ("监管函", "监管函"),
        ("事先告知书", "事先告知书"),
        ("审核意见函", "审核意见函"),
    ]
    for needle, label in rules:
        if needle in t:
            return label
    return "其他"


def load_classifications(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                raise ValueError(f"{path.name} 第 {line_no} 行缺少 id")
            idx = int(obj["id"])
            if idx in rows:
                raise ValueError(f"分类结果存在重复 id={idx}")
            rows[idx] = obj
    return rows


def build_entries(gt_path: Path, cls_path: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(gt_path, dtype={"secucode": str})
    required = {
        "secucode", "publish_date", "announcement_title",
        "regulatory_focus_points_json"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground truth 缺少字段: {sorted(missing)}")

    cls = load_classifications(cls_path)

    entries: list[dict] = []
    seen_case_ids: set[str] = set()
    global_id = 0
    n_text_mismatch = 0
    n_failed_cls = 0

    for row_idx, row in df.iterrows():
        fps = parse_focus_points(row["regulatory_focus_points_json"])
        fp_labels = []
        primary_labels = []
        secondary_labels = []

        for text in fps:
            if global_id not in cls:
                raise ValueError(
                    f"分类结果缺少 id={global_id}，对应事件行 {row_idx}，关注点：{text[:80]}"
                )
            c = cls[global_id]
            cls_text = str(c.get("text", "")).strip()
            if cls_text != text:
                n_text_mismatch += 1
                raise ValueError(
                    f"id={global_id} 文本错位。\n"
                    f"ground truth: {text}\nclassification: {cls_text}"
                )
            if not bool(c.get("ok", True)):
                n_failed_cls += 1

            primary_l2 = c.get("primary_theme_l2")
            secondary_l2 = c.get("secondary_theme_l2")
            if primary_l2:
                primary_labels.append(str(primary_l2))
            if secondary_l2:
                secondary_labels.append(str(secondary_l2))

            fp_labels.append({
                "focus_id": global_id,
                "text": text,
                "primary_theme_l1": c.get("primary_theme_l1"),
                "primary_theme_l2": primary_l2,
                "secondary_theme_l2": secondary_l2,
                "primary_question_type": c.get("primary_question_type"),
                "secondary_question_type": c.get("secondary_question_type"),
                "primary_stage": c.get("primary_stage"),
                "confidence": c.get("confidence"),
            })
            global_id += 1

        secucode = str(row["secucode"]).strip()
        publish_date = normalize_date(row["publish_date"])
        title = str(row["announcement_title"]).strip()
        date_digits = re.sub(r"\D", "", publish_date)
        case_id = f"IC-{secucode}-{date_digits}"

        # 同公司同日可能有多份函件时，避免 case_id 冲突
        base_id = case_id
        suffix = 2
        while case_id in seen_case_ids:
            case_id = f"{base_id}-{suffix}"
            suffix += 1
        seen_case_ids.add(case_id)

        p_labels = sorted(set(primary_labels))
        s_labels = sorted(set(secondary_labels))

        entries.append({
            "case_id": case_id,
            "company": secucode,
            "publish_date": publish_date,
            "inquiry_type": infer_inquiry_type(title),
            "title": title,
            "focus_points": fps,
            "taxonomy_labels": p_labels,
            "secondary_taxonomy_labels": s_labels,
            "focus_point_labels": fp_labels,
            # ground truth CSV 不含函件正文，避免伪造摘录；以后可由 PDF 回填
            "letter_excerpt": "",
        })

    expected_ids = set(range(global_id))
    actual_ids = set(cls)
    extra_ids = actual_ids - expected_ids
    missing_ids = expected_ids - actual_ids
    if extra_ids or missing_ids:
        raise ValueError(
            f"分类 id 集合不一致：missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    stats = {
        "n_cases": len(entries),
        "n_focus_points": global_id,
        "n_companies": len({e["company"] for e in entries}),
        "n_text_mismatch": n_text_mismatch,
        "n_failed_classification": n_failed_cls,
        "inquiry_types": {},
    }
    for e in entries:
        t = e["inquiry_type"]
        stats["inquiry_types"][t] = stats["inquiry_types"].get(t, 0) + 1

    return entries, stats


def case_to_embedding_text(entry: dict) -> str:
    label_text = "；".join(
        f"{code} {TAXONOMY_NAMES.get(code, '')}".strip()
        for code in entry.get("taxonomy_labels", [])
    )
    focus_text = "；".join(entry.get("focus_points", []))
    return (
        f"函件类型：{entry.get('inquiry_type', '')}。"
        f"监管风险标签：{label_text}。"
        f"监管关注点：{focus_text}"
    )


def build_vectors(entries: list[dict]) -> np.ndarray:
    # 延迟导入：metadata-only 模式不要求 torch/transformers
    from backend.config import EMBEDDING_BACKEND, EMBEDDING_MODEL
    from backend.skills.case_embedding import embed

    texts = [case_to_embedding_text(e) for e in entries]

    print(f"[embedding] backend = {EMBEDDING_BACKEND}")
    print(f"[embedding] model   = {EMBEDDING_MODEL}")
    print(f"[embedding] cases   = {len(texts)}")

    vectors = np.asarray(
        embed(texts, is_query=False),
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(f"Embedding 输出必须是二维矩阵，实际 shape={vectors.shape}")
    if vectors.shape[0] != len(entries):
        raise ValueError(
            f"Embedding 行数与案例数不一致: vectors={vectors.shape[0]}, cases={len(entries)}"
        )

    # 二次校验归一化，防止后端实现改动导致余弦检索口径漂移
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    vectors[valid] = vectors[valid] / norms[valid]

    return vectors.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="重建全量监管问询案例库")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--classify-results", required=True, type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只生成 case_db.json，不生成 case_vectors.npy",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="自定义 case_db.json 输出路径；默认使用 config.CASE_DB_PATH",
    )
    parser.add_argument(
        "--vec-path",
        type=Path,
        default=None,
        help="自定义 case_vectors.npy 输出路径；默认使用 config.CASE_VEC_PATH",
    )
    args = parser.parse_args()

    entries, stats = build_entries(args.ground_truth, args.classify_results)

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.metadata_only:
        # 仅写元数据，避免覆盖已有向量文件
        from backend.config import CASE_DB_PATH
        db_path = Path(args.db_path or CASE_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"case_db 已写入: {db_path}")
        print("注意：metadata-only 未重建向量。正式检索前必须再生成与当前 embedding 后端一致的 case_vectors.npy。")
        return

    from backend.config import EMBEDDING_BACKEND
    if EMBEDDING_BACKEND != "bge":
        print(
            f"警告：当前 EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}。"
            "正式案例检索建议使用 bge。"
        )

    vectors = build_vectors(entries)
    db_path, vec_path = vector_store.save(
        entries,
        vectors,
        db_path=args.db_path,
        vec_path=args.vec_path,
    )
    print(f"case_db: {db_path}")
    print(f"case_vectors: {vec_path}")
    print(f"vector shape: {vectors.shape}")


if __name__ == "__main__":
    main()
