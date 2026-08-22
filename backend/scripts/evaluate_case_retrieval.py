#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
案例检索 Agent 离线评测
======================

目的：
1) 给已经建好的 1,483 条案例库做模块级离线评测；
2) 同时比较 semantic-only / label-only / hybrid(RRF)；
3) 严格使用历史案例：candidate.publish_date < target.publish_date；
4) 可选排除同一公司，检验跨公司泛化；
5) 输出 Hit@5 / Strict Hit@5 / Label Recall@5 / MRR / nDCG@5。

重要说明：
- 这是“内部代理评测”，不是官方判定细则。
- 默认 query_mode=taxonomy：用目标案例的45类标签构造查询。
- query_mode=oracle_text 会额外使用真实问询关注点文本，只能视为检索器上限诊断，
  不能当作真实预警场景的端到端结果。
- 相关性由45类二级主题重合定义，因此必须在技术文档中明确写成 proxy relevance。

运行示例：
python -m backend.scripts.evaluate_case_retrieval --query-mode taxonomy
python -m backend.scripts.evaluate_case_retrieval --query-mode taxonomy --exclude-same-company
python -m backend.scripts.evaluate_case_retrieval --query-mode oracle_text --max-events 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.config import CASE_DB_PATH, CASE_VEC_PATH, OUTPUT_DIR, RRF_K
from backend.skills.embedding import embed
from backend.skills import vector_store
from backend.agents.case_retriever import CaseRetrieverAgent
from backend.agents.label_keywords_v2 import TAXONOMY_NAMES


def parse_date(value):
    s = str(value or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def target_labels(entry):
    return {
        str(x).strip()
        for x in entry.get("taxonomy_labels", [])
        if str(x).strip() in TAXONOMY_NAMES
    }


def query_text(entry, mode="taxonomy"):
    labels = sorted(target_labels(entry))
    label_text = "；".join(f"{c} {TAXONOMY_NAMES[c]}" for c in labels)

    if mode == "taxonomy":
        return label_text

    if mode == "oracle_text":
        fps = "；".join(str(x).strip() for x in entry.get("focus_points", []) if str(x).strip())
        return f"{label_text}；{fps}" if fps else label_text

    raise ValueError(f"未知 query_mode={mode}")


def relevance_grade(target_codes, candidate_codes):
    """
    0: 无L2重合
    1: 有至少1个L2重合
    2: 较强相关
    3: 高度相关

    规则：
    - 单标签目标只要精确命中该L2，记3；
    - 多标签目标：
      grade 3: 重合>=2 且目标覆盖率>=0.5
      grade 2: 重合>=2 或目标覆盖率>=0.5
      grade 1: 重合>=1
    """
    t = set(target_codes)
    c = set(candidate_codes)
    if not t or not c:
        return 0

    overlap = len(t & c)
    if overlap == 0:
        return 0

    if len(t) == 1:
        return 3

    coverage = overlap / len(t)

    if overlap >= 2 and coverage >= 0.5:
        return 3
    if overlap >= 2 or coverage >= 0.5:
        return 2
    return 1


def rrf(ranks_list, k=RRF_K):
    scores = {}
    for ranks in ranks_list:
        for idx, rank in ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def dcg(grades):
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        total += (2 ** grade - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(grades, ideal_grades):
    denom = dcg(ideal_grades)
    return dcg(grades) / denom if denom > 0 else 0.0


def ranks_from_scores(scores, eligible):
    order = sorted(eligible, key=lambda i: -float(scores[i]))
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def label_rank(agent, labels, entries, eligible):
    # 评测查询已经是精确45类标签，因此 raw_labels == taxonomy_codes
    return agent._label_rank(
        raw_labels=set(labels),
        taxonomy_codes=set(labels),
        entries=entries,
        eligible_indices=eligible,
    )


def evaluate(args):
    entries, vectors = vector_store.load(CASE_DB_PATH, CASE_VEC_PATH)
    if not entries or vectors is None:
        raise FileNotFoundError(
            f"未找到案例库或向量：\nCASE_DB_PATH={CASE_DB_PATH}\nCASE_VEC_PATH={CASE_VEC_PATH}"
        )

    if len(entries) != len(vectors):
        raise ValueError(f"案例数与向量行数不一致: {len(entries)} vs {len(vectors)}")

    if vectors.ndim != 2:
        raise ValueError(f"case_vectors.npy 应为二维矩阵，实际 {vectors.shape}")

    dates = [parse_date(e.get("publish_date")) for e in entries]
    labels_list = [target_labels(e) for e in entries]

    valid_targets = [
        i for i, (d, labs) in enumerate(zip(dates, labels_list))
        if d is not None and labs
    ]

    if args.max_events:
        valid_targets = valid_targets[: args.max_events]

    print(f"[eval] cases={len(entries)} vectors={vectors.shape}")
    print(f"[eval] valid targets={len(valid_targets)}")
    print(f"[eval] query_mode={args.query_mode}")
    print(f"[eval] exclude_same_company={args.exclude_same_company}")
    print("[eval] 注意：这是内部 proxy relevance 评测，不等同于官方判定口径。")

    queries = [query_text(entries[i], args.query_mode) for i in valid_targets]
    query_vecs = embed(
        queries,
        is_query=True,
        batch_size=args.batch_size,
    )

    if query_vecs.shape[1] != vectors.shape[1]:
        raise ValueError(
            f"查询向量与案例向量维度不一致: query={query_vecs.shape}, cases={vectors.shape}"
        )

    agent = CaseRetrieverAgent(top_k=args.k, rrf_k=RRF_K)

    per_event = []
    agg = defaultdict(float)
    n_eval = 0

    for pos, target_idx in enumerate(valid_targets):
        target = entries[target_idx]
        target_date = dates[target_idx]
        target_company = str(target.get("company", ""))
        tlabels = labels_list[target_idx]

        eligible = []
        for i, d in enumerate(dates):
            if d is None or d >= target_date:
                continue
            if args.exclude_same_company and str(entries[i].get("company", "")) == target_company:
                continue
            eligible.append(i)

        if not eligible:
            continue

        # Semantic channel
        sem_scores = vector_store.cosine_scores(query_vecs[pos], vectors)
        sem_rank = ranks_from_scores(sem_scores, eligible)

        # Label channel
        lab_rank = label_rank(agent, tlabels, entries, eligible)

        modes = {}
        modes["semantic"] = sem_rank
        modes["label"] = lab_rank
        modes["hybrid"] = rrf([sem_rank, lab_rank] if lab_rank else [sem_rank], k=RRF_K)

        event_row = {
            "case_id": target.get("case_id"),
            "company": target_company,
            "publish_date": target.get("publish_date"),
            "target_labels": ",".join(sorted(tlabels)),
            "candidate_count": len(eligible),
        }

        for mode in ("semantic", "label", "hybrid"):
            scores_or_ranks = modes[mode]

            if mode in ("semantic", "label"):
                # rank dict -> order by ascending rank
                top = [
                    idx for idx, _ in sorted(scores_or_ranks.items(), key=lambda kv: kv[1])
                ][: args.k]
            else:
                # RRF score dict -> descending score
                top = [
                    idx for idx, _ in sorted(scores_or_ranks.items(), key=lambda kv: -kv[1])
                ][: args.k]

            grades = [
                relevance_grade(tlabels, labels_list[i])
                for i in top
            ]

            loose_hit = int(any(g >= 1 for g in grades))
            strict_hit = int(any(g >= 2 for g in grades))

            # Top-K 对目标L2标签的覆盖率
            covered = set()
            for i in top:
                covered |= (tlabels & labels_list[i])
            label_recall = len(covered) / len(tlabels) if tlabels else 0.0

            # MRR：第一个 strict relevant；若不存在则0
            rr = 0.0
            for rank, g in enumerate(grades, start=1):
                if g >= 2:
                    rr = 1.0 / rank
                    break

            # nDCG：按所有 eligible 的 relevance grade 得到理想前K
            all_grades = sorted(
                (
                    relevance_grade(tlabels, labels_list[i])
                    for i in eligible
                ),
                reverse=True,
            )[: args.k]
            top_grades = grades + [0] * (args.k - len(grades))
            ideal = all_grades + [0] * (args.k - len(all_grades))
            ndcg = ndcg_at_k(top_grades[: args.k], ideal[: args.k])

            event_row[f"{mode}_hit@{args.k}"] = loose_hit
            event_row[f"{mode}_strict_hit@{args.k}"] = strict_hit
            event_row[f"{mode}_label_recall@{args.k}"] = round(label_recall, 6)
            event_row[f"{mode}_mrr"] = round(rr, 6)
            event_row[f"{mode}_ndcg@{args.k}"] = round(ndcg, 6)
            event_row[f"{mode}_top_case_ids"] = "|".join(
                str(entries[i].get("case_id", "")) for i in top
            )

            agg[f"{mode}_hit"] += loose_hit
            agg[f"{mode}_strict_hit"] += strict_hit
            agg[f"{mode}_label_recall"] += label_recall
            agg[f"{mode}_mrr"] += rr
            agg[f"{mode}_ndcg"] += ndcg

        per_event.append(event_row)
        n_eval += 1

        if n_eval % 100 == 0:
            print(f"[eval] {n_eval} events done")

    if n_eval == 0:
        raise RuntimeError("没有可评测目标事件。")

    summary = {
        "n_cases": len(entries),
        "n_evaluable_events": n_eval,
        "k": args.k,
        "query_mode": args.query_mode,
        "exclude_same_company": args.exclude_same_company,
        "relevance_definition": {
            "loose_hit": "Top-K中至少1个案例共享>=1个45类二级主题",
            "strict_hit": "Top-K中至少1个案例 relevance_grade>=2",
            "grade_3": "单标签精确命中；或多标签重合>=2且目标标签覆盖率>=0.5",
            "grade_2": "多标签重合>=2或目标标签覆盖率>=0.5",
            "grade_1": "至少共享1个二级主题",
        },
        "metrics": {},
    }

    for mode in ("semantic", "label", "hybrid"):
        summary["metrics"][mode] = {
            f"Hit@{args.k}": agg[f"{mode}_hit"] / n_eval,
            f"Strict_Hit@{args.k}": agg[f"{mode}_strict_hit"] / n_eval,
            f"Label_Recall@{args.k}": agg[f"{mode}_label_recall"] / n_eval,
            "MRR_strict": agg[f"{mode}_mrr"] / n_eval,
            f"nDCG@{args.k}": agg[f"{mode}_ndcg"] / n_eval,
        }

    out_dir = Path(args.output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "cross_company" if args.exclude_same_company else "history"
    stem = f"case_retrieval_eval_{args.query_mode}_{suffix}"

    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}_summary.json"

    if per_event:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_event[0].keys()))
            writer.writeheader()
            writer.writerows(per_event)

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐事件结果: {csv_path}")
    print(f"汇总结果:   {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-mode",
        choices=["taxonomy", "oracle_text"],
        default="taxonomy",
        help="taxonomy=仅用45类标签；oracle_text=标签+真实关注点文本（上限诊断）",
    )
    parser.add_argument(
        "--exclude-same-company",
        action="store_true",
        help="排除同一公司的历史案例，做更严格的跨公司泛化评测",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
