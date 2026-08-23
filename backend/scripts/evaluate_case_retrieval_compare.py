
from __future__ import annotations

import json
import random
from pathlib import Path
import sys
import numpy as np
# 添加项目根目录到 Python 搜索路径
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT)
)
from backend.skills.case_embedding import embed_one


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "backend" / "data" / "vector_db" / "case_db.json"
VEC_PATH = ROOT / "backend" / "data" / "vector_db" / "case_vectors.npy"


def load_data():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    vectors = np.load(VEC_PATH)

    return cases, vectors


def cosine_search(query_vec, vectors, topk=5):
    query_vec = query_vec / np.linalg.norm(query_vec)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    scores = vectors @ query_vec

    idx = np.argsort(-scores)[:topk]

    return idx.tolist()


def label_search(query_case, cases, topk=5):
    query_labels = set(query_case.get("taxonomy_labels", []))

    scores = []

    for i, c in enumerate(cases):
        labels = set(c.get("taxonomy_labels", []))

        overlap = len(query_labels & labels)

        scores.append((overlap, i))

    scores.sort(reverse=True)

    return [x[1] for x in scores[:topk]]


def rrf_merge(rank_lists, k=60, topk=5):
    score = {}

    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            score[idx] = score.get(idx, 0) + 1 / (k + rank)

    return [
        x[0]
        for x in sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True
        )[:topk]
    ]


def label_recall(result_ids, target_case):
    target = set(target_case.get("taxonomy_labels", []))

    if not target:
        return 0

    hit = 0

    for idx in result_ids:
        labels = set(cases[idx].get("taxonomy_labels", []))

        if labels & target:
            hit += 1

    return 1 if hit else 0


def evaluate(method_results, targets):
    hit5 = 0
    mrr = 0
    recall = 0

    for results, target in zip(method_results, targets):

        if target in results:
            hit5 += 1
            rank = results.index(target) + 1
            mrr += 1 / rank

        recall += label_recall(results, cases[target])

    n = len(targets)

    return {
        "Hit@5": round(hit5 / n, 4),
        "MRR": round(mrr / n, 4),
        "Label_Recall@5": round(recall / n, 4),
    }


if __name__ == "__main__":

    cases, vectors = load_data()

    random.seed(42)

    sample_size = min(300, len(cases))

    test_ids = random.sample(
        range(len(cases)),
        sample_size
    )

    semantic_results = []
    label_results = []
    hybrid_results = []

    for idx in test_ids:

        case = cases[idx]

        text = " ".join(
            case.get("focus_points", [])
        )

        query_vec = embed_one(
            text,
            is_query=True
        )

        semantic = cosine_search(
            query_vec,
            vectors
        )

        label = label_search(
            case,
            cases
        )

        hybrid = rrf_merge(
            [
                semantic,
                label
            ]
        )

        semantic_results.append(semantic)
        label_results.append(label)
        hybrid_results.append(hybrid)


    print("=" * 60)
    print("CaseRetriever Evaluation")
    print("=" * 60)

    print(f"Samples: {sample_size}")

    print("\n[Semantic Only]")
    print(
        evaluate(
            semantic_results,
            test_ids
        )
    )

    print("\n[Label Only]")
    print(
        evaluate(
            label_results,
            test_ids
        )
    )

    print("\n[Hybrid RRF]")
    print(
        evaluate(
            hybrid_results,
            test_ids
        )
    )
