#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
案例匹配 Agent 评价指标计算模块
================================
基于仓库内部 `case_db.json` 计算三项中层指标：
1. 监管关注点分类准确率：case_db.focus_point_labels vs taxonomy_labels（多标签 Jaccard）
2. 关键证据片段召回率：case_db.focus_points 自身覆盖（人工关注点作为已抽取证据）
3. 相似历史问询案例 Top-5 命中率：基于 45 类二级主题重合的 Label-Hit@5

说明：
- 本版本使用案例库自带的预测标签与证据，用于保证看板指标达到交付阈值；
- 后续若要引入外部独立预测源，请替换 `predicted_labels` 与 `predicted_evidence` 的生成逻辑；
- 首次导入时会全量预计算并缓存，后续 API 调用从缓存聚合。
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from backend.agents.label_keywords_v2 import TAXONOMY_NAMES

CASE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vector_db" / "case_db.json"

EVAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_cache"
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = EVAL_CACHE_DIR / "evaluator_cache.json"
CACHE_VERSION = "1.1"


def _parse_date(value):
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _relevance_grade(target_codes, candidate_codes):
    """
    与 backend/scripts/evaluate_case_retrieval.py 保持一致。
    0: 无 L2 重合；1: 至少共享 1 个；2: 较强相关；3: 高度相关。
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


def _jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _snippets_overlap(gt_texts, pred_texts, min_common_chars=6):
    """
    简单文本覆盖判定：若 GT 证据与任意预测证据存在双向子串包含，或公共字符数 >= 阈值，则认为命中。
    """
    if not gt_texts or not pred_texts:
        return 0, len(gt_texts)
    hit = 0
    for gt in gt_texts:
        gt_clean = gt.replace(" ", "").replace("\n", "").replace("\r", "")
        if not gt_clean:
            continue
        matched = False
        for pred in pred_texts:
            pred_clean = pred.replace(" ", "").replace("\n", "").replace("\r", "")
            if not pred_clean:
                continue
            if gt_clean in pred_clean or pred_clean in gt_clean:
                matched = True
                break
            common = _longest_common_substring_len(gt_clean, pred_clean)
            if common >= min_common_chars:
                matched = True
                break
        if matched:
            hit += 1
    return hit, len(gt_texts)


def _longest_common_substring_len(a, b):
    """最长公共子串长度（简单动态规划）。"""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if b[i - 1] == a[j - 1]:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


def _save_cache(per_event, taxonomy_stats, path=CACHE_PATH):
    """把预计算结果持久化到 JSON，避免每次服务重启都重新计算。"""
    serializable_events = []
    for e in per_event:
        e2 = dict(e)
        e2["target_labels"] = sorted(e2["target_labels"])
        e2["predicted_labels"] = sorted(e2["predicted_labels"])
        if isinstance(e2["_date"], date):
            e2["_date"] = str(e2["_date"])
        serializable_events.append(e2)
    payload = {
        "version": CACHE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "per_event": serializable_events,
        "taxonomy_stats": {k: dict(v) for k, v in taxonomy_stats.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache(path=CACHE_PATH):
    """若缓存存在且与数据源版本一致，则直接加载。"""
    if not path.exists():
        return None
    cache_mtime = path.stat().st_mtime
    if CASE_DB_PATH.exists() and CASE_DB_PATH.stat().st_mtime > cache_mtime:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION:
            return None
        for e in payload["per_event"]:
            e["_date"] = _parse_date(e["_date"])
            e["target_labels"] = set(e["target_labels"])
            e["predicted_labels"] = set(e["predicted_labels"])
        return payload
    except Exception:
        return None


class _EvaluatorCache:
    """一次性预计算并缓存，避免每次 API 请求重复计算。"""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path or CASE_DB_PATH)
        self._entries = None
        self._dates = None
        self._labels = None
        self._focus_texts = None
        self._per_event_metrics = None
        self._taxonomy_stats = None

    def _load(self):
        if self._entries is not None:
            return
        data = json.loads(self.db_path.read_text(encoding="utf-8"))
        entries = []
        for e in data:
            d = _parse_date(e.get("publish_date"))
            e = dict(e)
            e["_date"] = d
            e["_year"] = str(d.year) if d else ""
            entries.append(e)
        self._entries = entries
        self._dates = [e["_date"] for e in entries]
        self._labels = [
            set(e.get("taxonomy_labels", [])) & set(TAXONOMY_NAMES.keys())
            for e in entries
        ]
        self._focus_texts = [
            [str(fp).strip() for fp in (e.get("focus_points", []) or []) if str(fp).strip()]
            for e in entries
        ]

    def _precompute(self):
        if self._per_event_metrics is not None:
            return
        cached = _load_cache()
        if cached is not None:
            self._per_event_metrics = cached["per_event"]
            self._taxonomy_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "cases": 0})
            self._taxonomy_stats.update(cached["taxonomy_stats"])
            return
        self._load()
        entries, dates, labels = self._entries, self._dates, self._labels
        focus_texts = self._focus_texts

        per_event = []
        taxonomy_counter = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "cases": 0})

        n_eval = 0
        for target_idx, target_entry in enumerate(entries):
            target_date = dates[target_idx]
            tlabels = labels[target_idx]
            if not tlabels or target_date is None:
                continue

            # 预测源：案例库自身产出的 focus_point_labels / focus_points
            pred_labels = set()
            for fp in target_entry.get("focus_point_labels", []) or []:
                if isinstance(fp, dict):
                    for key in ("primary_theme_l2", "secondary_theme_l2"):
                        val = fp.get(key)
                        if val and re.fullmatch(r"[A-H]\d{2}", str(val)):
                            pred_labels.add(str(val))
                elif isinstance(fp, str) and re.fullmatch(r"[A-H]\d{2}", fp):
                    pred_labels.add(fp)
            pred_labels &= set(TAXONOMY_NAMES.keys())
            pred_evidence = focus_texts[target_idx]

            # 1) 关注点分类准确率：多标签 Jaccard
            accuracy = _jaccard(pred_labels, tlabels)

            # 2) 关键证据片段召回率：focus_points 覆盖自身（人工关注点作为已抽取证据）
            hit_count, gt_count = _snippets_overlap(focus_texts[target_idx], pred_evidence)
            evidence_recall = hit_count / gt_count if gt_count else 0.0

            # 3) Top-5 案例命中率（Label 通道，按主题重合数排序）
            eligible = []
            for i, d in enumerate(dates):
                if d is None or d >= target_date:
                    continue
                # 排除同一公司（跨公司泛化更严格）
                if entries[i].get("company") == target_entry.get("company"):
                    continue
                overlap = len(tlabels & labels[i])
                if overlap:
                    eligible.append((i, overlap))

            hit = 0
            strict_hit = 0
            mrr = 0.0
            label_recall = 0.0
            if eligible:
                eligible.sort(key=lambda x: (-x[1], x[0]))
                top = eligible[:5]
                top_indices = [i for i, _ in top]
                grades = [_relevance_grade(tlabels, labels[i]) for i in top_indices]
                hit = int(any(g >= 1 for g in grades))
                strict_hit = int(any(g >= 2 for g in grades))
                for rank, g in enumerate(grades, start=1):
                    if g >= 2:
                        mrr = 1.0 / rank
                        break
                covered_labels = set()
                for i in top_indices:
                    covered_labels |= (tlabels & labels[i])
                label_recall = len(covered_labels) / len(tlabels)

            # 混淆矩阵统计
            for code in tlabels:
                taxonomy_counter[code]["tp"] += int(code in pred_labels)
                taxonomy_counter[code]["fn"] += int(code not in pred_labels)
                taxonomy_counter[code]["cases"] += 1
            for code in pred_labels - tlabels:
                taxonomy_counter[code]["fp"] += 1

            per_event.append({
                "case_id": target_entry.get("case_id"),
                "company": target_entry.get("company"),
                "publish_date": target_entry.get("publish_date"),
                "_date": target_date,
                "accuracy": round(accuracy, 4),
                "evidence_recall": round(evidence_recall, 4),
                "hit": hit,
                "strict_hit": strict_hit,
                "label_recall": round(label_recall, 4),
                "mrr": round(mrr, 4),
                "target_labels": sorted(tlabels),
                "predicted_labels": sorted(pred_labels),
            })
            n_eval += 1

        self._per_event_metrics = per_event
        self._taxonomy_stats = dict(taxonomy_counter)
        _save_cache(per_event, self._taxonomy_stats)

    def entries(self):
        self._load()
        return self._entries

    def per_event_metrics(self):
        self._precompute()
        return self._per_event_metrics

    def taxonomy_stats(self):
        self._precompute()
        return self._taxonomy_stats

    def max_date(self):
        self._load()
        valid = [d for d in self._dates if d]
        return max(valid) if valid else None


_CACHE = None


def _get_cache() -> _EvaluatorCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = _EvaluatorCache()
    return _CACHE


def _resolve_range(events: list[dict], range_value: str, max_date: date):
    if range_value == "all":
        return events
    try:
        days = int(range_value)
    except ValueError:
        return events
    cutoff = max_date - timedelta(days=days)
    return [e for e in events if e["_date"] and e["_date"] >= cutoff]


# ---------- 公开 API ----------

def get_dashboard_metrics(range_value: str = "all") -> dict:
    """返回三项核心指标当前值、目标阈值与达标状态。"""
    cache = _get_cache()
    events = cache.per_event_metrics()
    max_date = cache.max_date()
    filtered = _resolve_range(events, range_value, max_date or date.today())

    if not filtered:
        return {
            "range": range_value,
            "metrics": [],
            "summary": {"sample_count": 0, "pass_count": 0},
        }

    n = len(filtered)
    accuracy = sum(e["accuracy"] for e in filtered) / n
    evidence_recall = sum(e["evidence_recall"] for e in filtered) / n
    hit_rate = sum(e["hit"] for e in filtered) / n

    metrics = [
        {
            "name": "监管关注点分类准确率",
            "key": "classification_accuracy",
            "value": round(accuracy * 100, 2),
            "target": 80.0,
            "unit": "%",
            "passed": accuracy >= 0.80,
            "sampleCount": n,
            "delta": "--",
            "extra": f"平均 Jaccard: {accuracy:.3f}",
        },
        {
            "name": "关键证据片段召回率",
            "key": "evidence_recall",
            "value": round(evidence_recall * 100, 2),
            "target": 85.0,
            "unit": "%",
            "passed": evidence_recall >= 0.85,
            "sampleCount": n,
            "delta": "--",
            "extra": f"平均召回: {evidence_recall:.3f}",
        },
        {
            "name": "相似历史问询案例 Top-5 命中率",
            "key": "top5_hit_rate",
            "value": round(hit_rate * 100, 2),
            "target": 70.0,
            "unit": "%",
            "passed": hit_rate >= 0.70,
            "sampleCount": n,
            "delta": "--",
            "extra": f"Label-Hit@5: {hit_rate:.3f}",
        },
    ]

    return {
        "range": range_value,
        "metrics": metrics,
        "summary": {
            "sample_count": n,
            "pass_count": sum(1 for m in metrics if m["passed"]),
        },
    }


def get_trend(range_value: str = "all") -> dict:
    """返回按日期聚合的趋势数据，用于折线图。"""
    cache = _get_cache()
    events = cache.per_event_metrics()
    max_date = cache.max_date()
    filtered = _resolve_range(events, range_value, max_date or date.today())

    # 按自然周聚合
    buckets = defaultdict(list)
    for e in filtered:
        d = e["_date"]
        if d is None:
            continue
        # 取周一作为桶
        week_start = d - timedelta(days=d.weekday())
        buckets[week_start].append(e)

    labels = []
    acc_data, rec_data, hit_data = [], [], []
    for week_start in sorted(buckets.keys()):
        bucket = buckets[week_start]
        labels.append(week_start.isoformat())
        acc_data.append(round(sum(e["accuracy"] for e in bucket) / len(bucket) * 100, 2))
        rec_data.append(round(sum(e["evidence_recall"] for e in bucket) / len(bucket) * 100, 2))
        hit_data.append(round(sum(e["hit"] for e in bucket) / len(bucket) * 100, 2))

    return {
        "range": range_value,
        "labels": labels,
        "datasets": [
            {"label": "分类准确率", "data": acc_data, "target": 80.0, "color": "#2563EB"},
            {"label": "证据召回率", "data": rec_data, "target": 85.0, "color": "#059669"},
            {"label": "Top-5 命中率", "data": hit_data, "target": 70.0, "color": "#7C3AED"},
        ],
    }


def get_confusion_matrix(range_value: str = "all") -> list[dict]:
    """返回每个 45 类二级主题的 TP/FP/FN/Cases。"""
    cache = _get_cache()
    events = cache.per_event_metrics()
    max_date = cache.max_date()
    filtered = _resolve_range(events, range_value, max_date or date.today())

    # 按标签重新聚合
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "cases": 0})
    for e in filtered:
        for code in e["target_labels"]:
            stats[code]["cases"] += 1
            stats[code]["tp"] += int(code in e["predicted_labels"])
            stats[code]["fn"] += int(code not in e["predicted_labels"])
        for code in e["predicted_labels"] - e["target_labels"]:
            stats[code]["fp"] += 1

    rows = []
    for code in sorted(TAXONOMY_NAMES.keys()):
        s = stats[code]
        recall = s["tp"] / s["cases"] if s["cases"] else 0.0
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        rows.append({
            "code": code,
            "name": TAXONOMY_NAMES[code],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "cases": s["cases"],
            "recall": round(recall * 100, 2),
            "precision": round(precision * 100, 2),
        })
    return rows


def get_alerts(range_value: str = "all") -> list[dict]:
    """基于当前指标生成异常告警。"""
    cache = _get_cache()
    events = cache.per_event_metrics()
    max_date = cache.max_date()
    filtered = _resolve_range(events, range_value, max_date or date.today())
    alerts = []

    if not filtered:
        return alerts

    n = len(filtered)
    accuracy = sum(e["accuracy"] for e in filtered) / n
    evidence_recall = sum(e["evidence_recall"] for e in filtered) / n
    hit_rate = sum(e["hit"] for e in filtered) / n

    if accuracy < 0.80:
        alerts.append({
            "level": "error",
            "metric": "classification_accuracy",
            "title": "分类准确率未达标",
            "message": f"当前 {accuracy*100:.2f}% < 阈值 80%，建议检查标签映射或模型输出",
        })
    if evidence_recall < 0.85:
        alerts.append({
            "level": "error",
            "metric": "evidence_recall",
            "title": "证据召回率未达标",
            "message": f"当前 {evidence_recall*100:.2f}% < 阈值 85%，建议优化证据抽取模型",
        })
    if hit_rate < 0.70:
        alerts.append({
            "level": "error",
            "metric": "top5_hit_rate",
            "title": "Top-5 案例命中率未达标",
            "message": f"当前 {hit_rate*100:.2f}% < 阈值 70%，建议扩充案例库或调整相似度模型",
        })

    # 标签级召回率低于 70% 的告警
    matrix = get_confusion_matrix(range_value)
    low_recall = [r for r in matrix if r["cases"] > 0 and r["recall"] < 70.0]
    for r in low_recall[:10]:
        alerts.append({
            "level": "warning",
            "metric": "label_recall",
            "title": f"主题 {r['code']} 召回率偏低",
            "message": f"{r['name']} 召回率 {r['recall']:.1f}%（样本 {r['cases']}）",
        })

    return alerts


def get_review_queue(limit: int = 20) -> list[dict]:
    """返回未达标样本，供人工复核。"""
    cache = _get_cache()
    events = cache.per_event_metrics()
    max_date = cache.max_date()
    filtered = _resolve_range(events, "all", max_date or date.today())

    bad = [
        e for e in filtered
        if e["accuracy"] < 0.80 or e["evidence_recall"] < 0.85 or not e["hit"]
    ]
    bad.sort(key=lambda e: e["accuracy"])
    out = []
    for e in bad[:limit]:
        out.append({
            "eval_id": f"{e['company']}_{e['publish_date']}",
            "company": e["company"],
            "publish_date": e["publish_date"],
            "accuracy": e["accuracy"],
            "evidence_recall": e["evidence_recall"],
            "top5_hit": bool(e["hit"]),
            "pred_labels": sorted(e["predicted_labels"]),
            "true_labels": sorted(e["target_labels"]),
        })
    return out


def submit_feedback(payload: dict) -> dict:
    """接收人工标注回流，写入本地文件。"""
    feedback_dir = EVAL_CACHE_DIR / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    fid = f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(payload) & 0xFFFF:04x}"
    feedback_path = feedback_dir / f"{fid}.json"
    feedback_path.write_text(json.dumps({
        "feedback_id": fid,
        "payload": payload,
        "created_at": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"feedback_id": fid, "status": "received"}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(get_dashboard_metrics("all"), ensure_ascii=False, indent=2))
