#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监管案例检索 Agent (CaseRetrieverAgent) —— BGE + 45类标签混合检索
=================================================================

职责：
    用目标公司风险画像（公告风险标签 + 财务异常）检索历史监管案例，
    双通道排序后用 RRF 融合，输出 Top-K 相似案例。

通道1：
    BGE 语义向量检索（query 使用 BGE query instruction）。

通道2：
    45类 taxonomy_labels 直接匹配优先；
    旧标签先映射到45类；
    关键词子串仅作为补充，不再作为标签通道的唯一依据。

时间过滤：
    execute(..., cutoff_date=...) 可显式传入预测/分析截点；
    也会尝试读取 ctx.cutoff_date / ctx.as_of_date。
    仅允许 publish_date < cutoff_date 的历史案例参与排序。
    若没有任何截点，保持兼容但打印警告——正式回测必须传入截点。

兼容：
    保留原 execute(company, ctx) 调用方式；
    保留 ctx.cases 原有字段，并新增 taxonomy_labels / matched_taxonomy_labels。
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from ..config import CASE_TOP_K, RRF_K
from ..skills import vector_store
from ..skills.case_embedding import embed_one
from .base import AgentBase
from .label_keywords_v2 import (
    LABEL_KEYWORDS,
    TAXONOMY_NAMES,
    expand_labels,
)


class CaseRetrieverAgent(AgentBase):
    name = "CaseRetriever"

    DIRECT_CODE_WEIGHT = 4.0
    MAPPED_CODE_WEIGHT = 2.0
    KEYWORD_WEIGHT = 0.25
    KEYWORD_HIT_CAP = 8

    def __init__(self, top_k=CASE_TOP_K, rrf_k=RRF_K):
        super().__init__()
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._cosine_scores = {}   # 语义通道余弦相似度（idx -> 0~1）
        self._db, self._vecs = None, None

    # ================= 数据加载 =================
    def _load_db(self):
        """懒加载案例库，并校验元数据与向量行数一致。"""
        if self._db is None:
            self._db, self._vecs = vector_store.load()

            if self._db and self._vecs is not None:
                if len(self._db) != len(self._vecs):
                    raise ValueError(
                        f"案例元数据与向量数量不一致: "
                        f"db={len(self._db)}, vectors={len(self._vecs)}"
                    )

        return self._db, self._vecs

    # ================= 标签规范化 =================
    @staticmethod
    def _raw_profile_labels(ctx) -> set[str]:
        """从公告风险与财务异常中收集上游原始标签。"""
        labels = set()

        for r in ctx.semantic.risk_factors:
            c = str(r.get("category", "")).strip()
            if c:
                labels.add(c)

        for a in ctx.financial.anomaly_list:
            l = str(a.get("label_ref", "")).strip()
            if l:
                labels.add(l)

        return labels

    @staticmethod
    def _taxonomy_codes(labels: Iterable[str]) -> set[str]:
        """只保留 expand_labels() 中的45类编码。"""
        return {
            lab for lab in expand_labels(labels)
            if lab in TAXONOMY_NAMES
        }

    # ================= 目标画像 =================
    def _risk_mapper_labels(self, ctx):
        """读取 RiskMapper 生成的45类监管标签。"""
        labels = getattr(getattr(ctx, "semantic", None), "risk_labels", [])

        codes = set()
        for item in labels:
            for code in item.get("taxonomy_labels", []):
                code = str(code).strip()
                if code in TAXONOMY_NAMES:
                    codes.add(code)

        return codes

    def _risk_mapper_label_text(self, ctx):
        """将RiskMapper标签转换为语义查询增强文本。"""
        codes = sorted(self._risk_mapper_labels(ctx))
        return " ".join(
            f"{c} {TAXONOMY_NAMES[c]}" for c in codes
        )

    def _profile_text(self, ctx):
        """
        目标公司风险画像文本（语义通道查询）。
        如果能映射到45类，同时加入标准主题名，提高 query 与案例库表述一致性。
        """
        parts = []

        for r in ctx.semantic.risk_factors:
            category = str(r.get("category", "")).strip()
            description = str(r.get("description", "")).strip()
            codes = self._taxonomy_codes([category])
            code_text = " / ".join(
                f"{c} {TAXONOMY_NAMES[c]}" for c in sorted(codes)
            )
            head = code_text or category
            if head or description:
                parts.append(f"{head}:{description}".strip(":"))

        for a in ctx.financial.anomaly_list:
            label = str(a.get("label_ref", "")).strip()
            evidence = str(a.get("evidence", "")).strip()
            codes = self._taxonomy_codes([label])
            code_text = " / ".join(
                f"{c} {TAXONOMY_NAMES[c]}" for c in sorted(codes)
            )
            head = code_text or label
            if head or evidence:
                parts.append(f"{head}:{evidence}".strip(":"))

        return "；".join(parts)

    def _profile_labels(self, ctx):
        """返回原始标签与45类编码集合。"""
        raw = self._raw_profile_labels(ctx)
        expanded = self._taxonomy_codes(raw)
        return raw, expanded

    # ================= 时间过滤 =================
    @staticmethod
    def _parse_date(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value

        s = str(value).strip()
        if not s:
            return None

        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y%m%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass

        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    def _resolve_cutoff_date(self, ctx, cutoff_date=None):
        """显式参数优先，其次尝试 Context 常见字段。"""
        candidates = [
            cutoff_date,
            getattr(ctx, "cutoff_date", None),
            getattr(ctx, "as_of_date", None),
            getattr(ctx, "as_of", None),
        ]

        for obj_name in ("prediction", "meta"):
            obj = getattr(ctx, obj_name, None)
            if obj is None:
                continue

            if isinstance(obj, dict):
                candidates.extend([
                    obj.get("cutoff_date"),
                    obj.get("as_of_date"),
                    obj.get("prediction_date"),
                ])
            else:
                candidates.extend([
                    getattr(obj, "cutoff_date", None),
                    getattr(obj, "as_of_date", None),
                    getattr(obj, "prediction_date", None),
                ])

        for value in candidates:
            parsed = self._parse_date(value)
            if parsed:
                return parsed

        return None

    def _eligible_indices(self, entries, cutoff_date=None):
        """
        返回可参与检索的原案例索引。
        有截点时严格使用 publish_date < cutoff_date，避免未来案例泄漏。
        """
        if cutoff_date is None:
            return list(range(len(entries)))

        eligible = []
        bad_dates = 0

        for i, e in enumerate(entries):
            d = self._parse_date(e.get("publish_date"))
            if d is None:
                bad_dates += 1
                continue
            if d < cutoff_date:
                eligible.append(i)

        if bad_dates:
            print(f"[CaseRetriever] 警告：{bad_dates} 条案例日期无法解析，已排除。")

        return eligible

    # ================= 通道1：语义向量 =================
    def _semantic_rank(self, query_vec, entries, vectors, eligible_indices):
        """余弦相似度排序，返回 {case_index: rank}；同时记录余弦相似度（0-1 直观值）。"""
        scores = vector_store.cosine_scores(query_vec, vectors)

        order = sorted(
            eligible_indices,
            key=lambda i: -float(scores[i]),
        )
        ranks = {i: r + 1 for r, i in enumerate(order)}
        # 余弦相似度映射：供展示"RRF 融合得分 + 余弦相似度"双口径
        self._cosine_scores = {i: round(float(scores[i]), 4) for i in eligible_indices}
        return ranks

    # ================= 通道2：45类标签 + 关键词辅助 =================
    def _label_rank(self, raw_labels, taxonomy_codes, entries, eligible_indices):
        """
        排序优先级：
        1) 上游直接给出的45类编码，与 case.taxonomy_labels 精确重合；
        2) 旧标签映射出的45类，与 case.taxonomy_labels 重合；
        3) LABEL_KEYWORDS 在 focus_points 中命中，作为低权重补充。
        """
        direct_codes = {
            lab for lab in raw_labels
            if lab in TAXONOMY_NAMES
        }
        mapped_codes = set(taxonomy_codes) - direct_codes

        kws = set()
        for lab in set(raw_labels) | set(taxonomy_codes):
            kws.update(LABEL_KEYWORDS.get(lab, [lab]))

        hit_scores = {}

        for i in eligible_indices:
            e = entries[i]
            case_codes = {
                str(x).strip()
                for x in e.get("taxonomy_labels", [])
                if str(x).strip()
            }

            direct_overlap = case_codes & direct_codes
            mapped_overlap = case_codes & mapped_codes

            fps = " ".join(e.get("focus_points", []))
            keyword_hits = sum(
                1 for kw in kws
                if kw and kw in fps
            )
            keyword_hits = min(keyword_hits, self.KEYWORD_HIT_CAP)

            score = (
                self.DIRECT_CODE_WEIGHT * len(direct_overlap)
                + self.MAPPED_CODE_WEIGHT * len(mapped_overlap)
                + self.KEYWORD_WEIGHT * keyword_hits
            )

            if score > 0:
                hit_scores[i] = score

        ranked = sorted(
            hit_scores.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return {i: r + 1 for r, (i, _) in enumerate(ranked)}

    # ================= RRF 融合 =================
    def _rrf(self, ranks_list):
        """score = Σ 1/(k + rank_i)。"""
        scores = {}
        for ranks in ranks_list:
            for idx, rank in ranks.items():
                scores[idx] = (
                    scores.get(idx, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )
        return scores


    # ================= 匹配解释 =================
    def _generate_match_reason(self, ctx, matched_codes, case):
        """
        生成案例匹配理由。
        基于：
        1. taxonomy标签命中
        2. 当前风险因素
        3. 历史案例关注点
        """
        reasons = []

        if matched_codes:
            labels = [
                TAXONOMY_NAMES.get(code, code)
                for code in matched_codes
            ]
            reasons.append(
                "命中监管关注标签：" + "、".join(labels)
            )

        risk_text = " ".join(
            [
                str(r.get("category", ""))
                + str(r.get("description", ""))
                for r in ctx.semantic.risk_factors
            ]
        )

        topic_text = " ".join(
            case.get("focus_points", [])
        )

        combined = risk_text + topic_text

        if any(k in combined for k in ["亏损", "利润", "盈利", "收入"]):
            reasons.append(
                "当前风险与历史案例均涉及盈利能力或经营业绩变化"
            )

        if any(k in combined for k in ["减值", "商誉", "资产"]):
            reasons.append(
                "当前风险与历史案例均涉及资产质量或减值事项"
            )

        if not reasons:
            reasons.append(
                "基于风险画像与历史问询文本语义相似性匹配"
            )

        return reasons

    # ================= 主入口 =================
    def execute(self, company, ctx, cutoff_date=None):
        entries, vectors = self._load_db()

        if not entries:
            ctx.cases = []
            return ctx

        profile = self._profile_text(ctx)
        raw_labels, taxonomy_codes = self._profile_labels(ctx)

        # RiskMapper输出优先进入标签通道
        mapper_codes = self._risk_mapper_labels(ctx)
        taxonomy_codes = taxonomy_codes | mapper_codes

        # 标签增强语义query
        mapper_text = self._risk_mapper_label_text(ctx)
        if mapper_text:
            profile = (profile + "；监管关注标签:" + mapper_text).strip("；")

        if not profile and not raw_labels:
            ctx.cases = []
            return ctx

        resolved_cutoff = self._resolve_cutoff_date(
            ctx,
            cutoff_date=cutoff_date,
        )

        if resolved_cutoff is None:
            print(
                "[CaseRetriever] 警告：未取得 cutoff_date，"
                "本次未执行时间过滤。正式回测必须传入预测截点。"
            )

        eligible_indices = self._eligible_indices(
            entries,
            cutoff_date=resolved_cutoff,
        )

        if not eligible_indices:
            ctx.cases = []
            return ctx

        ranks = []

        # 保存各检索通道排名，用于解释RRF融合
        semantic_ranks = {}
        label_ranks = {}

        # BGE 查询使用 query instruction；案例库向量是 document embedding。
        if profile:
            try:
                query_vec = embed_one(profile, is_query=True)
                semantic_ranks = self._semantic_rank(
                    query_vec,
                    entries,
                    vectors,
                    eligible_indices,
                )
                ranks.append(semantic_ranks)
            except Exception as exc:
                # 语义模型属于增强通道；本地权重缺失或网络不可达时退回标签通道。
                ctx.trace_log.append({
                    "agent": "CaseRetriever.Semantic",
                    "status": "skipped",
                    "reason": f"语义模型不可用，已退回标签检索：{type(exc).__name__}: {str(exc)[:180]}",
                    "trace_complete": True,
                })

        if raw_labels or taxonomy_codes:
            label_ranks = self._label_rank(
                raw_labels | mapper_codes,
                taxonomy_codes,
                entries,
                eligible_indices,
            )
            if label_ranks:
                ranks.append(label_ranks)

        if not ranks:
            ctx.cases = []
            return ctx

        fused = self._rrf(ranks)
        top = sorted(
            fused.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:self.top_k]

        ctx.cases = []

        for idx, score in top:
            e = entries[idx]
            case_codes = {
                str(x).strip()
                for x in e.get("taxonomy_labels", [])
                if str(x).strip()
            }
            matched_codes = sorted(case_codes & taxonomy_codes)

            ctx.cases.append({
                "case_id": e["case_id"],
                "company": e["company"],
                "inquiry_type": e["inquiry_type"],
                "publish_date": e["publish_date"],
                "topics": e.get("focus_points", [])[:5],
                # 兼容旧字段；其数值实际上是 RRF 融合分数。
                "similarity": round(float(score), 6),
                "rrf_score": round(float(score), 6),
                "cosine_similarity": self._cosine_scores.get(idx),  # 语义通道余弦相似度（0-1）
                "semantic_rank": semantic_ranks.get(idx),
                "label_rank": label_ranks.get(idx),
                "taxonomy_labels": sorted(case_codes),
                "matched_taxonomy_labels": matched_codes,
                "match_reason": self._generate_match_reason(ctx, matched_codes, e),
                "letter_excerpt": (e.get("letter_excerpt") or "")[:200],
            })

        return ctx


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context

    ctx = Context(company="000004.SZ")
    ctx.semantic.risk_factors = [
        {"category": "商誉减值", "description": "计提商誉减值损失4.5亿"},
        {"category": "收入确认", "description": "营收确认时点存在异常"},
        {"category": "内控", "description": "上年内控审计否定意见"},
    ]
    ctx.financial.anomaly_list = [
        {"label_ref": "盈利质量", "evidence": "净利润与经营现金流均为负"},
        {"label_ref": "盈利能力", "evidence": "ROE为负"},
    ]

    agent = CaseRetrieverAgent()
    agent.execute("000004.SZ", ctx, cutoff_date="2025-01-01")

    print(f"检索到 {len(ctx.cases)} 个相似案例：")
    for c in ctx.cases:
        print(
            f"  [{c['rrf_score']}] {c['company']} | "
            f"{c['inquiry_type']} | {c['publish_date']} | "
            f"命中标签: {','.join(c['matched_taxonomy_labels']) or '-'} | "
            f"关注点: {'、'.join(c['topics'][:3])} | "
            f"匹配原因: {'；'.join(c['match_reason'])}"
        )
