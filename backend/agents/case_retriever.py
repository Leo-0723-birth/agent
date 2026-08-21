#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监管案例检索 Agent (CaseRetrieverAgent) —— RAG 混合检索（RRF 融合）
====================================================================
职责：用目标公司风险画像（公告风险标签 + 财务异常）检索历史案例库，
      双通道检索 + RRF 融合排序，输出 Top-5 相似案例（含相似度与原文摘录）。
输入：company、ctx（读取 semantic.risk_factors + financial.anomaly_list）
输出：写入 ctx.cases（case_id/company/inquiry_type/publish_date/topics/similarity/letter_excerpt）

双通道检索：
  通道1 语义向量：embed(目标画像文本) → 案例库余弦相似 → 排名 R1
  通道2 标签重合：目标 label_ref/风险类别 与案例 focus_points 的子串重合数 → 排名 R2
RRF 融合：score(case) = Σ 1/(RRF_K + rank_i)，仅出现在任一通道的案例参与融合。

依赖：skills/embedding.py + skills/vector_store.py + 离线建好的案例库
      （scripts/build_case_vector_db.py 产出，官方标准答案对齐）。
"""
import sys
from pathlib import Path

from ..config import CASE_TOP_K, RRF_K
from ..skills import vector_store, concern_store
from ..skills.embedding import embed_one
from .base import AgentBase

# 财务侧标签 → 关键词（与案例 focus_points 官方关注点对齐；concern_dict 之外兜底）
FINANCIAL_LABEL_KEYWORDS = {
    "盈利质量": ["现金流", "净利", "利润", "亏损", "盈利质量", "经营现金"],
    "盈利能力": ["净利", "亏损", "盈利", "毛利率", "扣非", "营收"],
    "商誉减值": ["商誉", "减值"],
    "收入确认": ["收入", "确认", "营业收入"],
    "内控": ["内控", "内部控制"],
    "偿债能力": ["负债", "偿债", "债务", "流动"],
    "资金占用": ["资金占用", "占用", "往来款"],
    "担保": ["担保"],
    "信息披露": ["披露", "更正"],
    "关联交易": ["关联", "关联方"],
    "持续经营": ["持续经营", "退市", "风险警示"],
    "财务异常": ["异常", "变动", "下滑"],
}

# 任务1 官方标签体系关键词（risk_taxonomy + risk_dictionary；缺失时为空）
try:
    from ..skills.risk_labels import LABEL_KEYWORDS as _OFFICIAL_KWS
except Exception:
    _OFFICIAL_KWS = {}

_LABEL_KEYWORDS_CACHE = None


def label_keywords():
    """合并三源关键词：① 队友关注点词典（规则类别+风险标签 → 官方关注点词汇，覆盖率最高）
    ② 财务侧标签 ③ 任务1官方标签体系。任一缺失自动跳过。"""
    global _LABEL_KEYWORDS_CACHE
    if _LABEL_KEYWORDS_CACHE is None:
        kw = {}
        try:
            cd = concern_store.load()
            for cat, c in cd.get("categories", {}).items():
                kw[cat] = c.get("keywords", [])
            for lab, c in cd.get("labels", {}).items():
                kw[lab] = c.get("keywords", [])
        except Exception:
            pass
        kw.update(FINANCIAL_LABEL_KEYWORDS)
        kw.update(_OFFICIAL_KWS)
        _LABEL_KEYWORDS_CACHE = kw
    return _LABEL_KEYWORDS_CACHE


def expand_label_keywords(labels):
    """标签集合 → 关键词集合（无映射的中文标签退回标签本身）。"""
    kws = set()
    lk = label_keywords()
    for lab in labels or []:
        if not lab:
            continue
        mapped = lk.get(lab)
        if mapped:
            kws.update(mapped)
        elif not str(lab).isascii():
            kws.add(lab)
    return kws


class CaseRetrieverAgent(AgentBase):
    name = "CaseRetriever"

    def __init__(self, top_k=CASE_TOP_K, rrf_k=RRF_K):
        super().__init__()
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._db, self._vecs = None, None
        self._meta = {}

    # ================= 数据加载 =================
    def _load_db(self):
        """懒加载案例库 + 构建元数据（维度一致性校验用）。"""
        if self._db is None:
            self._db, self._vecs = vector_store.load()
            self._meta = vector_store.load_meta()
        return self._db, self._vecs

    # ================= 目标画像 =================
    def _profile_text(self, ctx):
        """目标公司风险画像文本（语义通道查询）。"""
        parts = []
        for r in ctx.semantic.risk_factors:
            parts.append(f"{r.get('category', '')}:{r.get('description', '')}")
        for a in ctx.financial.anomaly_list:
            parts.append(f"{a.get('label_ref', '')}:{a.get('evidence', '')}")
        return "；".join(parts)

    def _profile_labels(self, ctx):
        """目标公司风险标签集合（标签通道查询）。"""
        labels = set()
        for r in ctx.semantic.risk_factors:
            c = str(r.get("category", ""))
            if c:
                labels.add(c)
        for a in ctx.financial.anomaly_list:
            l = str(a.get("label_ref", ""))
            if l:
                labels.add(l)
        return labels

    # ================= 通道1: 语义向量检索 =================
    def _semantic_rank(self, query_vec, entries, vectors):
        """余弦相似度排序，返回 {case_index: rank}。"""
        scores = vector_store.cosine_scores(query_vec, vectors)
        order = sorted(range(len(entries)), key=lambda i: -scores[i])
        return {i: r + 1 for r, i in enumerate(order)}

    # ================= 通道2: 标签重合检索 =================
    def _label_rank(self, labels, entries):
        """目标标签（官方编码或俗称）→ 关键词，与案例 focus_points 子串重合数排序。"""
        kws = expand_label_keywords(labels)
        hit = {}
        for i, e in enumerate(entries):
            fps = " ".join(e.get("focus_points", []))
            c = sum(1 for kw in kws if kw and kw in fps)
            if c > 0:
                hit[i] = c
        ranked = sorted(hit.items(), key=lambda kv: -kv[1])
        return {i: r + 1 for r, (i, _) in enumerate(ranked)}

    # ================= RRF 融合 =================
    def _rrf(self, ranks_list):
        """score = Σ 1/(k + rank_i)。"""
        scores = {}
        for ranks in ranks_list:
            for idx, rank in ranks.items():
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        return scores

    # ================= 主入口 =================
    def execute(self, company, ctx):
        entries, vectors = self._load_db()
        if not entries:
            ctx.cases = []
            return ctx

        profile = self._profile_text(ctx)
        labels = self._profile_labels(ctx)
        if not profile and not labels:
            ctx.cases = []
            return ctx

        ranks = []
        if profile:
            query_vec = embed_one(profile)
            dim_ok = vectors is not None and len(vectors) > 0 and query_vec.shape[0] == vectors.shape[1]
            if not dim_ok:
                # 维度一致性守卫：query 与库必须同一 embedding 后端，否则语义通道禁用（仅标签通道兜底）
                backend = self._meta.get("embedding_backend", "?")
                dim = self._meta.get("embedding_dim", "?")
                print(f"[case_retriever] 向量维度不匹配：query={query_vec.shape[0]} vs "
                      f"库={vectors.shape[1]}（案例库 meta: {backend} {dim} 维）。"
                      f"语义通道禁用，仅标签通道。切 EMBEDDING_BACKEND=bge 或重建案例库后恢复。")
            else:
                ranks.append(self._semantic_rank(query_vec, entries, vectors))
        if labels:
            label_ranks = self._label_rank(labels, entries)
            if label_ranks:
                ranks.append(label_ranks)

        if not ranks:
            ctx.cases = []
            return ctx

        fused = self._rrf(ranks)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:self.top_k]

        ctx.cases = []
        for idx, score in top:
            e = entries[idx]
            ctx.cases.append({
                "case_id": e["case_id"],
                "company": e["company"],
                "inquiry_type": e["inquiry_type"],
                "publish_date": e["publish_date"],
                "topics": e["focus_points"][:5],
                "similarity": round(float(score), 4),
                "letter_excerpt": (e.get("letter_excerpt") or "")[:200],
            })
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.case_retriever）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context

    ctx = Context(company="000004.SZ")
    # 模拟公告研读 + 财务检测的产物（000004 国华网安的真实风险画像）
    ctx.semantic.risk_factors = [
        {"category": "商誉减值", "description": "计提商誉减值损失4.5亿"},
        {"category": "收入确认", "description": "营收同比下滑42%"},
        {"category": "内控", "description": "上年内控审计否定意见"},
    ]
    ctx.financial.anomaly_list = [
        {"label_ref": "盈利质量", "evidence": "净利润-639万与经营现金流-2799万均为负"},
        {"label_ref": "盈利能力", "evidence": "ROE=-7.05%"},
    ]
    agent = CaseRetrieverAgent()
    agent.execute("000004.SZ", ctx)
    print(f"检索到 {len(ctx.cases)} 个相似案例：")
    for c in ctx.cases:
        print(f"  [{c['similarity']}] {c['company']} | {c['inquiry_type']} | "
              f"{c['publish_date']} | 关注点: {'、'.join(c['topics'][:3])}")
