#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chunk 级检索 Agent (ChunkRetrieverAgent) —— 段落粒度先例/证据召回（§七 可选升级）
====================================================================================
职责：用目标公司风险画像（规则风险标签 + 财务异常）在 chunk 级索引上召回「最相似问询段落」
      → 写 ctx.chunks（chunk_id / company / publish_date / text / similarity）。

与 case_retriever（文档级）的区别：
  - case_retriever 召回 Top-5 相似「问询函」（整份文档，含关注点/摘录）；
  - chunk_retriever 召回 Top-K 相似「问询段落」（~420 字片段），更细粒度，用于证据/先例定位。

设计：
  - 语义通道：embed(风险画像文本) → chunk 向量余弦 → Top-K（排除目标公司自身，取他人先例）。
  - 优雅降级：chunk 索引未构建（data/chunk_vectors.npy 缺失）时直接跳过，不打断流水线。

依赖：skills/chunk_store.py + build_chunk_index.py 先构建索引。
"""
import logging
import sys
from pathlib import Path

from ..config import CHUNK_TOP_K
from ..skills import chunk_store
from ..skills.embedding import embed_one
from .base import AgentBase

_logger = logging.getLogger(__name__)


class ChunkRetrieverAgent(AgentBase):
    name = "ChunkRetriever"

    def __init__(self, top_k=CHUNK_TOP_K, run_config=None):
        super().__init__()
        self.run_config = run_config
        self.top_k = top_k
        self._entries, self._vecs = None, None

    # ================= 数据加载 =================
    def _load(self):
        if self._entries is None:
            self._entries, self._vecs = chunk_store.load()
        return self._entries, self._vecs

    # ================= 目标画像 =================
    @staticmethod
    def _profile_text(ctx):
        """目标公司风险画像文本（与 case_retriever 语义通道同源）。"""
        parts = []
        for r in ctx.semantic.risk_factors:
            parts.append(f"{r.get('category', '')}:{r.get('description', '')}")
        for a in ctx.financial.anomaly_list:
            parts.append(f"{a.get('label_ref', '')}:{a.get('evidence', '')}")
        return "；".join(parts)

    # ================= 主入口 =================
    def execute(self, company, ctx):
        entries, vectors = self._load()
        if not entries or vectors is None:
            ctx.chunks = []
            return ctx

        profile = self._profile_text(ctx)
        if not profile:
            ctx.chunks = []
            return ctx

        query_vec = embed_one(profile)
        if query_vec.shape[0] != vectors.shape[1]:
            _logger.warning("向量维度不匹配：query=%s vs 库=%s，请用相同 embedding 后端重建（python build_chunk_index.py）",
                            query_vec.shape[0], vectors.shape[1])
            ctx.chunks = []
            return ctx

        scores = chunk_store.cosine_scores(query_vec, vectors)
        order = sorted(range(len(entries)), key=lambda i: -scores[i])

        # 排除目标公司自身（先例 = 他人；自身段落已被规则风险/证据覆盖）
        target = str(ctx.company)
        chunks = []
        for i in order:
            if len(chunks) >= self.top_k:
                break
            e = entries[i]
            if str(e.get("company", "")) == target:
                continue
            chunks.append({
                "chunk_id": e.get("chunk_id"),
                "company": e.get("company"),
                "publish_date": e.get("publish_date"),
                "announcement_type": e.get("announcement_type"),
                "text": (e.get("text") or "")[:400],
                "similarity": round(float(scores[i]), 4),
            })
        ctx.chunks = chunks
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.chunk_retriever）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context

    ctx = Context(company="000004.SZ")
    ctx.semantic.risk_factors = [
        {"category": "商誉减值", "description": "计提商誉减值损失4.5亿"},
        {"category": "收入确认", "description": "营收同比下滑42%"},
    ]
    ctx.financial.anomaly_list = [
        {"label_ref": "盈利质量", "evidence": "净利润与经营现金流均为负"},
    ]
    agent = ChunkRetrieverAgent()
    agent.execute("000004.SZ", ctx)
    print(f"召回 {len(ctx.chunks)} 个相似问询段落：")
    for c in ctx.chunks:
        print(f"  [{c['similarity']}] {c['company']} | {c['publish_date']} | {c['text'][:50]}")
