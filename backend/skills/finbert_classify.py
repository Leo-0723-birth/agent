#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: finbert_classify —— FinBERT 风险粗分类（公告/文本预筛）
===============================================================
用 valuesimplex-ai-lab/FinBERT2-base（中文金融预训练 RoBERTa）对文本做
"风险关注点粗分类 + 风险相关性打分"，作为 LLM 精细抽取的【门控】信号。

与 LLM 的分工：
  - FinBERT: 快速、本地、免费地把文本归入风险类别（粗分类/预筛），输出相似度分数
  - LLM    : 只对"通过门控"的文本做细粒度要素抽取（描述/证据/严重度）

模型：valuesimplex-ai-lab/FinBERT2-base（首次加载从 HuggingFace 下载约 500MB，
国内默认走 hf-mirror.com，可用环境变量 HF_ENDPOINT 覆盖）。
迁移自：公告研读agents/finbert_client.py。
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass

# 国内默认走镜像，可被 HF_ENDPOINT 环境变量覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

MODEL_ID = "valuesimplex-ai-lab/FinBERT2-base"

# 监管关注点标签体系（风险原型描述，用于相似度粗分类）
RISK_CATEGORIES = {
    "财务异常": "营业收入、净利润、毛利率异常波动，业绩大幅下滑或亏损，现金流与利润背离",
    "披露矛盾": "不同公告或报告之间数据口径不一致，前后表述矛盾，信息披露不完整或更正",
    "关联交易": "关联方交易定价公允性存疑，关联交易金额异常或频繁",
    "担保事项": "对外担保余额过大、违规担保、担保逾期",
    "资金占用": "大股东或关联方非经营性占用上市公司资金",
    "业绩预告偏差": "业绩预告与实际情况差异较大，业绩变脸、下修",
    "会计处理争议": "会计政策或估计变更、商誉减值、收入确认、资产减值等会计处理存在争议",
    "公司治理": "内部控制缺陷、股权质押、董监高变动、诉讼仲裁、违规处罚",
    "并购重组": "并购标的估值、业绩承诺、重组进展存疑",
    "交易异动": "股票交易异常波动、股价大幅波动、成交量异常",
}


def _mean_pooling(model_output, attention_mask):
    """对 last_hidden_state 做 attention 加权的平均池化，得到句向量。"""
    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask).sum(1) / input_mask.sum(1).clamp(min=1e-9)


class FinBERTClient:
    """FinBERT2-base 封装：文本编码 + 风险粗分类。"""

    def __init__(self, model_id=MODEL_ID, max_length=512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.max_length = max_length
        print(f"[FinBERT] 加载模型 {model_id} ...（首次运行需下载权重）")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.category_names = list(RISK_CATEGORIES.keys())
        self.category_vecs = self.encode(list(RISK_CATEGORIES.values()))
        print(f"[FinBERT] 加载完成，风险类别数={len(self.category_names)}")

    def encode(self, texts):
        """把若干文本编码成 L2 归一化的向量 (n, dim)。空文本返回零向量。"""
        clean = [t if (t and t.strip()) else " " for t in texts]
        enc = self.tokenizer(
            clean, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        with self.torch.no_grad():
            out = self.model(**enc)
        emb = _mean_pooling(out, enc["attention_mask"])
        emb = self.torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.numpy()

    def classify(self, text, top_k=3):
        """对单段文本做风险粗分类，返回 top_k 个类别及相似度分数。"""
        if not text or not text.strip():
            return {"categories": [], "top_category": None, "max_score": 0.0}
        vec = self.encode([text])[0]
        sims = self.category_vecs @ vec
        order = np.argsort(-sims)
        categories = [
            {"category": self.category_names[i], "score": round(float(sims[i]), 4)}
            for i in order[:top_k]
        ]
        return {
            "categories": categories,
            "top_category": categories[0]["category"] if categories else None,
            "max_score": categories[0]["score"] if categories else 0.0,
        }


_SHARED_CLIENT: "FinBERTClient | None" = None


def get_finbert_client() -> "FinBERTClient":
    """进程级共享客户端：模型约 400MB，多 Agent 实例间必须复用一次加载。"""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = FinBERTClient()
    return _SHARED_CLIENT
