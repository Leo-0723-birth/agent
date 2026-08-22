#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: embedding —— 统一 Embedding 入口（正式版）
=============================================

后端：
- bge      : BAAI/bge-large-zh-v1.5，支持 CUDA + 批量编码
- fallback : 字符 bigram 哈希 TF，仅用于离线演示

设计原则：
1. 当 EMBEDDING_BACKEND=bge 时，BGE 加载/推理失败直接报错，不再静默退回 fallback；
   避免案例库使用 1024 维 BGE、查询却误退回 65536 维 fallback 的隐蔽错误。
2. 自动优先使用 CUDA；可通过 EMBEDDING_DEVICE=cpu 强制 CPU。
3. BGE 使用 CLS pooling + L2 normalize。
4. 支持批量编码，默认 batch_size=32。
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from ..config import EMBEDDING_BACKEND, EMBEDDING_MODEL

_BGE = {"tokenizer": None, "model": None, "device": None}
_FALLBACK_DIM = 1 << 16  # 65536

# BGE 中文检索任务常用 query instruction。
# 案例库文档向量不加；查询画像可选择加。
_BGE_QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION",
    "为这个句子生成表示以用于检索相关文章："
)


def _resolve_device() -> str:
    requested = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if requested:
        return requested

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _bge_load():
    if _BGE["model"] is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = _resolve_device()
        print(f"[embedding] 加载 BGE 模型: {EMBEDDING_MODEL}")
        print(f"[embedding] device = {device}")

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        model.eval()
        model.to(device)

        _BGE["tokenizer"] = tokenizer
        _BGE["model"] = model
        _BGE["device"] = device

        if device.startswith("cuda"):
            print(f"[embedding] CUDA = {torch.cuda.get_device_name(0)}")

    return _BGE["tokenizer"], _BGE["model"], _BGE["device"]


def _bge_embed(
    texts: list[str],
    batch_size: int = 32,
    is_query: bool = False,
) -> np.ndarray:

    tokenizer, model, device = _bge_load()

    if is_query:
        texts = [_BGE_QUERY_INSTRUCTION + t for t in texts]

    outputs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            # 4090 上启用 autocast，降低显存与推理耗时。
            if str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**enc)
            else:
                out = model(**enc)

        # BGE 系列使用 CLS token 作为句向量。
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        outputs.append(emb.float().cpu().numpy())

        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 10) == 0:
            print(f"[embedding] {done}/{len(texts)}")

    if not outputs:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(outputs).astype(np.float32)


def _char_bigrams(text: str) -> list[str]:
    s = (text or "").replace(" ", "").replace("\n", "").replace("\r", "")
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [s]
    return [s[i:i + 2] for i in range(n - 1)]


def _fallback_embed(texts: list[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), _FALLBACK_DIM), dtype=np.float32)

    for i, t in enumerate(texts):
        for g in _char_bigrams(t):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % _FALLBACK_DIM
            vecs[i, h] += 1.0

        if vecs[i].sum() > 0:
            vecs[i] = 1.0 + np.log1p(vecs[i])
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm

    return vecs


def embed(
    texts: Iterable[str],
    *,
    batch_size: int | None = None,
    is_query: bool = False,
) -> np.ndarray:
    """
    批量编码文本，返回 L2 归一化矩阵 (n, dim)。

    当 EMBEDDING_BACKEND=bge 时：
    - BGE 失败直接抛出异常；
    - 不再自动退回 fallback。
    """
    texts = [str(t) if t else " " for t in texts]
    batch_size = int(
        batch_size or os.getenv("EMBEDDING_BATCH_SIZE", "32")
    )

    if EMBEDDING_BACKEND == "bge":
        return _bge_embed(
            texts,
            batch_size=batch_size,
            is_query=is_query,
        )

    if EMBEDDING_BACKEND == "fallback":
        return _fallback_embed(texts)

    raise ValueError(
        f"未知 EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}，"
        "仅支持 'bge' 或 'fallback'"
    )


def embed_one(text: str, *, is_query: bool = False) -> np.ndarray:
    return embed([text], is_query=is_query)[0]
