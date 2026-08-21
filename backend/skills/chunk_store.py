#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: chunk_store —— chunk 级向量索引（监管问询段落召回）
====================================================================
替代/补充 document 级 case_vectors.npy：从 inquiry_embedding_index.jsonl
（Chroma 的源头，411,898 段 chunk 文本 + 元数据）重建 chunk 级 BGE 向量，
按段落粒度召回「最相似问询段落」（用于证据 / 先例定位，比文档级更细）。

持久化位置（config）：
  CHUNK_DB_PATH  = data/chunk_db.json      ← [{chunk_id, company, publish_date, text, ...}]
  CHUNK_VEC_PATH = data/chunk_vectors.npy  ← (n_chunks, 1024) BGE 向量（行对齐）

说明：
  - 原 Chroma HNSW 索引（index_metadata.pickle）版本不兼容（旧版 pickle，向量未随附），
    故此处直接从源头 JSONL 重嵌（build_chunk_index.py），不依赖 chromadb 运行时。
  - 接口对齐 vector_store：load / cosine_scores，可互换。
"""
import json
from pathlib import Path

import numpy as np

from ..config import CHUNK_DB_PATH, CHUNK_VEC_PATH


def save(entries, vectors, db_path=None, vec_path=None):
    """写入 chunk 索引（元数据 JSON + 向量 npy）。返回 (db_path, vec_path)。"""
    db_path = Path(db_path or CHUNK_DB_PATH)
    vec_path = Path(vec_path or CHUNK_VEC_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    np.save(vec_path, np.asarray(vectors, dtype=np.float32))
    return db_path, vec_path


def load(db_path=None, vec_path=None):
    """读取 chunk 索引。返回 (entries, vectors)；不存在返回 ([], None)。"""
    db_path = Path(db_path or CHUNK_DB_PATH)
    vec_path = Path(vec_path or CHUNK_VEC_PATH)
    if not db_path.exists() or not vec_path.exists():
        return [], None
    entries = json.loads(db_path.read_text(encoding="utf-8"))
    vectors = np.load(vec_path)
    return entries, vectors


def cosine_scores(query_vec, vectors):
    """查询向量 vs 库向量（均 L2 归一化），返回余弦相似度数组。"""
    if vectors is None or len(vectors) == 0:
        return np.zeros(0)
    q = np.asarray(query_vec, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    return vectors @ q
