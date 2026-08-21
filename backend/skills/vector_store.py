#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: vector_store —— 简单 numpy 向量库（案例库持久化 + 余弦检索）
====================================================================
说明：演示用轻量实现（JSON 元数据 + npy 向量，零额外依赖）；
生产可换 ChromaDB/Milvus，接口对齐 save/load/cosine_scores 即可。

持久化位置（config）：
  CASE_DB_PATH  = backend/data/vector_db/case_db.json
  CASE_VEC_PATH = backend/data/vector_db/case_vectors.npy
"""
import json
from pathlib import Path

import numpy as np

from ..config import CASE_DB_PATH, CASE_VEC_PATH, CASE_META_PATH


def save(entries, vectors, db_path=None, vec_path=None):
    """写入案例库（元数据 JSON + 向量 npy）。返回 (db_path, vec_path)。"""
    db_path = Path(db_path or CASE_DB_PATH)
    vec_path = Path(vec_path or CASE_VEC_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(vec_path, np.asarray(vectors, dtype=np.float32))
    return db_path, vec_path


def load(db_path=None, vec_path=None):
    """读取案例库。返回 (entries, vectors)；不存在返回 ([], None)。"""
    db_path = Path(db_path or CASE_DB_PATH)
    vec_path = Path(vec_path or CASE_VEC_PATH)
    if not db_path.exists() or not vec_path.exists():
        return [], None
    entries = json.loads(db_path.read_text(encoding="utf-8"))
    vectors = np.load(vec_path)
    return entries, vectors


def load_meta(db_path=None):
    """读取案例库构建元数据（embedding 后端/维度，检索一致性校验用）。"""
    path = Path(db_path or CASE_META_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cosine_scores(query_vec, vectors):
    """查询向量 vs 库向量（均为 L2 归一化），返回余弦相似度数组。"""
    if vectors is None or len(vectors) == 0:
        return np.zeros(0)
    q = np.asarray(query_vec, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    return vectors @ q
