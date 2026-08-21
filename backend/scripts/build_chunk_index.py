#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 chunk 级向量索引（监管问询段落召回，§七 可选升级）
========================================================
从 inquiry_embedding_index.jsonl（Chroma 的源头，411,898 段 chunk 文本 + 元数据）
重嵌 BGE 向量，产出 data/chunk_db.json + data/chunk_vectors.npy，供 chunk_retriever
按段落粒度召回「最相似问询段落」。

背景：原 Chroma HNSW 索引（index_metadata.pickle）是旧版 chromadb pickle，且向量未随附，
chromadb 1.5.x 无法加载（"Error loading hnsw index"），故直接从源头 JSONL 重嵌，零 chromadb 依赖。

用法：
    python build_chunk_index.py --limit 2000        # 快速验证（约 2000 段，本机实测约 15 分钟）
    python build_chunk_index.py                     # 全量（411,898 段，CPU 较慢，视机器而定）
    python build_chunk_index.py --fp16              # 全量 + float16 存储（省一半磁盘）

输出：data/chunk_db.json（元数据）+ data/chunk_vectors.npy（BGE 1024 维）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import CHUNK_INDEX_JSONL, CHUNK_DB_PATH, CHUNK_VEC_PATH
from backend.skills import chunk_store
from backend.skills.embedding import embed

# 只纳入问询函 chunk（对齐 README 口径：回复不混入），其余类型（inquiry_reply/其他）跳过
INCLUDE_TYPES = ("inquiry_letter",)

# 元数据落盘字段（chunk 召回展示所需的最小集合，不存 source_path 等冗余）
KEEP_FIELDS = ["chunk_id", "announcement_id", "company", "publish_date",
               "announcement_type", "text_type", "paragraph_id", "part_index", "text"]


def iter_chunks(path, include_types=INCLUDE_TYPES):
    """流式读取 chunk JSONL，只产出目标类型。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("announcement_type") not in include_types:
                continue
            yield obj


def main():
    parser = argparse.ArgumentParser(description="构建 chunk 级向量索引（监管问询段落召回）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 段（0=全量）")
    parser.add_argument("--fp16", action="store_true", help="向量以 float16 存储（省一半磁盘）")
    parser.add_argument("--out-db", default=None, help="覆盖 chunk_db.json 输出路径")
    parser.add_argument("--out-vec", default=None, help="覆盖 chunk_vectors.npy 输出路径")
    args = parser.parse_args()

    if not Path(CHUNK_INDEX_JSONL).exists():
        print(f"[build_chunk_index] 源 JSONL 不存在：{CHUNK_INDEX_JSONL}")
        sys.exit(1)

    entries = []
    vecs = []
    batch_texts = []
    skipped = 0

    def flush():
        nonlocal batch_texts
        if not batch_texts:
            return
        emb = embed(batch_texts)
        for v in emb:
            vecs.append(v.astype(np.float16 if args.fp16 else np.float32))
        batch_texts.clear()

    print(f"[build_chunk_index] 读取 {CHUNK_INDEX_JSONL.name} ...")
    for i, obj in enumerate(iter_chunks(CHUNK_INDEX_JSONL)):
        if args.limit and i >= args.limit:
            break
        text = obj.get("text") or ""
        if not text.strip():
            skipped += 1
            continue
        entries.append({k: obj.get(k) for k in KEEP_FIELDS})
        batch_texts.append(text)
        if len(batch_texts) >= 64:
            flush()
        if (i + 1) % 20000 == 0:
            print(f"  已处理 {i + 1} 段 ...")
    flush()

    if not entries:
        print("[build_chunk_index] 无可用 chunk（源文件为空或类型不符）")
        sys.exit(1)

    vectors = np.vstack(vecs)
    print(f"[build_chunk_index] 完成 {len(entries)} 段（跳过空文本 {skipped}），"
          f"向量 {vectors.shape} {vectors.dtype}")

    db_path, vec_path = chunk_store.save(entries, vectors,
                                         db_path=args.out_db, vec_path=args.out_vec)
    print(f"[build_chunk_index] 已写出：")
    print(f"  {db_path}")
    print(f"  {vec_path}")
    print(f"[build_chunk_index] 提示：运行 python -m backend.agents.chunk_retriever 自测召回")


if __name__ == "__main__":
    main()
