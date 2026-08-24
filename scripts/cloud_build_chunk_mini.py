#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云上最简版：chunk 库全量重建（单文件，无需仓库代码）

用法：
    python cloud_build_chunk_mini.py <inquiry_embedding_index.jsonl路径> [--fp16]

依赖：pip install transformers torch numpy
BGE 权重首次自动下载（HF_ENDPOINT 默认 hf-mirror.com，可覆盖）

产物（当前目录）：
    chunk_db.json      段落元数据（JSON 数组，字段与项目一致）
    chunk_vectors.npy  向量（默认 float32；--fp16 则 float16，省一半磁盘）

与项目兼容性：产物可直接替换 backend/data/vector_db/chunk_db.json + chunk_vectors.npy。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache"))

import numpy as np

INCLUDE_TYPES = ("inquiry_letter",)
KEEP_FIELDS = ["chunk_id", "announcement_id", "company", "publish_date",
               "announcement_type", "text_type", "paragraph_id", "part_index", "text"]
BATCH = 64


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    print("[1/3] 加载 BGE-large-zh-v1.5 ...")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to("cuda")
        print(f"    使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    使用 CPU（较慢，约 1~2 小时）")
    model.eval()
    return tok, model, torch, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.jsonl):
        print(f"[错误] 文件不存在: {args.jsonl}")
        sys.exit(1)

    tok, model, torch, device = load_model()

    print(f"[2/3] 读取 {args.jsonl} 中的问询函段落 ...")
    entries, texts = [], []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("announcement_type") not in INCLUDE_TYPES:
                continue
            text = obj.get("text") or ""
            if not text.strip():
                continue
            entries.append({k: obj.get(k) for k in KEEP_FIELDS})
            texts.append(text)
    print(f"    共 {len(entries)} 段（inquiry_letter）")

    print("[3/3] BGE 嵌入（GPU 加速）...")
    vecs, t0 = [], time.time()
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu()
        vecs.append(emb.numpy().astype(np.float16 if args.fp16 else np.float32))
        if (i // BATCH) % 200 == 0:
            print(f"    已嵌入 {i}/{len(texts)} 段（{time.time()-t0:.0f}s）", flush=True)

    vectors = np.vstack(vecs)
    json.dump(entries, open("chunk_db.json", "w", encoding="utf-8"), ensure_ascii=False)
    np.save("chunk_vectors.npy", vectors)

    companies = {e.get("company") for e in entries if e.get("company")}
    print(f"完成！{len(entries)} 段 | 向量 {vectors.shape} {vectors.dtype} | 覆盖公司 {len(companies)}")
    print("产物：chunk_db.json + chunk_vectors.npy（下载回本地替换 backend/data/vector_db/ 下同名文件）")


if __name__ == "__main__":
    main()
