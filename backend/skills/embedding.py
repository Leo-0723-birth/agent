#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: embedding —— 统一 Embedding 入口
=======================================
输入：文本列表
输出：L2 归一化向量 (n, dim)

两种后端（config.EMBEDDING_BACKEND）：
  - bge     : transformers 加载中文 BGE 模型（质量高，首次需下载权重，走 hf-mirror）
  - fallback: 字符 bigram 哈希 TF 向量（零依赖零下载，演示/离线可用；质量弱于 BGE）
加载失败自动退回 fallback，保证流程不断。

生产建议：EMBEDDING_BACKEND=bge，EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5（或 stella-base-zh-v3）。
"""
import hashlib
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 权重下载到项目内（backend/models/embedding/），随项目迁移，不占用户主目录
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "embedding"))

from ..config import EMBEDDING_BACKEND, EMBEDDING_MODEL

_BGE = {"tokenizer": None, "model": None}
_FALLBACK_DIM = 1 << 16  # 65536 维哈希空间


# ================= BGE 后端 =================
def _bge_local_snapshot_dir():
    """定位本地完整的 BGE snapshot 目录（含 config.json + 权重文件）。

    HF 缓存元数据可能损坏（snapshot 被拆散/缺 refs），from_pretrained 的缓存解析
    会失败并尝试联网（无外网时重试风暴卡死全流程）。直接指向快照目录可完全绕开。
    """
    base = Path(__file__).resolve().parent.parent / "models" / "embedding" \
        / "hub" / "models--BAAI--bge-large-zh-v1.5" / "snapshots"
    if not base.exists():
        return None
    for snap in sorted(base.iterdir()):
        if not snap.is_dir() or not (snap / "config.json").exists():
            continue
        weights = list(snap.glob("*.safetensors")) + list(snap.glob("pytorch_model.bin"))
        if weights:
            return str(snap)
    return None


def _bge_load():
    if _BGE["model"] is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        local_dir = _bge_local_snapshot_dir()
        if local_dir:
            print(f"[embedding] 加载 BGE 模型（本地快照: {Path(local_dir).parent.name}）")
            _BGE["tokenizer"] = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            _BGE["model"] = AutoModel.from_pretrained(local_dir, local_files_only=True)
        else:
            print(f"[embedding] 本地无 BGE 快照，尝试联网下载（HF_ENDPOINT={os.getenv('HF_ENDPOINT', '')}）")
            _BGE["tokenizer"] = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            _BGE["model"] = AutoModel.from_pretrained(EMBEDDING_MODEL)
        # 设备：优先 CUDA（云 GPU 环境自动加速），否则 CPU
        _BGE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if _BGE["device"] == "cuda":
            _BGE["model"] = _BGE["model"].to("cuda")
            print(f"[embedding] 使用 GPU: {torch.cuda.get_device_name(0)}")
        _BGE["model"].eval()
    return _BGE["tokenizer"], _BGE["model"]


def _bge_embed(texts):
    import torch
    tok, model = _bge_load()
    enc = tok(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    device = _BGE.get("device", "cpu")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


# ================= fallback 后端（字符 bigram 哈希 TF） =================
def _char_bigrams(text):
    """中文按字切分，生成字符 bigram 集合（英文/数字连在一起算整体）。"""
    s = (text or "").replace(" ", "").replace("\n", "").replace("\r", "")
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [s]
    return [s[i:i + 2] for i in range(n - 1)]


def _fallback_embed(texts):
    vecs = np.zeros((len(texts), _FALLBACK_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for g in _char_bigrams(t):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % _FALLBACK_DIM
            vecs[i, h] += 1.0
        if vecs[i].sum() > 0:
            vecs[i] = 1.0 + np.log1p(vecs[i])          # sublinear 缩放
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] /= n
    return vecs


# ================= 统一入口 =================
def embed(texts, allow_fallback=True):
    """把文本编码为向量；生产特征可禁止静默降级以避免混用向量空间。"""
    texts = [t if t else " " for t in texts]
    if EMBEDDING_BACKEND == "bge":
        try:
            return _bge_embed(texts)
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"BGE embedding failed: {e}") from e
            print(f"[embedding] BGE 加载失败，退回 fallback: {e}")
    return _fallback_embed(texts)


def embed_one(text):
    """单条文本编码。"""
    return embed([text])[0]
