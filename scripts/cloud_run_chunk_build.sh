#!/usr/bin/env bash
# ============================================================
# 云上 GPU 运行脚本：chunk 库全量重建（覆盖 219 → 1566 家公司）
# 适用：JupyterLab 终端（Linux，有 GPU 最佳）
#
# 用法：
#   bash scripts/cloud_run_chunk_build.sh \
#       /path/to/inquiry_embedding_index.jsonl
# 说明：
#   - 第一个参数 = 云上源 JSONL 路径（必填，或设环境变量 CHUNK_INDEX_JSONL）
#   - 会自动 clone 项目（若当前目录没有 agent/ 仓库）
#   - BGE 权重在云上默认联网下载（HF_ENDPOINT=hf-mirror.com）
#   - 产物：backend/data/vector_db/chunk_db.json + chunk_vectors.npy
# ============================================================
set -e

SRC_JSONL="${1:-$CHUNK_INDEX_JSONL}"
if [ -z "$SRC_JSONL" ]; then
    echo "[错误] 请传入源 JSONL 路径（参数 1 或环境变量 CHUNK_INDEX_JSONL）"
    exit 1
fi
if [ ! -f "$SRC_JSONL" ]; then
    echo "[错误] 源 JSONL 不存在: $SRC_JSONL"
    exit 1
fi
export CHUNK_INDEX_JSONL="$SRC_JSONL"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "========== 0/5 环境检查 =========="
nvidia-smi 2>/dev/null | head -4 || echo "（无 nvidia-smi，将走 CPU，仍可运行但较慢）"
python --version

echo "========== 1/5 准备项目代码 =========="
if [ ! -d agent ]; then
    git clone https://github.com/Leo-0723-birth/agent.git
fi
cd agent
git pull --ff-only || true

echo "========== 2/5 安装依赖 =========="
pip install -q -r requirements.txt
# 云端 torch 若为 CPU 版，下面一行可换装 CUDA 版（按你的 CUDA 版本调整）
# pip install -q torch --index-url https://download.pytorch.org/whl/cu121

echo "========== 3/5 校验 GPU 与嵌入设备 =========="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "========== 4/5 全量重建 chunk 库 =========="
echo "源 JSONL: $CHUNK_INDEX_JSONL"
python -m backend.scripts.build_chunk_index --fp16

echo "========== 5/5 校验产物 =========="
python - <<'PY'
import json, os
import numpy as np
db = json.load(open("backend/data/vector_db/chunk_db.json", encoding="utf-8"))
if isinstance(db, dict):
    db = db.get("chunks", [])
vec = np.load("backend/data/vector_db/chunk_vectors.npy", mmap_mode="r")
companies = {c.get("company") for c in db if c.get("company")}
print(f"chunk 段数: {len(db)}")
print(f"向量形状: {vec.shape} {vec.dtype}")
print(f"覆盖公司数: {len(companies)}（期望接近 1566）")
print("校验:", "✅ 通过" if len(db) > 2000 and len(companies) > 219 else "⚠️ 未达预期，请检查日志")
PY

echo "========== 完成 =========="
echo "请下载回本地并替换："
echo "  backend/data/vector_db/chunk_db.json"
echo "  backend/data/vector_db/chunk_vectors.npy"
