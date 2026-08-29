#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 训练同口径实时上游的子进程 worker（低内存模式）。

背景：8GB 机器上服务进程已驻留 FinBERT2+BGE（~1.7GB 模型权重）；
若 worker 再加载 BGE+reranker+finbert-tone（~4.6GB），合计超出物理
内存，Windows 原生层随机段错误（exit 139 / 0xC0000005，无法捕获）。

协议：父进程用已驻留的共享 BGE 预计算全部向量，worker 只加载
reranker + finbert-tone（~2.6GB 权重），峰值内存 ~3.3GB。
stdin 收 JSON {"docs": [{"announcement","chunk_texts","chunk_vectors"}...],
"query_vectors": [...], "company_code": "..."}，
stdout 输出 JSON {"rows": [...], "audit": {...}}；
模型加载期的 transformers/tqdm 打印全部重定向到 stderr。
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# torch 线程数压到 2：高线程数会显著放大本机的原生崩溃概率
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
# 低内存模式默认 bf16 驻留 reranker（省 1.1G 提交内存）
os.environ.setdefault("F1_RERANK_DTYPE", "bf16")


def main() -> int:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    documents = request.get("docs") or []
    query_vectors = request.get("query_vectors") or []
    company_code = request.get("company_code") or ""

    stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from backend.skills.fullrun_online_semantics import (
                FullRunOnlineSemanticPipeline,
            )

            rows, audit = FullRunOnlineSemanticPipeline().analyze_precomputed(
                documents, company_code, query_vectors
            )
    except Exception as exc:  # 崩溃/异常都要让调用方拿到结构化失败
        import traceback

        traceback.print_exc(file=sys.stderr)
        result = {
            "rows": [],
            "audit": {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "document_count": len(documents),
            },
        }
    else:
        result = {"rows": rows, "audit": audit}

    stdout.write(json.dumps(result, ensure_ascii=False))
    stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
