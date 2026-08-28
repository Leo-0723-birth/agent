#!/usr/bin/env python
"""Download and verify the pinned NLP models used by announcement processing.

Model weights are intentionally not committed to Git.  Every developer or
deployment environment should run this script once and share a persistent
Hugging Face cache directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS: dict[str, dict[str, str]] = {
    "bge": {
        "repo_id": "BAAI/bge-large-zh-v1.5",
        "revision": "79e7739b6ab944e86d6171e44d24c997fc1e0116",
        "purpose": "公告与风险主题向量编码",
    },
    "reranker": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "purpose": "候选证据精排",
    },
    "finbert": {
        "repo_id": "yiyanghkust/finbert-tone-chinese",
        "revision": "e91b1a3af10e1e8c9c03429d3cd7d5e9a1c8000d",
        "purpose": "中文金融文本分类门控",
    },
}


def _weight_files(snapshot: Path) -> list[Path]:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return [path for pattern in patterns for path in snapshot.rglob(pattern)]


def _verify_snapshot(snapshot: Path, spec: dict[str, str]) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"模型快照目录不存在: {snapshot}")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"模型缺少 config.json: {snapshot}")
    weights = _weight_files(snapshot)
    if not weights:
        raise FileNotFoundError(f"模型缺少权重文件: {snapshot}")

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "purpose": spec["purpose"],
        "snapshot": str(snapshot.resolve()),
        "model_type": config.get("model_type", "unknown"),
        "hidden_size": config.get("hidden_size"),
        "weight_files": [path.name for path in weights],
    }


def _cached_snapshot(cache_dir: Path, spec: dict[str, str]) -> Path:
    repo_dir = "models--" + spec["repo_id"].replace("/", "--")
    return cache_dir / "hub" / repo_dir / "snapshots" / spec["revision"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *MODELS),
        default="all",
        help="要处理的模型，默认全部",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face 缓存根目录；省略时使用 huggingface_hub 默认目录",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只校验本地缓存，不访问网络",
    )
    args = parser.parse_args()

    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    results: dict[str, dict[str, Any]] = {}

    if args.verify_only and args.cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        hub_cache = Path(HF_HUB_CACHE)
        cache_root = hub_cache.parent
    else:
        cache_root = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    for name, spec in selected.items():
        if args.verify_only:
            assert cache_root is not None
            snapshot = _cached_snapshot(cache_root, spec)
        else:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    cache_dir=str(cache_root / "hub") if cache_root else None,
                )
            )
        results[name] = _verify_snapshot(snapshot, spec)
        print(f"[OK] {name}: {snapshot}")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
