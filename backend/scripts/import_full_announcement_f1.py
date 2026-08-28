#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把全量公告流水线的公司季度 PCA50 特征接入模型训练原料。

只导入 F1 特征和主键，不导入上游 ``split`` / ``y_inquiry_next``，避免标签泄漏。
F2 的公司季度键仍是训练数据集的权威骨架；全量 F1 中无法匹配 F2 的额外记录
会在审计清单中计数，但不会被强行加入训练样本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "backend" / "data" / "modeling" / "raw"
SOURCE_FILENAME = "company_quarter_pca50.parquet"
OUTPUT_FILENAME = "F1_announcement_semantic_features.parquet"
MANIFEST_FILENAME = "f1_full_run_manifest.json"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["company_code"] = result["company_code"].astype(str).str.strip()
    result["report_period"] = (
        result["report_period"].astype(str).str.replace(".0", "", regex=False)
    )
    return result


def import_full_f1(source_dir: Path, raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    source_dir = Path(source_dir).expanduser().resolve()
    raw_dir = Path(raw_dir).expanduser().resolve()
    source_path = source_dir / SOURCE_FILENAME
    f2_path = raw_dir / "F2_financial_anomaly.csv"

    if not source_path.is_file():
        raise FileNotFoundError(f"找不到全量 F1 文件: {source_path}")
    if not f2_path.is_file():
        raise FileNotFoundError(f"找不到训练骨架 F2: {f2_path}")

    source = _normalise_keys(pd.read_parquet(source_path))
    pc_cols = [f"PC{i:02d}" for i in range(1, 51)]
    required = {"company_code", "report_period", *pc_cols}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"全量 F1 缺少必需字段: {missing}")
    if source.duplicated(["company_code", "report_period"]).any():
        count = int(source.duplicated(["company_code", "report_period"], keep=False).sum())
        raise ValueError(f"全量 F1 主键不唯一，重复行数: {count}")

    values = source[pc_cols].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"全量 F1 含 NaN/Inf，异常单元格: {bad}")

    f2 = _normalise_keys(
        pd.read_csv(
            f2_path,
            encoding="utf-8-sig",
            dtype={"company_code": str, "report_period": str},
        )
    )
    if f2.duplicated(["company_code", "report_period"]).any():
        raise ValueError("F2 训练骨架主键不唯一，停止覆盖 F1")

    keys = ["company_code", "report_period"]
    audit = f2[keys + ["split"]].merge(
        source[keys + (["split"] if "split" in source.columns else []) +
               (["has_announcement"] if "has_announcement" in source.columns else [])],
        on=keys,
        how="left",
        suffixes=("_f2", "_f1"),
        indicator=True,
    )
    missing_from_source = int((audit["_merge"] != "both").sum())
    if missing_from_source:
        raise ValueError(f"全量 F1 未覆盖 {missing_from_source} 条 F2 训练样本，停止覆盖")

    source_keys = pd.MultiIndex.from_frame(source[keys])
    f2_keys = pd.MultiIndex.from_frame(f2[keys])
    extra_source_rows = int((~source_keys.isin(f2_keys)).sum())

    split_mismatch = 0
    if "split_f1" in audit.columns:
        split_mismatch = int(
            (audit["split_f2"].astype(str) != audit["split_f1"].astype(str)).sum()
        )
        if split_mismatch:
            raise ValueError(f"全量 F1 与 F2 的 split 不一致: {split_mismatch} 条")

    rename = {col: f"announcement_semantic_{idx:03d}" for idx, col in enumerate(pc_cols)}
    output = source[keys + pc_cols].rename(columns=rename)
    output = output.rename(columns={"company_code": "stock_code", "report_period": "T_date"})
    output = output.sort_values(["stock_code", "T_date"], kind="stable").reset_index(drop=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / OUTPUT_FILENAME
    temp_path = raw_dir / f".{OUTPUT_FILENAME}.tmp"
    output.to_parquet(temp_path, index=False)
    temp_path.replace(output_path)

    has_announcement_count = None
    if "has_announcement" in audit.columns:
        has_announcement_count = int(audit["has_announcement"].fillna(False).astype(bool).sum())

    source_manifest = source_dir / "model_manifest.json"
    model_versions = {}
    if source_manifest.is_file():
        raw_models = json.loads(source_manifest.read_text(encoding="utf-8"))
        # 清单只保留可复核的模型标识与 revision，不传播上游机器的绝对路径。
        model_versions = {
            name: {key: value for key, value in details.items()
                   if key in {"repo_id", "revision"}}
            for name, details in raw_models.items()
        }

    variance_path = source_dir / "pca_explained_variance.csv"
    cumulative_explained_variance = None
    if variance_path.is_file():
        variance = pd.read_csv(variance_path)
        for candidate in ("cumulative_explained_variance", "cumulative_variance", "cumulative"):
            if candidate in variance.columns and len(variance):
                cumulative_explained_variance = float(variance[candidate].iloc[-1])
                break

    try:
        manifest_output_file = str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        manifest_output_file = str(output_path)

    manifest = {
        "schema_version": "f1-full-run-import-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_directory": str(source_dir),
        "source_file": SOURCE_FILENAME,
        "source_sha256": _sha256(source_path),
        "output_file": manifest_output_file,
        "output_sha256": _sha256(output_path),
        "feature_mapping": {
            col: rename[col] for col in pc_cols
        },
        "quality": {
            "source_rows": int(len(source)),
            "f2_skeleton_rows": int(len(f2)),
            "f2_rows_covered": int(len(f2) - missing_from_source),
            "missing_from_source": missing_from_source,
            "source_rows_not_in_f2_skeleton": extra_source_rows,
            "split_mismatch": split_mismatch,
            "duplicate_keys": 0,
            "non_finite_feature_cells": 0,
            "f2_rows_with_announcement": has_announcement_count,
            "f2_announcement_coverage_rate": (
                has_announcement_count / len(f2) if has_announcement_count is not None else None
            ),
        },
        "pca": {
            "components": 50,
            "cumulative_explained_variance": cumulative_explained_variance,
        },
        "models": model_versions,
        "leakage_guard": "未导入 source split、y_inquiry_next、industry 等字段；标签由本仓库 inquiry_events.csv 重建。",
    }
    manifest_path = raw_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="导入全量公告公司季度 PCA50 到模型 F1 原料")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=os.getenv("FULL_ANNOUNCEMENT_RUN_DIR"),
        help="全量公告流水线输出目录；也可设置 FULL_ANNOUNCEMENT_RUN_DIR",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()
    if args.source_dir is None:
        parser.error("必须传 --source-dir 或设置 FULL_ANNOUNCEMENT_RUN_DIR")
    manifest = import_full_f1(args.source_dir, args.raw_dir)
    quality = manifest["quality"]
    print(json.dumps({
        "status": "ok",
        "source_rows": quality["source_rows"],
        "f2_rows_covered": quality["f2_rows_covered"],
        "source_rows_not_in_f2_skeleton": quality["source_rows_not_in_f2_skeleton"],
        "features": manifest["pca"]["components"],
        "output_file": manifest["output_file"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
