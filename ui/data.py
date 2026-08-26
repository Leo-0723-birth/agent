from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "output"
MODELING_DIR = PROJECT_ROOT / "backend" / "data" / "modeling"
MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "predictor"


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def dataset_shape() -> tuple[int, int]:
    path = MODELING_DIR / "processed_dataset.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        columns = len(next(reader))
        rows = sum(1 for _ in reader)
    return rows, columns


@st.cache_data(show_spinner=False)
def load_model_summary() -> dict:
    path = MODELING_DIR / "output" / "model_summary.json"
    return dict(load_json(str(path)))


@st.cache_data(show_spinner=False)
def load_model_manifest() -> dict:
    path = MODEL_DIR / "models_manifest.json"
    return dict(load_json(str(path)))


@st.cache_data(show_spinner=False)
def load_risk_ranking(window: int = 60, limit: int = 10) -> pd.DataFrame:
    path = MODELING_DIR / "output" / f"risk_rank_{int(window)}d.csv"
    return pd.read_csv(path).head(limit)


@st.cache_data(show_spinner=False)
def latest_offline_report(company: str = "000004.SZ") -> dict:
    manifest_path = OUTPUT_DIR / "reports" / "manifest.json"
    manifest = load_json(str(manifest_path))
    normalized = company.upper().replace(".", "_")
    entry = next(
        (item for item in manifest if str(item.get("company", "")).upper().replace(".", "_") == normalized),
        None,
    )
    if not entry:
        raise FileNotFoundError(f"未找到 {company} 的离线报告")
    report_path = OUTPUT_DIR / "reports" / entry["json_file"]
    report = dict(load_json(str(report_path)))
    md_name = entry.get("md_file", "")
    md_path = OUTPUT_DIR / "reports" / md_name
    report["_markdown"] = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    report["_snapshot"] = {
        "mode": "offline",
        "label": "离线演示快照",
        "generated_at": report.get("generated_at", entry.get("generated_at", "")),
        "source_data_mode": report.get("data_source", "unknown"),
    }
    return report


def report_to_context(report: dict) -> dict:
    """Convert an archived report into the same display contract as a live Context."""
    profile = report.get("profile", {}) or {}
    semantic = report.get("semantic", {}) or {}
    return {
        "company": report.get("company", ""),
        "name": report.get("name", ""),
        "window": report.get("window", 60),
        "as_of": report.get("as_of", ""),
        "prediction": report.get("scorecard", {}) or {},
        "financial": report.get("financial", {}) or {},
        "semantic": {
            "stats": {
                "announcement_count": semantic.get("announcement_count", profile.get("announcement_count", 0)),
                "risk_factor_count": semantic.get("risk_factor_count", 0),
            },
            "risk_factors": semantic.get("risk_factors", []) or [],
            "data_quality": {"offline_snapshot_used": True, "snapshot_as_of": report.get("as_of", "")},
        },
        "cases": report.get("similar_cases", []) or [],
        "attribution": report.get("attribution", {}) or {},
        "trace_log": report.get("trace_log", []) or [],
        "report": {"json": report, "markdown": report.get("_markdown", "")},
        "snapshot": report.get("_snapshot", {}),
        "profile": profile,
    }


@st.cache_data(show_spinner=False)
def load_offline_context(company: str = "000004.SZ") -> dict:
    return report_to_context(latest_offline_report(company))

