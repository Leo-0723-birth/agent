from __future__ import annotations

import json
import csv
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_OUTPUT = PROJECT_ROOT / "backend" / "data" / "modeling" / "output"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "output"


@st.cache_data(show_spinner=False)
def load_model_summary() -> dict:
    path = MODELING_OUTPUT / "model_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_risk_ranking(window: int = 60, limit: int = 10) -> pd.DataFrame:
    path = MODELING_OUTPUT / f"risk_rank_{int(window)}d.csv"
    return pd.read_csv(path).head(limit)


@st.cache_data(show_spinner=False)
def dataset_shape() -> tuple[int, int]:
    path = PROJECT_ROOT / "backend" / "data" / "modeling" / "processed_dataset.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = len(next(reader))
        rows = sum(1 for _ in reader)
    return rows, columns


@st.cache_data(show_spinner=False)
def load_offline_context(company: str = "000063.SZ") -> dict:
    manifest_path = OUTPUT_DIR / "reports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = company.upper().replace(".", "_")
    entry = next((item for item in manifest if str(item.get("company", "")).upper().replace(".", "_") == normalized and (OUTPUT_DIR / "reports" / str(item.get("json_file", ""))).is_file()), None)
    if not entry:
        raise FileNotFoundError(f"未找到 {company} 的离线报告")
    report_path = OUTPUT_DIR / "reports" / entry["json_file"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    md_path = OUTPUT_DIR / "reports" / str(entry.get("md_file", ""))
    profile = report.get("profile", {}) or {}
    semantic = report.get("semantic", {}) or {}
    scorecard = report.get("scorecard", {}) or {}
    return {
        "company": report.get("company", ""), "name": report.get("name", ""),
        "window": report.get("window", 60), "as_of": report.get("as_of", ""),
        "prediction": scorecard, "financial": report.get("financial", {}) or {},
        "semantic": {"stats": {"announcement_count": semantic.get("announcement_count", profile.get("announcement_count", 0))}, "risk_factors": semantic.get("risk_factors", []) or [], "data_quality": {"offline_snapshot_used": True}},
        "cases": report.get("similar_cases", []) or [], "attribution": report.get("attribution", {}) or {},
        "trace_log": report.get("trace_log", []) or [], "report": {"json": report, "markdown": md_path.read_text(encoding="utf-8") if md_path.exists() else ""},
        "snapshot": {"mode": "offline", "label": "离线演示快照"},
    }
