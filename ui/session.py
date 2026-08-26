from __future__ import annotations

from datetime import date

import streamlit as st


PAGE_STATE_KEYS = (
    "announcement_analysis",
    "financial_analysis",
    "prediction_analysis",
    "case_analysis",
    "attribution_analysis",
    "report_analysis",
)


def publish_analysis(result: dict, *, source: str) -> None:
    """Publish one complete orchestration result for every Agent page."""
    for key in PAGE_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["dashboard_result"] = result
    st.session_state["active_analysis"] = result
    st.session_state["active_analysis_source"] = source
    st.session_state["active_analysis_revision"] = int(
        st.session_state.get("active_analysis_revision", 0)
    ) + 1


def active_analysis() -> dict | None:
    result = st.session_state.get("active_analysis") or st.session_state.get("dashboard_result")
    return result if isinstance(result, dict) else None


def hydrate_page_state(key: str) -> dict | None:
    """Use a page-local rerun when present; otherwise reuse the main Agent result."""
    local = st.session_state.get(key)
    if isinstance(local, dict):
        return local
    shared = active_analysis()
    if shared:
        st.session_state[key] = shared
        return shared
    return None


def restore_page_from_active(key: str) -> dict | None:
    shared = active_analysis()
    if shared:
        st.session_state[key] = shared
    return shared


def active_company(default: str = "000004.SZ") -> str:
    result = active_analysis() or {}
    return str(result.get("company") or default)


def active_as_of(default: date | None = None) -> date:
    result = active_analysis() or {}
    raw = str(result.get("as_of") or "")
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return default or date.today()


def shared_context_caption() -> str | None:
    result = active_analysis()
    if not result:
        return None
    company = result.get("company") or "—"
    name = result.get("name") or ""
    as_of = result.get("as_of") or "—"
    source = st.session_state.get("active_analysis_source", "主控 Agent")
    display = f"{name}（{company}）" if name else str(company)
    return f"已同步主控结果：{display} · 数据截止 {as_of} · 来源 {source}"
