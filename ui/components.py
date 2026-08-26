from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st


def _safe(value) -> str:
    return html.escape(str(value if value is not None else ""))


def render_page_header(title: str, subtitle: str, *, status: str = "", status_kind: str = "offline", metadata: list[str] | None = None) -> None:
    chips = ([f'<span class="risk-chip {_safe(status_kind)}">{_safe(status)}</span>'] if status else [])
    chips.extend(f'<span class="risk-chip">{_safe(item)}</span>' for item in (metadata or []) if item)
    st.html(f'<section class="risk-header"><div class="risk-eyebrow">Regulatory risk intelligence</div><div class="risk-title">{_safe(title)}</div><div class="risk-subtitle">{_safe(subtitle)}</div><div class="risk-meta">{"".join(chips)}</div></section>')


def render_metric_grid(items: list[dict]) -> None:
    cards = []
    for item in items:
        cards.append(f'<div class="risk-metric-card"><div class="risk-metric-label">{_safe(item.get("label"))}</div><div class="risk-metric-value">{_safe(item.get("value"))}<span class="risk-metric-unit">{_safe(item.get("unit"))}</span></div><div class="risk-metric-note">{_safe(item.get("note"))}</div></div>')
    st.html(f'<div class="risk-metric-grid">{"".join(cards)}</div>')


def render_trace(trace: list[dict]) -> None:
    steps = []
    for item in trace or []:
        status = item.get("status") or ("done" if item.get("trace_complete") else "error")
        cls = "" if status == "done" else (" error" if status == "error" else " skip")
        detail = f'{item.get("latency_ms")} ms' if item.get("latency_ms") not in (None, "") else status
        steps.append(f'<div class="risk-trace-step"><b><i class="risk-dot{cls}"></i>{_safe(item.get("agent", "Agent"))}</b><span>{_safe(detail)}</span></div>')
    st.html(f'<div class="risk-trace">{"".join(steps)}</div>')


def evidence_records(result: dict, limit: int = 8) -> list[dict]:
    records = []
    for index, item in enumerate(result.get("semantic", {}).get("risk_factors", []) or []):
        records.append({"id": f'ANN-{str(item.get("risk_id") or index + 1)[:8].upper()}', "severity": int(item.get("severity") or 0), "kind": item.get("taxonomy_l2") or item.get("category") or "公告风险", "title": item.get("announcement_title") or "公告原文证据", "quote": item.get("evidence") or item.get("description") or "", "source_url": item.get("source_url") or "", "page": "announcement"})
    for index, item in enumerate(result.get("financial", {}).get("anomaly_list", []) or []):
        records.append({"id": f"FIN-{index + 1:03d}", "severity": int(item.get("severity") or 0), "kind": item.get("type") or "财务异常", "title": item.get("indicator") or item.get("label_ref") or "财务指标异常", "quote": item.get("evidence") or "", "source_url": "", "page": "financial"})
    return sorted(records, key=lambda item: item["severity"], reverse=True)[:limit]


def render_evidence_cards(records: list[dict]) -> None:
    for item in records:
        level = "高风险" if item["severity"] >= 4 else ("中风险" if item["severity"] >= 2 else "提示")
        links = f'<a href="/{_safe(item["page"])}?evidence={quote(item["id"])}" target="_self">定位系统记录 ↗</a>'
        if item.get("source_url"):
            links += f' <a href="{_safe(item["source_url"])}" target="_blank" rel="noopener">查看公告原文 ↗</a>'
        st.html(f'<article class="risk-evidence {"high" if item["severity"] >= 4 else ""}"><div class="risk-evidence-title"><span>{_safe(item["kind"])} · {_safe(item["id"])}</span><span>{level}</span></div><div class="risk-evidence-meta">{_safe(item["title"])}</div><div class="risk-evidence-quote">{_safe(item["quote"])}</div><div class="risk-links">{links}</div></article>')


def render_source_index(company: str) -> None:
    code = quote(company)
    sources = [("公告原文与片段", "公告研读 Agent", f"/announcement?company={code}"), ("财务指标与计算", "财务异常 Agent", f"/financial?company={code}"), ("相似监管案例", "案例匹配 Agent", f"/case?company={code}"), ("推理与工具日志", "主控 Agent trace", f"/main?company={code}#agent-trace")]
    cards = "".join(f'<a class="risk-source-card" href="{href}" target="_self"><b>{_safe(title)} ↗</b><span>{_safe(detail)}</span></a>' for title, detail, href in sources)
    st.html(f'<div class="risk-source-grid">{cards}</div>')
