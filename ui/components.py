from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st

from backend.skills.announcement_context_filter import contextual_suppression_reason


def _safe(value) -> str:
    return html.escape(str(value if value is not None else ""))


def render_page_header(
    title: str,
    subtitle: str,
    *,
    status: str | None = None,
    status_kind: str = "offline",
    metadata: list[str] | None = None,
) -> None:
    chips = []
    if status:
        chips.append(f'<span class="risk-chip {_safe(status_kind)}">{_safe(status)}</span>')
    chips.extend(f'<span class="risk-chip">{_safe(item)}</span>' for item in (metadata or []) if item)
    st.html(
        f"""
        <section class="risk-header">
          <div class="risk-eyebrow">Regulatory risk intelligence</div>
          <div class="risk-title">{_safe(title)}</div>
          <div class="risk-subtitle">{_safe(subtitle)}</div>
          <div class="risk-meta">{''.join(chips)}</div>
        </section>
        """
    )


def render_metric_grid(items: list[dict]) -> None:
    cards = []
    for item in items:
        cards.append(
            "".join(
                [
                    '<div class="risk-metric-card">',
                    f'<div class="risk-metric-label">{_safe(item.get("label"))}</div>',
                    f'<div class="risk-metric-value">{_safe(item.get("value"))}<span class="risk-metric-unit">{_safe(item.get("unit"))}</span></div>',
                    f'<div class="risk-metric-note">{_safe(item.get("note"))}</div>',
                    "</div>",
                ]
            )
        )
    st.html(f'<div class="risk-metric-grid">{"".join(cards)}</div>')


def render_color_legend(items: list[tuple[str, str]], note: str = "") -> None:
    """Render an explicit swatch-to-label mapping below a chart."""
    entries = "".join(
        f'<span class="risk-color-legend-item"><i class="risk-color-swatch" '
        f'style="--legend-color:{_safe(color)}"></i>{_safe(label)}</span>'
        for label, color in items
    )
    note_html = f'<span class="risk-color-legend-note">{_safe(note)}</span>' if note else ""
    st.html(f'<div class="risk-color-legend">{entries}{note_html}</div>')


def render_trace(trace: list[dict]) -> None:
    steps = []
    for item in trace or []:
        status = item.get("status") or ("done" if item.get("trace_complete") else "error")
        dot = "" if status == "done" else (" error" if status == "error" else " skip")
        latency = item.get("latency_ms")
        detail = f"{latency} ms" if latency not in (None, "") else status
        summary = item.get("output_summary")
        if summary:
            detail += f" · {str(summary)[:42]}"
        steps.append(
            f'<div class="risk-trace-step"><b><i class="risk-dot{dot}"></i>{_safe(item.get("agent", "Agent"))}</b><span>{_safe(detail)}</span></div>'
        )
    st.html(f'<div class="risk-trace">{"".join(steps)}</div>')


def evidence_records(result: dict, limit: int = 8) -> list[dict]:
    records: list[dict] = []
    for index, item in enumerate(result.get("semantic", {}).get("risk_factors", []) or []):
        evidence = str(item.get("evidence") or "").strip()
        method = str(item.get("method") or "")
        agreement = str(item.get("agreement_status") or "")
        assertion_type = str(item.get("assertion_type") or "").lower()
        factual_channel = method == "llm_evidence_validated" or agreement == "rule_llm_agree"
        context_reason = contextual_suppression_reason(
            rule_id=str(item.get("rule_id") or ""),
            label=str(item.get("taxonomy_l2") or item.get("label") or ""),
            text=evidence,
            start=0,
            end=len(evidence),
        ) if evidence else "missing_evidence"
        if (
            not item.get("evidence_valid")
            or not evidence
            or not factual_channel
            or (assertion_type and not assertion_type.startswith("actual_event"))
            or context_reason
        ):
            continue
        verification_label = (
            "规则与 LLM 交叉一致" if agreement == "rule_llm_agree" else "原文事实语境已校验"
        )
        records.append(
            {
                "id": f"ANN-{str(item.get('risk_id') or index + 1)[:8].upper()}",
                "severity": int(item.get("severity") or 0),
                "kind": item.get("taxonomy_l2") or item.get("category") or "公告风险",
                "title": item.get("announcement_title") or "公告原文证据",
                "meta": " · ".join(filter(None, [item.get("announcement_date"), item.get("matched_keyword")])),
                "quote": evidence,
                "issue": f"问题：{item.get('description') or item.get('risk_label') or '公告风险事件'}",
                "keyword": item.get("matched_keyword") or "",
                "verification_label": verification_label,
                "source_url": item.get("source_url") or "",
                "pdf_url": item.get("pdf_url") or "",
                "page": "announcement",
            }
        )
    for index, item in enumerate(result.get("financial", {}).get("anomaly_list", []) or []):
        records.append(
            {
                "id": f"FIN-{index + 1:03d}",
                "severity": int(item.get("severity") or 0),
                "kind": item.get("type") or "财务异常",
                "title": item.get("indicator") or item.get("label_ref") or "财务指标异常",
                "meta": f"原始值 {item.get('value', '—')} · 阈值 {item.get('threshold', '—')}",
                "quote": item.get("evidence") or "",
                "issue": f"问题标记：{item.get('type', '财务异常')}；关联标签 {item.get('label_ref', '—')}。",
                "keyword": str(item.get("value", "")),
                "verification_label": "结构化指标记录",
                "source_url": item.get("source_url") or "",
                "pdf_url": item.get("pdf_url") or "",
                "page": "financial",
            }
        )
    records.sort(key=lambda item: item.get("severity", 0), reverse=True)
    return records[:limit]


def render_evidence_cards(records: list[dict]) -> None:
    if not records:
        st.info(
            "当前没有通过事实性语境校验的关键证据。规则命中、制度条款、职责条款和假设性描述"
            "仍保留在公告研读 Agent 的候选信号与审计记录中，但不在此处作为事实展示。",
            icon=":material/fact_check:",
        )
        return
    for item in records:
        quote_text = _safe(item.get("quote"))
        keyword = _safe(item.get("keyword"))
        if keyword and keyword in quote_text:
            quote_text = quote_text.replace(keyword, f"<mark>{keyword}</mark>", 1)
        severity = item.get("severity", 0)
        severity_text = "高风险" if severity >= 4 else ("中风险" if severity >= 2 else "提示")
        internal = f"/{_safe(item.get('page', 'main'))}?evidence={quote(str(item.get('id', '')))}"
        links = [f'<a href="{internal}" target="_self">定位系统记录 ↗</a>']
        if item.get("source_url"):
            links.append(f'<a href="{_safe(item["source_url"])}" target="_blank" rel="noopener">查看公告原文 ↗</a>')
        if item.get("pdf_url"):
            links.append(f'<a href="{_safe(item["pdf_url"])}" target="_blank" rel="noopener">打开官方 PDF ↗</a>')
        st.html(
            f"""
            <article class="risk-evidence {'high' if severity >= 4 else ''}">
              <div class="risk-evidence-title"><span>{_safe(item.get('kind'))} · {_safe(item.get('id'))}</span><span>{severity_text}</span></div>
              <div class="risk-evidence-meta"><span class="risk-evidence-check">{_safe(item.get('verification_label'))}</span>{_safe(item.get('title'))} · {_safe(item.get('meta'))}</div>
              <div class="risk-evidence-quote">{quote_text}</div>
              <div class="risk-evidence-issue">{_safe(item.get('issue'))}</div>
              <div class="risk-links">{''.join(links)}</div>
            </article>
            """
        )


def render_source_index(company: str) -> None:
    code = quote(company)
    sources = [
        ("公告原文与片段", "公告研读 Agent · ANN", f"/announcement?company={code}"),
        ("财务指标与计算", "财务异常 Agent · FIN", f"/financial?company={code}"),
        ("相似监管案例", "案例匹配 Agent · CASE", f"/case?company={code}"),
        ("推理与工具日志", "主控 Agent · trace_log", f"/main?company={code}#agent-trace"),
    ]
    cards = "".join(
        f'<a class="risk-source-card" href="{href}" target="_self"><b>{_safe(title)} ↗</b><span>{_safe(detail)}</span></a>'
        for title, detail, href in sources
    )
    st.html(f'<div class="risk-source-grid">{cards}</div>')

