#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""归因分析 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 归因分析agent.py --server.port 8506
说明：归因依赖上游公告研读 + 财务检测 + 预测建模（SHAP），本页自动先跑上游再归因。
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.agents.attributor import AttributorAgent
from backend.agents.case_retriever import CaseRetrieverAgent
from backend.agents.financial_detector import FinancialDetectorAgent
from backend.agents.predictor import PredictorAgent
from backend.config import ANNOUNCE_SOURCE
from backend.context import Context
from backend.skills.announcement_search import CninfoAnnouncementSource

st.set_page_config(
    page_title="归因分析 Agent",
    page_icon=":material/psychology:",
    layout="wide",
)


@st.cache_data(ttl="6h", max_entries=10, show_spinner=False)
def analyze_company(
    company: str,
    as_of: str,
    max_documents: int,
    use_ocr: bool,
    use_finbert: bool,
    use_llm: bool,
    top_k: int,
    shap_threshold: float,
) -> dict:
    """公告研读 → 财务检测 → 预测建模(SHAP) → 归因解释（SHAP + 证据白名单 + validate_narrative 防幻觉）。"""
    # 离线（ANNOUNCE_SOURCE=local）时不传 source，AnnouncementReaderAgent 自动走本地 PDF 扫描
    source = None
    if ANNOUNCE_SOURCE != "local":
        source = CninfoAnnouncementSource(max_documents=max_documents, ocr_enabled=use_ocr)
    ctx = Context(company=company, as_of=as_of)
    AnnouncementReaderAgent(source=source, use_finbert=use_finbert, use_llm=use_llm).run(company, ctx)
    FinancialDetectorAgent(use_llm=False).run(company, ctx)
    PredictorAgent().run(company, ctx)
    CaseRetrieverAgent().run(company, ctx)
    AttributorAgent(top_k=top_k, shap_threshold=shap_threshold, use_llm=use_llm).run(company, ctx)
    return asdict(ctx)


def factors_dataframe(factors: list[dict]) -> pd.DataFrame:
    rows = []
    for f in factors:
        rows.append(
            {
                "特征": f.get("feature", ""),
                "SHAP 贡献": f.get("shap"),
                "说明": f.get("desc") or f.get("description", ""),
                "标签引用": f.get("label_ref", ""),
                "风险主题": f"{f.get('taxonomy_l2', '')} {f.get('theme_name', '')}".strip(),
                "来源": f.get("source", ""),
                "降级归因": bool(f.get("is_fallback")),
                "证据 ID": f.get("evidence_id") or (
                    "无直接证据" if f.get("no_evidence") else ""
                ),
            }
        )
    return pd.DataFrame(rows)


st.title("归因分析 Agent")
st.caption("SHAP 特征贡献（LightGBM pred_contrib 代理）+ 证据白名单 + validate_narrative 防幻觉校验；无 SHAP 时自动降级为规则归因。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=date.today(), max_value=date.today())
    max_documents = st.slider("最多解析 PDF 数量", min_value=5, max_value=120, value=30, step=5)
    use_ocr = st.toggle("启用扫描 PDF OCR", value=True)
    use_finbert = st.toggle("启用 FinBERT", value=False)
    use_llm = st.toggle("启用 LLM 归因叙事", value=False, help="需要 DEEPSEEK_API_KEY。")
    top_k = st.slider("Top 诱因数", min_value=1, max_value=15, value=5)
    shap_threshold = st.slider("SHAP 贡献阈值", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    st.caption("端口约定：8506（独立运行：streamlit run 归因分析agent.py --server.port 8506）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000004.SZ",
        placeholder="例如：000004.SZ、国华网安",
    )
    submitted = st.form_submit_button("开始归因", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码或准确名称。", icon=":material/error:")
    else:
        try:
            with st.status("正在运行公告研读→财务检测→预测建模→归因……", expanded=True) as status:
                result = analyze_company(
                    normalized,
                    as_of_value.isoformat(),
                    max_documents,
                    use_ocr,
                    use_finbert,
                    use_llm,
                    top_k,
                    shap_threshold,
                )
                st.session_state["attribution_analysis"] = result
                status.update(label="归因分析完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("attribution_analysis", None)
            st.error(f"归因失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("attribution_analysis")
if result:
    att = result.get("attribution", {})
    pred = result.get("prediction", {})
    trace = result.get("trace_log", [])
    factors = att.get("top_risk_factors", [])
    citations = att.get("evidence_citations", [])
    case_links = att.get("case_links", [])

    with st.container(horizontal=True):
        st.metric("诱因数", len(factors), border=True)
        st.metric("证据引用", len(citations), border=True)
        st.metric("相似案例", len(case_links), border=True)
        st.metric("60 天概率", f"{pred.get('probability_60d'):.4f}" if pred.get("probability_60d") is not None else "—", border=True)
        st.metric("风险等级", pred.get("risk_level") or "—", border=True)

    if att.get("narrative"):
        st.subheader("归因叙事")
        st.write(att["narrative"])

    st.subheader("Top 风险诱因")
    if factors:
        st.dataframe(factors_dataframe(factors), hide_index=True)
    else:
        st.info("无归因诱因（上游无风险信号或 SHAP 贡献低于阈值）。")

    with st.expander("📎 证据池（原文证据白名单）"):
        for e in citations:
            st.markdown(f"- `{e.get('evidence_id')}` [{e.get('source')}] {e.get('snippet', '')[:200]}")

    st.subheader("相似历史案例")
    if case_links:
        for c in case_links:
            with st.container(border=True):
                st.markdown(f"**{c.get('company', '')}** · {c.get('inquiry_type', '')} · 相似度 {c.get('similarity')}")
                st.caption("关注点：" + "、".join(c.get("topics", [])[:3]))
    else:
        st.info("无高度重合的相似案例。")

    with st.expander("🔍 审计追踪（上游 + 本 Agent）"):
        st.json(trace, expanded=False)

    with st.expander("查看完整归因结果"):
        st.json(att, expanded=False)
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("归因叙事、Top 风险诱因（SHAP 贡献 + 标签引用）、原文证据白名单与完整推理追踪。")
        st.caption("先使用默认示例 000004.SZ，点击“开始归因”即可。首次运行需下载公告 PDF，可能等待数分钟。")
