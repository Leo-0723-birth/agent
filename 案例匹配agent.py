#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""案例匹配 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 案例匹配agent.py --server.port 8505
说明：案例检索依赖上游公告研读 + 财务检测产出的风险画像，本页自动先跑两个上游 Agent 再检索。
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
from backend.agents.case_retriever import CaseRetrieverAgent
from backend.agents.financial_detector import FinancialDetectorAgent
from backend.context import Context
from backend.skills.announcement_search import CninfoAnnouncementSource

st.set_page_config(
    page_title="案例匹配 Agent",
    page_icon=":material/compare_arrows:",
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
) -> dict:
    """公告研读 → 财务检测 → 案例检索（RRF 融合 + 三源标签通道 + 时间穿越控制）。"""
    source = CninfoAnnouncementSource(max_documents=max_documents, ocr_enabled=use_ocr)
    ctx = Context(company=company, as_of=as_of)
    AnnouncementReaderAgent(source=source, use_finbert=use_finbert, use_llm=use_llm).run(company, ctx)
    FinancialDetectorAgent(use_llm=False).run(company, ctx)
    CaseRetrieverAgent(top_k=top_k).run(company, ctx)
    return asdict(ctx)


def cases_dataframe(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cases:
        rows.append(
            {
                "公司": c.get("company", ""),
                "问询类型": c.get("inquiry_type", ""),
                "发布日期": c.get("publish_date", ""),
                "RRF融合得分": c.get("rrf_score"),
                "关注点": "；".join(str(t)[:40] for t in (c.get("topics") or [])[:3]),
            }
        )
    return pd.DataFrame(rows)


st.title("案例匹配 Agent")
st.caption("基于 1483 份历史监管问询函案例库，对目标公司风险画像做 RRF 融合检索（语义向量 + 标签重合），带时间穿越控制与防泄漏过滤。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=date.today(), max_value=date.today())
    max_documents = st.slider("最多解析 PDF 数量", min_value=5, max_value=120, value=30, step=5)
    use_ocr = st.toggle("启用扫描 PDF OCR", value=True)
    use_finbert = st.toggle("启用 FinBERT", value=False)
    use_llm = st.toggle("启用 LLM 精细抽取", value=False, help="需要 DEEPSEEK_API_KEY。")
    top_k = st.slider("返回相似案例数", min_value=1, max_value=20, value=5)
    st.caption("端口约定：8505（独立运行：streamlit run 案例匹配agent.py --server.port 8505）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000004.SZ",
        placeholder="例如：000004.SZ、国华网安",
    )
    submitted = st.form_submit_button("开始匹配", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码或准确名称。", icon=":material/error:")
    else:
        try:
            with st.status("正在研读公告与财务检测，随后检索案例库……", expanded=True) as status:
                result = analyze_company(
                    normalized,
                    as_of_value.isoformat(),
                    max_documents,
                    use_ocr,
                    use_finbert,
                    use_llm,
                    top_k,
                )
                st.session_state["case_analysis"] = result
                status.update(label="案例匹配完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("case_analysis", None)
            st.error(f"匹配失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("case_analysis")
if result:
    cases = result.get("cases", [])
    semantic = result.get("semantic", {})
    financial = result.get("financial", {})
    trace = result.get("trace_log", [])
    with st.container(horizontal=True):
        st.metric("相似案例", len(cases), border=True)
        st.metric("公告风险要素", len(semantic.get("risk_factors", [])), border=True)
        st.metric("财务异常", len(financial.get("anomaly_list", [])), border=True)
        st.metric("案例库规模", 1483, border=True)

    st.subheader("相似历史问询案例（Top 综合匹配）")
    if cases:
        st.dataframe(cases_dataframe(cases), hide_index=True)
        for c in cases:
            with st.expander(f"**{c.get('company')}**｜{c.get('inquiry_type')}｜{c.get('publish_date')}｜RRF融合得分 {c.get('rrf_score')}"):
                st.json(c, expanded=False)
    else:
        st.info("未检索到相似案例（画像为空或向量维度不匹配时语义通道被守卫拦截）。")

    with st.expander("🔍 审计追踪（上游 + 本 Agent）"):
        st.json(trace, expanded=False)
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("目标公司风险画像（公告 + 财务）→ RRF 融合检索 → Top 相似历史问询函（公司/类型/日期/RRF融合得分/关注点）。")
        st.caption("先使用默认示例 000004.SZ，点击“开始匹配”即可。首次运行需下载公告 PDF，可能等待数分钟（已缓存后秒级）。")
