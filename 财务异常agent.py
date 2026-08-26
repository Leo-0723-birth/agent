#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""财务异常检测 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 财务异常agent.py --server.port 8503
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

from backend.agents.financial_detector import FinancialDetectorAgent
from backend.context import Context
from ui.theme import apply_scan_theme

st.set_page_config(
    page_title="财务异常 Agent",
    page_icon=":material/account_balance:",
    layout="wide",
)
apply_scan_theme()


@st.cache_data(ttl="6h", max_entries=20, show_spinner=False)
def analyze_company(company: str, as_of: str, use_llm: bool) -> dict:
    """执行一次财务异常检测（东财实时数据 + 行业对标 Z-Score + 规则检测）。"""
    agent = FinancialDetectorAgent(use_llm=use_llm)
    ctx = Context(company=company, as_of=as_of)
    agent.run(company, ctx)
    payload = asdict(ctx)
    return payload


def anomaly_dataframe(anomalies: list[dict]) -> pd.DataFrame:
    rows = []
    for a in anomalies:
        rows.append(
            {
                "类型": a.get("type", ""),
                "严重度": a.get("severity", ""),
                "指标": a.get("indicator", ""),
                "数值": a.get("value", ""),
                "阈值": a.get("threshold", ""),
                "证据": a.get("evidence", ""),
                "标签引用": a.get("label_ref", ""),
            }
        )
    return pd.DataFrame(rows)


st.title("财务异常 Agent")
st.caption("东方财富实时财务数据 → F2/F3 特征 → 行业对标 Z-Score → 规则异常检测（含双负信号兜底）。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=date.today(), max_value=date.today())
    use_llm = st.toggle("启用 LLM 财务解读", value=False, help="需要 DEEPSEEK_API_KEY。")
    st.caption("端口约定：8503（独立运行：streamlit run 财务异常agent.py --server.port 8503）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000063.SZ",
        placeholder="例如：000063.SZ、中兴通讯",
    )
    submitted = st.form_submit_button("开始检测", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码或准确名称。", icon=":material/error:")
    else:
        try:
            with st.status("正在抓取财务数据并检测异常……", expanded=True) as status:
                result = analyze_company(normalized, as_of_value.isoformat(), use_llm)
                st.session_state["financial_analysis"] = result
                status.update(label="财务检测完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("financial_analysis", None)
            st.error(f"检测失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("financial_analysis")
if result:
    fin = result.get("financial", {})
    trace = result.get("trace_log", [])
    with st.container(horizontal=True):
        st.metric("风险等级", fin.get("risk_level") or "—", border=True)
        st.metric("异常信号", len(fin.get("anomaly_list", [])), border=True)
        st.metric("行业", fin.get("industry") or "—", border=True)
        st.metric("特征维度", len(fin.get("features", {})), border=True)
    if fin.get("skip"):
        st.info(f"财务分析跳过：{fin.get('skip_reason')}", icon=":material/info:")
        st.json(fin.get("indicators", {}), expanded=False)
        st.stop()

    st.subheader("异常信号清单")
    anomalies = fin.get("anomaly_list", [])
    if anomalies:
        st.dataframe(
            anomaly_dataframe(anomalies),
            hide_index=True,
            column_config={
                "严重度": st.column_config.NumberColumn("严重度", format="%d"),
            },
        )
    else:
        st.info("未发现规则异常信号。")

    tab1, tab2, tab3, tab4 = st.tabs(["原始指标", "行业对标", "特征", "审计追踪"])
    with tab1:
        st.json(fin.get("indicators", {}), expanded=False)
    with tab2:
        st.json(fin.get("benchmarks", {}), expanded=False)
    with tab3:
        features = fin.get("features", {})
        st.caption(f"共 {len(features)} 个特征。")
        st.json(features, expanded=False)
    with tab4:
        st.json(trace, expanded=False)
        st.download_button(
            "下载完整 JSON",
            data=__import__("json").dumps(result, ensure_ascii=False, indent=2),
            file_name=f"{result.get('company')}_financial.json",
            mime="application/json",
        )
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("风险等级、异常信号清单（类型/严重度/指标/阈值/证据）、原始指标、行业对标 Z-Score 与完整特征。")
        st.caption("先使用默认示例 000063.SZ（中兴通讯），点击“开始检测”即可。财务数据来自东方财富免费接口（实时）。")
