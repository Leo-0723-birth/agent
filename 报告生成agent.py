#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""报告生成 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 报告生成agent.py --server.port 8507
说明：报告依赖全流水线输出，本页直接跑 SweepingOrchestrator 全流程后渲染 Markdown/JSON 报告。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents import SweepingOrchestrator

st.set_page_config(
    page_title="报告生成 Agent",
    page_icon=":material/description:",
    layout="wide",
)


@st.cache_data(ttl="6h", max_entries=10, show_spinner=False)
def generate_report(company: str, as_of: str, window: int, use_llm: bool, use_finbert: bool) -> dict:
    """全流程（公告研读→财务检测→预测→案例→归因→报告）后渲染报告。"""
    orch = SweepingOrchestrator(use_llm=use_llm, use_finbert=use_finbert)
    ctx = orch.sweep_one(company, window=window, as_of=as_of)
    return ctx.to_dict()


st.title("报告生成 Agent")
st.caption("聚合全流水线结果（预测/财务/公告/案例/归因/trace）→ 六章 Markdown 报告 + 结构化 JSON。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=date(2025, 12, 2), max_value=date.today())
    window = st.selectbox("预测窗口（天）", [30, 60, 90], index=1)
    use_llm = st.toggle("启用 LLM 精细抽取", value=False, help="需要 DEEPSEEK_API_KEY。")
    use_finbert = st.toggle("启用 FinBERT", value=False)
    st.caption("端口约定：8507（独立运行：streamlit run 报告生成agent.py --server.port 8507）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000004.SZ",
        placeholder="例如：000004.SZ、国华网安",
    )
    submitted = st.form_submit_button("生成报告", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码或准确名称。", icon=":material/error:")
    else:
        try:
            with st.status("正在运行 6-Agent 全流程并渲染报告……", expanded=True) as status:
                result = generate_report(normalized, as_of_value.isoformat(), window, use_llm, use_finbert)
                st.session_state["report_analysis"] = result
                status.update(label="报告生成完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("report_analysis", None)
            st.error(f"报告生成失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("report_analysis")
if result:
    report = result.get("report", {})
    trace = result.get("trace_log", [])
    markdown = report.get("markdown", "")
    report_json = report.get("json", {})

    with st.container(horizontal=True):
        st.metric("流水线环节", len(trace), border=True)
        done = sum(1 for t in trace if t.get("status") == "done")
        st.metric("完成环节", done, border=True)
        st.metric("报告字数", len(markdown), border=True)

    with st.expander("🔍 流水线追踪", expanded=True):
        for t in trace:
            agent = t.get("agent", "?")
            status_icon = "✅" if t.get("status") == "done" else ("⏭️" if t.get("status") == "skipped" else "⚠️")
            st.markdown(f"{status_icon} **{agent}** ｜ {t.get('status')} ｜ {t.get('latency_ms', '')}ms ｜ {str(t.get('output_summary', ''))[:80]}")

    st.subheader("Markdown 报告")
    st.markdown(markdown)

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ 下载 Markdown 报告",
        data=markdown,
        file_name=f"{normalized}_risk_report.md",
        mime="text/markdown",
    )
    c2.download_button(
        "⬇️ 下载 JSON 报告",
        data=json.dumps(report_json, ensure_ascii=False, indent=2),
        file_name=f"{normalized}_risk_report.json",
        mime="application/json",
    )
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("六章 Markdown 报告（结论/财务/公告/案例/归因/推理链路）+ 可下载的 Markdown 与 JSON。")
        st.caption("先使用默认示例 000004.SZ，点击“生成报告”即可。首次运行需下载公告 PDF，可能等待数分钟。")
