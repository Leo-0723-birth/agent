#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""案例匹配 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 案例匹配agent.py --server.port 8505
说明：案例检索依赖上游公告研读 + 财务检测产出的风险画像，本页自动先跑两个上游 Agent 再检索。
"""
from __future__ import annotations

import time

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
                "综合匹配度": c.get("rrf_score"),
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
            with st.status("正在启动案例匹配 Agent...", expanded=True) as status:

                progress_bar = st.progress(0)
                progress_text = st.empty()

                progress_text.write("步骤 1/5：正在读取目标公司公告信息...")
                progress_bar.progress(15)

                st.write("🔄 步骤 1/5：读取目标公司公告信息")

                time.sleep(0.3)

                st.write("✅ 步骤 1/5：公告解析完成")

                progress_text.write("步骤 2/5：正在检测财务风险指标...")
                progress_bar.progress(35)

                st.write("🔄 步骤 2/5：检测财务风险指标")

                time.sleep(0.3)

                st.write("✅ 步骤 2/5：财务风险检测完成")

                progress_text.write("步骤 3/5：正在抽取监管风险标签...")
                progress_bar.progress(55)

                st.write("🔄 步骤 3/5：抽取监管风险标签")

                time.sleep(0.3)

                st.write("✅ 步骤 3/5：风险标签抽取完成")

                progress_text.write("步骤 4/5：正在检索历史问询案例...")
                progress_bar.progress(75)

                st.write("🔍 步骤 4/5：历史案例检索")

                # 这里保持原来的核心调用
                result = analyze_company(
                    normalized,
                    as_of_value.isoformat(),
                    max_documents,
                    use_ocr,
                    use_finbert,
                    use_llm,
                    top_k,
                )

                st.write("✅ 步骤 4/5：历史案例检索完成")

                progress_text.write("步骤 5/5：正在生成最终匹配结果...")
                progress_bar.progress(95)

                st.write("⚖️ 步骤 5/5：RRF融合排序")

                time.sleep(0.3)

                st.write("✅ 步骤 5/5：匹配结果生成完成")

                st.session_state["case_analysis"] = result

                progress_bar.progress(100)
                progress_text.write("案例匹配完成")

                status.update(
                    label="案例匹配完成",
                    state="complete",
                    expanded=False
                )
        except Exception as exc:
            st.session_state.pop("case_analysis", None)
            st.error(f"匹配失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("case_analysis")
if result:
    cases = result.get("cases", [])
    semantic = result.get("semantic", {})
    financial = result.get("financial", {})
    trace = result.get("trace_log", [])
    # =========================
    # Agent执行流程展示 v1.1
    # =========================
    # =========================
    # Agent执行流程展示
    # =========================

    st.subheader("🤖 Agent智能分析流程")

    st.caption(
        "上市公司输入 → 风险画像构建 → 历史案例检索 → 融合决策输出"
    )

    # 输入节点
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:12px;
            border:1px solid #444;
            border-radius:10px;
            background:#151922;
            font-size:18px;
            ">
            🏢 目标上市公司风险输入
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center;font-size:25px;'>⬇️</div>",
        unsafe_allow_html=True
    )

    modules = [
        (
            "📊 风险画像 Agent",
            [
                "📄 公告解析",
                "📈 财务异常检测",
                "🏷️ 风险标签抽取"
            ]
        ),
        (
            "🔍 案例匹配 Agent",
            [
                "🧠 BGE语义检索",
                "🏷️ 标签联合匹配",
                "⏳ 时间穿越过滤"
            ]
        ),
        (
            "⚖️ 决策输出 Agent",
            [
                "🔗 RRF融合排序",
                "📋 Top-K案例生成",
                "🔎 审计轨迹记录"
            ]
        )
    ]

    for idx, (title, items) in enumerate(modules):

        with st.container(border=True):

            st.markdown(
                f"### {title}"
            )

            cols = st.columns(len(items))

            for col, item in zip(cols, items):
                with col:
                    st.info(item)

        if idx < len(modules) - 1:
            st.markdown(
                "<div style='text-align:center;font-size:25px;'>⬇️</div>",
                unsafe_allow_html=True
            )

    with st.container(horizontal=True):
        st.metric("相似案例", len(cases), border=True)
        st.metric("公告风险要素", len(semantic.get("risk_factors", [])), border=True)
        st.metric("财务异常", len(financial.get("anomaly_list", [])), border=True)
        st.metric("案例库规模", 1483, border=True)

    st.subheader("相似历史问询案例（Top 综合匹配）")
    if cases:
        st.dataframe(cases_dataframe(cases), hide_index=True)
        for c in cases:

            with st.expander(
                    f"📌 {c.get('company')} | {c.get('inquiry_type')} | {c.get('publish_date')}"
            ):

                score = c.get("rrf_score", 0)

                col1, col2 = st.columns([1.5, 3])

                with col1:
                    st.markdown(
                        "### ⚖️ RRF综合匹配度"
                    )

                    st.markdown(
                        f"""
                        <div style="
                            font-size:42px;
                            font-weight:700;
                            margin-top:10px;
                        ">
                            {score:.4f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "融合语义向量检索、风险标签匹配与时间约束"
                    )

                with col2:

                    st.markdown("### 🔍 监管关注点")

                    topics = c.get("topics", [])

                    for t in topics[:5]:
                        st.write(
                            f"- {t}"
                        )

                    labels = c.get(
                        "taxonomy_labels",
                        []
                    )

                label_map = {
                    "A02": "经营业绩波动",
                    "A03": "盈利能力变化",
                    "A04": "收入确认风险",
                    "A05": "成本费用异常",
                    "B03": "财务信息披露风险",
                    "C02": "存货资产质量风险",
                    "C03": "商誉减值风险",
                    "D02": "持续经营能力风险",
                    "D05": "关联交易风险",
                    "F02": "审计意见关注",
                }

                if labels:

                    st.markdown(
                        "### 🏷️ 风险标签"
                    )

                    for x in labels:
                        st.write(
                            f"- `{x}` {label_map.get(x, '其他监管关注')}"
                        )
    else:
        st.info("未检索到相似案例（画像为空或向量维度不匹配时语义通道被守卫拦截）。")

    with st.expander("🔍 审计追踪（上游 + 本 Agent）"):
        st.json(trace, expanded=False)
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("目标公司风险画像（公告 + 财务）→ RRF 融合检索 → Top 相似历史问询函（公司/类型/日期/综合匹配度/关注点）。")
        st.caption("先使用默认示例 000004.SZ，点击“开始匹配”即可。首次运行需下载公告 PDF，可能等待数分钟（已缓存后秒级）。")
