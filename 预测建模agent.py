#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测建模 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 预测建模agent.py --server.port 8504
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

from backend.agents.predictor import PredictorAgent
from backend.context import Context

st.set_page_config(
    page_title="预测建模 Agent",
    page_icon=":material/model_training:",
    layout="wide",
)


@st.cache_data(ttl="6h", max_entries=20, show_spinner=False)
def analyze_company(company: str, as_of: str, window: int) -> dict:
    """查表推理：建模数据集最新一行特征 → RF/LGB/XGB 三模型集成 → 概率 + SHAP。"""
    agent = PredictorAgent(horizons=[30, 60, 90])
    ctx = Context(company=company, as_of=as_of, window=window)
    agent.run(company, ctx)
    payload = asdict(ctx)
    return payload


st.title("预测建模 Agent")
st.caption("按 (company, as_of) 从建模数据集取最新一行特征 → 三模型集成（RF+LightGBM+XGBoost）→ 30/60/90 天问询概率 + SHAP 归因。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("特征锚点 T", value=date(2025, 12, 2), max_value=date.today())
    st.caption("公司需在建模数据集 backend/data/modeling/processed_dataset.csv 内。")
    st.caption("端口约定：8504（独立运行：streamlit run 预测建模agent.py --server.port 8504）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码（带交易所后缀）",
        value="000004.SZ",
        placeholder="例如：000004.SZ",
    )
    submitted = st.form_submit_button("开始预测", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码。", icon=":material/error:")
    else:
        try:
            with st.status("正在查表并执行三模型集成推理……", expanded=True) as status:
                result = analyze_company(normalized, as_of_value.isoformat(), 60)
                st.session_state["prediction_analysis"] = result
                status.update(label="预测完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("prediction_analysis", None)
            st.error(f"预测失败：{type(exc).__name__}: {exc}", icon=":material/error:")

result = st.session_state.get("prediction_analysis")
if result:
    pred = result.get("prediction", {})
    trace = result.get("trace_log", [])
    st.subheader(f"问询概率（T={pred.get('feature_anchor') or '—'} 特征锚点）")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("30 天", f"{pred.get('probability_30d'):.4f}" if pred.get("probability_30d") is not None else "—", border=True)
    c2.metric("60 天", f"{pred.get('probability_60d'):.4f}" if pred.get("probability_60d") is not None else "—", border=True)
    c3.metric("90 天", f"{pred.get('probability_90d'):.4f}" if pred.get("probability_90d") is not None else "—", border=True)
    c4.metric("风险等级", pred.get("risk_level") or "—", border=True)
    c5.metric("置信度", f"{pred.get('confidence'):.2f}" if pred.get("confidence") is not None else "—", border=True)

    if pred.get("reason"):
        st.warning(pred["reason"], icon=":material/warning:")

    shap = pred.get("shap_features", []) or []
    if shap:
        st.subheader("SHAP 归因 Top 特征（集成贡献代理）")
        df = pd.DataFrame(shap, columns=["特征", "贡献值"]).head(15)
        st.bar_chart(df.set_index("特征"), y="贡献值", height=360, color="primary")
        st.dataframe(
            df,
            hide_index=True,
            column_config={"贡献值": st.column_config.NumberColumn("贡献值", format="%+.5f")},
        )
    else:
        st.info("无 SHAP 特征（模型未加载或公司不在建模数据集内）。")

    with st.expander("🔍 审计追踪"):
        st.json(trace, expanded=False)

    with st.expander("查看完整预测结果"):
        st.json(pred, expanded=False)
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("30/60/90 天监管问询概率、风险等级、置信度、SHAP 特征贡献图与推理追踪。")
        st.caption("先使用默认示例 000004.SZ，点击“开始预测”即可。若显示“未找到该股票特征”，说明该公司不在建模数据集内。")
