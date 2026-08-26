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
from ui.theme import apply_page_style
from ui.session import active_as_of, active_company, hydrate_page_state, shared_context_caption
from ui.components import render_trace

st.set_page_config(
    page_title="财务异常 Agent",
    page_icon=":material/account_balance:",
    layout="wide",
)
apply_page_style()


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


INDICATOR_META = {
    "report_period": ("报告期", "财务数据对应的报告截止期", "period"),
    "total_revenue": ("营业收入", "公司当期主营及其他营业收入", "money"),
    "net_profit": ("净利润", "公司当期归属经营成果", "money"),
    "operating_cash_flow": ("经营活动现金流", "经营活动产生的现金流量净额", "money"),
    "roe": ("净资产收益率（ROE）", "衡量股东权益的盈利能力", "percent"),
    "roa": ("总资产收益率（ROA）", "衡量资产整体盈利效率", "percent"),
    "debt_to_assets_ratio": ("资产负债率", "总负债占总资产的比例", "percent"),
    "revenue_yoy_growth": ("营业收入同比增速", "营业收入相较上年同期的变化", "percent"),
    "net_profit_yoy_growth": ("净利润同比增速", "净利润相较上年同期的变化", "percent"),
    "cf_to_profit": ("经营现金流／净利润", "判断利润是否有现金流支撑", "ratio"),
    "roe_trend_4q": ("ROE 近四期趋势", "每季度 ROE 的趋势斜率", "trend"),
    "industry": ("所属行业", "用于选择同行对标样本", "text"),
}

FEATURE_LABELS = {
    "anomaly_count": "异常信号数量",
    "max_severity": "最高异常严重度",
    "cf_income_ratio": "经营现金流／净利润",
    "roe": "净资产收益率（ROE）",
    "roe_trend_4q": "ROE 近四期趋势",
    "debt_to_assets_ratio": "资产负债率",
    "revenue_yoy_growth": "营业收入同比增速",
    "net_profit_yoy_growth": "净利润同比增速",
    "f2_beneish_m": "Beneish 盈余操纵评分",
    "f2_benford_flag": "Benford 数字分布异常标志",
    "f2_benford_max_dev": "Benford 最大偏离值",
    "f2_trend_deterioration": "经营趋势恶化信号数",
    "f2_loss_flag": "亏损标志",
    "f2_high_debt_flag": "高负债标志",
    "f2_p_score": "Piotroski 财务健康评分",
    "f2_ocf_to_revenue": "经营现金流／营业收入",
    "f2_ocf_to_profit": "经营现金流／净利润",
    "f2_net_margin": "净利率",
    "f2_accruals": "应计利润",
    "f2_accruals_to_assets": "应计利润／总资产",
    "f2_accruals_to_revenue": "应计利润／营业收入",
    "f2_profit_ocf_diverge": "利润与现金流背离标志",
    "f2_roe_industry_rank": "ROE 行业百分位",
    "f2_industry_outlier_count": "行业异常指标数量",
}

TOKEN_LABELS = {
    "roe": "净资产收益率", "roa": "总资产收益率", "profit": "利润", "revenue": "营业收入",
    "ocf": "经营现金流", "cash": "现金流", "debt": "负债", "ratio": "比例", "margin": "利润率",
    "growth": "增长率", "yoy": "同比", "qoq": "环比", "volatility": "波动率", "return": "收益率",
    "market": "市场", "cap": "市值", "rank": "排名", "flag": "标志", "score": "评分",
    "trend": "趋势", "decline": "下降", "streak": "连续期数", "industry": "行业", "outlier": "异常",
    "accruals": "应计利润", "assets": "资产", "extreme": "极端值", "negative": "负值",
    "high": "高位", "low": "低位", "sentiment": "舆情", "governance": "公司治理",
    "turnover": "换手率", "liquidity": "流动性", "illiquidity": "非流动性", "down": "下跌",
    "days": "天数", "count": "数量", "interval": "间隔", "inquiry": "问询", "last": "最近",
}


def _format_value(value, kind: str = "number") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "数据缺失"
    if kind == "period":
        raw = str(int(value)) if isinstance(value, float) else str(value)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 else raw
    if kind == "money":
        number = float(value)
        if abs(number) >= 1e8:
            return f"{number / 1e8:,.2f} 亿元"
        if abs(number) >= 1e4:
            return f"{number / 1e4:,.2f} 万元"
        return f"{number:,.2f} 元"
    if kind == "percent":
        return f"{float(value):,.2f}%"
    if kind == "trend":
        return f"{float(value):+.2f} 个百分点／季"
    if kind == "ratio":
        return f"{float(value):,.4f} 倍"
    if isinstance(value, bool) or (kind == "flag" and value in (0, 1)):
        return "是" if bool(value) else "否"
    if isinstance(value, (int, float)):
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    return str(value)


def indicator_dataframe(fin: dict) -> pd.DataFrame:
    indicators = fin.get("indicators", {}) or {}
    rows = []
    for key, value in indicators.items():
        label, meaning, kind = INDICATOR_META.get(key, ("补充财务指标", "用于财务风险计算的补充口径", "number"))
        rows.append({"指标": label, "当前值": _format_value(value, kind), "指标说明": meaning})
    if not rows:
        seen = set()
        for item in fin.get("anomaly_list", []) or []:
            key = str(item.get("indicator") or item.get("type") or "")
            if key in seen:
                continue
            seen.add(key)
            label = FEATURE_LABELS.get(key) or item.get("type") or "异常指标"
            rows.append({"指标": label, "当前值": _format_value(item.get("value")), "指标说明": item.get("evidence") or "由异常检测规则识别"})
    return pd.DataFrame(rows)


def benchmark_dataframe(benchmarks: dict) -> pd.DataFrame:
    rows = []
    for key, value in (benchmarks or {}).items():
        if key in {"note", "industry_peer_count", "peer_period"}:
            continue
        base = key.removesuffix("_zscore")
        label = INDICATOR_META.get(base, (FEATURE_LABELS.get(base, "补充指标"), "", ""))[0]
        if value is None:
            judgment = "样本或指标不足，未计算"
        elif abs(float(value)) > 2:
            judgment = "显著偏离同行均值，需重点核查"
        elif abs(float(value)) > 1:
            judgment = "与同行存在一定差异"
        else:
            judgment = "处于同行常见区间"
        rows.append({"对标指标": label, "Z-Score": _format_value(value), "业务判断": judgment})
    return pd.DataFrame(rows)


def _feature_category(name: str) -> str:
    if name.startswith("f2_") or name in FEATURE_LABELS:
        return "财务质量与异常"
    if name.startswith("mkt_"):
        return "市场交易行为"
    if name.startswith(("sent_", "f4_")):
        return "舆情表现"
    if name.startswith(("gov_", "governance_", "f5_")):
        return "公司治理"
    return "补充风险特征"


def _feature_label(name: str, index: int) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    parts = [part for part in name.split("_") if part not in {"f2", "f3", "f4", "f5", "mkt", "sent", "gov"}]
    translated = [TOKEN_LABELS.get(part) for part in parts]
    if translated and all(translated):
        return "·".join(translated)
    return f"{_feature_category(name)}指标 {index:02d}"


def feature_dataframe(features: dict) -> pd.DataFrame:
    rows = []
    for index, (name, value) in enumerate((features or {}).items(), start=1):
        category = _feature_category(name)
        kind = "flag" if name.endswith("_flag") else "number"
        rows.append({
            "特征类别": category,
            "业务指标": _feature_label(name, index),
            "当前值": _format_value(value, kind),
            "用途": "用于模型判断公司在该维度的风险强弱",
        })
    return pd.DataFrame(rows)


def trace_dataframe(trace: list[dict]) -> pd.DataFrame:
    status_labels = {"done": "完成", "skipped": "跳过", "error": "异常"}
    return pd.DataFrame([
        {
            "执行环节": item.get("agent") or "Agent",
            "状态": status_labels.get(item.get("status"), item.get("status") or "完成"),
            "耗时": f"{item.get('latency_ms')} ms" if item.get("latency_ms") is not None else "—",
            "处理结果": str(item.get("output_summary") or "该步骤已执行并写入共享上下文"),
        }
        for item in trace or []
    ])


def eastmoney_finance_url(company: str) -> str:
    raw = company.upper().replace(".", "")
    if raw.endswith("SZ"):
        code = f"SZ{raw[:6]}"
    elif raw.endswith("SH"):
        code = f"SH{raw[:6]}"
    elif raw.endswith("BJ"):
        code = f"BJ{raw[:6]}"
    else:
        code = raw
    return f"https://emweb.securities.eastmoney.com/PC_HSF10/FinanceAnalysis/Index?type=web&code={code}"


def render_source_link(company: str, report_period=None) -> None:
    period_text = _format_value(report_period, "period") if report_period else "当前可用报告期"
    st.caption(f"数据口径：东方财富公开财务数据 · 报告期 {period_text}。指标经规则引擎计算后进入异常检测，页面不展示程序源代码。")
    st.link_button("查看公开财务原始页面", eastmoney_finance_url(company), icon=":material/open_in_new:")


st.title("财务异常 Agent")
st.caption("东方财富实时财务数据 → F2/F3 特征 → 行业对标 Z-Score → 规则异常检测（含双负信号兜底）。")
if shared_context_caption():
    st.info(shared_context_caption(), icon=":material/sync:")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=active_as_of(), max_value=date.today())
    use_llm = st.toggle("启用 LLM 财务解读", value=False, help="需要 DEEPSEEK_API_KEY。")
    st.caption("端口约定：8503（独立运行：streamlit run 财务异常agent.py --server.port 8503）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value=active_company(),
        placeholder="例如：000004.SZ、国华网安",
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

result = hydrate_page_state("financial_analysis")
if result:
    fin = result.get("financial", {})
    trace = result.get("trace_log", [])
    with st.container(horizontal=True):
        st.metric("风险等级", fin.get("risk_level") or "—", border=True)
        st.metric("异常信号", len(fin.get("anomaly_list", [])), border=True)
        st.metric("行业", fin.get("industry") or "—", border=True)
        st.metric("特征维度", fin.get("features_count") or len(fin.get("features", {})), border=True)
    if fin.get("skip"):
        st.info(f"财务分析跳过：{fin.get('skip_reason')}", icon=":material/info:")

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
        st.link_button(
            "追溯异常指标的公开财务原始页面",
            eastmoney_finance_url(result.get("company", "")),
            icon=":material/open_in_new:",
        )
    else:
        st.info("未发现规则异常信号。")

    tab1, tab2, tab3, tab4 = st.tabs(["原始指标", "行业对标", "特征", "审计追踪"])
    with tab1:
        st.markdown("#### 财务指标原值")
        st.write("以下为本次分析实际采用的财务指标，金额和比例已转换为可直接阅读的口径。")
        indicator_df = indicator_dataframe(fin)
        if indicator_df.empty:
            st.info("当前结果没有可展示的财务指标。")
        else:
            st.dataframe(indicator_df, hide_index=True, width="stretch")
        render_source_link(result.get("company", ""), (fin.get("indicators", {}) or {}).get("report_period"))
    with tab2:
        st.markdown("#### 同行业偏离情况")
        benchmarks = fin.get("benchmarks", {}) or {}
        note = benchmarks.get("note")
        if note:
            st.info(f"本次对标说明：{note}")
        peer_count = benchmarks.get("industry_peer_count")
        peer_period = benchmarks.get("peer_period")
        if peer_count is not None:
            c1, c2 = st.columns(2)
            c1.metric("同行样本数", f"{peer_count} 家")
            c2.metric("样本时间口径", "同报告期" if peer_period == "same_period" else "可用历史报告期")
        benchmark_df = benchmark_dataframe(benchmarks)
        if benchmark_df.empty:
            st.write("当前快照未保存逐项 Z-Score；异常清单中已保留实际触发的行业偏离证据。")
        else:
            st.dataframe(benchmark_df, hide_index=True, width="stretch")
        st.caption("Z-Score 为正表示高于同行均值，为负表示低于同行均值；绝对值越大，偏离程度越高。")
        render_source_link(result.get("company", ""), (fin.get("indicators", {}) or {}).get("report_period"))
    with tab3:
        features = fin.get("features", {})
        feature_count = fin.get("features_count") or len(features)
        st.markdown("#### 模型使用的财务与市场特征")
        st.caption(f"本次结果共形成 {feature_count} 个特征。页面仅展示业务名称、当前值和用途，不展示程序字段或源代码。")
        feature_df = feature_dataframe(features)
        if feature_df.empty:
            st.info("当前为精简离线报告，未归档全部特征明细；下列为报告中保留的关键风险特征。")
            key_rows = [
                {
                    "业务特征": FEATURE_LABELS.get(str(item.get("indicator")), item.get("type") or "异常指标"),
                    "当前值": _format_value(item.get("value")),
                    "风险含义": item.get("evidence") or "触发财务异常规则",
                }
                for item in anomalies
            ]
            if key_rows:
                st.dataframe(pd.DataFrame(key_rows), hide_index=True, width="stretch")
        else:
            categories = feature_df["特征类别"].value_counts()
            st.write("；".join(f"{name} {count} 项" for name, count in categories.items()))
            st.dataframe(feature_df, hide_index=True, width="stretch", height=430)
        st.caption("特征用于风险排序与概率预测，不等同于单项违规结论；最终判断需结合异常证据和公告原文。")
        render_source_link(result.get("company", ""), (fin.get("indicators", {}) or {}).get("report_period"))
    with tab4:
        st.markdown("#### Agent 执行与证据链路")
        if trace:
            render_trace(trace)
            st.dataframe(trace_dataframe(trace), hide_index=True, width="stretch")
        else:
            st.info("当前结果未保存逐步耗时，但财务异常、阈值与证据均已保留。")
        with st.expander("查看财务异常检测口径", expanded=True):
            st.markdown(
                """
                1. **数据获取**：读取公开财务报表和公司行业信息，并锁定本次报告期。
                2. **指标计算**：统一计算盈利、现金流、偿债和增长指标。
                3. **同行比较**：同报告期同行样本充足时计算 Z-Score，样本不足则明确降级。
                4. **规则检测**：使用已披露阈值识别亏损、高负债、现金流背离、趋势恶化等信号。
                5. **结果留痕**：保存触发规则、指标值、阈值、文字证据、Agent 状态与耗时。
                """
            )
        if anomalies:
            st.markdown("#### 可复核证据")
            st.dataframe(
                anomaly_dataframe(anomalies)[["类型", "数值", "阈值", "证据", "标签引用"]],
                hide_index=True,
                width="stretch",
            )
        render_source_link(result.get("company", ""), (fin.get("indicators", {}) or {}).get("report_period"))
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("风险等级、异常信号清单（类型/严重度/指标/阈值/证据）、原始指标、行业对标 Z-Score 与完整特征。")
        st.caption("先使用默认示例 000004.SZ，点击“开始检测”即可。财务数据来自东方财富免费接口（实时）。")
