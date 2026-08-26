# -*- coding: utf-8 -*-
"""主控 Agent：评委演示首页与完整扫雷流水线。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agents import SweepingOrchestrator
from backend.config import OUTPUT_DIR
from ui.charts import case_score_chart, probability_chart, risk_severity_chart, shap_chart
from ui.components import evidence_records, render_evidence_cards, render_metric_grid, render_page_header, render_source_index, render_trace
from ui.data import dataset_shape, load_model_summary, load_offline_context
from ui.session import publish_analysis
from ui.theme import apply_page_style

st.set_page_config(page_title="监管问询风险简报", page_icon=":material/radar:", layout="wide")
apply_page_style()


def list_reports(max_n: int = 20) -> list[dict]:
    try:
        path = Path(OUTPUT_DIR) / "reports" / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return data[:max_n] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def load_report_text(filename: str) -> str:
    try:
        return (Path(OUTPUT_DIR) / "reports" / filename).read_text(encoding="utf-8")
    except OSError:
        return ""


def live_context_to_result(ctx, window: int) -> dict:
    financial, semantic = ctx.financial, ctx.semantic
    report = ctx.report or {}
    return {
        "company": ctx.company,
        "name": (report.get("json", {}) or {}).get("name", ""),
        "window": window,
        "as_of": ctx.as_of,
        "prediction": ctx.prediction or {},
        "financial": {"risk_level": financial.risk_level, "skip": financial.skip, "skip_reason": financial.skip_reason, "anomaly_list": financial.anomaly_list, "features_count": len(financial.features)},
        "semantic": {"stats": semantic.stats, "risk_factors": semantic.risk_factors, "data_quality": semantic.data_quality},
        "cases": ctx.cases or [],
        "attribution": ctx.attribution or {},
        "trace_log": ctx.trace_log or [],
        "report": report,
        "snapshot": {"mode": "live", "label": "本次实时运行", "generated_at": ""},
    }


if "dashboard_result" not in st.session_state:
    publish_analysis(load_offline_context("000004.SZ"), source="离线演示快照")

with st.sidebar:
    st.subheader("公司扫雷")
    st.caption("默认快照可离线秒开；需要时再启动完整流水线。")
    with st.form("sweep_form"):
        codes_text = st.text_area("公司代码（每行一个）", "000004.SZ", height=86)
        window = st.selectbox("预测窗口", [30, 60, 90], index=1, format_func=lambda x: f"{x} 天")
        use_llm = st.checkbox("启用 LLM 精细抽取", value=False, help="需要在 .env 中配置密钥")
        use_llm_summary = st.checkbox("生成 LLM 执行摘要", value=False)
        run_clicked = st.form_submit_button(":material/play_arrow: 运行完整扫雷", type="primary", width="stretch")
    if st.button(":material/offline_bolt: 恢复离线演示快照", width="stretch"):
        publish_analysis(load_offline_context("000004.SZ"), source="离线演示快照")
        st.rerun()

if run_clicked:
    codes = [code.strip() for code in codes_text.splitlines() if code.strip()]
    if not codes:
        st.warning("请输入至少一个公司代码。")
    else:
        orchestrator = SweepingOrchestrator(use_llm=use_llm, use_finbert=True)
        for code in codes:
            try:
                with st.status(f"正在分析 {code}", expanded=True) as status:
                    ctx = orchestrator.sweep_one(code, window=window, use_llm_summary=use_llm_summary)
                    for step in ctx.trace_log:
                        st.write(f"**{step.get('agent', 'Agent')}** · {step.get('status', 'done')} · {step.get('latency_ms', '—')} ms · {str(step.get('output_summary', ''))[:72]}")
                    status.update(label=f"{code} 分析完成", state="complete")
                publish_analysis(live_context_to_result(ctx, window), source="主控 Agent 实时分析")
            except Exception as exc:
                st.error(f"{code} 实时分析未完成：{exc}。页面继续保留上一次可用快照。")

result = st.session_state.dashboard_result
prediction = result.get("prediction", {})
financial = result.get("financial", {})
semantic = result.get("semantic", {})
report_json = result.get("report", {}).get("json", {}) or {}
snapshot = result.get("snapshot", {})
company = result.get("company") or "000004.SZ"
company_name = result.get("name") or "上市公司"

render_page_header(
    f"{company_name} · 监管问询风险简报",
    "用模型概率定位优先级，用公告、财务与历史案例解释风险，并保留可回溯证据链。",
    status=snapshot.get("label", "可用快照"),
    status_kind="live" if snapshot.get("mode") == "live" else "offline",
    metadata=[company, f"数据截止 {result.get('as_of', '—')}", f"{result.get('window', 60)} 天窗口"],
)

summary = report_json.get("executive_summary")
if summary:
    with st.container(border=True):
        st.markdown("#### 研判结论")
        st.write(summary)

model_summary = load_model_summary()
metrics_60 = model_summary.get("windows", {}).get("60", {}).get("Ensemble", {})
rows, columns = dataset_shape()
render_metric_grid([
    {"label": "60D 模型 AUC", "value": f"{metrics_60.get('AUC', 0):.4f}", "note": "集成模型判别能力"},
    {"label": "Top 10% 召回", "value": f"{metrics_60.get('Top10%Recall', 0):.1%}", "note": "高风险样本覆盖率"},
    {"label": "建模样本", "value": f"{rows:,}", "note": "离线训练与验证样本"},
    {"label": "建模特征", "value": f"{columns:,}", "unit": "维", "note": "含标识与标签字段"},
    {"label": "历史问询案例", "value": f"{4_785:,}", "note": "用于相似案例检索"},
])

st.markdown("### 风险判断与模型解释")
c1, c2, c3 = st.columns(3)
p60 = prediction.get("probability_60d")
c1.metric("60 天问询概率", f"{p60:.2%}" if p60 is not None else "未预测")
c2.metric("风险等级", prediction.get("risk_level") or financial.get("risk_level") or "—")
confidence = prediction.get("confidence")
c3.metric("模型置信度", f"{confidence:.1%}" if confidence is not None else "—")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### 30 / 60 / 90 天问询概率")
        st.altair_chart(probability_chart(prediction), width="stretch")
        st.caption("雾蓝、青绿与柔金分别对应 30、60、90 天预测窗口；柱顶为模型输出概率。")
with right:
    with st.container(border=True):
        st.markdown("#### Top 特征贡献")
        st.altair_chart(shap_chart(prediction.get("shap_features", [])), width="stretch")
        st.caption("珊瑚红表示推升风险，雾蓝表示降低风险；数值为 SHAP 局部贡献。")

st.markdown("### 关键证据与问题定位")
st.caption("每条证据同时标明问题、系统记录与原文入口，评委可直接追溯。")
render_evidence_cards(evidence_records(result, limit=6))

chart_left, chart_right = st.columns(2)
with chart_left:
    with st.container(border=True):
        st.markdown("#### 公告风险要素分布")
        factors = semantic.get("risk_factors", [])
        if factors:
            st.altair_chart(risk_severity_chart(factors), width="stretch")
            st.caption("珊瑚红对应高严重度，柔金对应关注项，青绿与雾蓝为较低等级。")
        else:
            st.info("当前结果未提取到公告风险要素。")
with chart_right:
    with st.container(border=True):
        st.markdown("#### 相似历史问询案例")
        cases = result.get("cases", [])
        if cases:
            st.altair_chart(case_score_chart(cases), width="stretch")
            st.caption("青绿色柱表示案例检索可信度；悬停可查看问询类型和精确得分。")
        else:
            st.info("当前结果未检索到相似案例。")

st.markdown('<div id="agent-trace"></div>', unsafe_allow_html=True)
st.markdown("### Agent 推理链路")
render_trace(result.get("trace_log", []))
with st.expander("查看完整 trace_log"):
    st.json(result.get("trace_log", []))

st.markdown("### 数据与证据入口")
render_source_index(company)

st.markdown("### 报告归档与下载")
reports = list_reports()
if reports:
    labels = {f"{item.get('report_id')} · P60 {item.get('probability_60d', '—')} · {item.get('risk_level', '—')}": item for item in reports}
    selected = labels[st.selectbox("已生成报告", list(labels))]
    d1, d2 = st.columns(2)
    d1.download_button(":material/download: 下载 Markdown", data=load_report_text(selected.get("md_file", "")), file_name=selected.get("md_file", "report.md"), mime="text/markdown", width="stretch")
    d2.download_button(":material/data_object: 下载索引 JSON", data=json.dumps(selected, ensure_ascii=False, indent=2), file_name=selected.get("json_file", "report.json"), mime="application/json", width="stretch")
    with st.expander("预览报告原文"):
        st.markdown(load_report_text(selected.get("md_file", "")))
else:
    st.info("暂无已归档报告。运行完整扫雷后会自动保存。")

st.caption("说明：本系统输出用于风险筛查与研究辅助，不构成投资建议或监管结论。")
