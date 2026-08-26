#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""报告生成 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 报告生成agent.py --server.port 8507
说明：报告依赖全流水线输出，本页直接跑 SweepingOrchestrator 全流程后渲染 Markdown/JSON 报告，
     并展示概率条 / SHAP 归因图与已生成报告文件列表（output/reports/ 存档）。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents import SweepingOrchestrator
from backend.config import OUTPUT_DIR
from ui.theme import apply_scan_theme

st.set_page_config(
    page_title="报告生成 Agent",
    page_icon=":material/description:",
    layout="wide",
)
apply_scan_theme()


@st.cache_data(ttl="6h", max_entries=10, show_spinner=False)
def generate_report(company: str, as_of: str, window: int, use_llm: bool,
                    use_finbert: bool, use_llm_summary: bool,
                    use_semantic_cases: bool) -> dict:
    """全流程（公告研读→财务检测→预测→案例→归因→报告）后渲染报告。"""
    orch = SweepingOrchestrator(
        use_llm=use_llm,
        use_finbert=use_finbert,
        use_semantic_cases=use_semantic_cases,
    )
    ctx = orch.sweep_one(company, window=window, as_of=as_of, use_llm_summary=use_llm_summary)
    return ctx.to_dict()


def list_reports(max_n: int = 10) -> list[dict]:
    try:
        mp = Path(OUTPUT_DIR) / "reports" / "manifest.json"
        if not mp.exists():
            return []
        data = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        reports_dir = mp.parent
        return [
            item for item in data
            if isinstance(item, dict)
            and (reports_dir / str(item.get("md_file", ""))).is_file()
            and (reports_dir / str(item.get("json_file", ""))).is_file()
        ][:max_n]
    except Exception:
        return []


st.title("报告生成 Agent")
st.caption("聚合全流水线结果（预测/财务/公告/案例/归因/trace）→ 八章风控函件式 Markdown 报告 + 结构化 JSON，自动归档 output/reports/。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("数据截止日", value=date.today(), max_value=date.today())
    window = st.selectbox("预测窗口（天）", [30, 60, 90], index=1)
    use_llm = st.toggle("启用 LLM 精细抽取", value=False, help="需要 DEEPSEEK_API_KEY。")
    use_finbert = st.toggle("启用 FinBERT", value=False)
    use_semantic_cases = st.toggle("启用 BGE 语义案例检索", value=True, key="report_use_semantic_cases")
    use_llm_summary = st.toggle(
        "启用 DeepSeek 执行摘要（deepseek-v4-flash）", value=False,
        help="用大模型生成风控函件式执行摘要；关闭时用规则拼装。演示时建议勾选。")
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
                result = generate_report(normalized, as_of_value.isoformat(), window,
                                         use_llm, use_finbert, use_llm_summary,
                                         use_semantic_cases)
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
    pred = result.get("prediction", {})
    att = result.get("attribution", {})
    if any(t.get("status") == "needs_choice" for t in trace):
        st.warning("BGE 语义检索未完成，本次暂使用标签检索。")
        if st.button("切换为快速标签检索并重新运行", key="report_fast_cases"):
            st.session_state["report_use_semantic_cases"] = False
            st.rerun()

    with st.container(horizontal=True):
        st.metric("流水线环节", len(trace), border=True)
        done = sum(1 for t in trace if t.get("status") == "done")
        st.metric("完成环节", done, border=True)
        st.metric("报告字数", len(markdown), border=True)
        st.metric("数据源", pred.get("data_source", "realtime"), border=True)

    # ---------- 可视化：概率条 + SHAP ----------
    colA, colB = st.columns(2)
    with colA:
        st.subheader("问询概率（30/60/90 天）")
        probs = {f"{w}天": pred.get(f"probability_{w}d") for w in (30, 60, 90)}
        if any(v is not None for v in probs.values()):
            st.bar_chart({k: [v] for k, v in probs.items() if v is not None}, height=220)
        else:
            st.info("未预测")
    with colB:
        st.subheader("SHAP 归因 Top 特征")
        shap = pred.get("shap_features", []) or []
        if shap:
            df = pd.DataFrame(shap, columns=["特征", "贡献值"]).head(10)
            st.bar_chart(df.set_index("特征"), y="贡献值", height=220, color="primary")
        else:
            st.info("无 SHAP 特征（规则降级归因见报告）")

    # ---------- 执行摘要 ----------
    if report_json.get("executive_summary"):
        with st.container(border=True):
            st.markdown("**一、执行摘要**")
            st.write(report_json["executive_summary"])

    with st.expander("🔍 流水线追踪", expanded=True):
        for t in trace:
            agent = t.get("agent", "?")
            status_icon = "✅" if t.get("status") == "done" else ("⏭️" if t.get("status") == "skipped" else "⚠️")
            st.markdown(f"{status_icon} **{agent}** ｜ {t.get('status')} ｜ {t.get('latency_ms', '')}ms ｜ {str(t.get('output_summary', ''))[:80]}")

    st.subheader("Markdown 报告（八章）")
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
        st.write("八章风控函件式报告（函件头/执行摘要/画像/评分卡/财务/公告/案例/证据链路与免责）+ 概率条与 SHAP 图 + 报告文件归档。")
        st.caption("先使用默认示例 000004.SZ，点击“生成报告”即可。首次运行需下载公告 PDF，可能等待数分钟。")

# ---------- 已生成报告文件列表 ----------
st.divider()
st.subheader("📁 已生成报告（output/reports/ 存档）")
for r in list_reports():
    report_path = Path(OUTPUT_DIR) / "reports" / str(r.get("md_file", ""))
    st.markdown(f"- **{r.get('report_id')}** ｜ 60d 概率 {r.get('probability_60d')} ｜ "
                f"{r.get('risk_level')} ｜ {r.get('generated_at')}")
    st.download_button(
        f"⬇️ {r.get('md_file')}",
        data=report_path.read_text(encoding="utf-8"),
        file_name=r.get("md_file", "report.md"),
        mime="text/markdown",
        key=f"dl_{r.get('report_id')}",
    )
