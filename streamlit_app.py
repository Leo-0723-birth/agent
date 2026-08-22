#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公告研读 Agent 的可审计 Streamlit 展示入口。"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


# 当前文件位于仓库根目录；显式加入根目录，兼容本地与 Streamlit Cloud。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.context import Context
from backend.dashboard_utils import (
    risk_theme_distribution_rows,
    risk_window_comparison_rows,
)
from backend.skills.announcement_search import CninfoAnnouncementSource


st.set_page_config(
    page_title="公告研读 Agent",
    page_icon=":material/article:",
    layout="wide",
)


SEVERITY_LABELS = {
    5: "高",
    4: "高",
    3: "中",
    2: "低",
    1: "提示",
}

FILTER_REASON_LABELS = {
    "governance_rules": "公司治理制度",
    "candidate_declaration": "候选人/提名人声明",
    "general_management_policy": "通用管理制度",
    "governance_eligibility_clause": "董监高职责或任职资格条款",
    "conditional_or_hypothetical_clause": "条件性或假设性描述",
    "prohibition_or_duty_clause": "禁止性或职责条款",
    "definition_or_template_clause": "定义或附件模板",
    "accounting_policy_or_table": "会计政策或报表模板",
    "missing_event_anchors": "缺少主体与现实事件动作",
    "negated_context": "否定语境",
    "rule_paragraph_exclusion": "风险词典段落排除规则",
    "excluded_context": "非事实语境",
}


@st.cache_data(ttl="6h", max_entries=20, show_spinner=False)
def analyze_company(
    company: str,
    as_of: str,
    max_documents: int,
    use_ocr: bool,
    use_finbert: bool,
    use_llm: bool,
) -> dict:
    """执行一次可复用的公告研读；结果按输入和开关缓存。"""
    source = CninfoAnnouncementSource(
        max_documents=max_documents,
        ocr_enabled=use_ocr,
    )
    agent = AnnouncementReaderAgent(
        source=source,
        use_finbert=use_finbert,
        use_llm=use_llm,
    )
    context = Context(company=company, as_of=as_of)
    result, trace = agent.run(company, context)
    payload = result.to_dict()
    payload["run_trace"] = trace
    return payload


def show_status_badges(channel_summary: dict, data_quality: dict) -> None:
    rule_status = channel_summary.get("rule", {}).get("status", "unknown")
    finbert_status = channel_summary.get("finbert", {}).get("status", "unknown")
    llm_status = channel_summary.get("llm", {}).get("status", "unknown")
    ocr_status = data_quality.get("ocr_status", "unknown")
    st.markdown(
        f":green-badge[巨潮官方主源] "
        f":blue-badge[规则 {rule_status}] "
        f":blue-badge[OCR {ocr_status}] "
        f":gray-badge[FinBERT {finbert_status}] "
        f":gray-badge[LLM {llm_status}]"
    )


def show_kpis(semantic: dict) -> None:
    stats = semantic.get("stats", {})
    quality = semantic.get("data_quality", {})
    ratio = quality.get("evidence_valid_ratio")
    ratio_text = "—" if ratio is None else f"{ratio:.1%}"
    with st.container(horizontal=True):
        st.metric("近一年公告", stats.get("announcement_count", 0), border=True)
        st.metric("风险事件", stats.get("risk_factor_count", 0), border=True)
        st.metric("高风险事件", stats.get("high_severity_count", 0), border=True)
        st.metric("有效证据比例", ratio_text, border=True)


def risk_dataframe(factors: list[dict]) -> pd.DataFrame:
    rows = []
    for item in factors:
        severity = int(item.get("severity") or 1)
        rows.append(
            {
                "日期": item.get("announcement_date", ""),
                "风险等级": SEVERITY_LABELS.get(severity, str(severity)),
                "L1": item.get("taxonomy_l1") or item.get("category", ""),
                "L2": item.get("taxonomy_l2") or item.get("label", ""),
                "风险描述": item.get("description", ""),
                "原文证据": item.get("evidence", ""),
                "证据有效": bool(item.get("evidence_valid")),
                "文本来源": item.get("text_extraction", ""),
                "OCR 置信度": item.get("ocr_mean_confidence"),
                "抽取通道": item.get("method", ""),
                "公告详情": item.get("source_url", ""),
                "官方 PDF": item.get("pdf_url", ""),
            }
        )
    return pd.DataFrame(rows)


def announcement_dataframe(announcements: list[dict]) -> pd.DataFrame:
    rows = []
    for item in announcements:
        rows.append(
            {
                "日期": item.get("date", ""),
                "公告标题": item.get("title", ""),
                "类型": item.get("type", ""),
                "研读状态": (
                    "已过滤"
                    if item.get("analysis_status") == "excluded_by_title"
                    else "进入研读"
                ),
                "过滤原因": FILTER_REASON_LABELS.get(
                    item.get("analysis_skip_reason", ""),
                    item.get("analysis_skip_reason", ""),
                ),
                "PDF 状态": item.get("text_status", ""),
                "OCR 状态": item.get("ocr_status", ""),
                "OCR 成功页": item.get("ocr_succeeded_pages", 0),
                "OCR 失败页": item.get("ocr_failed_pages", 0),
                "OCR 平均置信度": item.get("ocr_mean_confidence"),
                "提取字符数": item.get("char_count", 0),
                "公告详情": item.get("source_url", ""),
                "官方 PDF": item.get("pdf_url", ""),
                "SHA256": item.get("content_sha256", ""),
            }
        )
    return pd.DataFrame(rows)


def suppression_dataframe(per_announcement: dict, announcements: list[dict]) -> pd.DataFrame:
    metadata = {item.get("id"): item for item in announcements}
    rows = []
    for announcement_id, payload in per_announcement.items():
        announcement = metadata.get(announcement_id, {})
        for hit in payload.get("suppressed_rule_hits", []):
            reason = hit.get("suppression_reason", "")
            rows.append(
                {
                    "日期": announcement.get("date", ""),
                    "公告标题": announcement.get("title", ""),
                    "L2": hit.get("label", ""),
                    "命中词": hit.get("matched_keyword", ""),
                    "过滤原因": FILTER_REASON_LABELS.get(reason, reason),
                    "原文片段": hit.get("evidence", ""),
                    "公告详情": announcement.get("source_url", ""),
                }
            )
    return pd.DataFrame(rows)


st.title("公告研读 Agent")
st.caption("输入上市公司代码或准确名称，读取截止日以前近一年的巨潮官方公告并输出可核验证据。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input(
        "数据截止日",
        value=date.today(),
        max_value=date.today(),
        help="系统不会读取该日期之后的公告。",
    )
    max_documents = st.slider(
        "最多解析 PDF 数量",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="公告元数据仍覆盖近一年；为控制等待时间，只解析最新的指定数量 PDF。",
    )
    use_finbert = st.toggle(
        "启用 FinBERT",
        value=False,
        help="仅在已配置模型时生效；当前相似度不是风险概率。",
    )
    use_ocr = st.toggle(
        "启用扫描 PDF OCR",
        value=True,
        help="仅对没有有效文本层的图像页调用 RapidOCR，不重复识别正常文本页。",
    )
    use_llm = st.toggle(
        "启用 LLM 精细抽取",
        value=False,
        help="需要 DEEPSEEK_API_KEY；证据无法在原文逐字找到时会被拒绝。",
    )
    st.caption("公告事实主源：巨潮资讯网。规则/模型输出均为待复核风险信号。")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000001",
        placeholder="例如：000001、平安银行",
    )
    submitted = st.form_submit_button(
        "开始研读",
        type="primary",
        icon=":material/search:",
    )

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码或准确名称。", icon=":material/error:")
    else:
        try:
            with st.status("正在查询巨潮公告并核验证据……", expanded=True) as status:
                st.write("解析公司身份与近一年公告元数据")
                result = analyze_company(
                    normalized,
                    as_of_value.isoformat(),
                    max_documents,
                    use_ocr,
                    use_finbert,
                    use_llm,
                )
                st.write("下载或读取 PDF 缓存，执行风险抽取")
                st.session_state["announcement_analysis"] = result
                status.update(label="公告研读完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("announcement_analysis", None)
            st.error(
                f"研读失败：{type(exc).__name__}: {exc}",
                icon=":material/error:",
            )

result = st.session_state.get("announcement_analysis")
if result:
    semantic = result.get("semantic", {})
    quality = semantic.get("data_quality", {})
    features = semantic.get("f1_features", {})
    st.subheader(f"{result.get('name') or result.get('company')}（{result.get('company')}）")
    st.caption(
        f"特征锚点：{result.get('as_of')} · 回看 {quality.get('lookback_days', 365)} 天 · "
        f"来源：{quality.get('source', '巨潮资讯网')}"
    )
    show_status_badges(semantic.get("channel_summary", {}), quality)
    show_kpis(semantic)

    if quality.get("title_excluded_count", 0):
        st.info(
            f"已保留全部官方公告元数据，其中 {quality['title_excluded_count']} 份明确制度/候选人声明类公告"
            "未下载 PDF、未进入风险抽取；可在审计信息中核对原因。",
            icon=":material/filter_alt:",
        )

    if quality.get("document_limit_truncated"):
        st.warning(
            "本次只解析了设置数量以内的最新 PDF；公告元数据仍覆盖近一年。"
            "如需完整正文研读，请提高左侧 PDF 数量后重新运行。",
            icon=":material/warning:",
        )
    if quality.get("not_fulltext_count", 0):
        st.warning(
            f"已启用 OCR，但仍有 {quality['not_fulltext_count']} 份已尝试公告没有可用全文；"
            "请在公告清单中查看 OCR/PDF 失败状态。",
            icon=":material/document_scanner:",
        )
    if quality.get("ocr_status") == "not_available":
        st.error(
            "OCR 依赖不可用，请安装 requirements.txt 中的 rapidocr 与 onnxruntime。",
            icon=":material/error:",
        )
    elif quality.get("ocr_succeeded_pages", 0):
        st.success(
            f"RapidOCR 已成功识别 {quality['ocr_succeeded_pages']} 个扫描页；"
            f"失败 {quality.get('ocr_failed_pages', 0)} 页。",
            icon=":material/document_scanner:",
        )

    overview_tab, risk_tab, announcement_tab, audit_tab = st.tabs(
        [
            ":material/analytics: 风险概览",
            ":material/warning: 风险证据",
            ":material/article: 公告清单",
            ":material/fact_check: 审计信息",
        ]
    )

    with overview_tab:
        scalar = features.get("scalar_features", {})
        comparison_data = pd.DataFrame(risk_window_comparison_rows(scalar))
        with st.container(border=True):
            st.subheader("30/60/90 天风险数量对比")
            st.caption(
                "同时展示公告覆盖量、风险事件和其中的高风险事件；"
                "一份公告可能对应多个风险事件，因此该图是数量对比，不是风险概率。"
            )
            st.bar_chart(
                comparison_data,
                x="时间窗口",
                y=["公告总数", "风险事件", "高风险事件"],
                y_label="数量",
                color=["gray", "orange", "red"],
                stack=False,
                sort=False,
                height=330,
            )
            with st.expander(
                "查看三个窗口的精确数据",
                icon=":material/table_chart:",
            ):
                st.dataframe(
                    comparison_data,
                    hide_index=True,
                    column_config={
                        "公告总数": st.column_config.NumberColumn(format="%d"),
                        "风险事件": st.column_config.NumberColumn(format="%d"),
                        "高风险事件": st.column_config.NumberColumn(format="%d"),
                        "每份公告风险事件": st.column_config.NumberColumn(
                            "风险事件/公告",
                            format="%.2f",
                            help="风险事件数除以公告数，仅表示信号密度，不是概率。",
                        ),
                    },
                )

        category_counts = features.get("category_event_counts", {})
        with st.container(border=True):
            window_label = st.segmented_control(
                "统计窗口",
                ["最近 30 天", "最近 60 天", "最近 90 天"],
                default="最近 90 天",
                key="risk_theme_window",
            )
            window_days = {
                "最近 30 天": 30,
                "最近 60 天": 60,
                "最近 90 天": 90,
            }[window_label or "最近 90 天"]
            rows = risk_theme_distribution_rows(category_counts, window_days)
            chart_data = pd.DataFrame(rows)
            st.subheader(f"{window_label or '最近 90 天'}风险主题分布")
            st.caption("按累计时间窗口统计；同一公告、同一风险主题的重复命中已去重。")
            with st.container(horizontal=True):
                st.metric(
                    "风险事件",
                    scalar.get(f"risk_event_count_{window_days}d", 0),
                    border=True,
                )
                st.metric(
                    "高风险事件",
                    scalar.get(f"high_risk_event_count_{window_days}d", 0),
                    border=True,
                )
                st.metric("涉及主题", len(rows), border=True)

            if chart_data.empty:
                st.info(
                    f"{window_label or '最近 90 天'}内没有识别到风险事件。",
                    icon=":material/info:",
                )
            else:
                st.bar_chart(
                    chart_data,
                    x="图表标签",
                    y="事件数",
                    x_label="风险主题",
                    y_label="事件数",
                    horizontal=True,
                    sort="-事件数",
                    color="primary",
                    height=max(260, len(chart_data) * 48),
                )
                with st.expander(
                    "查看精确计数与占比",
                    icon=":material/table_chart:",
                ):
                    st.dataframe(
                        chart_data[["主题代码", "风险主题", "事件数", "占比"]],
                        hide_index=True,
                        column_config={
                            "事件数": st.column_config.NumberColumn("事件数", format="%d"),
                            "占比": st.column_config.ProgressColumn(
                                "主题计数占比",
                                format="percent",
                                min_value=0,
                                max_value=1,
                            ),
                        },
                    )
        with st.container(border=True):
            st.markdown("**口径说明**")
            st.write(features.get("window_semantics", ""))
            st.caption(features.get("probability_status", "F1 是文本特征，不是风险概率。"))

    with risk_tab:
        risk_df = risk_dataframe(semantic.get("risk_factors", []))
        if risk_df.empty:
            st.info("规则和已启用模型均未产生有证据的风险信号。", icon=":material/check_circle:")
        else:
            st.dataframe(
                risk_df,
                hide_index=True,
                column_config={
                    "证据有效": st.column_config.CheckboxColumn("证据有效"),
                    "OCR 置信度": st.column_config.NumberColumn(
                        "OCR 置信度", format="percent"
                    ),
                    "公告详情": st.column_config.LinkColumn("公告详情", display_text="打开"),
                    "官方 PDF": st.column_config.LinkColumn("官方 PDF", display_text="查看"),
                },
            )
            st.caption("风险信号必须结合公告原文复核，不构成事实认定或投资建议。")

    with announcement_tab:
        announcement_df = announcement_dataframe(semantic.get("announcements", []))
        st.dataframe(
            announcement_df,
            hide_index=True,
            column_config={
                "提取字符数": st.column_config.NumberColumn("提取字符数", format="%d"),
                "OCR 成功页": st.column_config.NumberColumn("OCR 成功页", format="%d"),
                "OCR 失败页": st.column_config.NumberColumn("OCR 失败页", format="%d"),
                "OCR 平均置信度": st.column_config.NumberColumn(
                    "OCR 平均置信度", format="percent"
                ),
                "公告详情": st.column_config.LinkColumn("公告详情", display_text="打开"),
                "官方 PDF": st.column_config.LinkColumn("官方 PDF", display_text="查看"),
            },
        )

    with audit_tab:
        rule_summary = semantic.get("channel_summary", {}).get("rule", {})
        llm_summary = semantic.get("channel_summary", {}).get("llm", {})
        with st.container(horizontal=True):
            st.metric(
                "标题过滤公告",
                quality.get("title_excluded_count", 0),
                border=True,
            )
            st.metric(
                "规则语境过滤",
                rule_summary.get("suppressed_count", 0),
                border=True,
            )
            st.metric(
                "LLM 非事实语境拒绝",
                llm_summary.get("rejected_nonfactual_context", 0),
                border=True,
            )
        left, right = st.columns(2)
        with left.container(border=True):
            st.subheader("三通道状态")
            st.json(semantic.get("channel_summary", {}), expanded=True)
        with right.container(border=True):
            st.subheader("数据质量")
            st.json(quality, expanded=True)
        announcements = semantic.get("announcements", [])
        filtered_announcements = [
            item for item in announcements
            if item.get("analysis_status") == "excluded_by_title"
        ]
        if filtered_announcements:
            with st.expander(
                "查看按标题过滤的公告",
                icon=":material/filter_alt:",
            ):
                filtered_df = announcement_dataframe(filtered_announcements)
                st.dataframe(
                    filtered_df[
                        ["日期", "公告标题", "过滤原因", "公告详情", "官方 PDF"]
                    ],
                    hide_index=True,
                    column_config={
                        "公告详情": st.column_config.LinkColumn(
                            "公告详情", display_text="打开"
                        ),
                        "官方 PDF": st.column_config.LinkColumn(
                            "官方 PDF", display_text="查看"
                        ),
                    },
                )
        suppressed_df = suppression_dataframe(
            semantic.get("per_announcement", {}), announcements
        )
        if not suppressed_df.empty:
            with st.expander(
                "查看被规则过滤的候选证据",
                icon=":material/rule:",
            ):
                st.dataframe(
                    suppressed_df,
                    hide_index=True,
                    column_config={
                        "公告详情": st.column_config.LinkColumn(
                            "公告详情", display_text="打开"
                        )
                    },
                )
        with st.expander("查看完整 F1 特征", icon=":material/data_object:"):
            st.json(features, expanded=False)
        st.info(semantic.get("source_policy", ""), icon=":material/verified:")
        st.download_button(
            "下载完整 JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"announcement_f1_{result.get('company', 'company')}_{result.get('as_of', '')}.json",
            mime="application/json",
            icon=":material/download:",
        )
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("30/60/90 天风险数量、风险主题分布、逐条原文证据、官方公告链接、PDF 状态和三通道审计记录。")
        st.caption("先使用默认示例 000001，点击“开始研读”即可。首次运行需要下载 PDF，可能等待数分钟。")
