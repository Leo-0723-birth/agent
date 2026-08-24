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
    risk_interval_comparison_rows,
    risk_monthly_severity_rows,
    risk_theme_distribution_rows,
    risk_theme_heatmap_rows,
)
from backend.skills.announcement_context_filter import FILTER_VERSION
from backend.skills.announcement_search import CninfoAnnouncementSource
from backend.skills.competition_history import CompetitionAwareAnnouncementSource
from backend.skills.offline_announcement_snapshot import OfflineAnalysisSnapshotStore


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


def analyze_company(
    company: str,
    as_of: str,
    use_ocr: bool,
    use_finbert: bool,
    use_llm: bool,
    filter_version: str,
    progress_callback=None,
) -> dict:
    """执行公告研读；PDF/解析文件仍使用磁盘缓存，进度按真实阶段回传。"""
    cached = OfflineAnalysisSnapshotStore().lookup(
        company,
        as_of,
        use_ocr,
        use_finbert,
        use_llm,
        filter_version,
    )
    if cached.get("status") == "hit":
        if progress_callback is not None:
            progress_callback({"event": "offline_analysis_completed", **cached})
        return cached["result"]
    source = CompetitionAwareAnnouncementSource(
        CninfoAnnouncementSource(
            max_documents=None,
            ocr_enabled=use_ocr,
            progress_callback=progress_callback,
        ),
        progress_callback=progress_callback,
    )
    agent = AnnouncementReaderAgent(
        source=source,
        use_finbert=use_finbert,
        use_llm=use_llm,
        progress_callback=progress_callback,
    )
    context = Context(company=company, as_of=as_of)
    result, trace = agent.run(company, context)
    payload = result.to_dict()
    payload["announcement_filter_version"] = filter_version
    payload["run_trace"] = trace
    return payload


def build_progress_handler(
    status,
    history_line,
    online_line,
    merge_line,
    analysis_line,
    progress_bar,
):
    """把后端真实事件转换成三阶段可见进度，不使用虚假计时。"""
    current_company = {"name": "", "code": ""}

    def update(event: dict) -> None:
        event_name = event.get("event", "")
        if event_name == "history_check_started":
            status.update(label="步骤 1/3 · 正在检查比赛历史库……")
            history_line.markdown(
                ":material/hourglass_top: **步骤 1 · 比赛历史库** — 正在按本次输入检索"
            )
            progress_bar.progress(0.04, text="正在检查历史数据")
        elif event_name == "offline_analysis_completed":
            result = event.get("result") or {}
            semantic = result.get("semantic") or {}
            quality = semantic.get("data_quality") or {}
            stats = semantic.get("stats") or {}
            status.update(label="已命中 000004.SZ 离线分析快照")
            history_line.markdown(
                f":material/check_circle: **步骤 1 · 比赛历史库** — 已加载快照中的历史候选"
            )
            online_line.markdown(
                f":material/offline_bolt: **步骤 2 · 巨潮官方公告离线快照** — 数据锚点 "
                f"{quality.get('snapshot_as_of') or result.get('as_of')}，读取 "
                f"{stats.get('announcement_count', 0)} 份官方公告"
            )
            merge_line.markdown(
                ":material/layers: **步骤 3 · 分层结果** — 直接加载已核验的规则分析结果；"
                "历史候选与当前公告仍保持分层"
            )
            analysis_line.markdown(
                f":material/check_circle: **快速研读完成** — "
                f"{stats.get('risk_factor_count', 0)} 条待复核风险事件"
            )
            progress_bar.progress(1.0, text="离线快照加载完成")
        elif event_name == "offline_snapshot_started":
            status.update(label="步骤 2/3 · 正在检查 000004.SZ 离线公告快照……")
            online_line.markdown(
                ":material/offline_bolt: **步骤 2 · 本地快照** — 正在匹配公司、截止日和回看窗口"
            )
            progress_bar.progress(0.16, text="正在检查离线公告快照")
        elif event_name == "offline_snapshot_completed":
            current_company["name"] = str(event.get("company_name") or "")
            current_company["code"] = str(event.get("secucode") or "")
            online_line.markdown(
                f":material/check_circle: **步骤 2 · 巨潮官方公告离线快照** — 命中 "
                f"{current_company['name']}（{current_company['code']}），数据锚点 "
                f"{event.get('snapshot_as_of')}，读取 {event.get('announcement_count', 0)} 份公告"
            )
            progress_bar.progress(0.70, text="离线官方公告读取完成")
        elif event_name == "offline_snapshot_missed":
            online_line.markdown(
                ":material/cloud_sync: **步骤 2 · 本地快照** — 未覆盖本次公司或截止日，转为查询巨潮"
            )
        elif event_name == "history_check_completed":
            match_status = event.get("status")
            if match_status == "hit":
                history_line.markdown(
                    f":material/check_circle: **步骤 1 · 比赛历史库** — 命中 {event.get('document_count', 0)} 份历史公告，"
                    f"覆盖 {event.get('date_start') or '未知'} 至 {event.get('date_end') or '未知'}"
                )
            elif match_status == "unavailable":
                history_line.markdown(
                    ":material/database_off: **步骤 1 · 比赛历史库** — 本机未配置数据目录，将继续查询巨潮"
                )
            else:
                history_line.markdown(
                    ":material/info: **步骤 1 · 比赛历史库** — 输入暂未直接命中，解析公司身份后再按代码复查"
                )
            progress_bar.progress(0.12, text="历史库初次检索完成")
        elif event_name == "history_identity_recheck_started":
            history_line.markdown(
                f":material/manage_search: **步骤 1 · 比赛历史库** — 正在按解析后的代码 {event.get('secucode', '')} 复查"
            )
        elif event_name == "online_company_started":
            status.update(label="步骤 2/3 · 正在查询巨潮近一年公告……")
            online_line.markdown(
                ":material/hourglass_top: **步骤 2 · 巨潮资讯网** — 正在解析公司代码与名称"
            )
            progress_bar.progress(0.16, text="正在解析公司身份")
        elif event_name == "online_company_completed":
            current_company["name"] = str(event.get("company_name") or "")
            current_company["code"] = str(event.get("secucode") or "")
            online_line.markdown(
                f":material/check_circle: **步骤 2 · 巨潮资讯网** — 已解析为 "
                f"{current_company['name']}（{current_company['code']}），正在查询近一年公告"
            )
            progress_bar.progress(0.22, text="公司身份解析完成")
        elif event_name == "online_metadata_started":
            online_line.markdown(
                f":material/cloud_download: **步骤 2 · 巨潮资讯网** — 已解析为 "
                f"{current_company['name']}（{current_company['code']}），正在分页获取近一年公告"
            )
            progress_bar.progress(0.27, text="正在获取公告元数据")
        elif event_name == "online_metadata_completed":
            online_line.markdown(
                f":material/check_circle: **步骤 2 · 巨潮资讯网** — 解析为 "
                f"{current_company['name']}（{current_company['code']}），取得 "
                f"{event.get('announcement_count', 0)} 份近一年公告元数据；"
                f"{event.get('eligible_count', 0)} 份可研读，准备处理 {event.get('pdf_count', 0)} 份 PDF"
            )
            progress_bar.progress(0.36, text="公告元数据获取完成")
        elif event_name == "pdf_processing":
            current = int(event.get("current") or 0)
            total = max(1, int(event.get("total") or 1))
            title = str(event.get("title") or "")
            online_line.markdown(
                f":material/picture_as_pdf: **步骤 2 · 巨潮资讯网** — 正在读取 PDF "
                f"{current}/{total} · {title[:48]}"
            )
            progress_bar.progress(
                min(0.70, 0.36 + 0.34 * current / total),
                text=f"正在处理 PDF {current}/{total}",
            )
        elif event_name == "pdf_processing_completed":
            online_line.markdown(
                f":material/check_circle: **步骤 2 · 巨潮资讯网** — "
                f"{current_company['name']}（{current_company['code']}）近一年公告读取完成，"
                f"已处理 {event.get('total', 0)} 份 PDF"
            )
            progress_bar.progress(0.70, text="最新公告读取完成")
        elif event_name == "source_merge_completed":
            if event.get("history_status") == "hit":
                history_line.markdown(
                    f":material/check_circle: **步骤 1 · 比赛历史库** — 命中 "
                    f"{event.get('historical_document_count', 0)} 份历史公告"
                )
            elif event.get("history_status") == "unavailable":
                history_line.markdown(
                    ":material/database_off: **步骤 1 · 比赛历史库** — 不可用，本次仅使用巨潮最新公告"
                )
            else:
                history_line.markdown(
                    ":material/search_off: **步骤 1 · 比赛历史库** — 未收录该公司"
                )
            merge_line.markdown(
                f":material/layers: **步骤 3 · 分层合并** — 历史公告 "
                f"{event.get('historical_document_count', 0)} 份、巨潮近一年公告 "
                f"{event.get('current_announcement_count', 0)} 份；历史候选单独展示，"
                "当前 F1 与 30/60/90 天统计只使用巨潮近一年公告"
            )
        elif event_name == "rule_analysis_started":
            status.update(label="步骤 3/3 · 正在执行风险证据抽取……")
            analysis_line.markdown(
                f":material/rule: **规则通道**：正在分析 {event.get('document_count', 0)} 份公告"
            )
            progress_bar.progress(0.74, text="正在执行规则通道")
        elif event_name == "rule_analysis_completed":
            analysis_line.markdown(
                f":material/check_circle: **规则通道完成**：保留 {event.get('factor_count', 0)} 条候选，"
                f"过滤 {event.get('suppressed_count', 0)} 条否定或非事实语境"
            )
            progress_bar.progress(0.80, text="规则通道完成")
        elif event_name == "finbert_started" and event.get("enabled"):
            analysis_line.markdown(":material/model_training: **FinBERT 通道**：正在生成实验性文本信号")
        elif event_name == "finbert_completed":
            progress_bar.progress(0.85, text=f"FinBERT：{event.get('status', '')}")
        elif event_name == "llm_started":
            analysis_line.markdown(
                f":material/psychology: **LLM 精细通道**：正在核验 {event.get('document_count', 0)} 份公告原文"
            )
        elif event_name == "llm_completed":
            analysis_line.markdown(
                f":material/check_circle: **模型通道完成**：LLM {event.get('status', '')}，"
                f"接受 {event.get('factor_count', 0)} 条逐字证据"
            )
            progress_bar.progress(0.94, text="模型通道完成")
        elif event_name == "finalizing":
            analysis_line.markdown(":material/analytics: **正在汇总**：生成 30/60/90 天 F1 与审计信息")
            progress_bar.progress(0.97, text="正在生成最终结果")
        elif event_name == "analysis_completed":
            analysis_line.markdown(
                f":material/check_circle: **研读完成**：当前公告形成 {event.get('risk_factor_count', 0)} 条待复核风险事件"
            )
            progress_bar.progress(1.0, text="历史数据与最新公告已完成分层合并")

    return update


def show_status_badges(channel_summary: dict, data_quality: dict) -> None:
    rule_status = channel_summary.get("rule", {}).get("status", "unknown")
    finbert_status = channel_summary.get("finbert", {}).get("status", "unknown")
    llm_status = channel_summary.get("llm", {}).get("status", "unknown")
    ocr_status = data_quality.get("ocr_status", "unknown")
    history_status = data_quality.get("competition_history_match_status", "unknown")
    st.markdown(
        f":green-badge[巨潮官方主源] "
        f":violet-badge[历史库 {history_status}] "
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


def historical_risk_dataframe(history: dict) -> pd.DataFrame:
    """只展示历史旧规则候选，不转换成当前风险事件。"""
    rows = []
    for item in history.get("risk_candidates", []):
        rows.append(
            {
                "历史日期": item.get("date", ""),
                "公告标题": item.get("title", ""),
                "旧规则标签": item.get("risk_label", ""),
                "旧严重度": item.get("severity", ""),
                "历史原文摘录": item.get("evidence", ""),
                "页码": item.get("page"),
                "文档 ID": item.get("doc_id", ""),
                "数据层级": item.get("source_tier", ""),
                "复核状态": item.get("verification_status", ""),
            }
        )
    return pd.DataFrame(rows)


st.title("公告研读 Agent")
st.caption("输入上市公司代码或准确名称：先查比赛历史库，再读取巨潮近一年官方公告，分层输出可核验证据。")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input(
        "数据截止日",
        value=date.today(),
        max_value=date.today(),
        help="系统不会读取该日期之后的公告。",
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
    st.caption("查询顺序：比赛历史库 → 巨潮资讯网。当前事实主源仍是巨潮；历史旧规则信号单独展示。")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码或准确名称",
        value="000004SZ",
        placeholder="例如：000004SZ、国华退",
        help="000004SZ 内置截至 2026-08-24 的可核验离线快照，可用于快速演示。",
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
        progress_slot = st.empty()
        try:
            with progress_slot.container():
                with st.status("准备开始分层查询……", expanded=True) as status:
                    history_line = st.empty()
                    online_line = st.empty()
                    merge_line = st.empty()
                    analysis_line = st.empty()
                    progress_bar = st.progress(0.0, text="正在初始化公告研读")
                    progress_callback = build_progress_handler(
                        status,
                        history_line,
                        online_line,
                        merge_line,
                        analysis_line,
                        progress_bar,
                    )
                    result = analyze_company(
                        normalized,
                        as_of_value.isoformat(),
                        use_ocr,
                        use_finbert,
                        use_llm,
                        FILTER_VERSION,
                        progress_callback,
                    )
                    st.session_state["announcement_analysis"] = result
            progress_slot.empty()
        except Exception as exc:
            progress_slot.empty()
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

    if quality.get("offline_snapshot_used"):
        st.info(
            f"本次使用仓库内巨潮官方公告离线快照，数据锚点为 "
            f"{quality.get('snapshot_as_of') or result.get('as_of')}。公告详情链接、官方 PDF 链接、"
            "正文 SHA-256 与 OCR 审计信息均保留；该快照不是锚点日期之后的实时数据。",
            icon=":material/offline_bolt:",
        )

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

    overview_tab, risk_tab, history_tab, announcement_tab, audit_tab = st.tabs(
        [
            ":material/analytics: 风险概览",
            ":material/warning: 风险证据",
            ":material/history: 比赛历史库",
            ":material/article: 公告清单",
            ":material/fact_check: 审计信息",
        ]
    )

    with overview_tab:
        scalar = features.get("scalar_features", {})
        current_factors = semantic.get("risk_factors", [])
        current_announcements = semantic.get("announcements", [])

        monthly_data = pd.DataFrame(
            risk_monthly_severity_rows(current_factors, result.get("as_of", ""))
        )
        with st.container(border=True):
            st.subheader("近一年风险事件时间轴")
            st.caption(
                "按最近 12 个自然月统计当前巨潮公告中的去重风险事件；"
                "颜色表示规则或模型给出的待复核严重度，不是风险概率。"
            )
            if monthly_data.empty or not monthly_data["事件总数"].sum():
                st.info("近一年没有可绘制的风险事件。", icon=":material/info:")
            else:
                st.bar_chart(
                    monthly_data,
                    x="月份",
                    y=["高风险", "中风险", "低风险"],
                    y_label="风险事件数",
                    color=["#C2410C", "#F59E0B", "#60A5FA"],
                    stack=True,
                    sort=False,
                    height=350,
                )
                with st.expander("查看月度精确计数", icon=":material/table_chart:"):
                    st.dataframe(
                        monthly_data,
                        hide_index=True,
                        column_config={
                            column: st.column_config.NumberColumn(column, format="%d")
                            for column in ("高风险", "中风险", "低风险", "事件总数")
                        },
                        key="monthly_risk_counts",
                    )

        interval_data = pd.DataFrame(
            risk_interval_comparison_rows(
                current_announcements,
                current_factors,
                result.get("as_of", ""),
            )
        )
        with st.container(border=True):
            st.subheader("最近 90 天风险节奏")
            st.caption(
                "三个区间互不重叠，分别统计最近 1–30 天、此前 31–60 天和此前 61–90 天；"
                "堆叠柱总高度等于该区间风险事件总数。"
            )
            if interval_data.empty:
                st.info("没有可用的日期数据。", icon=":material/info:")
            else:
                st.bar_chart(
                    interval_data,
                    x="时间区间",
                    y=["高风险事件", "中低风险事件"],
                    y_label="风险事件数",
                    color=["#C2410C", "#FDBA74"],
                    stack=True,
                    sort=False,
                    height=320,
                )
                with st.expander("查看区间精确数据", icon=":material/table_chart:"):
                    st.dataframe(
                        interval_data[
                            [
                                "时间区间",
                                "公告总数",
                                "风险事件",
                                "高风险事件",
                                "中低风险事件",
                                "风险事件/公告",
                            ]
                        ],
                        hide_index=True,
                        column_config={
                            "公告总数": st.column_config.NumberColumn(format="%d"),
                            "风险事件": st.column_config.NumberColumn(format="%d"),
                            "高风险事件": st.column_config.NumberColumn(format="%d"),
                            "中低风险事件": st.column_config.NumberColumn(format="%d"),
                            "风险事件/公告": st.column_config.NumberColumn(
                                format="%.2f",
                                help="风险事件数除以同一非重叠区间的公告总数；仅为信号密度。",
                            ),
                        },
                        key="interval_risk_counts",
                    )

        category_counts = features.get("category_event_counts", {})
        with st.container(border=True):
            st.subheader("风险主题分析")
            theme_view = st.segmented_control(
                "主题视图",
                ["月份热力图", "主题排名"],
                default="月份热力图",
                key="risk_theme_view",
            )
            if theme_view == "主题排名":
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
                st.caption("累计窗口统计；同一公告、同一风险主题的重复命中已去重。")
                if chart_data.empty:
                    st.info("所选窗口内没有风险主题。", icon=":material/info:")
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
                    with st.expander("查看精确计数与占比", icon=":material/table_chart:"):
                        st.dataframe(
                            chart_data[["主题代码", "风险主题", "事件数", "占比"]],
                            hide_index=True,
                            column_config={
                                "事件数": st.column_config.NumberColumn(format="%d"),
                                "占比": st.column_config.ProgressColumn(
                                    "主题计数占比",
                                    format="percent",
                                    min_value=0,
                                    max_value=1,
                                ),
                            },
                            key="theme_ranking_counts",
                        )
            else:
                heatmap_rows = risk_theme_heatmap_rows(
                    current_factors,
                    result.get("as_of", ""),
                    max_themes=8,
                )
                heatmap_data = pd.DataFrame(heatmap_rows)
                st.caption(
                    "最近 12 个自然月 Top 8 风险主题；颜色越深表示当月该主题的去重事件越多，"
                    "空白格表示 0。"
                )
                if heatmap_data.empty:
                    st.info("近一年没有可绘制的风险主题。", icon=":material/info:")
                else:
                    month_order = list(dict.fromkeys(heatmap_data["月份"].tolist()))
                    theme_order = list(dict.fromkeys(heatmap_data["风险主题"].tolist()))
                    st.vega_lite_chart(
                        heatmap_data,
                        {
                            "mark": {"type": "rect", "cornerRadius": 2},
                            "encoding": {
                                "x": {
                                    "field": "月份",
                                    "type": "ordinal",
                                    "sort": month_order,
                                    "axis": {"labelAngle": -35, "title": "月份"},
                                },
                                "y": {
                                    "field": "风险主题",
                                    "type": "nominal",
                                    "sort": theme_order,
                                    "axis": {"title": "风险主题", "labelLimit": 300},
                                },
                                "color": {
                                    "field": "事件数",
                                    "type": "quantitative",
                                    "scale": {"range": ["#FFF7ED", "#C2410C"]},
                                    "legend": {"title": "事件数"},
                                },
                                "tooltip": [
                                    {"field": "月份", "type": "ordinal"},
                                    {"field": "风险主题", "type": "nominal"},
                                    {"field": "事件数", "type": "quantitative"},
                                    {"field": "主题全年合计", "type": "quantitative"},
                                ],
                            },
                            "height": max(260, len(theme_order) * 42),
                        },
                        key="risk_theme_heatmap",
                    )
        with st.container(border=True):
            st.markdown("**口径说明**")
            st.write(
                "时间轴和热力图使用最近 12 个自然月；90 天节奏图使用三个互不重叠的 30 天区间。"
            )
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

    with history_tab:
        history = semantic.get("historical_context", {})
        if history.get("match_status") != "hit":
            st.info(
                history.get("message", "比赛历史库未命中该公司，本次仅分析巨潮最新公告。"),
                icon=":material/search_off:",
            )
        else:
            with st.container(horizontal=True):
                st.metric("历史公告", history.get("document_count", 0), border=True)
                st.metric("旧规则命中文档", history.get("risk_document_count", 0), border=True)
                st.metric("旧规则候选片段", history.get("risk_candidate_count", 0), border=True)
                semantic_feature = history.get("semantic_feature", {})
                st.metric("历史语义维度", semantic_feature.get("feature_count", 0), border=True)
            st.warning(history.get("warning", "历史候选不等于当前事实。"), icon=":material/warning:")
            st.caption(
                f"数据区间：{history.get('date_start')} 至 {history.get('date_end')} · "
                f"旧词典版本：{', '.join(history.get('dictionary_versions', []))} · "
                f"历史特征锚点：{semantic_feature.get('anchor_date') or '无'}"
            )
            historical_df = historical_risk_dataframe(history)
            if historical_df.empty:
                st.info("历史库包含该公司，但旧规则没有候选命中。", icon=":material/check_circle:")
            else:
                st.dataframe(
                    historical_df,
                    hide_index=True,
                    column_config={"页码": st.column_config.NumberColumn("页码", format="%d")},
                    key="historical_risk_candidates",
                )
                if history.get("risk_candidates_truncated"):
                    st.caption("页面只展示最新 200 条历史候选；完整计数保留在下载 JSON 中。")
            with st.expander("查看历史公告清单", icon=":material/article:"):
                historical_announcements = pd.DataFrame(history.get("announcements", []))
                if historical_announcements.empty:
                    st.info("没有可展示的历史公告元数据。")
                else:
                    historical_announcements = historical_announcements.rename(
                        columns={
                            "date": "历史日期",
                            "title": "公告标题",
                            "doc_id": "文档 ID",
                            "doc_type": "历史类型",
                            "parse_status": "解析状态",
                            "has_old_rule_candidate": "有旧规则候选",
                            "old_rule_candidate_count": "候选数",
                            "source_tier": "数据层级",
                        }
                    )
                    st.dataframe(
                        historical_announcements,
                        hide_index=True,
                        column_config={
                            "有旧规则候选": st.column_config.CheckboxColumn("有旧规则候选"),
                            "候选数": st.column_config.NumberColumn("候选数", format="%d"),
                        },
                        key="historical_announcements",
                    )
            with st.expander("历史语义特征说明", icon=":material/data_object:"):
                st.json(semantic_feature, expanded=False)

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
