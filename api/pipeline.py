"""离线数据 → 前端 JSON 的桥梁。

方案 B：从 backend/data/output/reports/ 读取已生成的离线报告，
映射为前端 AnalyzeResponse 格式，秒级响应，不调模型。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，以便 import backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .models import (
    AnalyzeResponse,
    FactorItem,
    AnnouncementRiskItem,
    AttributionEvidenceItem,
    SimilarCaseItem,
)

REPORTS_DIR = PROJECT_ROOT / "backend" / "data" / "output" / "reports"

# ---------- 公司代码 → 最新离线报告 ----------

def _load_manifest() -> list[dict]:
    path = REPORTS_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_report_file(company: str) -> Path | None:
    """从 manifest 找某公司最新一份报告 JSON。"""
    manifest = _load_manifest()
    normalized = company.upper().replace(".", "_")
    entries = [
        e for e in manifest
        if str(e.get("company", "")).upper().replace(".", "_") == normalized
        and (REPORTS_DIR / str(e.get("json_file", ""))).is_file()
    ]
    if not entries:
        return None
    # 按 generated_at 取最新
    entries.sort(key=lambda e: str(e.get("generated_at", "")), reverse=True)
    return REPORTS_DIR / entries[0]["json_file"]


def _load_report(company: str) -> dict | None:
    path = _latest_report_file(company)
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_report_md(company: str) -> str:
    path = _latest_report_file(company)
    if not path:
        return ""
    md_path = path.with_suffix(".md")
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return ""


# ---------- 字段映射 ----------

_LEVEL_MAP = {"低": "low", "中": "mid", "高": "high"}
_LEVEL_TEXT = {"低": "低风险", "中": "中风险", "高": "高风险"}
_COLOR_MAP = {"low": "#10B981", "mid": "#F59E0B", "high": "#EF4444"}


def _map_risk_level(raw: str) -> str:
    for k, v in _LEVEL_MAP.items():
        if k in (raw or ""):
            return v
    return "low"


def _map_shap_to_factors(report: dict) -> list[FactorItem]:
    """把 scorecard.shap_features + attribution.top_risk_factors 映射为 FactorItem。"""
    scorecard = report.get("scorecard", {}) or {}
    attribution = report.get("attribution", {}) or {}
    top_factors = attribution.get("top_risk_factors", []) or []
    shap_features = scorecard.get("shap_features", []) or []

    # 人读名称映射
    name_map = {
        "mkt_volume_ratio_20d": "20 日量比",
        "f6_last_inquiry_interval_days": "最近问询间隔",
        "f6_inquiry_count_60m": "60 月问询次数",
        "sent_guba_negative_ratio_30d": "股吧负面占比 30d",
        "mkt_market_cap": "总市值",
        "mkt_log_market_cap": "总市值（对数）",
        "f6_inquiry_count_12m": "12 月问询次数",
        "f6_inquiry_count_24m": "24 月问询次数",
        "f6_inquiry_count_36m": "36 月问询次数",
        "f6_inquiry_count_48m": "48 月问询次数",
        "mkt_turnover_20d": "20 日换手率",
        "mkt_volatility_20d": "20 日波动率",
    }
    tag_map = {
        "市场异动": ("market", "市场"),
        "问询历史": ("text", "问询历史"),
        "舆情": ("text", "舆情"),
        "市值": ("market", "市值"),
        "财务": ("finance", "财务"),
        "历史问询": ("text", "历史问询"),
        "市场情绪": ("text", "舆情"),
    }
    desc_map = {
        "mkt_volume_ratio_20d": "当日成交量 / 前 19 日均量异常",
        "f6_last_inquiry_interval_days": "距最近一次监管问询天数",
        "f6_inquiry_count_60m": "历史监管问询频次",
        "sent_guba_negative_ratio_30d": "近 30 天负面舆情比例",
        "mkt_market_cap": "当前总市值水平",
        "mkt_log_market_cap": "当前总市值水平（对数）",
    }

    factors = []
    # 优先用 attribution.top_risk_factors（有 description 和 label_ref）
    for f in top_factors[:6]:
        feat = f.get("feature", "")
        shap_val = f.get("shap", 0)
        desc = f.get("description", name_map.get(feat, feat))
        label_ref = f.get("label_ref", "")
        tag, tag_text = tag_map.get(label_ref, ("finance", "财务"))
        score = min(abs(shap_val), 1.0)
        if score > 0.3:
            color = "#EF4444"
        elif score > 0.15:
            color = "#F59E0B"
        else:
            color = "#10B981"
        factors.append(FactorItem(
            name=desc, tag=tag, tagText=tag_text,
            desc=desc_map.get(feat, f.get("theme_name", desc)),
            score=round(score, 2), color=color,
        ))

    # 如果 attribution 为空，从 shap_features 补
    if not factors:
        tag_pool = ["市场异动", "问询历史", "问询历史", "舆情", "市值", "财务"]
        for i, (feat, val) in enumerate(shap_features[:6]):
            tag_label = tag_pool[i % len(tag_pool)]
            tag, tag_text = tag_map.get(tag_label, ("finance", "财务"))
            score = min(abs(val), 1.0)
            if score > 0.3:
                color = "#EF4444"
            elif score > 0.15:
                color = "#F59E0B"
            else:
                color = "#10B981"
            factors.append(FactorItem(
                name=name_map.get(feat, feat), tag=tag, tagText=tag_text,
                desc=desc_map.get(feat, "SHAP 特征贡献"),
                score=round(score, 2), color=color,
            ))
    return factors


def _map_financial_table(report: dict) -> list[list[str]]:
    """financial.anomaly_list → [[指标, 本期值, 行业均值/阈值, 偏离度], ...]"""
    anomalies = report.get("financial", {}).get("anomaly_list", []) or []
    rows = []
    for a in anomalies[:6]:
        indicator = a.get("indicator", "—")
        value = str(a.get("value", "—"))
        threshold = str(a.get("threshold", "—"))
        evidence = a.get("evidence", "")
        # 从 evidence 里提取偏离度（如果有的话）
        deviation = "—"
        if "，" in evidence:
            parts = evidence.split("，")
            for p in parts:
                if "%" in p or "倍" in p:
                    deviation = p.strip()
                    break
        rows.append([indicator, value, threshold, deviation])
    return rows


def _map_attribution_text(report: dict) -> str:
    """attribution.top_risk_factors → 归因文本。"""
    factors = report.get("attribution", {}).get("top_risk_factors", []) or []
    if not factors:
        return "暂无归因数据"
    parts = []
    for f in factors[:5]:
        desc = f.get("description", f.get("feature", ""))
        shap_val = f.get("shap", 0)
        pct = abs(shap_val) * 100
        parts.append(f"{desc}（{pct:.1f}%）")
    return "、".join(parts)


def _build_text_summary(report: dict) -> str:
    """semantic → 文本摘要。"""
    sem = report.get("semantic", {}) or {}
    risk_factors = sem.get("risk_factors", []) or []
    ann_count = sem.get("announcement_count", 0)
    rf_count = len(risk_factors)
    if rf_count:
        categories = list({f.get("category", "") for f in risk_factors if f.get("category")})
        cat_text = "、".join(categories[:3]) if categories else "多类"
        return (
            f"近一年共研读公告 <b>{ann_count}</b> 份，经规则引擎 + FinBERT + LLM 三通道抽取，"
            f"识别出 <b>{rf_count}</b> 条风险信号，涵盖 {cat_text} 等类别。"
        )
    return f"近一年共研读公告 <b>{ann_count}</b> 份，未识别出显著风险信号。"


# ---------- 新增：公告研读风险证据表 ----------

_LEVEL_FROM_SEVERITY = {1: "低", 2: "低", 3: "中", 4: "高", 5: "高"}


def _map_announcement_risks(report: dict) -> list[AnnouncementRiskItem]:
    """semantic.risk_factors → 公告研读风险证据表。"""
    sem = report.get("semantic", {}) or {}
    risk_factors = sem.get("risk_factors", []) or []
    if not risk_factors:
        return []

    # 按 severity 降序、confidence 降序取前 20 条（去重：相同证据只保留一条）
    sorted_factors = sorted(
        risk_factors,
        key=lambda f: (f.get("severity", 0), f.get("confidence", 0)),
        reverse=True,
    )
    seen = set()
    items = []
    for f in sorted_factors:
        evidence = f.get("evidence", "").strip()
        key = (f.get("announcement_id"), evidence[:40])
        if key in seen:
            continue
        seen.add(key)
        severity = f.get("severity", 3)
        items.append(AnnouncementRiskItem(
            date=f.get("announcement_date", "") or "—",
            level=_LEVEL_FROM_SEVERITY.get(severity, "中"),
            l1=f.get("taxonomy_l1", "") or f.get("category", "") or "—",
            l2=f.get("taxonomy_l2", "") or f.get("label", "") or "—",
            description=f.get("description", "").strip() or "—",
            evidence=evidence or "—",
            title=f.get("announcement_title", "") or "—",
            sourceUrl=f.get("source_url", "") or "",
        ))
        if len(items) >= 20:
            break
    return items


# ---------- 新增：风险归因原文 ----------

def _map_attribution_evidence(report: dict) -> list[AttributionEvidenceItem]:
    """attribution.top_risk_factors + semantic.risk_factors → 归因原文。"""
    attribution = report.get("attribution", {}) or {}
    top_factors = attribution.get("top_risk_factors", []) or []
    semantic = report.get("semantic", {}) or {}
    risk_factors = semantic.get("risk_factors", []) or []

    # 证据引用（若存在）
    citations = attribution.get("evidence_citations", []) or []
    if citations:
        return [
            AttributionEvidenceItem(
                factor=c.get("factor", "") or "风险因子",
                evidence=c.get("text", "").strip() or c.get("evidence", "").strip() or "—",
                source=c.get("source", ""),
                anchor=f"evidence-{i}",
            )
            for i, c in enumerate(citations[:10])
            if c.get("text") or c.get("evidence")
        ]

    # 无显式引用时，按 top_factors 的主题/分类去 semantic.risk_factors 中找最接近证据
    results = []
    used = set()
    for i, f in enumerate(top_factors[:6]):
        feat = f.get("feature", "")
        desc = f.get("description", feat)
        taxonomy_l1 = f.get("taxonomy_l1", "")
        taxonomy_l2 = f.get("taxonomy_l2", "")
        theme = f.get("theme_name", "")

        # 优先匹配 taxonomy_l2，其次 label_ref/theme
        matched = None
        for rf in risk_factors:
            if rf.get("risk_id") in used:
                continue
            rf_l2 = rf.get("taxonomy_l2", "")
            rf_l1 = rf.get("taxonomy_l1", "")
            rf_cat = rf.get("category", "")
            if taxonomy_l2 and rf_l2 == taxonomy_l2:
                matched = rf
                break
            if taxonomy_l1 and (rf_l1 == taxonomy_l1 or rf_cat == taxonomy_l1):
                matched = rf
                break
        if not matched and risk_factors:
            matched = risk_factors[0]

        if matched:
            used.add(matched.get("risk_id"))
            evidence = matched.get("evidence", "").strip()
            results.append(AttributionEvidenceItem(
                factor=desc,
                evidence=evidence or desc,
                source=matched.get("announcement_title", ""),
                anchor=f"evidence-{i}",
            ))
        else:
            results.append(AttributionEvidenceItem(
                factor=desc,
                evidence=desc,
                source="",
                anchor=f"evidence-{i}",
            ))
    return results


# ---------- 新增：全部风险因子（查看全部用） ----------

def _map_risk_factor_details(report: dict) -> list[FactorItem]:
    """返回更完整的因子列表（最多 12 个），含证据锚点。"""
    scorecard = report.get("scorecard", {}) or {}
    attribution = report.get("attribution", {}) or {}
    top_factors = attribution.get("top_risk_factors", []) or []
    shap_features = scorecard.get("shap_features", []) or []

    name_map = {
        "mkt_volume_ratio_20d": "20 日量比",
        "f6_last_inquiry_interval_days": "最近问询间隔",
        "f6_inquiry_count_60m": "60 月问询次数",
        "sent_guba_negative_ratio_30d": "股吧负面占比 30d",
        "mkt_market_cap": "总市值",
        "mkt_log_market_cap": "总市值（对数）",
        "f6_inquiry_count_12m": "12 月问询次数",
        "f6_inquiry_count_24m": "24 月问询次数",
        "f6_inquiry_count_36m": "36 月问询次数",
        "f6_inquiry_count_48m": "48 月问询次数",
        "mkt_turnover_20d": "20 日换手率",
        "mkt_volatility_20d": "20 日波动率",
        "mkt_pe_ratio": "市盈率",
        "mkt_return_60d": "60 日收益率",
        "f2_p_roa": "ROA",
        "f2_beneish_m": "Beneish M-Score",
        "debt_to_assets_ratio": "资产负债率",
    }
    tag_map = {
        "市场异动": ("market", "市场"),
        "问询历史": ("text", "问询历史"),
        "舆情": ("text", "舆情"),
        "市值": ("market", "市值"),
        "财务": ("finance", "财务"),
        "历史问询": ("text", "历史问询"),
        "市场情绪": ("text", "舆情"),
        "偿债能力": ("finance", "财务"),
        "盈利能力": ("finance", "财务"),
        "市场市值": ("market", "市值"),
    }
    desc_map = {
        "mkt_volume_ratio_20d": "当日成交量 / 前 19 日均量异常",
        "f6_last_inquiry_interval_days": "距最近一次监管问询天数",
        "f6_inquiry_count_60m": "历史监管问询频次",
        "sent_guba_negative_ratio_30d": "近 30 天负面舆情比例",
        "mkt_market_cap": "当前总市值水平",
        "mkt_log_market_cap": "当前总市值水平（对数）",
        "mkt_pe_ratio": "当前市盈率水平",
        "mkt_return_60d": "近 60 日股价收益率",
        "f2_p_roa": "资产回报率",
        "f2_beneish_m": "盈余操纵嫌疑评分",
        "debt_to_assets_ratio": "资产负债率水平",
    }

    factors = []
    for i, f in enumerate(top_factors[:12]):
        feat = f.get("feature", "")
        shap_val = f.get("shap", 0)
        desc = f.get("description", name_map.get(feat, feat))
        label_ref = f.get("label_ref", "")
        tag, tag_text = tag_map.get(label_ref, ("finance", "财务"))
        score = min(abs(shap_val), 1.0)
        if score > 0.3:
            color = "#EF4444"
        elif score > 0.15:
            color = "#F59E0B"
        else:
            color = "#10B981"
        factors.append(FactorItem(
            name=desc,
            tag=tag,
            tagText=tag_text,
            desc=desc_map.get(feat, f.get("theme_name", desc)),
            score=round(score, 2),
            color=color,
            evidenceAnchor=f"evidence-{i}",
        ))

    # 若 attribution 为空，从 shap_features 补
    if not factors:
        for i, (feat, val) in enumerate(shap_features[:12]):
            score = min(abs(val), 1.0)
            if score > 0.3:
                color = "#EF4444"
            elif score > 0.15:
                color = "#F59E0B"
            else:
                color = "#10B981"
            factors.append(FactorItem(
                name=name_map.get(feat, feat),
                tag="finance",
                tagText="财务",
                desc=desc_map.get(feat, "SHAP 特征贡献"),
                score=round(score, 2),
                color=color,
                evidenceAnchor=f"evidence-{i}",
            ))
    return factors


# ---------- 新增：相似监管案例（结构化） ----------

def _map_similar_cases(report: dict) -> list[SimilarCaseItem]:
    raw = report.get("similar_cases", []) or []
    items = []
    for i, c in enumerate(raw[:10]):
        items.append(SimilarCaseItem(
            caseId=c.get("case_id", f"case-{i}"),
            company=c.get("company", ""),
            publishDate=c.get("publish_date", "") or "—",
            inquiryType=c.get("inquiry_type", "") or "监管问询函",
            topics=[t for t in (c.get("topics", []) or []) if t][:5],
            similarity=round(c.get("similarity", 0) or c.get("cosine_similarity", 0) or 0, 4),
            matchReason=[r for r in (c.get("match_reason", []) or []) if r][:3],
        ))
    return items


def offline_to_response(company: str) -> AnalyzeResponse | None:
    """核心映射：离线报告 JSON → AnalyzeResponse。"""
    report = _load_report(company)
    if not report:
        return None

    scorecard = report.get("scorecard", {}) or {}
    risk_level_raw = scorecard.get("risk_level", "低")
    level = _map_risk_level(risk_level_raw)
    prob_60d = scorecard.get("probability_60d", 0) or 0
    confidence = scorecard.get("confidence", 0) or 0
    financial = report.get("financial", {}) or {}
    semantic = report.get("semantic", {}) or {}
    similar_cases = report.get("similar_cases", []) or []
    attribution = report.get("attribution", {}) or {}

    risk_factors_count = len(semantic.get("risk_factors", []) or [])
    anomaly_count = len(financial.get("anomaly_list", []) or [])
    total_factors = risk_factors_count + anomaly_count

    # 构建 summary（带 HTML）
    prob_pct = prob_60d * 100
    top_factor_names = [
        f.get("description", f.get("feature", ""))
        for f in (attribution.get("top_risk_factors", []) or [])[:3]
    ]
    factor_text = "、".join(top_factor_names) if top_factor_names else "综合多维度指标"
    summary = (
        f"未来 60 天收到问询函的概率为 <b>{prob_pct:.1f}%</b>。"
        f"主要风险来源为{factor_text}。"
    )

    # 构建 conclusion
    conclusion = report.get("executive_summary", "")
    if not conclusion:
        conclusion = f"风险等级为{_LEVEL_TEXT.get(risk_level_raw[0] if risk_level_raw else '低', '低风险')}，建议持续跟踪。"

    factors_list = _map_shap_to_factors(report)
    return AnalyzeResponse(
        code=report.get("company", company),
        name=report.get("name", company),
        risk=round(prob_pct, 1),
        level=level,
        levelText=_LEVEL_TEXT.get(risk_level_raw[0] if risk_level_raw else "低", "低风险"),
        confidence=round(confidence, 2),
        factors=total_factors,
        summary=summary,
        factorsList=factors_list,
        financialTable=_map_financial_table(report),
        textSummary=_build_text_summary(report),
        caseMatch=f"{len(similar_cases)} 起",
        attribution=_map_attribution_text(report),
        conclusion=conclusion[:200],
        advice=report.get("disclaimer", "本预测结果仅供参考，不构成投资建议。"),
        reportMarkdown=_load_report_md(company),
        traceLog=report.get("trace_log", []) or [],
        similarCases=_map_similar_cases(report),
        generatedAt=report.get("generated_at", ""),
        announcementRisks=_map_announcement_risks(report),
        attributionEvidence=_map_attribution_evidence(report),
        riskFactorDetails=_map_risk_factor_details(report),
    )


async def run_pipeline(codes: list, window: int = 60, use_llm: bool = False, use_bge: bool = True):
    """方案 B：离线数据快速查询。"""
    results = []
    for code in codes:
        resp = offline_to_response(code)
        if resp:
            results.append(resp)
    return results


def list_available_companies() -> list[dict]:
    """返回所有有离线报告的公司列表。"""
    manifest = _load_manifest()
    seen = {}
    for e in manifest:
        code = e.get("company", "")
        if code and code not in seen:
            seen[code] = {
                "code": code,
                "name": e.get("name", code),
                "generated_at": e.get("generated_at", ""),
            }
    return list(seen.values())


# ============================================================
# 方案 C：实时扫雷管道 + WebSocket 进度推送
# ============================================================

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .models import ProgressMessage, ScanRequest


# ---------- Orchestrator 全局单例（避免每次请求都重新加载模型） ----------

_orchestrator_pool: dict[str, "SweepingOrchestrator"] = {}


def get_orchestrator(use_llm: bool, use_finbert: bool, use_semantic_cases: bool = True, max_documents: int | None = 5):
    """获取全局单例 Orchestrator。key 按参数组合区分。"""
    key = f"llm={use_llm}_finbert={use_finbert}_bge={use_semantic_cases}_docs={max_documents}"
    if key not in _orchestrator_pool:
        # 延迟导入，避免模块级报错
        from backend.agents.orchestrator import SweepingOrchestrator
        _orchestrator_pool[key] = SweepingOrchestrator(
            use_llm=use_llm,
            use_finbert=use_finbert,
            use_semantic_cases=use_semantic_cases,
            max_documents=max_documents,
        )
    return _orchestrator_pool[key]


# ---------- StreamingOrchestrator：在关键节点推送进度 ----------

class StreamingOrchestrator:
    """包装 SweepingOrchestrator，在每个 Agent 前后回调进度。

    注意：execute() 整体仍是同步阻塞的，必须在线程池中运行。
    回调函数会被从工作线程调用，因此需要线程安全地写入 queue。
    """

    _AGENT_ORDER = [
        ("AnnouncementReader", "公告研读", "读取并解析最新公告、抽取语义风险"),
        ("FinancialDetector", "财务异动", "检测财务指标异常与偏离度"),
        ("Predictor", "预测建模", "XGBoost + SHAP 计算问询概率"),
        ("CaseRetriever", "案例匹配", "BGE 语义检索历史问询案例"),
        ("ChunkRetriever", "段落召回", "chunk 级证据召回（可选）"),
        ("Attributor", "归因分析", "聚合归因解释与风险叙事"),
        ("Reporter", "报告生成", "渲染风控简报并落盘"),
    ]

    def __init__(self, callback: Callable[[ProgressMessage], None]):
        self.callback = callback

    def run(self, company: str, window: int = 60, as_of: str | None = None,
            use_llm: bool = False, use_bge: bool = True, max_documents: int | None = 5):
        from backend.agents.orchestrator import SweepingOrchestrator
        from backend.context import Context

        orch = get_orchestrator(
            use_llm=use_llm,
            use_finbert=False,           # 前端未暴露 FinBERT 开关，固定关闭以提速
            use_semantic_cases=use_bge,
            max_documents=max_documents,
        )

        ctx = Context(company=company, window=window, as_of=as_of or str(date.today()))
        ctx.use_llm_summary = bool(use_llm)
        ctx.use_llm = bool(use_llm)
        ctx.max_documents = max_documents

        total = len(self._AGENT_ORDER)
        start_total = time.time()

        for idx, (agent_name, display_name, desc) in enumerate(self._AGENT_ORDER, start=1):
            step_start = time.time()
            self._emit(idx, total, agent_name, "running", f"{display_name} Agent 正在执行：{desc}", 0)

            try:
                runner = getattr(self, f"_run_{agent_name.lower()}", None)
                if runner is None:
                    # 兜底：用 orchestrator 内部方法
                    runner = getattr(orch, f"_run_{agent_name.lower()}", None)
                if runner is None:
                    raise RuntimeError(f"未找到 Agent {agent_name} 的执行入口")
                runner(company, ctx)
            except Exception as e:
                latency = int((time.time() - step_start) * 1000)
                # 段落召回失败可跳过，不打断流水线
                if agent_name == "ChunkRetriever":
                    ctx.trace_log.append({"agent": agent_name, "status": "skipped", "reason": str(e), "trace_complete": True})
                    self._emit(idx, total, agent_name, "skipped", f"{display_name} Agent 跳过：{e}", latency)
                else:
                    self._emit(idx, total, agent_name, "error", f"{display_name} Agent 失败：{e}", latency)
                    raise
            else:
                latency = int((time.time() - step_start) * 1000)
                self._emit(idx, total, agent_name, "done", f"{display_name} Agent 完成（{latency} ms）", latency)

        ctx.meta = {"total_elapsed_ms": int((time.time() - start_total) * 1000)}
        return ctx

    def _emit(self, step: int, total: int, agent: str, status: str, message: str, elapsed_ms: int):
        self.callback(ProgressMessage(
            type="progress",
            step=step,
            total=total,
            agent=agent,
            status=status,
            message=message,
            elapsed_ms=elapsed_ms,
        ))

    # 下面这些方法与 SweepingOrchestrator 内部保持一致，便于注入回调
    def _run_announcementreader(self, company, ctx):
        from backend.agents.announcement_reader import AnnouncementReaderAgent
        agent = AnnouncementReaderAgent(
            max_documents=getattr(ctx, "max_documents", 5),
            use_finbert=False,
            use_llm=getattr(ctx, "use_llm", False),
            use_rule=True,
        )
        agent.run(company, ctx)

    def _run_financialdetector(self, company, ctx):
        from backend.agents.financial_detector import FinancialDetectorAgent
        agent = FinancialDetectorAgent(use_llm=False, rate_limit=0.5)
        agent.run(company, ctx)

    def _run_predictor(self, company, ctx):
        from backend.agents.predictor import PredictorAgent
        agent = PredictorAgent()
        agent.run(company, ctx)

    def _run_caseretriever(self, company, ctx):
        from backend.agents.case_retriever import CaseRetrieverAgent
        agent = CaseRetrieverAgent(use_semantic=True)
        agent.run(company, ctx)

    def _run_chunkretriever(self, company, ctx):
        from backend.agents.chunk_retriever import ChunkRetrieverAgent
        agent = ChunkRetrieverAgent()
        agent.run(company, ctx)

    def _run_attributor(self, company, ctx):
        from backend.agents.attributor import AttributorAgent
        agent = AttributorAgent(use_llm=getattr(ctx, "use_llm", False))
        agent.run(company, ctx)

    def _run_reporter(self, company, ctx):
        from backend.agents.reporter import ReporterAgent
        agent = ReporterAgent()
        agent.run(company, ctx)


# ---------- 报告 ctx -> AnalyzeResponse ----------

def report_ctx_to_response(ctx) -> AnalyzeResponse:
    """把 SweepingOrchestrator 跑完后的 ctx.report['json'] 映射为前端格式。"""
    report = ctx.report["json"]
    return offline_to_response_from_report(report)


def offline_to_response_from_report(report: dict) -> AnalyzeResponse:
    """复用方案 B 的字段映射逻辑，但直接接收 report dict。"""
    scorecard = report.get("scorecard", {}) or {}
    risk_level_raw = scorecard.get("risk_level", "低")
    level = _map_risk_level(risk_level_raw)
    prob_60d = scorecard.get("probability_60d", 0) or 0
    confidence = scorecard.get("confidence", 0) or 0
    financial = report.get("financial", {}) or {}
    semantic = report.get("semantic", {}) or {}
    similar_cases = report.get("similar_cases", []) or []
    attribution = report.get("attribution", {}) or {}

    risk_factors_count = len(semantic.get("risk_factors", []) or [])
    anomaly_count = len(financial.get("anomaly_list", []) or [])
    total_factors = risk_factors_count + anomaly_count

    prob_pct = prob_60d * 100
    top_factor_names = [
        f.get("description", f.get("feature", ""))
        for f in (attribution.get("top_risk_factors", []) or [])[:3]
    ]
    factor_text = "、".join(top_factor_names) if top_factor_names else "综合多维度指标"
    summary = (
        f"未来 60 天收到问询函的概率为 <b>{prob_pct:.1f}%</b>。"
        f"主要风险来源为{factor_text}。"
    )

    conclusion = report.get("executive_summary", "")
    if not conclusion:
        conclusion = f"风险等级为{_LEVEL_TEXT.get(risk_level_raw[0] if risk_level_raw else '低', '低风险')}，建议持续跟踪。"

    factors_list = _map_shap_to_factors(report)
    return AnalyzeResponse(
        code=report.get("company", ""),
        name=report.get("name", ""),
        risk=round(prob_pct, 1),
        level=level,
        levelText=_LEVEL_TEXT.get(risk_level_raw[0] if risk_level_raw else "低", "低风险"),
        confidence=round(confidence, 2),
        factors=total_factors,
        summary=summary,
        factorsList=factors_list,
        financialTable=_map_financial_table(report),
        textSummary=_build_text_summary(report),
        caseMatch=f"{len(similar_cases)} 起",
        attribution=_map_attribution_text(report),
        conclusion=conclusion[:200],
        advice=report.get("disclaimer", "本预测结果仅供参考，不构成投资建议。"),
        reportMarkdown=report.get("markdown", ""),
        traceLog=report.get("trace_log", []) or [],
        similarCases=_map_similar_cases(report),
        generatedAt=report.get("generated_at", ""),
        announcementRisks=_map_announcement_risks(report),
        attributionEvidence=_map_attribution_evidence(report),
        riskFactorDetails=_map_risk_factor_details(report),
    )


# ---------- 任务状态管理 ----------

@dataclass
class TaskState:
    task_id: str
    code: str
    status: str = "pending"           # pending / running / completed / failed
    progress: list[ProgressMessage] = field(default_factory=list)
    result: AnalyzeResponse | None = None
    error: str | None = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


_task_store: dict[str, TaskState] = {}

# 实时扫雷串行锁：SweepingOrchestrator 内部有全局模型/状态，并发容易竞争/死锁，
# 因此同一时刻只跑一个实时任务，其他任务排队等待。
_scan_lock = asyncio.Lock()


def create_task(code: str) -> TaskState:
    task_id = f"scan_{uuid.uuid4().hex[:12]}"
    state = TaskState(task_id=task_id, code=code)
    _task_store[task_id] = state
    return state


def get_task(task_id: str) -> TaskState | None:
    return _task_store.get(task_id)


def _thread_safe_emit(loop: asyncio.AbstractEventLoop, state: TaskState, msg: ProgressMessage):
    """从工作线程向 asyncio.Queue 写消息的线程安全方式。"""
    asyncio.run_coroutine_threadsafe(state.queue.put(msg), loop)


async def run_scan_task(state: TaskState, req: ScanRequest):
    """在后台执行实时扫雷，并通过 queue 推送进度。"""
    loop = asyncio.get_running_loop()

    def callback(msg: ProgressMessage):
        _thread_safe_emit(loop, state, msg)

    # 排队等待：同一时刻只跑一个实时任务
    state.status = "pending"
    _thread_safe_emit(loop, state, ProgressMessage(
        type="progress",
        step=0,
        total=7,
        agent="SweepingOrchestrator",
        status="running",
        message=f"任务 {state.task_id} 已排队，正在等待模型资源（当前有任务正在执行）...",
        elapsed_ms=0,
    ))

    async with _scan_lock:
        state.status = "running"
        _thread_safe_emit(loop, state, ProgressMessage(
            type="progress",
            step=0,
            total=7,
            agent="SweepingOrchestrator",
            status="running",
            message=f"正在为 {req.code} 启动 7-Agent 实时扫雷流水线（首次加载 BGE/OCR 模型约需 15-30 秒）...",
            elapsed_ms=0,
        ))

        try:
            streamer = StreamingOrchestrator(callback=callback)
            ctx = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: streamer.run(
                        company=req.code,
                        window=req.window,
                        use_llm=req.use_llm,
                        use_bge=req.use_bge,
                        max_documents=req.max_documents,
                    ),
                ),
                timeout=600,  # 10 分钟硬上限
            )
            result = report_ctx_to_response(ctx)
            state.result = result
            state.status = "completed"
            complete_msg = ProgressMessage(
                type="complete",
                step=7,
                total=7,
                agent="Reporter",
                status="done",
                message="报告生成完成",
                elapsed_ms=0,
                result=result,
            )
            _thread_safe_emit(loop, state, complete_msg)
            # 给 WebSocket handler 留时间把 complete 发出去再结束任务
            await asyncio.sleep(0.5)
        except asyncio.TimeoutError:
            state.status = "failed"
            state.error = "实时扫雷超过 10 分钟超时，请尝试减少公告数量或关闭 LLM/BGE。"
            _thread_safe_emit(loop, state, ProgressMessage(
                type="error",
                step=0,
                total=7,
                agent="SweepingOrchestrator",
                status="error",
                message=state.error,
                elapsed_ms=0,
                error=state.error,
            ))
        except Exception as e:
            state.status = "failed"
            state.error = f"{type(e).__name__}: {e}"
            _thread_safe_emit(loop, state, ProgressMessage(
                type="error",
                step=0,
                total=7,
                agent="SweepingOrchestrator",
                status="error",
                message=state.error,
                elapsed_ms=0,
                error=state.error,
            ))
