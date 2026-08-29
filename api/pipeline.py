"""离线数据 → 前端 JSON 的桥梁。

方案 B：从 backend/data/output/reports/ 读取已生成的离线报告，
映射为前端 AnalyzeResponse 格式，秒级响应，不调模型。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

_logger = logging.getLogger(__name__)

# 把项目根目录加入 sys.path，以便 import backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .models import (
    AnalyzeResponse,
    FactorItem,
    AnnouncementRiskItem,
    AttributionEvidenceItem,
    FinancialAnomalyItem,
    SimilarCaseItem,
)
from backend.config import PREDICTOR_HORIZONS, PREDICTOR_MODEL_DIR, RISK_THRESHOLDS
from backend.skills.evidence_policy import publishable_evidence

REPORTS_DIR = PROJECT_ROOT / "backend" / "data" / "output" / "reports"
MODELS_MANIFEST_PATH = PREDICTOR_MODEL_DIR / "models_manifest.json"

# ---------- 公司代码 → 最新离线报告 ----------

# manifest.json 内存缓存（mtime 失效）：避免每次离线查询都全量解析清单
_manifest_cache: list[dict] | None = None
_manifest_cache_mtime: float | None = None

# 单个报告 JSON 内存缓存（mtime 失效）：避免同一公司多次查询都读盘
_report_cache: dict[str, tuple[float, dict]] = {}


def _load_manifest() -> list[dict]:
    """读取 manifest.json（带 mtime 失效的内存缓存）。"""
    global _manifest_cache, _manifest_cache_mtime
    path = REPORTS_DIR / "manifest.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if _manifest_cache is not None and _manifest_cache_mtime == mtime:
        return _manifest_cache
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _manifest_cache = payload if isinstance(payload, list) else []
    except (OSError, UnicodeError, json.JSONDecodeError):
        _manifest_cache = []
    _manifest_cache_mtime = mtime
    return _manifest_cache


def _read_report_path(path: Path) -> dict | None:
    """读取报告 JSON，按 mtime 缓存；损坏文件视为不可用。"""
    global _report_cache
    try:
        key = str(path)
        mtime = path.stat().st_mtime
        cached = _report_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        _report_cache[key] = (mtime, data)
        return data
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


def _is_usable_report(report: dict | None) -> bool:
    """拒绝损坏或由全源失败产生的空报告，防止其污染 latest。"""
    if not report:
        return False
    quality = report.get("quality", {}) or {}
    if quality.get("publishable") is False or quality.get("status") == "invalid":
        return False
    scorecard = report.get("scorecard", {}) or {}
    # 兼容早期/测试报告契约；新版报告必须有评分卡概率。
    if not scorecard and report.get("risk") is not None:
        return True
    probabilities = [scorecard.get(f"probability_{w}d") for w in (30, 60, 90)]
    if not any(value is not None for value in probabilities):
        return False
    semantic = report.get("semantic", {}) or {}
    financial = report.get("financial", {}) or {}
    name = str(report.get("name") or "").strip()
    company = str(report.get("company") or "").strip()
    empty_failed_runtime = (
        report.get("data_source") == "offline_lookup"
        and int(semantic.get("announcement_count", 0) or 0) == 0
        and bool(financial.get("skip"))
        and (not name or name == company)
    )
    return not empty_failed_runtime


def _candidate_entries(company: str) -> list[dict]:
    manifest = _load_manifest()
    normalized = company.upper().replace(".", "_")
    return [
        e for e in manifest
        if str(e.get("company", "")).upper().replace(".", "_") == normalized
        and (REPORTS_DIR / str(e.get("json_file", ""))).is_file()
    ]


def _select_report_entry(company: str, as_of: str | None = None) -> tuple[dict, dict] | None:
    """选择可用报告；历史查询按 report.as_of<=请求日取最近一份。"""
    candidates: list[tuple[str, str, dict, dict]] = []
    for entry in _candidate_entries(company):
        path = REPORTS_DIR / str(entry.get("json_file", ""))
        report = _read_report_path(path)
        if not _is_usable_report(report):
            continue
        report_as_of = str(entry.get("as_of") or report.get("as_of") or "")[:10]
        if as_of and (not report_as_of or report_as_of > as_of):
            continue
        candidates.append((report_as_of, str(entry.get("generated_at", "")), entry, report))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def _latest_report_file(company: str, as_of: str | None = None) -> Path | None:
    selected = _select_report_entry(company, as_of)
    return REPORTS_DIR / selected[0]["json_file"] if selected else None


def get_model_metrics() -> dict:
    """读取模型评估指标（三窗口 AUC/F1/Top10%Recall/threshold）。

    优先读 models_manifest.json（train_models.py 产出，含 metrics 子键）；
    manifest 缺失时兜底读 model_summary.json 取 Ensemble 指标（同为训练真实产物）。
    返回 {"30d": {"AUC":..., "F1":..., "Top10%Recall":..., "threshold":...}, ...}，
    均找不到时返回空 dict（前端硬编码兜底）。
    """
    out = {}
    # 1) 优先 models_manifest.json
    if MODELS_MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(MODELS_MANIFEST_PATH.read_text(encoding="utf-8"))
            for h in PREDICTOR_HORIZONS:
                w = h.replace("d", "")
                cfg = manifest.get("windows", {}).get(w, {})
                metrics = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
                if metrics:
                    out[h] = {
                        "AUC": metrics.get("AUC"),
                        "F1": metrics.get("F1"),
                        "Top10%Recall": metrics.get("Top10%Recall"),
                        "threshold": metrics.get("threshold"),
                    }
            if manifest.get("metadata"):
                out["metadata"] = manifest["metadata"]
        except (OSError, json.JSONDecodeError):
            pass
    # 2) 兜底 model_summary.json（取每窗口 Ensemble 指标）
    if not out:
        summary_path = PREDICTOR_MODEL_DIR / "model_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                for h in PREDICTOR_HORIZONS:
                    w = h.replace("d", "")
                    ens = (summary.get("windows", {}) or {}).get(w, {}).get("Ensemble", {})
                    if ens:
                        out[h] = {
                            "AUC": ens.get("AUC"),
                            "F1": ens.get("F1"),
                            "Top10%Recall": ens.get("Top10%Recall"),
                            "threshold": ens.get("threshold"),
                        }
                if summary.get("generated_at"):
                    out["metadata"] = {"generated_at": summary["generated_at"],
                                       "models": summary.get("models", []),
                                       "source": "model_summary.json"}
            except (OSError, json.JSONDecodeError):
                pass
    return out


def get_report_download_path(company: str, fmt: str = "md") -> Path | None:
    """从 manifest 找某公司最新一份报告的下载路径（md/json）。"""
    selected = _select_report_entry(company)
    if not selected:
        return None
    entry, _ = selected
    file_key = "md_file" if fmt == "md" else "json_file"
    path = REPORTS_DIR / str(entry.get(file_key, ""))
    return path if path.is_file() else None


def _load_report(company: str, as_of: str | None = None) -> dict | None:
    """读取公司最新报告 JSON，带 mtime 失效的内存缓存。"""
    selected = _select_report_entry(company, as_of)
    return selected[1] if selected else None


def _load_report_md(company: str, as_of: str | None = None) -> str:
    path = _latest_report_file(company, as_of)
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

# SHAP 因子的人读名称/标签/描述映射（单一来源，_map_shap_to_factors 与
# _map_risk_factor_details 共用，避免双份维护）。
_SHAP_NAME_MAP = {
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
_SHAP_TAG_MAP = {
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
_SHAP_DESC_MAP = {
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


def _map_risk_level(raw: str) -> str:
    for k, v in _LEVEL_MAP.items():
        if k in (raw or ""):
            return v
    return "low"


_WINDOW_PROB_KEY = {30: "probability_30d", 60: "probability_60d", 90: "probability_90d"}


def _prob_for_window(scorecard: dict, window: int) -> float:
    """按预测窗口取对应概率（0~1），未知窗口回落 60d。"""
    key = _WINDOW_PROB_KEY.get(window, "probability_60d")
    return scorecard.get(key, 0) or 0


def _level_from_prob(prob: float) -> str:
    """基于概率（0~1）+ RISK_THRESHOLDS 重算风险等级。"""
    if prob >= RISK_THRESHOLDS["high"]:
        return "high"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "mid"
    return "low"


_LEVEL_TEXT_EN = {"low": "低风险", "mid": "中风险", "high": "高风险"}


def _factors_from_importance_csv(window: int, name_map, tag_map, desc_map) -> list:
    """读 feature_importance_{window}d.csv（训练产出的全局特征重要性）兜底 per-window SHAP。

    离线报告 scorecard 无 per-window shap 时使用；importance 归一化到 0~1 作 score。
    """
    csv_path = PREDICTOR_MODEL_DIR / f"feature_importance_{window}d.csv"
    if not csv_path.is_file():
        return []
    try:
        rows = []
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                feat = (r.get("feature") or "").strip()
                try:
                    imp = float(r.get("importance_ensemble") or 0)
                except (TypeError, ValueError):
                    imp = 0
                if feat and imp > 0:
                    rows.append((feat, imp))
    except (OSError, ValueError):
        return []
    if not rows:
        return []
    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:6]
    max_imp = top[0][1] or 1
    tag_pool = ["市场异动", "问询历史", "问询历史", "舆情", "市值", "财务"]
    factors = []
    for i, (feat, imp) in enumerate(top):
        tag_label = tag_pool[i % len(tag_pool)]
        tag, tag_text = tag_map.get(tag_label, ("finance", "财务"))
        score = round(min(imp / max_imp, 1.0), 2)
        color = "#EF4444" if score > 0.3 else "#F59E0B" if score > 0.15 else "#10B981"
        factors.append(FactorItem(
            name=name_map.get(feat, feat), tag=tag, tagText=tag_text,
            desc=desc_map.get(feat, "特征重要性贡献"),
            score=score, color=color,
        ))
    return factors


def _map_shap_to_factors(report: dict, window: int = 60) -> list[FactorItem]:
    """把 scorecard 的 SHAP + attribution.top_risk_factors 映射为 FactorItem（per-window）。

    优先级：
    1. scorecard[f"shap_features_{window}d"]（predictor 实时 per-window SHAP）
    2. scorecard["shap_features"]（旧报告仅 60d；非 60d 窗口跳过避免串窗口）
    3. feature_importance_{window}d.csv（训练全局重要性，per-window 真实兜底）
    attribution.top_risk_factors 仅 60d 归因，只在 window==60 时使用。
    """
    scorecard = report.get("scorecard", {}) or {}
    attribution = report.get("attribution", {}) or {}

    # per-window SHAP：优先 shap_features_{window}d；旧报告无该键时仅 60d 回退 shap_features
    shap_features = scorecard.get(f"shap_features_{window}d")
    if not shap_features and window == 60:
        shap_features = scorecard.get("shap_features", []) or []
    if not shap_features:
        shap_features = []

    name_map = _SHAP_NAME_MAP
    tag_map = _SHAP_TAG_MAP
    desc_map = _SHAP_DESC_MAP

    factors = []
    # attribution.top_risk_factors 仅 60d 归因，只在 window==60 时用（避免 30/90 显示 60d 归因）
    if window == 60:
        top_factors = attribution.get("top_risk_factors", []) or []
        for f in top_factors[:6]:
            feat = f.get("feature", "")
            shap_val = f.get("shap") or 0
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

    # shap_features 补
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

    # CSV 兜底（per-window 训练全局重要性）
    if not factors:
        factors = _factors_from_importance_csv(window, name_map, tag_map, desc_map)
    return factors


def _map_financial_table(report: dict) -> list[list[str]]:
    """financial.anomaly_list → [[指标, 本期值, 行业均值/阈值, 偏离度], ...]"""
    anomalies = report.get("financial", {}).get("anomaly_list", []) or []
    rows = []
    for a in anomalies[:6]:
        indicator = a.get("indicator", "—")
        value = str(a.get("value", "—"))
        threshold = str(a.get("threshold", "—"))
        evidence = a.get("evidence") or ""
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


def _map_financial_anomalies(report: dict) -> list[FinancialAnomalyItem]:
    """financial.anomaly_list → 前端财务异常信号对象列表。"""
    anomalies = report.get("financial", {}).get("anomaly_list", []) or []
    items = []
    for a in anomalies:
        items.append(FinancialAnomalyItem(
            type=a.get("type", "—"),
            severity=int(a.get("severity", 0) or 0),
            indicator=a.get("indicator", "—"),
            value=a.get("value", "—"),
            threshold=a.get("threshold", "—"),
            evidence=a.get("evidence", "—"),
            label_ref=a.get("label_ref", "—"),
        ))
    return items


def _map_attribution_text(report: dict) -> str:
    """attribution.top_risk_factors → 归因文本。"""
    factors = report.get("attribution", {}).get("top_risk_factors", []) or []
    if not factors:
        return "暂无归因数据"
    parts = []
    for f in factors[:5]:
        desc = f.get("description", f.get("feature", ""))
        shap_val = f.get("shap") or 0
        pct = abs(shap_val) * 100
        parts.append(f"{desc}（{pct:.1f}%）")
    return "、".join(parts)


def _build_text_summary(report: dict) -> str:
    """semantic → 文本摘要。"""
    sem = report.get("semantic", {}) or {}
    risk_factors = publishable_evidence(sem.get("risk_factors", []) or [])
    ann_count = sem.get("announcement_count", 0)
    rf_count = len(risk_factors)
    if rf_count:
        categories = list({f.get("category", "") for f in risk_factors if f.get("category")})
        cat_text = "、".join(categories[:3]) if categories else "多类"
        return (
            f"近一年共研读公告 <b>{ann_count}</b> 份，经规则引擎 + FinBERT + LLM 三通道抽取，"
            f"识别出 <b>{rf_count}</b> 条已核验风险证据，涵盖 {cat_text} 等类别。"
        )
    return f"近一年共研读公告 <b>{ann_count}</b> 份，未识别出显著风险信号。"


# ---------- 新增：公告研读风险证据表 ----------

_LEVEL_FROM_SEVERITY = {1: "低", 2: "低", 3: "中", 4: "高", 5: "高"}


def _map_announcement_risks(report: dict) -> list[AnnouncementRiskItem]:
    """semantic.risk_factors → 公告研读风险证据表。"""
    sem = report.get("semantic", {}) or {}
    risk_factors = publishable_evidence(sem.get("risk_factors", []) or [])
    if not risk_factors:
        return []

    # 按 severity 降序、confidence 降序取前 20 条（去重：相同证据只保留一条）
    sorted_factors = sorted(
        risk_factors,
        key=lambda f: (f.get("severity") or 0, f.get("confidence") or 0),
        reverse=True,
    )
    seen = set()
    items = []
    for f in sorted_factors:
        evidence = (f.get("evidence") or "").strip()
        key = (f.get("announcement_id"), evidence[:40])
        if key in seen:
            continue
        seen.add(key)
        severity = f.get("severity") or 3
        items.append(AnnouncementRiskItem(
            date=f.get("announcement_date") or "—",
            level=_LEVEL_FROM_SEVERITY.get(severity, "中"),
            l1=f.get("taxonomy_l1") or f.get("category") or "—",
            l2=f.get("taxonomy_l2") or f.get("label") or "—",
            description=(f.get("description") or "").strip() or "—",
            evidence=evidence or "—",
            title=f.get("announcement_title") or "—",
            sourceUrl=f.get("source_url") or "",
            pdfUrl=f.get("pdf_url") or "",
        ))
        if len(items) >= 20:
            break
    return items


def _map_announcement_review(report: dict) -> dict:
    """把公告扫描覆盖、低风险观察和未发布候选整理为前端审计信息。

    低风险观察只描述扫描覆盖和证据核验结果，不把规则候选伪装成公司风险事实；
    被标题、语境或发布门槛排除的内容以聚合审计项单独展示。
    """
    semantic = report.get("semantic", {}) or {}
    quality = semantic.get("data_quality", {}) or {}
    channels = semantic.get("channel_summary", {}) or {}
    rule_channel = channels.get("rule", {}) or {}

    def _count(value) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    as_of = str(quality.get("as_of") or report.get("as_of") or "")[:10] or "—"
    lookback_days = _count(quality.get("lookback_days")) or 365
    reviewed_count = _count(
        quality.get("announcement_count", semantic.get("announcement_count", 0))
    )
    eligible_count = _count(quality.get("analysis_eligible_count"))
    attempted_count = _count(
        quality.get("pdf_attempted_count", quality.get("analyzed_document_count", 0))
    )
    parsed_count = _count(quality.get("pdf_parsed_count"))
    title_excluded_count = _count(quality.get("title_excluded_count"))
    not_fulltext_count = _count(quality.get("not_fulltext_count"))
    suppressed_count = _count(rule_channel.get("suppressed_count"))
    candidate_count = _count(semantic.get("candidate_count"))
    verified_count = len(publishable_evidence(semantic.get("risk_factors", []) or []))
    unpublished_count = max(0, candidate_count - verified_count)

    low_signals: list[dict] = []
    if reviewed_count:
        low_signals.append({
            "date": as_of,
            "level": "低",
            "l1": "审计观察",
            "l2": "公告覆盖",
            "description": f"近 {lookback_days} 天已获取 {reviewed_count} 份公告，{eligible_count} 份进入风险语义分析。",
            "evidence": "该项反映公告披露与扫描覆盖情况，属于常规观察，不代表已发生风险事件。",
            "title": "公告扫描覆盖汇总",
            "sourceUrl": "",
            "isObservation": True,
        })
    if attempted_count:
        coverage_pct = round(parsed_count / attempted_count * 100) if attempted_count else 0
        low_signals.append({
            "date": as_of,
            "level": "低",
            "l1": "审计观察",
            "l2": "正文解析",
            "description": f"深读 {attempted_count} 份公告，成功解析 {parsed_count} 份，正文解析覆盖率约 {coverage_pct}%。",
            "evidence": f"未取得完整正文的 {not_fulltext_count} 份公告不会被直接判定为风险事实。",
            "title": "公告正文解析质量",
            "sourceUrl": "",
            "isObservation": True,
        })
    if candidate_count and not verified_count:
        low_signals.append({
            "date": as_of,
            "level": "低",
            "l1": "审计观察",
            "l2": "交叉核验",
            "description": f"规则通道产生 {candidate_count} 条待复核候选，本次没有候选达到事实证据发布门槛。",
            "evidence": "单通道关键词命中仅作召回线索；未经原文事实语境及多通道核验，不作为风险事实展示。",
            "title": "候选信号交叉核验结果",
            "sourceUrl": "",
            "isObservation": True,
        })

    excluded: list[dict] = []
    if title_excluded_count:
        excluded.append({
            "category": "标题过滤",
            "count": title_excluded_count,
            "reason": "公司章程、议事规则、候选人声明等制度或程序性公告不进入风险抽取。",
            "basis": f"标题过滤规则 {quality.get('title_filter_version') or '当前版本'}",
        })
    if suppressed_count:
        excluded.append({
            "category": "语境排除",
            "count": suppressed_count,
            "reason": "否定表述、假设条款、法规引用或报表模板命中已被抑制。",
            "basis": "规则引擎事实语境与否定检测",
        })
    if unpublished_count:
        excluded.append({
            "category": "未发布候选",
            "count": unpublished_count,
            "reason": "仅规则命中或尚未取得 LLM 原文核验一致，不作为事实风险对外发布。",
            "basis": "证据发布策略：LLM 原文校验或规则与 LLM 交叉一致",
        })
    if not_fulltext_count:
        excluded.append({
            "category": "全文不足",
            "count": not_fulltext_count,
            "reason": "未获得可核验完整正文，保留为覆盖缺口，不据此形成风险结论。",
            "basis": "逐字证据完整性校验",
        })

    return {
        "reviewedCount": reviewed_count,
        "eligibleCount": eligible_count,
        "verifiedCount": verified_count,
        "candidateCount": candidate_count,
        "lowRiskSignals": low_signals,
        "excludedSignals": excluded,
        "excludedCount": sum(item["count"] for item in excluded),
        "source": quality.get("source") or "公告扫描审计",
        "asOf": as_of,
    }


# ---------- 新增：风险归因原文 ----------

def _map_attribution_evidence(report: dict) -> list[AttributionEvidenceItem]:
    """attribution.top_risk_factors + semantic.risk_factors → 归因原文。"""
    attribution = report.get("attribution", {}) or {}
    top_factors = attribution.get("top_risk_factors", []) or []
    semantic = report.get("semantic", {}) or {}
    risk_factors = publishable_evidence(semantic.get("risk_factors", []) or [])

    # 证据引用（若存在）
    citations = attribution.get("evidence_citations", []) or []
    if citations:
        allowed = {
            (str(f.get("evidence") or "").strip(), str(f.get("announcement_title") or "").strip())
            for f in risk_factors if str(f.get("evidence") or "").strip()
        }
        for anomaly in (report.get("financial", {}) or {}).get("anomaly_list", []) or []:
            evidence = str(anomaly.get("evidence") or "").strip()
            if evidence:
                allowed.add((evidence, "财务异常检测"))
        filtered = []
        for citation in citations:
            text_value = str(citation.get("text") or citation.get("evidence") or citation.get("snippet") or "").strip()
            source_value = str(citation.get("source") or "").strip()
            if any(text_value == evidence and (not source_value or not title or source_value == title)
                   for evidence, title in allowed):
                filtered.append(citation)
        return [
            AttributionEvidenceItem(
                factor=c.get("factor", "") or "风险因子",
                evidence=(c.get("text") or c.get("evidence") or c.get("snippet") or "").strip() or "—",
                source=c.get("source") or "",
                anchor=f"evidence-{i}",
            )
            for i, c in enumerate(filtered[:10])
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

        # 优先匹配 taxonomy_l2，其次 taxonomy_l1/category，最后 theme；
        # 均不命中则不再兜底到 risk_factors[0]（避免证据错配到不相关因子）
        matched = None
        for rf in risk_factors:
            if rf.get("risk_id") in used:
                continue
            rf_l2 = rf.get("taxonomy_l2", "")
            rf_l1 = rf.get("taxonomy_l1", "")
            rf_cat = rf.get("category", "")
            rf_label = rf.get("label", "")
            if taxonomy_l2 and rf_l2 == taxonomy_l2:
                matched = rf
                break
            if taxonomy_l1 and (rf_l1 == taxonomy_l1 or rf_cat == taxonomy_l1):
                matched = rf
                break
            if theme and (rf.get("theme_name") == theme or rf_label == theme):
                matched = rf
                break

        if matched:
            used.add(matched.get("risk_id"))
            evidence = (matched.get("evidence") or "").strip()
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

    name_map = _SHAP_NAME_MAP
    tag_map = _SHAP_TAG_MAP
    desc_map = _SHAP_DESC_MAP

    factors = []
    for i, f in enumerate(top_factors[:12]):
        feat = f.get("feature", "")
        shap_val = f.get("shap") or 0
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


def offline_to_response(company: str, window: int = 60, as_of: str | None = None) -> AnalyzeResponse | None:
    """核心映射：离线报告 JSON → AnalyzeResponse（按 window 取对应天数概率）。"""
    report = _load_report(company, as_of)
    if not report:
        return None
    resp = offline_to_response_from_report(report, window=window)
    md = _load_report_md(company, as_of)
    if md:
        resp.reportMarkdown = md
    return resp


def run_pipeline(codes: list, window: int = 60, use_llm: bool = False, use_bge: bool = True):
    """方案 B：离线数据快速查询（同步实现，由 FastAPI 放入线程池执行）。"""
    results = []
    for code in codes:
        resp = offline_to_response(code, window=window)
        if resp:
            results.append(resp)
    return results


def list_available_companies() -> list[dict]:
    """返回所有有离线报告的公司列表。"""
    manifest = _load_manifest()
    codes = []
    for e in manifest:
        code = e.get("company", "")
        if code and code not in codes:
            codes.append(code)
    results = []
    for code in codes:
        selected = _select_report_entry(code)
        if not selected:
            continue
        entry, report = selected
        results.append({
            "code": code,
            "name": report.get("name") or entry.get("name") or code,
            "generated_at": entry.get("generated_at", ""),
        })
    return results


# ============================================================
# 方案 C：实时扫雷管道 + WebSocket 进度推送
# ============================================================

import asyncio
import copy
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .models import ProgressMessage, ScanRequest


# ---------- Orchestrator 全局 LRU 缓存（避免每次请求都重新加载模型） ----------

ORCHESTRATOR_POOL_SIZE = int(os.getenv("ORCHESTRATOR_POOL_SIZE", "8"))
_orchestrator_pool: OrderedDict[str, "SweepingOrchestrator"] = OrderedDict()
_orchestrator_pool_lock = threading.Lock()


def get_orchestrator(use_llm: bool, use_finbert: bool, use_semantic_cases: bool = True, max_documents: int | None = 5):
    """获取带 LRU 淘汰的 Orchestrator。key 按参数组合区分。

    限制最大缓存数量，避免参数组合过多时内存无限增长；命中时移动到队尾。
    线程安全：加锁防止并发创建/淘汰竞态（创建在锁外，避免长时间持锁）。
    """
    key = f"llm={use_llm}_finbert={use_finbert}_bge={use_semantic_cases}_docs={max_documents}"
    with _orchestrator_pool_lock:
        if key in _orchestrator_pool:
            _orchestrator_pool.move_to_end(key)
            return _orchestrator_pool[key]
    # 延迟导入，避免模块级报错（锁外创建）
    from backend.agents.orchestrator import SweepingOrchestrator
    orchestrator = SweepingOrchestrator(
        use_llm=use_llm,
        use_finbert=use_finbert,
        use_semantic_cases=use_semantic_cases,
        max_documents=max_documents,
    )
    with _orchestrator_pool_lock:
        _orchestrator_pool[key] = orchestrator
        while len(_orchestrator_pool) > ORCHESTRATOR_POOL_SIZE:
            _orchestrator_pool.popitem(last=False)
    return orchestrator


def clear_orchestrator_pool() -> None:
    """清空 Orchestrator 缓存，用于测试或内存回收。"""
    with _orchestrator_pool_lock:
        _orchestrator_pool.clear()


# ---------- StreamingOrchestrator：在关键节点推送进度 ----------

class StreamingOrchestrator:
    """包装 SweepingOrchestrator，在每个 Agent 前后回调进度。

    注意：execute() 整体仍是同步阻塞的，必须在线程池中运行。
    回调函数会被从工作线程调用，因此需要线程安全地写入 queue。
    """

    _AGENT_ORDER = [
        ("AnnouncementReader", "announcement", "公告研读", "读取并解析最新公告、抽取语义风险"),
        ("FinancialDetector", "financial", "财务异常", "检测财务指标异常与偏离度"),
        ("Predictor", "prediction", "预测建模", "XGBoost + SHAP 计算问询概率"),
        ("CaseRetriever", "case", "案例匹配", "BGE 语义检索历史问询案例"),
        ("ChunkRetriever", "chunk", "段落召回", "chunk 级证据召回（可选）"),
        ("Attributor", "attribution", "归因分析", "聚合归因解释与风险叙事"),
        ("Reporter", "report", "报告生成", "渲染风控简报并落盘"),
    ]

    _AGENT_TIMEOUTS = {
        "AnnouncementReader": 420,   # 公告研读最长（含 PDF/OCR/LLM）
        "FinancialDetector": 240,
        "Predictor": 120,
        "CaseRetriever": 180,
        "ChunkRetriever": 60,
        "Attributor": 60,
        "Reporter": 60,
    }

    @staticmethod
    def _agent_timeout(agent_name: str) -> int:
        """按运行模式给 Agent 设置可配置上限。

        训练同口径实时 F1 在 CPU 上需要依次运行三套模型，不能沿用普通
        规则研读的 420 秒上限，否则会把正常的长任务误判成无公告。
        """
        if agent_name == "AnnouncementReader" and os.getenv(
            "F1_ONLINE_SEMANTICS_ENABLED", "auto"
        ).lower() in {"1", "true", "yes", "on"}:
            return int(os.getenv("F1_ANNOUNCEMENT_TIMEOUT_SECONDS", "3600"))
        return StreamingOrchestrator._AGENT_TIMEOUTS.get(agent_name, 120)

    # Agent 类名 → SweepingOrchestrator 的 dispatch 方法名（单一来源的映射）
    _ORCH_DISPATCH = {
        "AnnouncementReader": "_run_announcement",
        "FinancialDetector": "_run_financial",
        "Predictor": "_run_predict",
        "CaseRetriever": "_run_cases",
        "ChunkRetriever": "_run_chunks",
        "Attributor": "_run_attribution",
        "Reporter": "_run_report",
    }

    def __init__(self, callback: Callable[[ProgressMessage], None]):
        self.callback = callback
        self._active_agents: dict[str, tuple] = {}
        self._closed_agent_keys: set[str] = set()

    def _run_agent(self, runner, company, ctx, timeout):
        """在 daemon 线程执行单个 Agent，join(timeout) 看门狗。

        返回 (outcome, error)：outcome ∈ {"ok", "timeout", "error"}。
        超时后主流程跳过该 Agent 继续，避免单个挂死节点永久阻塞整条流水线
        （配合 cancel_event 让整体超时后的旧线程在下一节点边界退出）。
        """
        box = {}
        isolated_ctx = copy.copy(ctx)
        # 给隔离 ctx 独立的 cancel_event：继承主状态，超时后只设 isolated 的，
        # 让 Agent 内部 _check_cancel 在下个检查点主动退出，不影响主流水线 cancel_event
        main_cancel = getattr(ctx, "cancel_event", None)
        isolated_cancel = threading.Event()
        if main_cancel is not None and main_cancel.is_set():
            isolated_cancel.set()
        if ctx is not None:
            class _CombinedCancelEvent:
                def is_set(self):
                    return isolated_cancel.is_set() or (
                        main_cancel is not None and main_cancel.is_set()
                    )

            for key, value in vars(ctx).items():
                if key == "cancel_event":
                    # 同时监听节点看门狗与用户取消，避免界面显示已取消但底层
                    # BGE/reranker/FinBERT 线程仍长期占用模型资源。
                    setattr(isolated_ctx, key, _CombinedCancelEvent())
                else:
                    setattr(isolated_ctx, key, copy.deepcopy(value))

        def worker():
            try:
                runner(company, isolated_ctx)
            except Exception as e:  # noqa: BLE001  记堆栈便于复盘，不阻断看门狗
                _logger.warning("Agent worker 异常: %s", e, exc_info=True)
                box["error"] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # 超时：发独立 cancel 信号，Agent 内部 _check_cancel 下个检查点主动退出
            # （Python 无法强杀线程，但 Agent 在耗时步骤间检查 cancel_event 可尽快退出，避免长期占用资源）
            isolated_cancel.set()
            _logger.warning("Agent 看门狗超时 %ss（runner=%s），已发独立 cancel 信号", timeout, runner)
            return "timeout", None
        if "error" in box:
            return "error", box["error"]
        # 只有按时成功的节点才原子合入共享 Context；超时线程的迟到写入被隔离丢弃。
        if ctx is not None:
            for key, value in vars(isolated_ctx).items():
                if key != "cancel_event":
                    setattr(ctx, key, value)
        return "ok", None

    def run(self, company: str, window: int = 60, as_of: str | None = None,
            use_llm: bool = False, use_bge: bool = True, max_documents: int | None = 5,
            cancel_event: threading.Event | None = None):
        from backend.context import Context

        orch = get_orchestrator(
            use_llm=use_llm,
            use_finbert=False,           # 前端未暴露 FinBERT 开关，固定关闭以提速
            use_semantic_cases=use_bge,
            max_documents=max_documents,
        )
        # 复用 SweepingOrchestrator 的 dispatch（单一来源），临时注入进度回调；
        # 实时任务由 _scan_lock 串行，finally 恢复避免单例回调泄漏。
        orch.progress_callback = self._detail_callback
        try:
            ctx = Context(company=company, window=window, as_of=as_of or str(date.today()))
            ctx.use_llm_summary = bool(use_llm)
            ctx.use_llm = bool(use_llm)
            ctx.use_bge = bool(use_bge)
            ctx.max_documents = max_documents
            ctx.cancel_event = cancel_event

            total = len(self._AGENT_ORDER)
            start_total = time.time()

            for idx, (agent_name, agent_key, display_name, desc) in enumerate(self._AGENT_ORDER, start=1):
                if getattr(ctx, "cancel_event", None) is not None and ctx.cancel_event.is_set():
                    raise PipelineCancelled("任务已取消")
                step_start = time.time()
                self._active_agents[agent_key] = (idx, total, agent_name, agent_key, display_name)
                self._closed_agent_keys.discard(agent_key)
                self._emit(idx, total, agent_name, agent_key, "running", f"{display_name} Agent 正在执行：{desc}", 0, 0)

                runner = getattr(orch, self._ORCH_DISPATCH.get(agent_name, ""), None)
                if runner is None:
                    raise RuntimeError(f"未找到 Agent {agent_name} 的执行入口")

                # 节点级看门狗：单个 Agent 挂死只跳过它，不阻塞后续环节
                timeout = self._agent_timeout(agent_name)
                if agent_name == "Reporter":
                    ctx.meta["runtime_quality"] = _evaluate_runtime_quality(ctx)
                outcome, error = self._run_agent(runner, company, ctx, timeout)
                latency = int((time.time() - step_start) * 1000)
                self._closed_agent_keys.add(agent_key)

                if outcome == "timeout":
                    ctx.meta.setdefault("agent_timeouts", {})[agent_name] = {
                        "timeout_seconds": timeout,
                        "reason": f"{display_name}语义模型处理超过 {timeout}s",
                    }
                    ctx.trace_log.append({"agent": agent_name, "status": "timeout",
                                          "reason": f"{display_name} 超过 {timeout}s 看门狗触发", "trace_complete": True})
                    self._emit(idx, total, agent_name, agent_key, "skipped",
                               f"{display_name} Agent 超时跳过（{latency} ms）", latency, 100)
                elif outcome == "error":
                    # 段落召回失败可跳过，不打断流水线
                    if agent_name == "ChunkRetriever":
                        ctx.trace_log.append({"agent": agent_name, "status": "skipped", "reason": str(error), "trace_complete": True})
                        self._emit(idx, total, agent_name, agent_key, "skipped", f"{display_name} Agent 跳过：{error}", latency, 100)
                    else:
                        self._emit(idx, total, agent_name, agent_key, "error", f"{display_name} Agent 失败：{error}", latency, 100)
                        raise error
                else:
                    self._emit(idx, total, agent_name, agent_key, "done", f"{display_name} Agent 完成（{latency} ms）", latency, 100)

            ctx.meta["total_elapsed_ms"] = int((time.time() - start_total) * 1000)
            return ctx
        finally:
            orch.progress_callback = None

    def _emit(self, step: int, total: int, agent: str, agent_key: str, status: str,
              message: str, elapsed_ms: int, progress_percent: int):
        self.callback(ProgressMessage(
            type="progress",
            step=step,
            total=total,
            agent=agent,
            agent_key=agent_key,
            status=status,
            progress_percent=progress_percent,
            message=message,
            elapsed_ms=elapsed_ms,
        ))

    def _detail_callback(self, payload):
        """把 Agent 内部事件翻译为统一 WebSocket 进度消息。"""
        payload = payload or {}
        agent_key = str(payload.get("agent_key") or "")
        if not agent_key or agent_key in self._closed_agent_keys:
            return
        metadata = self._active_agents.get(agent_key)
        if metadata is None:
            return
        idx, total, agent, agent_key, display_name = metadata
        event = str(payload.get("event", "agent_progress"))
        percent = int(payload.get("percent", 0) or 0)
        message = str(payload.get("message", "") or "")
        if event == "pdf_processing":
            current = int(payload.get("current", 0) or 0)
            count = max(1, int(payload.get("total", 1) or 1))
            percent = 15 + int(25 * current / count)
            message = f"正在下载/解析第 {current}/{count} 份公告 PDF"
        event_map = {
            "offline_snapshot_started": (8, "正在检查官方公告离线快照"),
            "offline_snapshot_completed": (35, "已加载官方公告离线快照"),
            "online_company_started": (8, "正在校验公司与交易所代码"),
            "online_metadata_started": (12, "正在获取公告列表"),
            "online_metadata_completed": (15, f"已获取公告列表，共 {payload.get('announcement_count', 0)} 份"),
            "pdf_processing": (percent, message),
            "pdf_processing_completed": (42, "公告 PDF 下载与 OCR 解析完成"),
            "rule_analysis_started": (45, "正在匹配风险词典"),
            "rule_analysis_completed": (65, "风险词典匹配完成"),
            "finbert_started": (70, "正在执行 FinBERT 语义筛查"),
            "finbert_completed": (82, "FinBERT 语义筛查完成"),
            "llm_started": (84, "正在执行 LLM 精细抽取"),
            "llm_completed": (92, "LLM 精细抽取完成"),
            "finalizing": (96, "正在汇总公告风险证据"),
            "analysis_completed": (100, "公告研读完成"),
            "agent_progress": (percent, message),
        }
        percent, default_message = event_map.get(event, (percent or 20, message or f"{display_name} Agent 处理中"))
        channel_status = str(payload.get("status") or "").lower()
        display_status = "running"
        if event in {"finbert_started", "finbert_completed", "llm_started", "llm_completed"}:
            if channel_status in {"disabled", "not_configured", "skipped"} or payload.get("enabled") is False:
                display_status = "skipped"
                label = "FinBERT" if event.startswith("finbert") else "LLM"
                suffix = "已禁用" if channel_status == "disabled" or payload.get("enabled") is False else "未配置"
                message = f"{label} {suffix}"
            elif channel_status == "failed":
                display_status = "error"
        self._emit(idx, total, agent, agent_key, display_status, message or default_message, 0, max(0, min(100, percent)))


def _evaluate_runtime_quality(ctx) -> dict:
    """判定本次实时运行能否作为新的事实快照发布。"""
    reasons = []
    announcement_timeout = ((getattr(ctx, "meta", {}) or {}).get(
        "agent_timeouts", {}
    ) or {}).get("AnnouncementReader")
    if announcement_timeout:
        reasons.append(announcement_timeout.get("reason") or "公告研读语义模型处理超时")
    semantic_quality = getattr(ctx.semantic, "data_quality", {}) or {}
    announcement_count = int((getattr(ctx.semantic, "stats", {}) or {}).get("announcement_count", 0) or 0)
    semantic_available = bool(semantic_quality.get("current_data_available")) or announcement_count > 0
    if not semantic_available:
        reasons.append("当前公告事实源无可用数据")
    financial_available = bool(getattr(ctx.financial, "features", {}) or {}) and not bool(ctx.financial.skip)
    if not financial_available:
        reasons.append(ctx.financial.skip_reason or "当前财务特征不可用")
    source = str((ctx.prediction or {}).get("data_source") or "unavailable")
    if source != "realtime":
        reasons.extend((ctx.prediction or {}).get("degraded_reasons", []) or [])
        reasons.append(f"预测数据源为 {source}，不是实时同口径模型输入")
    reasons = list(dict.fromkeys(str(reason) for reason in reasons if reason))
    return {
        "publishable": semantic_available and financial_available and source == "realtime",
        "status": "valid" if not reasons else "degraded",
        "degraded_reasons": reasons,
        "announcement_available": semantic_available,
        "financial_available": financial_available,
        "prediction_source": source,
    }

# ---------- 报告 ctx -> AnalyzeResponse ----------

def report_ctx_to_response(ctx) -> AnalyzeResponse:
    """把 SweepingOrchestrator 跑完后的 ctx.report['json'] 映射为前端格式。"""
    report = ctx.report["json"]
    return offline_to_response_from_report(report)


def offline_to_response_from_report(report: dict, window: int = 60) -> AnalyzeResponse:
    """复用方案 B 的字段映射逻辑，但直接接收 report dict；按 window 取对应天数概率。"""
    scorecard = report.get("scorecard", {}) or {}
    prob = _prob_for_window(scorecard, window)
    level = _level_from_prob(prob)
    confidence = scorecard.get("confidence", 0) or 0
    financial = report.get("financial", {}) or {}
    semantic = report.get("semantic", {}) or {}
    similar_cases = report.get("similar_cases", []) or []
    attribution = report.get("attribution", {}) or {}
    quality = report.get("quality", {}) or {}
    model_version = str(scorecard.get("model_version") or report.get("model_version") or "")
    data_source = str(report.get("data_source") or scorecard.get("data_source") or "unknown")
    degraded_reasons = list(quality.get("degraded_reasons") or scorecard.get("degraded_reasons") or [])
    if not quality or not model_version:
        data_source = "legacy_snapshot_unverified"
        degraded_reasons.append("历史报告未记录运行质量或模型版本，来源状态不可追溯")
        model_version = model_version or "legacy-unversioned"
    degraded_reasons = list(dict.fromkeys(degraded_reasons))

    risk_factors_count = len(publishable_evidence(semantic.get("risk_factors", []) or []))
    anomaly_count = len(financial.get("anomaly_list", []) or [])
    total_factors = risk_factors_count + anomaly_count

    prob_pct = prob * 100
    top_factor_names = [
        f.get("description", f.get("feature", ""))
        for f in (attribution.get("top_risk_factors", []) or [])[:3]
    ]
    factor_text = "、".join(top_factor_names) if top_factor_names else "综合多维度指标"
    summary = (
        f"未来 {window} 天收到问询函的概率为 <b>{prob_pct:.1f}%</b>。"
        f"主要风险来源为{factor_text}。"
    )

    conclusion = report.get("executive_summary", "")
    if not conclusion:
        conclusion = f"风险等级为{_LEVEL_TEXT_EN.get(level, '低风险')}，建议持续跟踪。"

    # 三档窗口概率（前端切换窗口用，单位 %）
    risk_by_window = {
        f"{w}d": round(_prob_for_window(scorecard, w) * 100, 1) for w in (30, 60, 90)
    }

    factors_list = _map_shap_to_factors(report, 60)
    # 三窗口模型评估指标（优先 models_manifest.json，兜底 model_summary.json Ensemble）
    model_metrics = get_model_metrics() or {}
    # 三窗口完整预测（供前端 renderPredictorDetail 30/60/90 切换：风险概率 + 置信度 + 该窗口评估指标 + per-window SHAP）
    window_predictions = []
    for w in (30, 60, 90):
        p = _prob_for_window(scorecard, w)
        lvl = _level_from_prob(p)
        h_key = f"{w}d"
        window_predictions.append({
            "window": w,
            "risk": round(p * 100, 1),
            "confidence": round(confidence, 2),
            "level": lvl,
            "levelText": _LEVEL_TEXT_EN.get(lvl, "低风险"),
            "factors": _map_shap_to_factors(report, w),  # per-window SHAP（实时/CSV 兜底）
            "metrics": model_metrics.get(h_key, {}) if isinstance(model_metrics, dict) else {},
        })
    return AnalyzeResponse(
        code=report.get("company", ""),
        name=report.get("name", ""),
        risk=round(prob_pct, 1),
        level=level,
        levelText=_LEVEL_TEXT_EN.get(level, "低风险"),
        confidence=round(confidence, 2),
        confidenceMeaning=scorecard.get("confidence_meaning", "predicted_class_score"),
        dataSource=data_source,
        dataCoverage=(report.get("profile", {}) or {}).get("coverage") or scorecard.get("coverage") or {},
        degradedReasons=degraded_reasons,
        featureAnchor=str(scorecard.get("feature_anchor") or ""),
        modelVersion=model_version,
        factors=total_factors,
        summary=summary,
        factorsList=factors_list,
        financialTable=_map_financial_table(report),
        financialAnomalies=_map_financial_anomalies(report),
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
        announcementReview=_map_announcement_review(report),
        attributionEvidence=_map_attribution_evidence(report),
        riskFactorDetails=_map_risk_factor_details(report),
        riskByWindow=risk_by_window,
        windowPredictions=window_predictions,
        modelMetrics={k: v for k, v in (model_metrics or {}).items() if k != "metadata"} if isinstance(model_metrics, dict) else {},
    )


# ---------- 任务状态管理 ----------


def agent_metadata() -> list[dict]:
    """返回 7 个 Agent 的元数据（key/agent_key/显示名/描述），作为前端 Agent 清单的单一来源。"""
    return [
        {"key": agent_name, "agent_key": agent_key, "name": display_name, "description": desc}
        for (agent_name, agent_key, display_name, desc) in StreamingOrchestrator._AGENT_ORDER
    ]


AGENT_TOTAL = len(StreamingOrchestrator._AGENT_ORDER)
TERMINAL_STATUSES = {"completed", "failed", "fallback", "cancelled"}
RESULT_CACHE_SIZE = 20
TERMINAL_TASK_TTL = 600.0   # 终态任务保留时长（秒），供断线重连 / 状态查询
MAX_TERMINAL_TASKS = 20     # 最多保留的终态任务数


class PipelineCancelled(RuntimeError):
    pass


@dataclass
class TaskState:
    task_id: str
    code: str
    status: str = "pending"           # pending / running / completed / failed / fallback / cancelled
    progress: list[ProgressMessage] = field(default_factory=list)
    result: AnalyzeResponse | None = None
    error: str | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_task_store: dict[str, TaskState] = {}
_task_handles: dict[str, asyncio.Task] = {}
_result_cache: OrderedDict[str, AnalyzeResponse] = OrderedDict()

# 实时扫雷串行锁：SweepingOrchestrator 内部有全局模型/状态，并发容易竞争/死锁，
# 因此同一时刻只跑一个实时任务，其他任务排队等待。
_scan_lock = asyncio.Lock()


def _cache_key(code: str, window: int, as_of: str | None = None) -> str:
    """实时结果缓存 key：公司代码 + 窗口 + 截止日期。"""
    parts = [str(code).strip().upper(), str(int(window))]
    if as_of:
        parts.append(str(as_of).strip())
    return "_".join(parts)


def get_cached_result(code: str, window: int = 60, as_of: str | None = None) -> AnalyzeResponse | None:
    key = _cache_key(code, window, as_of)
    result = _result_cache.get(key)
    if result is None:
        return None
    _result_cache.move_to_end(key)
    return result.model_copy(deep=True)


def cache_result(code: str, window: int, result: AnalyzeResponse, as_of: str | None = None) -> None:
    key = _cache_key(code, window, as_of)
    _result_cache[key] = result.model_copy(deep=True)
    _result_cache.move_to_end(key)
    while len(_result_cache) > RESULT_CACHE_SIZE:
        _result_cache.popitem(last=False)


def get_offline_result(code: str, window: int = 60, as_of: str | None = None) -> AnalyzeResponse | None:
    # 历史查询选择 report.as_of <= 请求日的最近一份可用报告。
    cached = get_cached_result(code, window, as_of)
    if cached is not None:
        return cached
    try:
        result = offline_to_response(code, window=window, as_of=as_of)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        result = None
    if result is not None:
        cache_result(code, window, result, as_of)
    return result


def _prune_finished_tasks() -> None:
    """清理已终态的任务，避免 state.progress + result 永久驻留内存。

    策略：超过 TERMINAL_TASK_TTL 的终态任务直接删除；否则仅保留最近
    MAX_TERMINAL_TASKS 个（超出部分删最老的），保证状态查询与断线重连仍可用。
    """
    now = time.time()
    for state in list(_task_store.values()):
        if state.status in TERMINAL_STATUSES:
            if state.finished_at and now - state.finished_at > TERMINAL_TASK_TTL:
                _task_store.pop(state.task_id, None)
    still_finished = sorted(
        (s for s in _task_store.values() if s.status in TERMINAL_STATUSES),
        key=lambda s: s.finished_at or 0,
    )
    overflow = len(still_finished) - MAX_TERMINAL_TASKS
    for state in still_finished[:max(0, overflow)]:
        _task_store.pop(state.task_id, None)


def _persist_trace(company: str, task_id: str, trace_log: list) -> None:
    """把 ctx.trace_log 落盘为 JSONL（赛后复盘审计），失败静默不影响主流程。"""
    try:
        from backend.config import TRACE_DIR
        safe_code = str(company).replace(".", "_")
        path = TRACE_DIR / f"{safe_code}_{task_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for trace in trace_log:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def create_task(code: str) -> TaskState:
    _prune_finished_tasks()
    task_id = f"scan_{uuid.uuid4().hex[:12]}"
    state = TaskState(task_id=task_id, code=code)
    _task_store[task_id] = state
    return state


def get_task(task_id: str) -> TaskState | None:
    return _task_store.get(task_id)


def active_tasks() -> list[TaskState]:
    return [state for state in _task_store.values() if state.status in {"pending", "running"}]


def bind_task_handle(task_id: str, handle: asyncio.Task) -> None:
    _task_handles[task_id] = handle


def subscribe_task(state: TaskState) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    state.subscribers.add(queue)
    return queue


def unsubscribe_task(state: TaskState, queue: asyncio.Queue) -> None:
    state.subscribers.discard(queue)


def _record_message(state: TaskState, msg: ProgressMessage) -> None:
    # 历史由生产端记录，不依赖 WebSocket 是否已连接。
    state.progress.append(msg)
    for subscriber in tuple(state.subscribers):
        try:
            subscriber.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def emit_message(state: TaskState, msg: ProgressMessage) -> None:
    _record_message(state, msg)


def _thread_safe_emit(loop: asyncio.AbstractEventLoop, state: TaskState, msg: ProgressMessage):
    """从工作线程写入历史并广播给所有 WebSocket 订阅者。"""
    if not loop.is_closed():
        loop.call_soon_threadsafe(_record_message, state, msg)


async def cancel_task(task_id: str) -> TaskState | None:
    state = get_task(task_id)
    if state is None:
        return None
    if state.status in TERMINAL_STATUSES:
        return state

    state.cancel_event.set()
    previous_status = state.status
    state.status = "cancelled"
    state.finished_at = time.time()
    emit_message(state, ProgressMessage(
        type="cancelled",
        step=0,
        total=AGENT_TOTAL,
        agent="SweepingOrchestrator",
        agent_key="orchestrator",
        status="cancelled",
        progress_percent=0,
        message=f"任务 {task_id} 已取消",
    ))

    # 排队中的协程可直接取消；已进入同步 Agent 的任务采用边界点协作取消，
    # 避免释放串行锁后底层线程仍在运行、与新任务抢模型资源。
    handle = _task_handles.get(task_id)
    if previous_status == "pending" and handle is not None and not handle.done():
        handle.cancel()
    return state


async def run_scan_task(state: TaskState, req: ScanRequest):
    """在后台执行实时扫雷，并通过 queue 推送进度。"""
    loop = asyncio.get_running_loop()

    def callback(msg: ProgressMessage):
        _thread_safe_emit(loop, state, msg)

    # 排队等待：同一时刻只跑一个实时任务
    state.status = "pending"
    emit_message(state, ProgressMessage(
        type="progress",
        step=0,
        total=AGENT_TOTAL,
        agent="SweepingOrchestrator",
        agent_key="orchestrator",
        status="running",
        progress_percent=0,
        message=f"任务 {state.task_id} 已排队，正在等待模型资源（当前有任务正在执行）...",
        elapsed_ms=0,
    ))
    try:
        async with _scan_lock:
            if state.cancel_event.is_set():
                raise PipelineCancelled("任务已取消")
            state.status = "running"
            emit_message(state, ProgressMessage(
                type="progress",
                step=0,
                total=AGENT_TOTAL,
                agent="SweepingOrchestrator",
                agent_key="orchestrator",
                status="running",
                progress_percent=0,
                message=f"正在为 {req.code} 启动 {AGENT_TOTAL}-Agent 实时扫雷流水线...",
                elapsed_ms=0,
            ))

            streamer = StreamingOrchestrator(callback=callback)
            # 去掉 asyncio.shield：超时后让 wait_for 及时返回并释放串行锁，
            # 底层线程由 StreamingOrchestrator 的节点看门狗 + cancel_event 保证有界退出。
            fullrun_enabled = os.getenv(
                "F1_ONLINE_SEMANTICS_ENABLED", "auto"
            ).lower() in {"1", "true", "yes", "on"}
            pipeline_timeout = int(os.getenv(
                "REALTIME_PIPELINE_TIMEOUT_SECONDS", "3900" if fullrun_enabled else "600"
            ))
            ctx = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: streamer.run(
                        company=req.code,
                        window=req.window,
                        as_of=req.as_of,
                        use_llm=req.use_llm,
                        use_bge=req.use_bge,
                        max_documents=req.max_documents,
                        cancel_event=state.cancel_event,
                    ),
                ),
                timeout=pipeline_timeout,
            )
            if state.cancel_event.is_set():
                raise PipelineCancelled("任务已取消")
            quality = (getattr(ctx, "meta", {}) or {}).get("runtime_quality", {}) or {}
            if quality.get("publishable") is False:
                reasons = quality.get("degraded_reasons", []) or ["实时数据质量未达到发布条件"]
                await asyncio.to_thread(_persist_trace, req.code, state.task_id, getattr(ctx, "trace_log", []) or [])
                await _fallback_after_error(state, req, "；".join(reasons))
                return
            result = report_ctx_to_response(ctx)
            await asyncio.to_thread(_persist_trace, req.code, state.task_id, getattr(ctx, "trace_log", []) or [])
            state.result = result
            state.status = "completed"
            state.finished_at = time.time()
            cache_result(req.code, req.window, result, req.as_of)
            complete_msg = ProgressMessage(
                type="complete",
                step=AGENT_TOTAL,
                total=AGENT_TOTAL,
                agent="Reporter",
                agent_key="report",
                status="done",
                progress_percent=100,
                message="报告生成完成",
                elapsed_ms=0,
                result=result,
            )
            emit_message(state, complete_msg)
    except (PipelineCancelled, asyncio.CancelledError):
        state.cancel_event.set()
        if state.status != "cancelled":
            state.status = "cancelled"
            state.finished_at = time.time()
            emit_message(state, ProgressMessage(
                type="cancelled", total=AGENT_TOTAL,
                agent="SweepingOrchestrator", agent_key="orchestrator",
                status="cancelled", message=f"任务 {state.task_id} 已取消",
            ))
    except asyncio.TimeoutError:
        state.cancel_event.set()
        fullrun_enabled = os.getenv(
            "F1_ONLINE_SEMANTICS_ENABLED", "auto"
        ).lower() in {"1", "true", "yes", "on"}
        timeout = os.getenv(
            "REALTIME_PIPELINE_TIMEOUT_SECONDS", "3900" if fullrun_enabled else "600"
        )
        await _fallback_after_error(
            state, req, f"实时扫雷超过 {timeout} 秒运行上限。"
        )
    except Exception as exc:
        if state.cancel_event.is_set():
            state.status = "cancelled"
            state.finished_at = state.finished_at or time.time()
        else:
            await _fallback_after_error(state, req, f"{type(exc).__name__}: {exc}")
    finally:
        current = _task_handles.get(state.task_id)
        if current is asyncio.current_task():
            _task_handles.pop(state.task_id, None)


async def _fallback_after_error(state: TaskState, req: ScanRequest, error: str) -> None:
    state.error = error
    emit_message(state, ProgressMessage(
        type="error", total=AGENT_TOTAL,
        agent="SweepingOrchestrator", agent_key="orchestrator",
        status="error", progress_percent=0,
        message=error, error=error, fatal=False,
    ))
    offline = get_offline_result(req.code, req.window, req.as_of)
    state.finished_at = time.time()
    if offline is not None:
        state.result = offline
        state.status = "fallback"
        emit_message(state, ProgressMessage(
            type="fallback", total=AGENT_TOTAL,
            agent="SweepingOrchestrator", agent_key="orchestrator",
            status="done", progress_percent=100,
            message="实时扫雷失败，已自动切换离线快照",
            error=error, result=offline,
        ))
    else:
        state.status = "failed"
        emit_message(state, ProgressMessage(
            type="error", total=AGENT_TOTAL,
            agent="SweepingOrchestrator", agent_key="orchestrator",
            status="error", progress_percent=0,
            message=f"{error}；且未找到可用离线快照",
            error=error, fatal=True,
        ))
