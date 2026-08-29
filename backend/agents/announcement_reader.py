#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公告研读 Agent：巨潮事实主源 → 三通道风险抽取 → 可审计 F1。"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import Counter
from datetime import date

from ..config import (
    ANNOUNCE_MAX_DOCUMENTS,
    ANNOUNCE_SOURCE,
    ANNOUNCE_WINDOW_DAYS,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    F1_DECAY_HALF_LIFE_DAYS,
    FINBERT_GATE,
    FINBERT_ENABLED,
    FINBERT_GATE_ENABLED,
    MAX_TEXT_CHARS,
)
from ..llm import chat_json
from ..run_config import RunConfig
from ..skills.announcement_context_filter import (
    FILTER_VERSION,
    apply_title_policy,
    contextual_suppression_reason,
    is_analysis_eligible,
)
from ..skills.rule_risk_extract import RuleRiskExtractor
from .base import AgentBase

_logger = logging.getLogger(__name__)


SEVERITY_NUMBER = {"critical": 5, "high": 5, "medium": 3, "low": 2}
WINDOWS = (30, 60, 90)


def _severity(value):
    if isinstance(value, (int, float)):
        return max(1, min(5, int(value)))
    return SEVERITY_NUMBER.get(str(value or "").lower(), 2)


def _event_key(announcement_id, label):
    return hashlib.sha256(f"{announcement_id}|{label}".encode("utf-8")).hexdigest()[:20]


def _risk_id(event_key, method, evidence_start):
    return hashlib.sha256(
        f"{event_key}|{method}|{evidence_start}".encode("utf-8")
    ).hexdigest()[:20]


def _within(value, cutoff, days):
    delta = (cutoff - date.fromisoformat(str(value)[:10])).days
    return 0 <= delta < days


def _age_days(value, cutoff):
    """事件距 as_of 的天数（越小越新）。"""
    return (cutoff - date.fromisoformat(str(value)[:10])).days


def _decay_weight(age_days, half_life=F1_DECAY_HALF_LIFE_DAYS):
    """时间衰减权重：age=0 → 1.0；age=half_life → 0.5；指数衰减 2^(-age/half_life)。"""
    return 2.0 ** (-max(age_days, 0) / max(half_life, 1))


class AnnouncementReaderAgent(AgentBase):
    """只写 ``ctx.semantic``，不负责财务、预测、案例检索或报告。"""

    name = "AnnouncementReader"

    def __init__(
        self,
        data_root=None,
        use_finbert=None,
        use_llm=None,
        use_rule=None,
        max_text_chars=MAX_TEXT_CHARS,
        gate_threshold=FINBERT_GATE,
        source=None,
        max_documents=ANNOUNCE_MAX_DOCUMENTS,
        rule_extractor=None,
        llm_callable=None,
        progress_callback=None,
        run_config=None,
    ):
        super().__init__()
        rc = run_config or RunConfig()
        self.data_root = data_root
        # 公共开关：显式传入优先，否则从 RunConfig 读（默认值 = 历史无参行为）
        self.use_finbert = bool(rc.use_finbert if use_finbert is None else use_finbert)
        self.use_llm = bool(rc.use_llm if use_llm is None else use_llm)
        self.use_rule = bool(rc.use_rule if use_rule is None else use_rule)
        self.run_config = rc
        self.max_text_chars = int(max_text_chars)
        self.gate_threshold = float(gate_threshold)
        self.rule_extractor = rule_extractor or RuleRiskExtractor()
        self.progress_callback = progress_callback   # 需在 _default_source() 之前赋值
        self.max_documents = None if max_documents is None else int(max_documents)
        self.source = source or self._default_source()
        self.llm_callable = llm_callable or chat_json
        self.llm_configured = bool(llm_callable is not None or os.getenv("DEEPSEEK_API_KEY"))
        self.finbert = None
        self.finbert_error = ""
        if self.use_finbert and FINBERT_ENABLED:
            try:
                from ..skills.finbert_classify import FinBERTClient

                self.finbert = FinBERTClient()
            except Exception as exc:
                self.finbert_error = f"{type(exc).__name__}: {exc}"

    def _default_source(self):
        if ANNOUNCE_SOURCE == "local":
            return None
        from ..skills.announcement_search import CninfoAnnouncementSource
        from ..skills.competition_history import CompetitionAwareAnnouncementSource

        return CompetitionAwareAnnouncementSource(
            CninfoAnnouncementSource(
                # 必须用 self.max_documents（含 ScanRequest.max_documents 的实际请求值），
                # 否则源始终按 ANNOUNCE_MAX_DOCUMENTS=120 下载 PDF，请求的 5 份限制
                # 只截断元数据列表，PDF 已全部下载完。
                max_documents=self.max_documents,
                progress_callback=self.progress_callback,
            ),
            progress_callback=self.progress_callback,
        )

    def _emit_progress(self, event, **payload):
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **payload})

    def _load_announcements(self, company, as_of):
        if self.source is not None:
            return self.source.search(company, days=ANNOUNCE_WINDOW_DAYS, as_of=as_of)
        from pathlib import Path

        from ..config import DATA_RAW, INDEX_DIR
        from ..skills.announcement_search import AnnouncementStore

        code = re.search(r"\d{6}", str(company))
        if not code:
            raise ValueError("本地公告模式需要六位公司代码")
        secucode = str(company) if "." in str(company) else code.group(0)
        cache = Path(INDEX_DIR) / f"{secucode.replace('.', '_')}_index.json"
        store = AnnouncementStore(Path(self.data_root or DATA_RAW) / secucode, str(cache))
        announcements = store.search(days=ANNOUNCE_WINDOW_DAYS, as_of=as_of)
        if self.max_documents is not None:
            announcements = announcements[:self.max_documents]
        identity = {
            "code": code.group(0),
            "secucode": secucode,
            "company_name": secucode,
            "exchange": "",
            "org_id": "",
            "source_url": str(Path(self.data_root or DATA_RAW).resolve()),
        }
        for item in announcements:
            item.setdefault("announcement_id", item["id"])
            item.setdefault("published_at", item.get("date"))
            item.setdefault("source_name", "本地官方公告副本")
            item.setdefault("source_tier", "official_local_copy")
            item.setdefault("source_url", "")
            item.setdefault("pdf_url", "")
            item.setdefault("text_status", "local_parsed" if item.get("text") else "local_empty_text")
            item.setdefault("content_sha256", "")
        return identity, announcements

    def _rule_extract(self, announcements, cancel_event=None):
        factors, suppressed = [], 0
        per_announcement = {}
        for announcement in announcements:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消")
            source_text = announcement.get("text") or announcement.get("title") or ""
            evidence_field = "pdf_text" if announcement.get("text") else "title"
            hits = self.rule_extractor.extract(source_text)
            accepted, suppressed_hits = [], []
            for hit in hits:
                if hit.get("negated") or hit.get("excluded"):
                    suppressed += 1
                    suppressed_hits.append(
                        {
                            "rule_id": hit.get("rule_id", ""),
                            "label": hit.get("label", ""),
                            "matched_keyword": hit.get("matched_key", ""),
                            "evidence": hit.get("evidence", ""),
                            "suppression_reason": hit.get("suppression_reason")
                            or ("negated_context" if hit.get("negated") else "excluded_context"),
                        }
                    )
                    continue
                label = str(hit.get("label") or hit.get("rule_id") or "OTHER")
                event_key = _event_key(announcement["id"], label)
                factor = {
                    "risk_id": _risk_id(event_key, "deterministic_rule", hit.get("evidence_start", 0)),
                    "event_key": event_key,
                    "category": str(hit.get("category_id") or "OTHER"),
                    "label": label,
                    "risk_label": label,
                    "description": f"风险词典命中 {hit.get('matched_key')}；仅为候选信号，需复核原文。",
                    "severity": _severity(hit.get("severity")),
                    "confidence": 0.84 if evidence_field == "pdf_text" else 0.70,
                    "matched_keyword": hit.get("matched_key"),
                    "evidence": hit.get("evidence") or "",
                    "evidence_start": hit.get("evidence_start", 0),
                    "evidence_end": hit.get("evidence_end", 0),
                    "evidence_field": evidence_field,
                    "text_extraction": (
                        "rapidocr" if announcement.get("ocr_succeeded_pages", 0) else evidence_field
                    ),
                    "ocr_status": announcement.get("ocr_status", "not_reported"),
                    "ocr_mean_confidence": announcement.get("ocr_mean_confidence"),
                    "evidence_valid": bool(hit.get("evidence_valid")),
                    "announcement_id": announcement["id"],
                    "announcement_title": announcement.get("title", ""),
                    "announcement_date": announcement.get("date", ""),
                    "source_url": announcement.get("source_url", ""),
                    "pdf_url": announcement.get("pdf_url", ""),
                    "method": "deterministic_rule_with_negation",
                    "rule_id": hit.get("rule_id", ""),
                    "taxonomy_l1": hit.get("category_id", ""),
                    "taxonomy_l2": label,
                    "negation_checked": True,
                    "agreement_status": "rule_only",
                }
                factors.append(factor)
                accepted.append(factor)
            per_announcement[announcement["id"]] = {
                "rule_factors": accepted,
                "suppressed_rule_hits": suppressed_hits,
            }
        return factors, per_announcement, suppressed

    def _finbert_classify(self, announcements):
        if not self.use_finbert:
            return [], "disabled"
        if self.finbert is None:
            return [
                {"announcement_id": item["id"], "categories": [], "max_score": None}
                for item in announcements
            ], "not_configured"
        signals = []
        for item in announcements:
            try:
                result = self.finbert.classify(
                    (item.get("text") or item.get("title") or "")[: self.max_text_chars]
                )
                signals.append({"announcement_id": item["id"], **result})
            except Exception as exc:
                signals.append(
                    {
                        "announcement_id": item["id"],
                        "categories": [],
                        "max_score": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return signals, "experimental_unvalidated"

    def _llm_extract(self, company_name, announcements, cancel_event=None):
        factors, rejected, rejected_context, failed, per_announcement = [], 0, 0, 0, {}
        # 子进度只对有正文的公告计数：无正文的直接标记 no_full_text，
        # 避免进度显示“第 1/147 份”而其中绝大多数实际是秒过跳过。
        total_docs = sum(1 for item in announcements if (item.get("text") or "").strip())
        processed = 0
        for item in announcements:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消")
            text = item.get("text") or ""
            if not text:
                per_announcement[item["id"]] = {"llm_factors": [], "status": "no_full_text"}
                continue
            processed += 1
            self._emit_progress(
                "llm_processing",
                current=processed,
                total=total_docs,
                title=item.get("title", ""),
            )
            prompt_text = text[: self.max_text_chars]
            prompt = f"""你是上市公司公告证据抽取器。只能依据给出的公告正文，不得补充外部事实。
公司：{company_name}
公告：{item.get('title', '')}
日期：{item.get('date', '')}
正文：\n{prompt_text}

只输出 JSON：
{{"risk_factors":[{{"taxonomy_l1":"A-H或OTHER","taxonomy_l2":"如A03或OTHER","description":"风险描述","evidence":"正文中连续原文","severity":1,"assertion_type":"actual_event","subject":"实际发生事件的主体","event_action":"已经发生的动作"}}]}}
要求：
1. evidence 必须逐字存在于正文，severity 为1到5；没有事实证据返回空数组。
2. 只有公司、控股股东、实际控制人或董监高已经发生的现实事件，assertion_type 才能写 actual_event。
3. 法规引用、公司章程、管理制度、董监高职责或任职资格、禁止性/条件性条款、会计政策、表头、目录、附件模板不得识别为风险。
4. “如发生、若发生、存在下列情形之一、不得、应当、有权”等假设或规范描述不是已发生事实。
5. 描述其他公司、行业或历史案例时不得归因于本公司；无法区分时不输出。"""
            try:
                # max_tokens 需容纳 thinking(reasoning) + 答案：v4-flash 推理会占用约 2000 token
                result = self.llm_callable("", prompt, max_tokens=6000)
            except Exception as exc:
                failed += 1
                per_announcement[item["id"]] = {
                    "llm_factors": [],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            if not isinstance(result, dict) or "risk_factors" not in result:
                failed += 1
                per_announcement[item["id"]] = {
                    "llm_factors": [],
                    "status": "invalid_response",
                }
                continue
            accepted = []
            for raw in result.get("risk_factors", []):
                evidence = str(raw.get("evidence") or "").strip()
                start = prompt_text.find(evidence) if evidence else -1
                if start < 0:
                    rejected += 1
                    continue
                l1 = str(raw.get("taxonomy_l1") or "OTHER").upper()
                if l1 not in {*"ABCDEFGH", "OTHER"}:
                    l1 = "OTHER"
                label = str(raw.get("taxonomy_l2") or "OTHER").upper()
                if not re.fullmatch(r"(?:[A-H]\d{2}(?:-CANDIDATE)?|OTHER)", label):
                    label = "OTHER"
                assertion_type = str(raw.get("assertion_type") or "").lower()
                context_reason = contextual_suppression_reason(
                    label=label,
                    text=prompt_text,
                    start=start,
                    end=start + len(evidence),
                )
                if (assertion_type and assertion_type != "actual_event") or context_reason:
                    rejected_context += 1
                    continue
                event_key = _event_key(item["id"], label)
                factor = {
                    "risk_id": _risk_id(event_key, "llm_evidence_validated", start),
                    "event_key": event_key,
                    "category": l1,
                    "label": label,
                    "risk_label": label,
                    "description": str(raw.get("description") or ""),
                    "severity": _severity(raw.get("severity")),
                    "confidence": None,
                    "matched_keyword": "",
                    "evidence": evidence,
                    "evidence_start": start,
                    "evidence_end": start + len(evidence),
                    "evidence_field": "pdf_text",
                    "text_extraction": (
                        "rapidocr" if item.get("ocr_succeeded_pages", 0) else "pdf_text"
                    ),
                    "ocr_status": item.get("ocr_status", "not_reported"),
                    "ocr_mean_confidence": item.get("ocr_mean_confidence"),
                    "evidence_valid": True,
                    "announcement_id": item["id"],
                    "announcement_title": item.get("title", ""),
                    "announcement_date": item.get("date", ""),
                    "source_url": item.get("source_url", ""),
                    "pdf_url": item.get("pdf_url", ""),
                    "method": "llm_evidence_validated",
                    "rule_id": "",
                    "taxonomy_l1": l1,
                    "taxonomy_l2": label,
                    "negation_checked": False,
                    "agreement_status": "llm_only",
                    "assertion_type": assertion_type or "actual_event_validated",
                    "event_subject": str(raw.get("subject") or ""),
                    "event_action": str(raw.get("event_action") or ""),
                }
                factors.append(factor)
                accepted.append(factor)
            per_announcement[item["id"]] = {"llm_factors": accepted, "status": "ok"}
        return factors, per_announcement, rejected, rejected_context, failed

    def _build_f1(self, announcements, factors, as_of):
        cutoff = date.fromisoformat(str(as_of)[:10])
        valid = [factor for factor in factors if factor.get("evidence_valid")]
        scalar, category_counts = {}, {}
        for days in WINDOWS:
            window_announcements = [item for item in announcements if _within(item["date"], cutoff, days)]
            eligible_announcements = [
                item for item in window_announcements if is_analysis_eligible(item)
            ]
            window_factors = [item for item in valid if _within(item["announcement_date"], cutoff, days)]
            unique_events = {item["event_key"] for item in window_factors}
            high_events = {item["event_key"] for item in window_factors if item["severity"] >= 4}
            scalar[f"announcement_count_{days}d"] = len(window_announcements)
            scalar[f"analyzed_announcement_count_{days}d"] = len(eligible_announcements)
            scalar[f"risk_event_count_{days}d"] = len(unique_events)
            scalar[f"high_risk_event_count_{days}d"] = len(high_events)
            category_counts[f"{days}d"] = dict(
                sorted(Counter(item["taxonomy_l2"] for item in window_factors).items())
            )
        recent = [item for item in valid if _within(item["announcement_date"], cutoff, 90)]
        scalar["max_severity_90d"] = max((item["severity"] for item in recent), default=0)
        scalar["avg_severity_90d"] = round(
            sum(item["severity"] for item in recent) / len(recent), 4
        ) if recent else 0.0
        scalar["evidence_valid_ratio"] = round(
            len(valid) / len(factors), 4
        ) if factors else 1.0
        for label, count in Counter(item["taxonomy_l2"] for item in recent).items():
            scalar[f"label_{label}_count_90d"] = count

        # ---- 近一年时间衰减特征（Q2：时间权重） ----
        year_factors = [item for item in valid
                        if _within(item["announcement_date"], cutoff, ANNOUNCE_WINDOW_DAYS)]
        if year_factors:
            # 加权事件计数（指数衰减，半衰期 F1_DECAY_HALF_LIFE_DAYS）
            scalar["weighted_risk_event_count_365d"] = round(
                sum(_decay_weight(_age_days(f["announcement_date"], cutoff))
                    for f in year_factors), 4)
            scalar["weighted_high_risk_event_count_365d"] = round(
                sum(_decay_weight(_age_days(f["announcement_date"], cutoff))
                    for f in year_factors if f["severity"] >= 4), 4)
            # 加权 L2 计数
            _l2w = Counter()
            for f in year_factors:
                _l2w[f["taxonomy_l2"]] += _decay_weight(
                    _age_days(f["announcement_date"], cutoff))
            for label, wcount in _l2w.items():
                scalar[f"weighted_label_{label}_count_365d"] = round(wcount, 4)
            # 分桶离散计数（近一年 3 桶）
            for bname, (lo, hi) in {
                "0_90": (0, 90), "91_180": (91, 180), "181_365": (181, ANNOUNCE_WINDOW_DAYS)
            }.items():
                scalar[f"risk_event_count_{bname}d"] = sum(
                    1 for f in year_factors
                    if lo <= _age_days(f["announcement_date"], cutoff) < hi)
            # recency
            scalar["days_since_last_risk_event_365d"] = min(
                _age_days(f["announcement_date"], cutoff) for f in year_factors)
        else:
            for k in ("weighted_risk_event_count_365d",
                      "weighted_high_risk_event_count_365d",
                      "risk_event_count_0_90d", "risk_event_count_91_180d",
                      "risk_event_count_181_365d", "days_since_last_risk_event_365d"):
                scalar[k] = 0.0
        return {
            "feature_version": "f1_announcement_evidence_v2",
            "as_of": cutoff.isoformat(),
            "window_semantics": "累计窗口[as_of-days+1, as_of]；同一公告同一L2标签跨通道去重",
            "scalar_features": scalar,
            "category_event_counts": category_counts,
            "vector_names": list(scalar),
            "vector_values": list(scalar.values()),
            "probability": None,
            "probability_status": "F1是文本特征，不是风险概率",
        }

    @staticmethod
    def _check_cancel(ctx):
        """取消信号检查：任务被取消时抛异常，让 execute 尽快中断（最终由调度层标记 cancelled）。"""
        cancel = getattr(ctx, "cancel_event", None)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("任务已取消")

    def execute(self, company, ctx):
        _t_total = time.perf_counter()
        as_of = str(ctx.as_of or date.today().isoformat())[:10]
        identity, announcements = self._load_announcements(company, as_of)
        _t_fetch = time.perf_counter()
        self._check_cancel(ctx)
        historical_context = getattr(self.source, "last_history", {}) or {}
        query_trace = getattr(self.source, "last_query_trace", []) or []
        ctx.company = identity["secucode"]
        ctx.name = identity["company_name"]
        for item in announcements:
            apply_title_policy(item)
        eligible_announcements = [
            item for item in announcements if is_analysis_eligible(item)
        ]

        # 深读集合：PDF 下载/规则/FinBERT/LLM/全量F1上游只处理前 max_documents 份
        # （与下载源的选择顺序一致）；announcement_count_*d 等计数特征与 F6 问询
        # 特征仍用全量元数据列表，保持与离线训练口径一致。
        analyzed_announcements = eligible_announcements
        if self.max_documents is not None and len(eligible_announcements) > self.max_documents:
            analyzed_announcements = eligible_announcements[: self.max_documents]

        self._emit_progress("rule_analysis_started", document_count=len(analyzed_announcements))
        _t_rule = time.perf_counter()
        if self.use_rule:
            rule_factors, per_announcement, suppressed = self._rule_extract(
                analyzed_announcements,
                cancel_event=getattr(ctx, "cancel_event", None),
            )
        else:
            rule_factors, suppressed = [], 0
            per_announcement = {
                item["id"]: {"rule_factors": [], "suppressed_rule_hits": []}
                for item in analyzed_announcements
            }
        _t_finbert = time.perf_counter()
        self._emit_progress("rule_analysis_completed", factor_count=len(rule_factors), suppressed_count=suppressed)
        self._check_cancel(ctx)
        self._emit_progress("finbert_started", enabled=bool(self.use_finbert))
        finbert_signals, finbert_status = self._finbert_classify(analyzed_announcements)
        _t_llm = time.perf_counter()
        self._emit_progress("finbert_completed", status=finbert_status, signal_count=len(finbert_signals))
        self._check_cancel(ctx)
        signal_map = {item["announcement_id"]: item for item in finbert_signals}
        rule_ids = {item["announcement_id"] for item in rule_factors}
        gate_active = bool(
            FINBERT_GATE_ENABLED and self.finbert is not None and self.use_finbert
        )
        if self.use_llm and self.llm_configured:
            if gate_active:
                llm_candidates = [
                    item for item in analyzed_announcements
                    if item["id"] in rule_ids
                    or (signal_map.get(item["id"], {}).get("max_score") or 0) >= self.gate_threshold
                ]
            else:
                llm_candidates = analyzed_announcements
            self._emit_progress("llm_started", document_count=len(llm_candidates))
            (
                llm_factors,
                llm_per_announcement,
                rejected_llm,
                rejected_llm_context,
                failed_llm,
            ) = self._llm_extract(
                identity["company_name"], llm_candidates,
                cancel_event=getattr(ctx, "cancel_event", None),
            )
            llm_status = "partial_failed" if failed_llm else "enabled"
        else:
            llm_candidates, llm_factors, llm_per_announcement = [], [], {}
            rejected_llm, rejected_llm_context, failed_llm = 0, 0, 0
            llm_status = "disabled" if not self.use_llm else "not_configured"
        _t_f1 = time.perf_counter()
        self._emit_progress(
            "llm_completed",
            status=llm_status,
            processed_count=len(llm_candidates),
            factor_count=len(llm_factors),
        )
        self._check_cancel(ctx)

        for announcement_id, payload in llm_per_announcement.items():
            per_announcement.setdefault(announcement_id, {}).update(payload)
        llm_event_keys = {item["event_key"] for item in llm_factors}
        rule_event_keys = {item["event_key"] for item in rule_factors}
        for factor in rule_factors:
            if factor["event_key"] in llm_event_keys:
                factor["agreement_status"] = "rule_llm_agree"
        for factor in llm_factors:
            if factor["event_key"] in rule_event_keys:
                factor["agreement_status"] = "rule_llm_agree"

        factors, seen = [], set()
        for factor in rule_factors + llm_factors:
            key = factor["event_key"]
            if key in seen:
                continue
            seen.add(key)
            factors.append(factor)

        f1 = self._build_f1(announcements, factors, as_of)
        self._emit_progress("finalizing", risk_factor_count=len(factors))
        f1_vector = None
        f1_vector_backend = "not_generated: EMBEDDING_BACKEND is not bge"
        if EMBEDDING_BACKEND == "bge" and factors:
            texts = [
                f"{item['taxonomy_l2']}:{item['description']}:{item['evidence']}"
                for item in factors
            ]
            try:
                from ..skills.embedding import embed

                vectors = embed(texts, allow_fallback=False)
                f1_vector = vectors.mean(axis=0).tolist()
                f1_vector_backend = f"bge:{EMBEDDING_MODEL}:risk_factor_mean"
            except Exception as exc:
                f1_vector_backend = f"not_generated:{type(exc).__name__}:{str(exc)[:120]}"
        parsed = [
            item for item in analyzed_announcements
            if "_parsed" in item.get("text_status", "")
        ]
        attempted = [
            item for item in analyzed_announcements
            if item.get("text_status") != "not_fetched"
        ]
        title_excluded = [
            item for item in announcements if not is_analysis_eligible(item)
        ]
        evidence = [item for item in factors if item.get("evidence_valid")]
        ocr_candidate_pages = sum(item.get("ocr_candidate_pages", 0) for item in attempted)
        ocr_attempted_pages = sum(item.get("ocr_attempted_pages", 0) for item in attempted)
        ocr_succeeded_pages = sum(item.get("ocr_succeeded_pages", 0) for item in attempted)
        ocr_failed_pages = sum(item.get("ocr_failed_pages", 0) for item in attempted)
        ocr_skipped_pages = sum(item.get("ocr_skipped_pages", 0) for item in attempted)
        ocr_states = {item.get("ocr_status", "") for item in attempted}
        if not ocr_candidate_pages:
            aggregate_ocr_status = "not_needed"
        elif "not_available" in ocr_states:
            aggregate_ocr_status = "not_available"
        elif "disabled" in ocr_states:
            aggregate_ocr_status = "disabled"
        elif ocr_skipped_pages:
            aggregate_ocr_status = "partial_truncated"
        elif ocr_failed_pages and ocr_succeeded_pages:
            aggregate_ocr_status = "partial_failed"
        elif ocr_failed_pages:
            aggregate_ocr_status = "failed"
        else:
            aggregate_ocr_status = "completed"

        ctx.semantic.announcements = [
            {key: value for key, value in item.items() if key != "text"}
            for item in announcements
        ]
        ctx.semantic.finbert_signals = finbert_signals
        ctx.semantic.risk_factors = factors
        ctx.semantic.evidence_snippets = [
            {
                "announcement_id": item["announcement_id"],
                "label": item["taxonomy_l2"],
                "text": item["evidence"],
                "start": item["evidence_start"],
                "end": item["evidence_end"],
                "source_url": item["source_url"],
            }
            for item in evidence
        ]
        ctx.semantic.per_announcement = per_announcement
        retrieval_mode = identity.get("retrieval_mode", "online")
        current_data_available = retrieval_mode != "online_unavailable"
        # 可选的训练同口径实时 F1 上游。默认 auto：只有显式配置三套本地模型
        # 时启动，避免普通 UI 查询意外下载数 GB 权重；设
        # F1_ONLINE_SEMANTICS_ENABLED=true 可允许 HuggingFace 首次下载。
        ctx.semantic.f1_announcement_risk_rows = []
        semantic_mode = os.getenv("F1_ONLINE_SEMANTICS_ENABLED", "auto").lower()
        configured_models = all(
            os.getenv(name) and os.path.exists(os.getenv(name))
            for name in (
                "F1_BGE_MODEL_PATH", "F1_RERANK_MODEL_PATH", "F1_FINBERT_MODEL_PATH"
            )
        )
        fullrun_enabled = semantic_mode in {"1", "true", "yes", "on"} or (
            semantic_mode == "auto" and configured_models
        )
        fullrun_upstream_audit = {
            "status": "disabled",
            "reason": (
                "未启用训练同口径三模型上游；设置 F1_ONLINE_SEMANTICS_ENABLED=true，"
                "或配置三套 F1_*_MODEL_PATH。"
            ),
        }
        if fullrun_enabled and current_data_available:
            fullrun_documents = [
                item for item in analyzed_announcements if (item.get("text") or "").strip()
            ]
            try:
                self._emit_progress(
                    "fullrun_f1_started", document_count=len(fullrun_documents)
                )
                from ..skills.fullrun_online_semantics import FullRunOnlineSemanticPipeline

                rows, fullrun_upstream_audit = FullRunOnlineSemanticPipeline().analyze(
                    fullrun_documents, identity.get("secucode") or ctx.company
                )
                ctx.semantic.f1_announcement_risk_rows = rows
            except Exception as exc:
                fullrun_upstream_audit = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "document_count": len(fullrun_documents),
                }
                _logger.warning("[训练同口径实时 F1 上游失败] %s", exc)
            self._emit_progress(
                "fullrun_f1_completed",
                status=fullrun_upstream_audit.get("status"),
                output_rows=len(ctx.semantic.f1_announcement_risk_rows),
            )
        # 实时源不可达时，不能把“未取得公告”当作“公告数为 0”送入模型。
        ctx.semantic.f1_features = f1 if current_data_available else {}
        # 注意：f1_features 是本 Agent 的规则/LLM标量，不等价于模型训练所用的
        # announcement_semantic_000~049。严禁按公司或日期从全量历史 PCA 表查值后
        # 冒充当前在线特征；在完整同口径流水线接入前，显式保留为空并写审计原因。
        ctx.semantic.f1_model_features = {}
        ctx.semantic.f1_model_audit = {
            "status": "not_generated",
            "source": "online_fullrun_rows_pending_pca" if ctx.semantic.f1_announcement_risk_rows else "online_rule_llm_scalars_only",
            "required_pipeline": "BGE-CLS→主题召回→reranker→FinBERT联合打分→208维聚合→冻结PCA50",
            "expected_dimensions": 50,
            "produced_dimensions": 0,
            "static_full_run_lookup_used": False,
            "reason": (
                "已生成公告×主题行，等待财务 Agent 完成后由 Predictor 执行冻结 PCA50。"
                if ctx.semantic.f1_announcement_risk_rows else
                "避免将历史全量 PCA50 或规则标量伪装为当前实时模型输入。"
            ),
            "upstream": fullrun_upstream_audit,
        }
        ctx.semantic.f1_vector = f1_vector
        ctx.semantic.f1_vector_backend = f1_vector_backend
        # F6 监管问询函特征：由本 Agent 的公告列表（巨潮官方源）计算，
        # 供预测建模使用（口径与离线 F6 表一致；窗口内无问询公告 → 全 0）
        try:
            from datetime import date as _date
            from ..skills.inquiry_features import compute_f6_from_announcements
            ctx.semantic.f6_features = (
                compute_f6_from_announcements(
                    announcements, _date.fromisoformat(str(as_of)[:10])
                )
                if current_data_available
                else {}
            )
        except Exception as e:
            _logger.warning("[F6 问询特征计算失败] %s", e)
            ctx.semantic.f6_features = {}
        # 分步耗时归因：区分网络抓取/PDF下载与规则、FinBERT、LLM 各通道，
        # 否则 Reader 慢时无法定位是巨潮接口慢还是 LLM 慢。
        _t_end = time.perf_counter()
        source_timing = (
            getattr(self.source, "last_timing_ms", None)
            or getattr(getattr(self.source, "online_source", None), "last_timing_ms", None)
            or {}
        )
        timing_ms = {
            "fetch_total_ms": int((_t_fetch - _t_total) * 1000),
            "metadata_ms": source_timing.get("metadata_ms"),
            "pdf_ms": source_timing.get("pdf_ms"),
            "pdf_downloaded": source_timing.get("pdf_downloaded"),
            "pdf_total": source_timing.get("pdf_total"),
            "rule_ms": int((_t_finbert - _t_rule) * 1000),
            "finbert_ms": int((_t_llm - _t_finbert) * 1000),
            "llm_ms": int((_t_f1 - _t_llm) * 1000),
            "aggregate_ms": int((_t_end - _t_f1) * 1000),
            "total_ms": int((_t_end - _t_total) * 1000),
        }
        _logger.info(
            "[AnnouncementReader] 分步耗时: 公告检索+PDF=%.1fs (元数据=%.1fs, PDF=%.1fs, %s/%s 份) "
            "规则=%.2fs FinBERT=%.2fs LLM=%.1fs 汇总=%.2fs 总计=%.1fs",
            timing_ms["fetch_total_ms"] / 1000,
            (timing_ms["metadata_ms"] or 0) / 1000,
            (timing_ms["pdf_ms"] or 0) / 1000,
            timing_ms["pdf_downloaded"], timing_ms["pdf_total"],
            timing_ms["rule_ms"] / 1000,
            timing_ms["finbert_ms"] / 1000,
            timing_ms["llm_ms"] / 1000,
            timing_ms["aggregate_ms"] / 1000,
            timing_ms["total_ms"] / 1000,
        )
        ctx.semantic.timing_ms = timing_ms
        ctx.semantic.channel_summary = {
            "rule": {
                "status": "enabled" if self.use_rule else "disabled",
                "dictionary_version": self.rule_extractor.version,
                "factor_count": len(rule_factors),
                "suppressed_count": suppressed,
            },
            "finbert": {
                "status": finbert_status,
                "gate_active": gate_active,
                "gate_threshold": self.gate_threshold if gate_active else None,
                "warning": "未经公告标注集校准，不得把相似度称为概率",
                "error": self.finbert_error,
            },
            "llm": {
                "status": llm_status,
                "processed_count": len(llm_candidates),
                "accepted_factor_count": len(llm_factors),
                "rejected_nonverbatim_evidence": rejected_llm,
                "rejected_nonfactual_context": rejected_llm_context,
                "failed_document_count": failed_llm,
            },
        }
        ctx.semantic.historical_context = historical_context
        ctx.semantic.query_trace = query_trace
        offline_snapshot = retrieval_mode == "offline_official_snapshot"
        if offline_snapshot:
            current_source_policy = (
                f"本次命中仓库内巨潮官方公告离线快照，数据锚点为{identity.get('snapshot_as_of')}；"
            )
        elif not current_data_available:
            current_source_policy = (
                "本次巨潮实时接口连接失败，未取得当前公告；公告相关特征按缺失处理，"
                "零条公告不表示公司没有风险；"
            )
        else:
            current_source_policy = "本次联网访问巨潮并读取截止日以前的最新公告；"
        ctx.semantic.source_policy = (
            "查询先检查比赛历史库，再读取巨潮官方公告。"
            + current_source_policy
            + "当前事实与近30/60/90天F1仅来自巨潮官方公告和官方PDF正文；"
            "比赛库中的2020—2024历史旧规则命中只作候选证据，单独展示且不计入当前风险。"
            "规则或模型只生成待复核信号，不构成事实认定。"
            "制度类公告和规范性段落会保留审计记录但不计入风险。"
        )
        ctx.semantic.data_quality = {
            "source": (
                "巨潮官方公告（仓库离线快照）"
                if offline_snapshot
                else (
                    "巨潮资讯网（本次连接失败）"
                    if not current_data_available
                    else ("巨潮资讯网" if self.source is not None else "本地官方公告副本")
                )
            ),
            "retrieval_mode": retrieval_mode,
            "current_data_available": current_data_available,
            "network_error": identity.get("network_error", ""),
            "offline_snapshot_used": offline_snapshot,
            "snapshot_id": identity.get("snapshot_id", ""),
            "snapshot_as_of": identity.get("snapshot_as_of", ""),
            "snapshot_created_at": identity.get("snapshot_created_at", ""),
            "as_of": as_of,
            "lookback_days": ANNOUNCE_WINDOW_DAYS,
            "announcement_count": len(announcements),
            "analysis_eligible_count": len(eligible_announcements),
            "analyzed_document_count": len(analyzed_announcements),
            "title_excluded_count": len(title_excluded),
            "title_filter_version": FILTER_VERSION,
            "pdf_attempted_count": len(attempted),
            "pdf_parsed_count": len(parsed),
            "pdf_parsed_ratio": round(len(parsed) / len(attempted), 4) if attempted else None,
            "not_fetched_count": len(analyzed_announcements) - len(attempted),
            "not_fulltext_count": len(attempted) - len(parsed),
            "document_limit_truncated": len(attempted) < len(eligible_announcements),
            "evidence_valid_ratio": f1["scalar_features"]["evidence_valid_ratio"],
            "ocr_status": aggregate_ocr_status,
            "ocr_engine": next(
                (item.get("ocr_engine") for item in attempted if item.get("ocr_engine")), ""
            ),
            "ocr_candidate_pages": ocr_candidate_pages,
            "ocr_attempted_pages": ocr_attempted_pages,
            "ocr_succeeded_pages": ocr_succeeded_pages,
            "ocr_failed_pages": ocr_failed_pages,
            "ocr_skipped_pages": ocr_skipped_pages,
            "f1_vector_backend": f1_vector_backend,
            "f1_model_feature_status": ctx.semantic.f1_model_audit,
            "competition_history_available": historical_context.get("available", False),
            "competition_history_match_status": historical_context.get("match_status", "not_configured"),
            "competition_history_document_count": historical_context.get("document_count", 0),
        }
        ctx.semantic.stats = {
            "announcement_count": len(announcements),
            "analysis_eligible_count": len(eligible_announcements),
            "analyzed_document_count": len(analyzed_announcements),
            "title_excluded_count": len(title_excluded),
            "llm_processed_count": len(llm_candidates),
            "gated_count": len(analyzed_announcements) - len(llm_candidates) if gate_active else 0,
            "risk_factor_count": len(factors),
            "high_severity_count": sum(item["severity"] >= 4 for item in factors),
            "risk_event_count_30d": f1["scalar_features"]["risk_event_count_30d"],
            "risk_event_count_60d": f1["scalar_features"]["risk_event_count_60d"],
            "risk_event_count_90d": f1["scalar_features"]["risk_event_count_90d"],
            "window_days": ANNOUNCE_WINDOW_DAYS,
            "as_of": as_of,
        }
        self._emit_progress("analysis_completed", risk_factor_count=len(factors))
        return ctx
