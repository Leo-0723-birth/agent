#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""比赛历史库检索：先查本地历史，再由巨潮事实源补充最新公告。

历史规则命中来自 2020—2024 年离线产物，只作为可追溯的候选信号；
本模块不会把它们并入当前 30/60/90 天风险事件，也不会称为概率。
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import requests

from ..config import COMPETITION_RULE_RISKS, COMPETITION_SEMANTIC_FEATURES
from .stock_code import StockCodeError, normalize_stock_code


_CODE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_ALIAS_PATTERNS = (
    re.compile(r"(?:证券简称|股票简称|公司简称)\s*[:：]\s*(.{2,20}?)(?=\s+(?:公告编号|证券代码|股票代码)\s*[:：]|$)"),
    re.compile(r"^([^：:（）()]{2,16})\s*[:：]\s*"),
)


def _six_digit(value: object) -> str:
    match = _CODE.search(str(value or ""))
    return match.group(1) if match else ""


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[\s*·•．.()（）]", "", text)


def _aliases_from_title(title: object) -> set[str]:
    text = str(title or "").strip()
    aliases = set()
    for pattern in _ALIAS_PATTERNS:
        match = pattern.search(text)
        if match:
            alias = match.group(1).strip(" -—_：:")
            if 2 <= len(alias) <= 16 and not any(word in alias for word in ("公告", "报告", "证券代码")):
                aliases.add(alias)
    return aliases


def _open_jsonl(path: Path):
    """打开普通或 gzip 压缩 JSONL，保持逐行读取且不落地临时副本。"""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


@lru_cache(maxsize=4)
def _build_index(path_text: str, mtime_ns: int) -> dict:
    """把 42MB JSONL 压缩为公司级索引；mtime 变化时自动失效。"""
    del mtime_ns
    companies: dict[str, dict] = {}
    alias_codes: dict[str, set[str]] = defaultdict(set)
    path = Path(path_text)
    with _open_jsonl(path) as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            raw_code = str(row.get("stock_code") or "").strip()
            code = _six_digit(raw_code)
            if not code:
                continue
            company = companies.setdefault(
                code,
                {
                    "stock_code": raw_code or code,
                    "document_count": 0,
                    "risk_document_count": 0,
                    "risk_candidate_count": 0,
                    "dates": [],
                    "aliases": set(),
                    "dictionary_versions": set(),
                    "rule_engine_versions": set(),
                    "category_counts": Counter(),
                    "announcements": [],
                    "candidates": [],
                },
            )
            company["document_count"] += 1
            published = str(row.get("publish_date") or "")[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
                company["dates"].append(published)
            aliases = _aliases_from_title(row.get("title"))
            company["aliases"].update(aliases)
            for alias in aliases:
                alias_codes[_normalize_name(alias)].add(code)
            company["dictionary_versions"].add(str(row.get("dictionary_version") or "unknown"))
            company["rule_engine_versions"].add(str(row.get("rule_engine_version") or "unknown"))
            matches = row.get("risk_matches") or []
            company["announcements"].append(
                {
                    "date": published,
                    "title": str(row.get("title") or ""),
                    "doc_id": str(row.get("doc_id") or ""),
                    "doc_type": str(row.get("doc_type") or ""),
                    "parse_status": str(row.get("source_parse_status") or ""),
                    "has_old_rule_candidate": bool(matches),
                    "old_rule_candidate_count": len(matches),
                    "source_tier": "competition_historical_derived",
                }
            )
            if matches:
                company["risk_document_count"] += 1
            company["risk_candidate_count"] += len(matches)
            for match in matches:
                label = str(match.get("risk_label") or match.get("label") or "未分类")
                category = str(match.get("category_id") or match.get("risk_category") or match.get("category") or "")
                company["category_counts"][label] += 1
                company["candidates"].append(
                    {
                        "date": published,
                        "title": str(row.get("title") or ""),
                        "doc_id": str(row.get("doc_id") or ""),
                        "doc_type": str(row.get("doc_type") or ""),
                        "risk_label": label,
                        "risk_category": category,
                        "severity": str(match.get("severity") or ""),
                        "evidence": str(match.get("evidence_text") or match.get("evidence") or match.get("text") or ""),
                        "matched_pattern": str(match.get("matched_pattern") or match.get("matched_keyword") or ""),
                        "page": match.get("page_no", match.get("page")),
                        "paragraph_id": match.get("paragraph_id"),
                        "source_artifact": str(path.resolve()),
                        "source_line": line_no,
                        "source_tier": "competition_historical_derived",
                        "verification_status": "历史旧规则候选，未按当前规则复核",
                    }
                )
    return {"companies": companies, "alias_codes": dict(alias_codes)}


class CompetitionHistoryStore:
    """对比赛历史产物做精确代码/唯一简称检索。"""

    def __init__(self, rule_path=COMPETITION_RULE_RISKS, semantic_path=COMPETITION_SEMANTIC_FEATURES):
        self.rule_path = Path(rule_path)
        self.semantic_path = Path(semantic_path)

    @property
    def available(self) -> bool:
        return self.rule_path.is_file()

    def _index(self) -> dict:
        if not self.available:
            return {"companies": {}, "alias_codes": {}}
        return _build_index(str(self.rule_path.resolve()), self.rule_path.stat().st_mtime_ns)

    def _semantic_feature(self, stock_code: str) -> dict:
        if not self.semantic_path.is_file():
            return {"available": False, "status": "file_not_found"}
        try:
            import pandas as pd

            frame = pd.read_parquet(
                self.semantic_path,
                filters=[("stock_code", "==", stock_code)],
            )
            if frame.empty:
                code = _six_digit(stock_code)
                frame = pd.read_parquet(self.semantic_path)
                frame = frame[frame["stock_code"].astype(str).str.contains(code, regex=False)]
            if frame.empty:
                return {"available": False, "status": "company_not_found"}
            frame = frame.sort_values("T_date")
            row = frame.iloc[-1]
            feature_cols = [name for name in frame.columns if str(name).startswith("semantic_")]
            return {
                "available": True,
                "status": "historical_uncalibrated_feature",
                "anchor_date": str(row.get("T_date"))[:10],
                "feature_count": len(feature_cols),
                "feature_names": feature_cols,
                "feature_values": [float(row[name]) for name in feature_cols],
                "model": "BAAI/bge-large-zh-v1.5 + train-only incremental PCA",
                "historical_window": "[T-180天, T)",
                "source_artifact": str(self.semantic_path.resolve()),
                "warning": "历史语义向量不是风险概率；PCA 分量正负不表示风险方向。",
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "read_failed",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            }

    def lookup(self, query: object, include_semantic: bool = True) -> dict:
        raw = str(query or "").strip()
        base = {
            "available": self.available,
            "query": raw,
            "match_status": "unavailable" if not self.available else "miss",
            "source_artifact": str(self.rule_path.resolve()) if self.available else str(self.rule_path),
        }
        if not self.available:
            base["message"] = "本机未配置比赛历史库；仍可继续查询巨潮最新公告。"
            return base
        index = self._index()
        code = _six_digit(raw)
        resolved_by = "stock_code" if code else ""
        if not code:
            codes = sorted(index["alias_codes"].get(_normalize_name(raw), set()))
            if len(codes) > 1:
                return {**base, "match_status": "ambiguous", "candidate_codes": codes, "message": "历史简称对应多个代码，交由巨潮解析身份后再按代码复查。"}
            if len(codes) == 1:
                code, resolved_by = codes[0], "unique_historical_alias"
        company = index["companies"].get(code) if code else None
        if not company:
            return {**base, "message": "比赛历史库未命中该输入。"}
        dates = company["dates"]
        candidates = sorted(company["candidates"], key=lambda item: item.get("date") or "", reverse=True)
        announcements = sorted(company["announcements"], key=lambda item: item.get("date") or "", reverse=True)
        stock_code = company["stock_code"]
        result = {
            **base,
            "match_status": "hit",
            "resolved_by": resolved_by,
            "stock_code": stock_code,
            "aliases": sorted(company["aliases"]),
            "document_count": company["document_count"],
            "risk_document_count": company["risk_document_count"],
            "risk_candidate_count": company["risk_candidate_count"],
            "date_start": min(dates) if dates else "",
            "date_end": max(dates) if dates else "",
            "dictionary_versions": sorted(company["dictionary_versions"]),
            "rule_engine_versions": sorted(company["rule_engine_versions"]),
            "category_counts": dict(company["category_counts"].most_common()),
            "announcements": announcements,
            "risk_candidates": candidates[:200],
            "risk_candidates_truncated": len(candidates) > 200,
            "source_tier": "competition_historical_derived",
            "warning": "历史规则命中仅为候选信号，不计入当前 30/60/90 天风险，不是风险概率。",
        }
        result["semantic_feature"] = self._semantic_feature(stock_code) if include_semantic else {"available": False, "status": "not_requested"}
        return result


class CompetitionAwareAnnouncementSource:
    """保持巨潮为事实主源，同时把历史库检索轨迹暴露给 Agent。"""

    def __init__(self, online_source, history_store=None, progress_callback=None):
        self.online_source = online_source
        self.history_store = history_store or CompetitionHistoryStore()
        self.progress_callback = progress_callback
        if hasattr(self.online_source, "progress_callback"):
            self.online_source.progress_callback = progress_callback
        self.last_history = {}
        self.last_query_trace = []

    def _emit_progress(self, event, **payload):
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **payload})

    def search(self, user_input, days=365, as_of=None):
        self._emit_progress("history_check_started", query=str(user_input))
        initial = self.history_store.lookup(user_input, include_semantic=False)
        self._emit_progress(
            "history_check_completed",
            status=initial.get("match_status", "miss"),
            document_count=initial.get("document_count", 0),
            date_start=initial.get("date_start", ""),
            date_end=initial.get("date_end", ""),
            message=initial.get("message", ""),
        )
        self.last_query_trace = [
            {
                "step": 1,
                "source": "比赛历史库",
                "status": initial.get("match_status"),
                "detail": initial.get("message") or f"命中 {initial.get('document_count', 0)} 份历史公告",
            }
        ]
        try:
            identity, announcements = self.online_source.search(
                user_input, days=days, as_of=as_of
            )
        except (requests.RequestException, TimeoutError, ConnectionError, OSError) as exc:
            # 官方站点不可达时，精确股票代码仍可继续后续财务/预测/案例流程。
            # 此处绝不把历史公告伪装成当前公告，当前公告列表保持为空。
            candidate = initial.get("stock_code") or user_input
            try:
                secucode = normalize_stock_code(candidate)
            except StockCodeError:
                raise exc
            code, suffix = secucode.split(".", 1)
            exchange = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}[suffix]
            aliases = initial.get("aliases") or []
            identity = {
                "code": code,
                "secucode": secucode,
                "company_name": aliases[0] if aliases else secucode,
                "exchange": exchange,
                "org_id": "",
                "resolved_from": str(user_input),
                "source_url": "https://www.cninfo.com.cn/",
                "retrieval_mode": "online_unavailable",
                "current_data_available": False,
                "network_error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }
            announcements = []
            self.last_history = self.history_store.lookup(
                secucode, include_semantic=True
            )
            self.last_query_trace.extend(
                [
                    {
                        "step": 2,
                        "source": "巨潮资讯网",
                        "status": "unavailable",
                        "detail": "官方实时接口连接超时；本次未取得当前公告。",
                    },
                    {
                        "step": 3,
                        "source": "降级运行",
                        "status": (
                            "history_only"
                            if self.last_history.get("match_status") == "hit"
                            else "no_current_announcement_data"
                        ),
                        "detail": "后续财务、模型与案例环节继续；公告零条不解释为无风险。",
                    },
                ]
            )
            self._emit_progress(
                "online_source_unavailable",
                secucode=secucode,
                reason=identity["network_error"],
            )
            self._emit_progress(
                "source_merge_completed",
                history_status=self.last_history.get("match_status", "miss"),
                historical_document_count=self.last_history.get("document_count", 0),
                current_announcement_count=0,
                current_data_available=False,
            )
            return identity, announcements
        offline_snapshot = identity.get("retrieval_mode") == "offline_official_snapshot"
        self.last_query_trace.append(
            {
                "step": 2,
                "source": "巨潮官方公告离线快照" if offline_snapshot else "巨潮资讯网",
                "status": "success",
                "detail": (
                    f"命中锚点 {identity.get('snapshot_as_of')} 的仓库快照，读取 {len(announcements)} 份官方公告"
                    if offline_snapshot
                    else f"解析为 {identity.get('company_name')}（{identity.get('secucode')}），取得 {len(announcements)} 份近一年公告元数据"
                ),
            }
        )
        identity_code = _six_digit(identity.get("secucode"))
        initial_code = _six_digit(initial.get("stock_code"))
        if initial.get("match_status") != "hit" or identity_code != initial_code:
            self._emit_progress("history_identity_recheck_started", secucode=identity.get("secucode", ""))
            self.last_history = self.history_store.lookup(identity.get("secucode"), include_semantic=True)
        else:
            self.last_history = self.history_store.lookup(initial.get("stock_code"), include_semantic=True)
        self.last_query_trace.append(
            {
                "step": 3,
                "source": "分层合并",
                "status": "history_and_current" if self.last_history.get("match_status") == "hit" else "current_only",
                "detail": "历史候选单独展示；当前 F1 与 30/60/90 天统计只使用巨潮近一年公告。",
            }
        )
        self._emit_progress(
            "source_merge_completed",
            history_status=self.last_history.get("match_status", "miss"),
            historical_document_count=self.last_history.get("document_count", 0),
            current_announcement_count=len(announcements),
        )
        return identity, announcements
