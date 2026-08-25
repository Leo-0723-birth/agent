#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仓库内公告离线快照：只接受可核验的官方公告缓存。"""
from __future__ import annotations

import copy
import gzip
import json
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from ..config import ANNOUNCE_OFFLINE_SNAPSHOT_DIR


SNAPSHOT_SCHEMA = "cninfo_offline_announcement_snapshot_v1"
ANALYSIS_SCHEMA = "announcement_analysis_snapshot_v1"
_CODE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _normalize_alias(value: object) -> str:
    return re.sub(r"[\s.*·•．.()（）_\-]+", "", str(value or "")).lower()


@lru_cache(maxsize=16)
def _load_gzip_json(path_text: str, mtime_ns: int) -> dict:
    del mtime_ns
    with gzip.open(path_text, mode="rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load(path: Path) -> dict:
    return _load_gzip_json(str(path.resolve()), path.stat().st_mtime_ns)


def _matches(payload: dict, query: object) -> bool:
    raw = str(query or "").strip()
    requested_code = (_CODE.search(raw).group(1) if _CODE.search(raw) else "")
    company = payload.get("company") or {}
    code = str(company.get("code") or "")
    if requested_code:
        return requested_code == code
    aliases = {_normalize_alias(item) for item in payload.get("aliases") or []}
    aliases.update(
        _normalize_alias(company.get(key))
        for key in ("company_name", "secucode", "code")
    )
    return _normalize_alias(raw) in aliases


class OfflineAnnouncementSnapshotStore:
    """按公司、截止日与窗口读取仓库内官方公告全文快照。"""

    def __init__(self, root=ANNOUNCE_OFFLINE_SNAPSHOT_DIR):
        self.root = Path(root)

    def lookup(self, query: object, days=365, as_of=None) -> dict:
        cutoff = date.fromisoformat(str(as_of or date.today().isoformat())[:10])
        requested_start = cutoff - timedelta(days=int(days) - 1)
        if not self.root.is_dir():
            return {"status": "unavailable", "reason": "snapshot_directory_not_found"}
        matched = False
        for path in sorted(self.root.glob("*_announcements_*.json.gz"), reverse=True):
            payload = _load(path)
            if payload.get("schema_version") != SNAPSHOT_SCHEMA or not _matches(payload, query):
                continue
            matched = True
            coverage = payload.get("coverage") or {}
            snapshot_start = date.fromisoformat(str(coverage.get("query_start"))[:10])
            snapshot_end = date.fromisoformat(str(coverage.get("query_end"))[:10])
            if cutoff > snapshot_end:
                continue
            if requested_start < snapshot_start:
                continue
            announcements = [
                copy.deepcopy(item)
                for item in payload.get("announcements") or []
                if requested_start
                <= date.fromisoformat(str(item.get("date"))[:10])
                <= cutoff
            ]
            company = copy.deepcopy(payload.get("company") or {})
            company.update(
                {
                    "resolved_from": str(query),
                    "retrieval_mode": "offline_official_snapshot",
                    "snapshot_id": payload.get("snapshot_id", path.stem),
                    "snapshot_as_of": snapshot_end.isoformat(),
                    "snapshot_created_at": payload.get("created_at", ""),
                    "snapshot_path": path.name,
                }
            )
            for item in announcements:
                item["retrieval_mode"] = "offline_official_snapshot"
                item["snapshot_as_of"] = snapshot_end.isoformat()
                item["source_tier"] = "official_current_snapshot"
                item["cache_path"] = ""
                item["extraction_cache_path"] = ""
            return {
                "status": "hit",
                "company": company,
                "announcements": announcements,
                "snapshot_id": company["snapshot_id"],
                "snapshot_as_of": snapshot_end.isoformat(),
                "snapshot_created_at": payload.get("created_at", ""),
                "snapshot_path": str(path),
            }
        return {
            "status": "miss",
            "reason": "snapshot_window_not_covered" if matched else "company_not_in_snapshot",
        }


class OfflineAnalysisSnapshotStore:
    """读取与界面运行参数完全一致的预计算公告研读结果。"""

    def __init__(self, root=ANNOUNCE_OFFLINE_SNAPSHOT_DIR):
        self.root = Path(root)

    def lookup(
        self,
        query: object,
        as_of: str,
        use_ocr: bool,
        use_finbert: bool,
        use_llm: bool,
        filter_version: str,
    ) -> dict:
        if not self.root.is_dir():
            return {"status": "unavailable", "reason": "snapshot_directory_not_found"}
        expected = {
            "as_of": str(as_of)[:10],
            "use_ocr": bool(use_ocr),
            "use_finbert": bool(use_finbert),
            "use_llm": bool(use_llm),
            "filter_version": str(filter_version),
        }
        for path in sorted(self.root.glob("*_analysis_*.json.gz"), reverse=True):
            payload = _load(path)
            if payload.get("schema_version") != ANALYSIS_SCHEMA or not _matches(payload, query):
                continue
            options = payload.get("analysis_options") or {}
            if any(options.get(key) != value for key, value in expected.items()):
                continue
            return {
                "status": "hit",
                "result": copy.deepcopy(payload.get("result") or {}),
                "snapshot_id": payload.get("snapshot_id", path.stem),
                "snapshot_as_of": expected["as_of"],
                "snapshot_created_at": payload.get("created_at", ""),
                "snapshot_path": str(path),
            }
        return {"status": "miss", "reason": "no_exact_analysis_snapshot"}
