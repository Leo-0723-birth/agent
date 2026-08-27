#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
import json

from backend.skills.offline_announcement_snapshot import (
    ANALYSIS_SCHEMA,
    SNAPSHOT_SCHEMA,
    OfflineAnalysisSnapshotStore,
    OfflineAnnouncementSnapshotStore,
)
from backend.skills.announcement_search import CninfoAnnouncementSource


def _write(path, payload):
    with gzip.open(path, mode="wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def test_offline_announcement_snapshot_matches_code_alias_and_window(tmp_path):
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_id": "000004_SZ_2026-08-24",
        "created_at": "2026-08-24T10:00:00+08:00",
        "company": {"code": "000004", "secucode": "000004.SZ", "company_name": "国华退"},
        "aliases": ["000004SZ", "国华退", "*ST国华"],
        "coverage": {"query_start": "2025-08-25", "query_end": "2026-08-24"},
        "announcements": [
            {"id": "a1", "date": "2026-08-20", "title": "公告一", "text": "原文"},
            {"id": "a2", "date": "2026-06-01", "title": "公告二", "text": "原文"},
        ],
    }
    _write(tmp_path / "000004_SZ_announcements_2026-08-24.json.gz", payload)
    store = OfflineAnnouncementSnapshotStore(tmp_path)

    result = store.lookup("000004SZ", days=90, as_of="2026-08-24")
    by_name = store.lookup("*ST国华", days=90, as_of="2026-08-24")

    assert result["status"] == "hit"
    assert by_name["status"] == "hit"
    assert [item["id"] for item in result["announcements"]] == ["a1", "a2"]
    assert result["announcements"][0]["source_tier"] == "official_current_snapshot"


def test_offline_announcement_snapshot_rejects_future_cutoff(tmp_path):
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "company": {"code": "000004", "secucode": "000004.SZ"},
        "aliases": ["000004"],
        "coverage": {"query_start": "2025-08-25", "query_end": "2026-08-24"},
        "announcements": [],
    }
    _write(tmp_path / "000004_SZ_announcements_2026-08-24.json.gz", payload)

    result = OfflineAnnouncementSnapshotStore(tmp_path).lookup(
        "000004", days=365, as_of="2026-08-25"
    )

    assert result["status"] == "miss"
    assert result["reason"] == "snapshot_window_not_covered"


def test_cninfo_source_returns_snapshot_without_network_request(tmp_path):
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_id": "000004_SZ_2026-08-24",
        "created_at": "2026-08-24T10:00:00+08:00",
        "company": {
            "code": "000004",
            "secucode": "000004.SZ",
            "company_name": "国华退",
            "source_url": "https://www.cninfo.com.cn/new/information/topSearch/query",
        },
        "aliases": ["000004SZ", "国华退"],
        "coverage": {"query_start": "2025-08-25", "query_end": "2026-08-24"},
        "announcements": [
            {
                "id": "a1",
                "date": "2026-08-20",
                "title": "重大诉讼公告",
                "text": "公司发生重大诉讼。",
                "source_url": "https://www.cninfo.com.cn/a1",
                "pdf_url": "https://static.cninfo.com.cn/a1.pdf",
            }
        ],
    }
    _write(tmp_path / "000004_SZ_announcements_2026-08-24.json.gz", payload)

    class NoNetworkSession:
        def request(self, *args, **kwargs):
            raise AssertionError("offline hit must not call network")

    source = CninfoAnnouncementSource(
        session=NoNetworkSession(),
        offline_snapshot_dir=tmp_path,
        prefer_offline=True,
        ocr_enabled=True,
    )
    company, announcements = source.search("000004SZ", as_of="2026-08-24")

    assert company["retrieval_mode"] == "offline_official_snapshot"
    assert announcements[0]["text"] == "公司发生重大诉讼。"


def test_cninfo_source_keeps_offline_mode_when_ocr_is_disabled(tmp_path):
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "company": {"code": "000004", "secucode": "000004.SZ", "company_name": "国华退"},
        "aliases": ["000004SZ"],
        "coverage": {"query_start": "2025-08-25", "query_end": "2026-08-24"},
        "announcements": [{"id": "a1", "date": "2026-08-20", "title": "公告一", "text": "原文"}],
    }
    _write(tmp_path / "000004_SZ_announcements_2026-08-24.json.gz", payload)

    class NoNetworkSession:
        def request(self, *args, **kwargs):
            raise AssertionError("OCR disabled must not disable offline snapshot")

    source = CninfoAnnouncementSource(
        session=NoNetworkSession(),
        offline_snapshot_dir=tmp_path,
        prefer_offline=True,
        ocr_enabled=False,
    )
    company, announcements = source.search("000004SZ", as_of="2026-08-24")

    assert company["retrieval_mode"] == "offline_official_snapshot"
    assert announcements[0]["id"] == "a1"


def test_analysis_snapshot_requires_exact_runtime_options(tmp_path):
    payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "snapshot_id": "analysis",
        "company": {"code": "000004", "secucode": "000004.SZ", "company_name": "国华退"},
        "aliases": ["000004SZ", "国华退"],
        "analysis_options": {
            "as_of": "2026-08-24",
            "use_ocr": True,
            "use_finbert": False,
            "use_llm": False,
            "filter_version": "v1",
        },
        "result": {"company": "000004.SZ", "semantic": {"stats": {"announcement_count": 2}}},
    }
    _write(tmp_path / "000004_SZ_analysis_2026-08-24.json.gz", payload)
    store = OfflineAnalysisSnapshotStore(tmp_path)

    hit = store.lookup("国华退", "2026-08-24", True, False, False, "v1")
    miss = store.lookup("000004", "2026-08-24", True, True, False, "v1")

    assert hit["status"] == "hit"
    assert hit["result"]["company"] == "000004.SZ"
    assert miss["status"] == "miss"
