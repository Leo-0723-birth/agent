#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
import json

from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.context import Context
from backend.skills.competition_history import (
    CompetitionAwareAnnouncementSource,
    CompetitionHistoryStore,
)


def _write_history(path):
    rows = [
        {
            "doc_id": "old-1",
            "stock_code": "000004.SZ",
            "doc_type": "risk_warning",
            "title": "证券代码：000004 证券简称：ST 国华 公告编号：2024-033",
            "publish_date": "2024-10-17",
            "risk_matches": [
                {
                    "category_id": "operation_market",
                    "risk_label": "abnormal_stock_trading",
                    "severity": "medium",
                    "matched_pattern": "股票交易异常波动",
                    "evidence_text": "一、股票交易异常波动的情况",
                    "paragraph_id": "para_1",
                    "page_no": 1,
                }
            ],
            "dictionary_version": "1.0.0",
            "rule_engine_version": "1.0.0",
        },
        {
            "doc_id": "old-2",
            "stock_code": "000004.SZ",
            "doc_type": "annual_report",
            "title": "证券代码：000004 证券简称：国华网安 公告编号：2023-010",
            "publish_date": "2023-04-01",
            "risk_matches": [],
            "dictionary_version": "1.0.0",
            "rule_engine_version": "1.0.0",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_history_store_supports_code_and_unique_alias(tmp_path):
    risk_path = tmp_path / "risks.jsonl"
    _write_history(risk_path)
    store = CompetitionHistoryStore(risk_path, tmp_path / "missing.parquet")

    by_code = store.lookup("000004", include_semantic=False)
    by_name = store.lookup("ST国华", include_semantic=False)

    assert by_code["match_status"] == "hit"
    assert by_name["match_status"] == "hit"
    assert by_name["resolved_by"] == "unique_historical_alias"
    assert by_code["document_count"] == 2
    assert len(by_code["announcements"]) == 2
    assert by_code["risk_candidate_count"] == 1
    assert by_code["risk_candidates"][0]["evidence"] == "一、股票交易异常波动的情况"
    assert by_code["risk_candidates"][0]["source_tier"] == "competition_historical_derived"


def test_history_store_supports_repository_gzip_package(tmp_path):
    plain_path = tmp_path / "risks.jsonl"
    gzip_path = tmp_path / "risks.jsonl.gz"
    _write_history(plain_path)
    with plain_path.open("rb") as source, gzip.open(gzip_path, "wb") as target:
        target.write(source.read())

    store = CompetitionHistoryStore(gzip_path, tmp_path / "missing.parquet")
    result = store.lookup("000004", include_semantic=False)

    assert result["match_status"] == "hit"
    assert result["document_count"] == 2
    assert result["risk_candidates"][0]["source_artifact"].endswith("risks.jsonl.gz")


class _OnlineSource:
    def search(self, user_input, days=365, as_of=None):
        return (
            {
                "code": "000004",
                "secucode": "000004.SZ",
                "company_name": "国华网安",
                "exchange": "SZSE",
                "org_id": "fixture",
                "source_url": "https://www.cninfo.com.cn/fixture",
            },
            [
                {
                    "id": "current-1",
                    "announcement_id": "current-1",
                    "title": "关于收到立案告知书的公告",
                    "date": "2026-08-17",
                    "text": "公司收到中国证监会立案告知书，中国证监会决定对公司立案调查。",
                    "text_status": "fixture_parsed",
                    "source_url": "https://www.cninfo.com.cn/current-1",
                    "pdf_url": "https://static.cninfo.com.cn/current-1.pdf",
                }
            ],
        )


def test_history_is_visible_but_not_merged_into_current_f1(tmp_path):
    risk_path = tmp_path / "risks.jsonl"
    _write_history(risk_path)
    store = CompetitionHistoryStore(risk_path, tmp_path / "missing.parquet")
    progress_events = []
    source = CompetitionAwareAnnouncementSource(
        _OnlineSource(), store, progress_callback=progress_events.append
    )
    agent = AnnouncementReaderAgent(
        source=source,
        use_finbert=False,
        use_llm=False,
        progress_callback=progress_events.append,
    )

    context = agent.execute("ST国华", Context(as_of="2026-08-20"))

    assert context.semantic.historical_context["document_count"] == 2
    assert context.semantic.historical_context["risk_candidate_count"] == 1
    assert len(context.semantic.query_trace) == 3
    assert context.semantic.f1_features["scalar_features"]["risk_event_count_30d"] == 1
    assert {item["announcement_id"] for item in context.semantic.risk_factors} == {"current-1"}
    assert all(item.get("source_tier") != "competition_historical_derived" for item in context.semantic.risk_factors)
    event_names = [item["event"] for item in progress_events]
    assert event_names[0] == "history_check_started"
    assert "history_check_completed" in event_names
    assert "source_merge_completed" in event_names
    assert event_names[-1] == "analysis_completed"


def test_missing_history_store_fails_open_for_online_query(tmp_path):
    store = CompetitionHistoryStore(tmp_path / "missing.jsonl", tmp_path / "missing.parquet")
    source = CompetitionAwareAnnouncementSource(_OnlineSource(), store)

    _, announcements = source.search("000004", as_of="2026-08-20")

    assert len(announcements) == 1
    assert source.last_history["match_status"] == "unavailable"
    assert source.last_query_trace[-1]["status"] == "current_only"
