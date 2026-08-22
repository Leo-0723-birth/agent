#!/usr/bin/env python
# -*- coding: utf-8 -*-
from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.context import Context
from backend.skills.announcement_search import _extract_date, _six_digit_code


class FakeSource:
    def __init__(self, announcements=None):
        self.announcements = announcements

    def search(self, company, days, as_of):
        identity = {
            "code": "000001",
            "secucode": "000001.SZ",
            "company_name": "测试公司",
            "exchange": "SZSE",
            "org_id": "fixture",
            "source_url": "https://www.cninfo.com.cn/fixture",
        }
        announcements = self.announcements or [
            {
                "id": "a1",
                "announcement_id": "a1",
                "secucode": "000001.SZ",
                "company_name": "测试公司",
                "title": "业绩预告",
                "date": "2026-08-15",
                "published_at": "2026-08-15",
                "type": "公告",
                "source_name": "巨潮资讯网",
                "source_tier": "official_current",
                "source_url": "https://www.cninfo.com.cn/a1",
                "pdf_url": "https://static.cninfo.com.cn/a1.pdf",
                "official": True,
                "text": "公司预计亏损1.2亿元，主要原因正在进一步核查。",
                "text_status": "fixture_parsed",
                "char_count": 25,
                "content_sha256": "sha-a1",
                "cache_path": "fixture/a1.pdf",
            },
            {
                "id": "a2",
                "announcement_id": "a2",
                "secucode": "000001.SZ",
                "company_name": "测试公司",
                "title": "核查说明",
                "date": "2026-07-10",
                "published_at": "2026-07-10",
                "type": "公告",
                "source_name": "巨潮资讯网",
                "source_tier": "official_current",
                "source_url": "https://www.cninfo.com.cn/a2",
                "pdf_url": "https://static.cninfo.com.cn/a2.pdf",
                "official": True,
                "text": "经核查，公司不存在商誉减值风险。",
                "text_status": "fixture_parsed",
                "char_count": 18,
                "content_sha256": "sha-a2",
                "cache_path": "fixture/a2.pdf",
            },
            {
                "id": "a3",
                "announcement_id": "a3",
                "secucode": "000001.SZ",
                "company_name": "测试公司",
                "title": "旧公告",
                "date": "2026-05-01",
                "published_at": "2026-05-01",
                "type": "公告",
                "source_name": "巨潮资讯网",
                "source_tier": "official_current",
                "source_url": "https://www.cninfo.com.cn/a3",
                "pdf_url": "https://static.cninfo.com.cn/a3.pdf",
                "official": True,
                "text": "公司计提坏账准备。",
                "text_status": "fixture_parsed",
                "char_count": 10,
                "content_sha256": "sha-a3",
                "cache_path": "fixture/a3.pdf",
            },
        ]
        return identity, announcements


class EmptyRuleExtractor:
    version = "fixture"

    def extract(self, text):
        return []


def test_company_code_and_date_parsing():
    assert _six_digit_code("平安银行 000001.SZ") == "000001"
    assert _extract_date("公告编号：2024-029\n二〇二四年三月二十六日") == "2024-03-26"


def test_rule_channel_builds_cutoff_safe_f1_and_suppresses_negation():
    agent = AnnouncementReaderAgent(
        source=FakeSource(), use_finbert=False, use_llm=False
    )
    ctx = agent.execute("000001", Context(as_of="2026-08-20"))
    scalar = ctx.semantic.f1_features["scalar_features"]
    assert ctx.company == "000001.SZ"
    assert scalar["risk_event_count_30d"] == 1
    assert scalar["risk_event_count_60d"] == 1
    assert scalar["risk_event_count_90d"] == 1
    assert ctx.semantic.channel_summary["rule"]["suppressed_count"] >= 1
    assert all(item["evidence_valid"] for item in ctx.semantic.risk_factors)
    assert all(item["source_url"].startswith("https://www.cninfo.com.cn/") for item in ctx.semantic.risk_factors)
    assert ctx.semantic.data_quality["pdf_parsed_ratio"] == 1.0


def test_llm_nonverbatim_evidence_is_rejected():
    def fake_chat_json(*args, **kwargs):
        return {
            "risk_factors": [
                {
                    "taxonomy_l1": "A",
                    "taxonomy_l2": "A03",
                    "description": "有原文",
                    "evidence": "公司预计亏损1.2亿元",
                    "severity": 5,
                },
                {
                    "taxonomy_l1": "G",
                    "taxonomy_l2": "G07",
                    "description": "虚构处罚",
                    "evidence": "公司受到监管处罚",
                    "severity": 5,
                },
            ]
        }

    agent = AnnouncementReaderAgent(
        source=FakeSource(),
        rule_extractor=EmptyRuleExtractor(),
        use_finbert=False,
        use_llm=True,
        llm_callable=fake_chat_json,
    )
    ctx = agent.execute("000001", Context(as_of="2026-08-20"))
    assert len(ctx.semantic.risk_factors) == 1
    assert ctx.semantic.risk_factors[0]["evidence"] == "公司预计亏损1.2亿元"
    assert ctx.semantic.channel_summary["llm"]["rejected_nonverbatim_evidence"] == 5


def test_f1_fields_survive_context_serialization():
    agent = AnnouncementReaderAgent(
        source=FakeSource(), use_finbert=False, use_llm=False
    )
    payload = agent.execute("000001", Context(as_of="2026-08-20")).to_dict()
    assert payload["semantic"]["f1_features"]["feature_version"] == "f1_announcement_evidence_v2"
    assert payload["semantic"]["source_policy"]
    assert payload["semantic"]["data_quality"]["source"] == "巨潮资讯网"


def test_data_quality_counts_only_attempted_documents_as_missing_fulltext():
    def announcement(identifier, published, title, text):
        return {
            "id": identifier,
            "announcement_id": identifier,
            "title": title,
            "date": published,
            "published_at": published,
            "type": "公告",
            "source_url": f"https://www.cninfo.com.cn/{identifier}",
            "pdf_url": f"https://static.cninfo.com.cn/{identifier}.pdf",
            "text": text,
            "text_status": "downloaded_native_parsed" if text else "not_fetched",
            "char_count": len(text),
            "content_sha256": f"sha-{identifier}",
        }

    announcements = [
        announcement("a1", "2026-08-10", "重大诉讼", "公司发生重大诉讼"),
        announcement("a2", "2026-08-09", "未抓取公告", ""),
    ]
    announcements[0]["text_status"] = "downloaded_native_parsed"
    announcements[0].update(
        {
            "ocr_status": "not_needed",
            "ocr_candidate_pages": 0,
            "ocr_attempted_pages": 0,
            "ocr_succeeded_pages": 0,
            "ocr_failed_pages": 0,
            "ocr_skipped_pages": 0,
        }
    )
    announcements[1]["text_status"] = "not_fetched"
    source = FakeSource(announcements)
    agent = AnnouncementReaderAgent(
        source=source,
        rule_extractor=EmptyRuleExtractor(),
        use_finbert=False,
        use_llm=False,
    )

    context = agent.execute("000001", Context(as_of="2026-08-20"))

    assert context.semantic.data_quality["not_fetched_count"] == 1
    assert context.semantic.data_quality["not_fulltext_count"] == 0
    assert context.semantic.data_quality["ocr_status"] == "not_needed"


def test_rule_channel_can_be_disabled_by_orchestrator():
    agent = AnnouncementReaderAgent(
        source=FakeSource(),
        use_rule=False,
        use_finbert=False,
        use_llm=False,
    )

    context = agent.execute("000001", Context(as_of="2026-08-20"))

    assert context.semantic.risk_factors == []
    assert context.semantic.channel_summary["rule"]["status"] == "disabled"
    assert all(
        item["rule_factors"] == []
        for item in context.semantic.per_announcement.values()
    )


def test_title_policy_excludes_governance_document_from_all_channels():
    announcements = [
        {
            "id": "policy",
            "announcement_id": "policy",
            "title": "独立董事候选人声明",
            "date": "2026-08-18",
            "text": "候选人因涉嫌证券期货违法犯罪，被中国证监会立案调查的不得任职。",
            "text_status": "fixture_parsed",
            "source_url": "https://www.cninfo.com.cn/policy",
            "pdf_url": "https://static.cninfo.com.cn/policy.pdf",
        },
        {
            "id": "actual",
            "announcement_id": "actual",
            "title": "关于收到立案告知书的公告",
            "date": "2026-08-17",
            "text": "公司收到中国证监会立案告知书，中国证监会决定对公司立案调查。",
            "text_status": "fixture_parsed",
            "source_url": "https://www.cninfo.com.cn/actual",
            "pdf_url": "https://static.cninfo.com.cn/actual.pdf",
        },
    ]
    agent = AnnouncementReaderAgent(
        source=FakeSource(announcements), use_finbert=False, use_llm=False
    )

    context = agent.execute("000001", Context(as_of="2026-08-20"))

    assert context.semantic.data_quality["title_excluded_count"] == 1
    assert context.semantic.data_quality["analysis_eligible_count"] == 1
    assert {item["announcement_id"] for item in context.semantic.risk_factors} == {
        "actual"
    }


def test_llm_normative_evidence_is_rejected_even_when_verbatim():
    evidence = "因涉嫌证券期货违法犯罪，被中国证监会立案调查的不得担任董事"

    def fake_chat_json(*args, **kwargs):
        return {
            "risk_factors": [
                {
                    "taxonomy_l1": "G",
                    "taxonomy_l2": "G07",
                    "description": "错误地把任职条款当作现实立案",
                    "evidence": evidence,
                    "severity": 5,
                    "assertion_type": "actual_event",
                    "subject": "候选人",
                    "event_action": "被立案调查",
                }
            ]
        }

    announcements = [
        {
            "id": "board",
            "announcement_id": "board",
            "title": "董事会决议公告",
            "date": "2026-08-18",
            "text": evidence,
            "text_status": "fixture_parsed",
            "source_url": "https://www.cninfo.com.cn/board",
            "pdf_url": "https://static.cninfo.com.cn/board.pdf",
        }
    ]
    agent = AnnouncementReaderAgent(
        source=FakeSource(announcements),
        rule_extractor=EmptyRuleExtractor(),
        use_finbert=False,
        use_llm=True,
        llm_callable=fake_chat_json,
    )

    context = agent.execute("000001", Context(as_of="2026-08-20"))

    assert context.semantic.risk_factors == []
    assert context.semantic.channel_summary["llm"]["rejected_nonfactual_context"] == 1
