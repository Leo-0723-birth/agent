from backend.skills.announcement_context_filter import classify_announcement_title
from backend.skills.announcement_search import CninfoAnnouncementSource
from backend.skills.rule_risk_extract import RuleRiskExtractor


def test_governance_and_candidate_titles_are_excluded():
    excluded = [
        "独立董事候选人声明与承诺",
        "公司章程（2026年修订）",
        "关于修订《公司章程》的公告",
        "董事会议事规则",
        "信息披露管理制度",
    ]

    assert all(
        classify_announcement_title(title)["decision"] == "exclude"
        for title in excluded
    )


def test_real_risk_title_overrides_governance_words():
    result = classify_announcement_title(
        "关于公司董事收到中国证监会立案告知书的公告"
    )

    assert result["decision"] == "analyze"
    assert result["reason"] == "risk_title_override"


def test_title_exclusion_does_not_consume_pdf_download_limit():
    source = CninfoAnnouncementSource.__new__(CninfoAnnouncementSource)
    source.max_documents = 1
    source.resolve_company = lambda value: {"code": "000001"}
    source._list_metadata = lambda *args: [
        {
            "id": "policy",
            "title": "独立董事候选人声明",
            "text_status": "not_fetched",
            "ocr_status": "not_fetched",
        },
        {
            "id": "risk",
            "title": "关于收到立案告知书的公告",
            "text_status": "not_fetched",
            "ocr_status": "not_fetched",
        },
    ]
    processed = []

    def fake_process(item):
        processed.append(item["id"])
        item["text_status"] = "fixture_parsed"

    source._process_pdf = fake_process

    _, announcements = source.search("000001", as_of="2026-08-22")

    assert processed == ["risk"]
    assert announcements[0]["analysis_status"] == "excluded_by_title"
    assert announcements[0]["text_status"] == "skipped_title_policy"


def test_unlimited_pdf_mode_processes_all_eligible_announcements():
    source = CninfoAnnouncementSource.__new__(CninfoAnnouncementSource)
    source.max_documents = None
    source.resolve_company = lambda value: {"code": "000001"}
    source._list_metadata = lambda *args: [
        {"id": "risk-1", "title": "关于收到立案告知书的公告"},
        {"id": "risk-2", "title": "重大诉讼公告"},
    ]
    processed = []
    source._process_pdf = lambda item: processed.append(item["id"])

    source.search("000001", as_of="2026-08-23")

    assert processed == ["risk-1", "risk-2"]


def test_governance_eligibility_clause_does_not_become_investigation_risk():
    text = (
        "独立董事候选人不得存在下列情形：（二）因涉嫌证券期货违法犯罪，"
        "被中国证监会立案调查或者被司法机关立案侦查，尚未有明确结论意见。"
    )

    hits = RuleRiskExtractor().extract(text)
    hit = next(item for item in hits if item["label"] == "G07")

    assert hit["excluded"] is True
    assert hit["suppression_reason"] == "governance_eligibility_clause"


def test_actual_investigation_with_subject_and_action_is_retained():
    text = (
        "公司于2026年8月20日收到中国证监会《立案告知书》，"
        "因涉嫌信息披露违法违规，中国证监会决定对公司立案调查。"
    )

    hits = RuleRiskExtractor().extract(text)
    hit = next(item for item in hits if item["label"] == "G07")

    assert hit["excluded"] is False
    assert hit["suppression_reason"] == ""


def test_accounting_policy_and_table_are_not_impairment_events():
    extractor = RuleRiskExtractor()
    policy_hits = extractor.extract(
        "固定资产减值准备按单项资产计提，存在减值迹象时应当计提减值准备。"
    )
    table_hits = extractor.extract("信用减值损失（损失以“-”号填列）")

    assert policy_hits and all(item["excluded"] for item in policy_hits)
    assert table_hits and all(item["excluded"] for item in table_hits)


def test_actual_impairment_event_is_retained():
    hits = RuleRiskExtractor().extract("公司本期计提固定资产减值准备500万元。")
    hit = next(item for item in hits if item["label"] == "C04")

    assert hit["excluded"] is False
