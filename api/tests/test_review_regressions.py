"""代码审查 P0/P1 问题的回归测试。"""
import asyncio
import json
import time

from fastapi.testclient import TestClient

from api import main, pipeline
from api.models import ScanRequest
from backend.agents.reporter import ReporterAgent
from backend.context import Context
from backend.skills.evidence_policy import publishable_evidence


def _report(company: str, as_of: str, probability: float) -> dict:
    return {
        "company": company,
        "name": "测试公司",
        "as_of": as_of,
        "generated_at": as_of + "T12:00:00",
        "data_source": "offline_lookup",
        "scorecard": {
            "probability_30d": probability,
            "probability_60d": probability,
            "probability_90d": probability,
            "confidence": 0.6,
        },
        "semantic": {"announcement_count": 1, "risk_factors": []},
        "financial": {"skip": False, "anomaly_list": []},
    }


def test_mock_unknown_company_never_relabels_other_company():
    with TestClient(main.app) as client:
        response = client.get("/api/mock/000002.SZ")
    assert response.status_code == 404


def test_invalid_as_of_is_422():
    with TestClient(main.app) as client:
        response = client.post("/api/scan", json={"code": "300577.SZ", "as_of": "not-a-date"})
    assert response.status_code == 422


def test_as_of_selects_latest_report_not_after_requested_date(tmp_path, monkeypatch):
    old = _report("300577.SZ", "2026-08-20", 0.2)
    new = _report("300577.SZ", "2026-08-28", 0.8)
    (tmp_path / "old.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    manifest = [
        {"company": "300577.SZ", "as_of": "2026-08-20", "generated_at": old["generated_at"], "json_file": "old.json"},
        {"company": "300577.SZ", "as_of": "2026-08-28", "generated_at": new["generated_at"], "json_file": "new.json"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    pipeline._manifest_cache = None
    pipeline._manifest_cache_mtime = None
    pipeline._report_cache.clear()

    selected = pipeline._load_report("300577.SZ", "2026-08-25")
    assert selected["as_of"] == "2026-08-20"


def test_timeout_discards_late_context_mutation():
    ctx = Context(company="300577.SZ")
    ctx.name = "before"

    def mutate_late(company, isolated):
        time.sleep(0.1)
        isolated.name = "mutated_after_timeout"

    streamer = pipeline.StreamingOrchestrator(callback=lambda message: None)
    outcome, _ = streamer._run_agent(mutate_late, ctx.company, ctx, 0.01)
    time.sleep(0.15)
    assert outcome == "timeout"
    assert ctx.name == "before"


def test_rule_only_candidate_is_not_publishable():
    factors = [
        {"evidence_valid": True, "agreement_status": "rule_only", "assertion_type": "actual_event"},
        {"evidence_valid": True, "agreement_status": "rule_llm_agree", "assertion_type": "actual_event"},
    ]
    assert publishable_evidence(factors) == [factors[1]]


def test_unpublishable_runtime_falls_back_instead_of_completed(monkeypatch):
    ctx = Context(company="300577.SZ")
    ctx.meta["runtime_quality"] = {"publishable": False, "degraded_reasons": ["事实源不可用"]}

    def fake_run(*args, **kwargs):
        return ctx

    monkeypatch.setattr(pipeline.StreamingOrchestrator, "run", fake_run)
    monkeypatch.setattr(pipeline, "_persist_trace", lambda *args, **kwargs: None)
    state = pipeline.create_task("300577.SZ")
    asyncio.run(pipeline.run_scan_task(state, ScanRequest(code="300577.SZ", realtime=True)))
    assert state.status == "fallback"
    assert state.result is not None


def test_reporter_does_not_save_unpublishable_report(tmp_path, monkeypatch):
    import backend.agents.reporter as reporter_module

    monkeypatch.setattr(reporter_module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(reporter_module, "MANIFEST_PATH", tmp_path / "manifest.json")
    ctx = Context(company="300577.SZ")
    ctx.meta["runtime_quality"] = {"publishable": False, "degraded_reasons": ["事实源不可用"]}
    ReporterAgent()._save_report(ctx, {"report_id": "blocked", "generated_at": "2026-08-28"}, "blocked")
    assert not (tmp_path / "blocked.json").exists()


def test_disabled_model_channel_is_not_reported_as_completed():
    messages = []
    streamer = pipeline.StreamingOrchestrator(callback=messages.append)
    streamer._active_agents["announcement"] = (1, 7, "AnnouncementReader", "announcement", "公告研读")
    streamer._detail_callback({"agent_key": "announcement", "event": "finbert_completed", "status": "disabled"})
    assert messages[-1].status == "skipped"
    assert "已禁用" in messages[-1].message


def test_legacy_report_is_explicitly_unverified():
    response = pipeline.offline_to_response_from_report(_report("300577.SZ", "2026-08-20", 0.2))
    assert response.dataSource == "legacy_snapshot_unverified"
    assert response.modelVersion == "legacy-unversioned"
    assert response.degradedReasons


def test_announcement_review_keeps_low_risk_and_excluded_audit_visible():
    report = _report("000415.SZ", "2026-08-29", 0.2)
    report["semantic"] = {
        "announcement_count": 202,
        "risk_factors": [],
        "candidate_count": 12,
        "risk_factor_candidates": [],
        "channel_summary": {"rule": {"suppressed_count": 3}},
        "data_quality": {
            "as_of": "2026-08-29",
            "lookback_days": 365,
            "announcement_count": 202,
            "analysis_eligible_count": 181,
            "title_excluded_count": 21,
            "pdf_attempted_count": 100,
            "pdf_parsed_count": 60,
            "not_fulltext_count": 40,
            "title_filter_version": "test-v1",
        },
    }

    response = pipeline.offline_to_response_from_report(report)
    review = response.announcementReview

    assert review["reviewedCount"] == 202
    assert len(review["lowRiskSignals"]) == 3
    assert {item["category"] for item in review["excludedSignals"]} == {
        "标题过滤", "语境排除", "未发布候选", "全文不足",
    }
    assert review["excludedCount"] == 76
