from ui import session


def test_publish_analysis_replaces_stale_page_results(monkeypatch):
    state = {key: {"company": "OLD"} for key in session.PAGE_STATE_KEYS}
    monkeypatch.setattr(session.st, "session_state", state)

    result = {"company": "000004.SZ", "as_of": "2026-08-23"}
    session.publish_analysis(result, source="主控 Agent 实时分析")

    assert state["dashboard_result"] is result
    assert state["active_analysis"] is result
    assert all(key not in state for key in session.PAGE_STATE_KEYS)
    assert session.hydrate_page_state("financial_analysis") is result
    assert state["financial_analysis"] is result


def test_active_defaults_follow_shared_company(monkeypatch):
    state = {
        "active_analysis": {"company": "600000.SH", "as_of": "2026-06-30"},
        "active_analysis_source": "主控 Agent 实时分析",
    }
    monkeypatch.setattr(session.st, "session_state", state)

    assert session.active_company() == "600000.SH"
    assert session.active_as_of().isoformat() == "2026-06-30"
    assert "600000.SH" in session.shared_context_caption()
