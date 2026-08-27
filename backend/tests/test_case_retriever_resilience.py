import numpy as np

from backend.agents import case_retriever as module
from backend.agents.case_retriever import CaseRetrieverAgent
from backend.context import Context


def test_embedding_failure_falls_back_to_label_retrieval(monkeypatch):
    entries = [
        {
            "case_id": "case-1",
            "company": "历史公司",
            "inquiry_type": "年报问询",
            "publish_date": "2025-01-01",
            "focus_points": ["盈利能力"],
            "taxonomy_labels": ["A03"],
            "letter_excerpt": "历史问询原文摘要",
        }
    ]
    agent = CaseRetrieverAgent(top_k=1)
    monkeypatch.setattr(
        agent,
        "_load_db",
        lambda: (entries, np.zeros((1, 1024), dtype=np.float32)),
    )
    monkeypatch.setattr(
        module,
        "embed_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("WinError 10060")),
    )
    ctx = Context(company="000001.SZ", as_of="2026-08-26")
    ctx.semantic.risk_factors = [
        {"category": "A03", "description": "利润显著波动"}
    ]

    result = agent.execute("000001.SZ", ctx)

    assert [item["case_id"] for item in result.cases] == ["case-1"]
    assert result.cases[0]["label_rank"] == 1
    assert result.cases[0]["semantic_rank"] is None
    assert any(
        item.get("agent") == "CaseRetriever.Semantic"
        and item.get("status") == "skipped"
        for item in result.trace_log
    )
