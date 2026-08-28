from ui.components import evidence_records


def _result_with_factors(*factors: dict) -> dict:
    return {"semantic": {"risk_factors": list(factors)}, "financial": {"anomaly_list": []}}


def test_evidence_records_excludes_rule_only_candidate() -> None:
    result = _result_with_factors(
        {
            "risk_id": "rule-1",
            "severity": 5,
            "taxonomy_l2": "A06",
            "evidence": "报告期内公司股票被实施退市风险警示或其他风险警示",
            "evidence_valid": True,
            "method": "deterministic_rule_with_negation",
            "agreement_status": "rule_only",
        }
    )

    assert evidence_records(result) == []


def test_evidence_records_excludes_normative_reporting_duty() -> None:
    result = _result_with_factors(
        {
            "risk_id": "llm-1",
            "severity": 5,
            "taxonomy_l2": "F06",
            "evidence": "发现内部控制存在重大缺陷或者重大风险，应当及时向董事会或者审计委员会报告",
            "evidence_valid": True,
            "method": "llm_evidence_validated",
            "agreement_status": "llm_only",
            "assertion_type": "actual_event",
        }
    )

    assert evidence_records(result) == []


def test_evidence_records_keeps_validated_actual_event() -> None:
    result = _result_with_factors(
        {
            "risk_id": "llm-2",
            "severity": 5,
            "taxonomy_l2": "G07",
            "description": "公司收到行政处罚决定书",
            "evidence": "公司于2026年5月8日收到中国证监会出具的《行政处罚决定书》。",
            "evidence_valid": True,
            "method": "llm_evidence_validated",
            "agreement_status": "rule_llm_agree",
            "assertion_type": "actual_event",
            "source_url": "https://example.test/announcement",
        }
    )

    records = evidence_records(result)

    assert len(records) == 1
    assert records[0]["verification_label"] == "规则与 LLM 交叉一致"
    assert records[0]["quote"].startswith("公司于2026年5月8日收到")
