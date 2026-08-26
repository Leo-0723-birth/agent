from datetime import date

import pandas as pd
import pytest

from backend.agents.predictor import PredictorAgent
from backend.context import Context
from backend.skills.announcement_search import _market
from backend.skills.feature_loader import _safe_normalize_company_code
from backend.skills.financial_data_fetch import DataFetcher, _market_of, _to_6digit
from backend.skills.stock_code import (
    StockCodeError,
    normalize_company_input,
    normalize_stock_code,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("000004.SZ", "000004.SZ"),
        ("000004", "000004.SZ"),
        ("000004sz", "000004.SZ"),
        ("SZ000004", "000004.SZ"),
        ("000004-SZ", "000004.SZ"),
        ("０００００４．ｓｚ", "000004.SZ"),
        ("4", "000004.SZ"),
        (4.0, "000004.SZ"),
        ("600000", "600000.SH"),
        ("200000", "200000.SZ"),
        ("920000", "920000.BJ"),
        ("BJ920000", "920000.BJ"),
    ],
)
def test_normalize_stock_code_accepts_common_forms(raw, expected):
    assert normalize_stock_code(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "123456", "600000.SZ", "920000.SH", "000004.SZZ", "股票000004"],
)
def test_normalize_stock_code_rejects_empty_unknown_or_conflicting_input(raw):
    with pytest.raises(StockCodeError):
        normalize_stock_code(raw)


def test_company_names_are_only_allowed_explicitly():
    assert normalize_company_input("国华退", allow_name=True) == "国华退"
    assert normalize_company_input("360安全", allow_name=True) == "360安全"
    with pytest.raises(StockCodeError):
        normalize_company_input("国华退")


def test_all_exchange_helpers_share_the_same_rules():
    assert _to_6digit("SZ000004") == "000004"
    assert _market_of("920000") == "BJ"
    assert DataFetcher()._secucode("920000") == "920000.BJ"
    assert _market("200000")[1] == "SZ"


def test_bad_feature_table_key_is_skipped_instead_of_crashing():
    assert _safe_normalize_company_code(pd.NA) == ""
    assert _safe_normalize_company_code("not-a-code") == ""


def test_predictor_normalizes_before_lookup(monkeypatch):
    agent = PredictorAgent()
    seen = {}
    monkeypatch.setattr(agent, "_load_survival", lambda: None)
    monkeypatch.setattr(agent, "_load_manifest", lambda: {"windows": {}})

    def fake_lookup(ctx, manifest, company, as_of):
        seen["company"] = company
        return ctx

    monkeypatch.setattr(agent, "_execute_lookup", fake_lookup)
    ctx = Context(as_of=date(2026, 8, 26))

    result = agent.execute("SZ000004", ctx)

    assert seen["company"] == "000004.SZ"
    assert result.company == "000004.SZ"
