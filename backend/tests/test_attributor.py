#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试：归因解释 Agent（backend/agents/attributor.py）
====================================================
覆盖：
  1. SHAP 路径：特征 → 可读因素 + 证据定位（结构统一）
  2. 降级路径：无 shap_features 时用异常/风险标签，且标记 is_fallback
  3. 叙事防幻觉校验：证据池内引用通过、池外引用被识别
  4. execute 输出结构：validation 字段 + evidence_citations 白名单过滤
"""
import pytest

from backend.context import Context
from backend.agents.attributor import AttributorAgent


def _make_ctx(with_shap=True):
    ctx = Context(company="000004.SZ", window=60)
    ctx.prediction = {
        "probability_60d": 0.72,
        "risk_level": "高",
        "confidence": 0.87,
        "shap_features": (
            [("cf_income_ratio", -0.21), ("roe", -0.18), ("anomaly_count", 0.12)]
            if with_shap else []
        ),
    }
    ctx.financial.anomaly_list = [
        {"type": "双负信号", "severity": 4, "indicator": "cf_income_ratio",
         "evidence": "净利润与经营现金流均为负", "label_ref": "盈利质量"},
        {"type": "亏损", "severity": 4, "indicator": "roe",
         "evidence": "ROE=-7.05%", "label_ref": "盈利能力"},
    ]
    ctx.semantic.risk_factors = [
        {"category": "C", "taxonomy_l2": "C03", "severity": 4,
         "description": "计提商誉减值损失4.5亿"},
    ]
    ctx.semantic.evidence_snippets = [
        {"category": "C", "text": "本年计提商誉减值损失45,057.53万元"},
    ]
    ctx.cases = [
        {"case_id": "IC-000005-2023", "company": "ST星源", "inquiry_type": "年报问询函",
         "similarity": 0.81, "topics": ["盈利质量", "资金占用"]},
    ]
    return ctx


def test_shap_path_maps_factors():
    ctx = _make_ctx(with_shap=True)
    agent = AttributorAgent(use_llm=False)
    agent.execute("000004.SZ", ctx)

    factors = ctx.attribution["top_risk_factors"]
    assert len(factors) >= 1
    # 结构统一：description（非 desc）+ evidence_id + is_fallback=False
    assert "description" in factors[0]
    assert "desc" not in factors[0]
    assert factors[0]["is_fallback"] is False
    # 财务因素按 indicator 绑定证据
    assert factors[0]["evidence_id"] is not None
    assert factors[0]["evidence_id"].startswith("fin_")


def test_fallback_path_when_no_shap():
    ctx = _make_ctx(with_shap=False)
    agent = AttributorAgent(use_llm=False)
    agent.execute("000004.SZ", ctx)

    factors = ctx.attribution["top_risk_factors"]
    assert len(factors) >= 1
    # 降级路径：全部标记 is_fallback=True
    assert all(f.get("is_fallback") for f in factors)
    # 降级来源覆盖财务 + 语义
    sources = {f["source"] for f in factors}
    assert "financial" in sources


def test_validate_narrative_whitelist():
    agent = AttributorAgent(use_llm=False)
    pool = [{"evidence_id": "fin_000", "snippet": "净利润为负"},
            {"evidence_id": "sem_000", "snippet": "商誉减值"}]

    cited, invalid = agent.validate_narrative("证据 fin_000 表明盈利质量差", pool)
    assert cited == {"fin_000"}
    assert invalid == set()

    cited, invalid = agent.validate_narrative("证据 fin_999 与 sem_000 均异常", pool)
    assert "sem_000" in cited
    assert invalid == {"fin_999"}


def test_execute_output_structure():
    ctx = _make_ctx(with_shap=False)
    agent = AttributorAgent(use_llm=False)
    agent.execute("000004.SZ", ctx)

    att = ctx.attribution
    # 输出字段完整
    for key in ("top_risk_factors", "evidence_citations", "case_links",
                "narrative", "confidence", "validation"):
        assert key in att
    # use_llm=False 时 narrative 为空 → 无引用 → evidence_citations 为空（白名单过滤）
    assert att["narrative"] == ""
    assert att["evidence_citations"] == []
    assert att["validation"]["narrative_validated"] is True
    # 案例链接保留
    assert len(att["case_links"]) >= 1
