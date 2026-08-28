#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一的公告事实证据发布策略。

规则通道负责高召回候选，不等同于事实认定。面向 API、报告和归因展示的
证据必须经过 LLM 原文校验，或规则与 LLM 对同一事件达成交叉一致。
"""
from __future__ import annotations


_PUBLISHABLE_AGREEMENTS = {"rule_llm_agree", "llm_only"}
_PUBLISHABLE_ASSERTIONS = {"actual_event", "actual_event_validated"}


def is_publishable_evidence(factor: dict | None) -> bool:
    """判断风险因子是否可以作为已核验事实证据对外展示。"""
    factor = factor or {}
    if not factor.get("evidence_valid"):
        return False
    if factor.get("suppression_reason") or factor.get("contextual_suppression_reason"):
        return False

    agreement = str(factor.get("agreement_status") or "").strip().lower()
    method = str(factor.get("method") or "").strip().lower()
    assertion = str(factor.get("assertion_type") or "").strip().lower()

    if agreement == "rule_llm_agree":
        return True
    if agreement == "llm_only" or method == "llm_evidence_validated":
        return assertion in _PUBLISHABLE_ASSERTIONS
    return False


def publishable_evidence(factors: list[dict] | None) -> list[dict]:
    """返回可对外展示的事实证据；候选信号仍保留在原始审计数据中。"""
    return [factor for factor in (factors or []) if is_publishable_evidence(factor)]
