#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试：共享 Context（backend/context.py）
=======================================
"""
import pytest
from backend.context import Context, Semantic, Financial


def test_context_defaults():
    ctx = Context()
    assert ctx.window == 60
    assert ctx.semantic.risk_factors == []
    assert ctx.trace_log == []
    assert ctx.semantic.f1_features == {}


def test_context_write_and_to_dict():
    ctx = Context(company="000004.SZ", window=60)
    ctx.semantic.risk_factors.append({"category": "财务异常"})
    d = ctx.to_dict()
    assert d["company"] == "000004.SZ"
    assert d["semantic"]["risk_factors"][0]["category"] == "财务异常"
    assert "f1_features" in d["semantic"]
    assert "data_quality" in d["semantic"]
