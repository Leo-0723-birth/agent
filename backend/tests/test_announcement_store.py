#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试：公告检索（skills/announcement_search 迁移后）
==================================================
重点覆盖：中文数字日期解析、ASCII 日期解析、窗口边界、索引缓存。

TODO: 迁移 公告研读agents/announcement_store.py 后填充断言。
"""
import pytest


def test_date_extraction():
    # TODO: 断言 _extract_date 对 "2024年3月26日" / "二〇二四年三月二十六日" 的解析
    pass


def test_window_search():
    # TODO: 断言 search(days=365, as_of=...) 的时间窗口过滤
    pass
