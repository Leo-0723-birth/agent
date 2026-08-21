#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: contradiction_detect —— 跨公告矛盾检测
============================================
输入：多份公告文本（或结构化段落）
输出：矛盾点列表 [{type, doc_a, doc_b, description, evidence}]

实现思路：关键数字/口径抽取后做跨文档比对（数值不一致、前后表述矛盾），
可用 LLM 辅助判断（backend.llm.chat）。

TODO: 由后续开发填充实现。
"""


def detect(announcements):
    """返回矛盾点列表。"""
    raise NotImplementedError
