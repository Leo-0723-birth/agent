#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: concern_store —— 关注点词典加载（规则类别 ↔ 官方关注点关键词）
=====================================================================
读取 build_concern_dict.py 产出的 concern_dict.json（规则类别/风险标签 → 关键词），
供案例检索标签通道（case_retriever 通道2）把目标公司的规则风险标签对齐到官方关注点词汇。

结构：
  categories: {category_name: {"category_id", "keywords"}}
  labels:     {risk_label: {"category", "keywords"}}

缺失 concern_dict.json 时返回空词典（标签通道退化为「标签名即关键词」，不报错）。
"""
import json
from pathlib import Path

from ..config import CONCERN_DICT_PATH

_DICT = None


def load():
    """懒加载关注点词典。返回 {categories, labels}。"""
    global _DICT
    if _DICT is None:
        if Path(CONCERN_DICT_PATH).exists():
            _DICT = json.loads(Path(CONCERN_DICT_PATH).read_text(encoding="utf-8"))
        else:
            _DICT = {"categories": {}, "labels": {}}
    return _DICT


def category_keywords(category):
    """规则类别 → 关键词列表（无映射时返回 [类别名] 兜底）。"""
    c = load()["categories"].get(category)
    return c["keywords"] if c else [category]


def label_keywords(label):
    """风险标签（risk_label）→ 关键词列表；无映射返回 []。"""
    l = load()["labels"].get(label)
    return l["keywords"] if l else []
