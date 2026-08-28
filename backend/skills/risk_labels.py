#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: risk_labels —— 官方标签体系加载器（任务1交付包）
========================================================
把 backend/data/labels/ 下的官方标签资产加载为全系统统一标签语言：
  - TAXONOMY        标签体系 A-H 一级 + 45 二级（risk_taxonomy.yaml）
  - QUESTION_TYPES  Q01-13 质疑性质
  - LABEL_NAMES     二级编码 → 中文名（C03 → 商誉与商誉减值）
  - LABEL_KEYWORDS  标签 → 关键词（词典 keywords + 标签名拆分）
  - ALIAS_LABEL_MAP 我方俗称标签 → 官方二级编码（盈利质量→D01/C01）
  - expand_label_keywords(labels)  标签集合 → 关键词集合（案例检索/归因共用）

数据位置：backend/data/labels/（来自任务1交付包）
"""
import json
from pathlib import Path

import yaml

from ..config import BASE_DIR

LABELS_DIR = Path(BASE_DIR) / "backend" / "data" / "labels"
TAXONOMY_PATH = LABELS_DIR / "risk_taxonomy.yaml"
DICT_PATH = Path(__import__("backend.config", fromlist=["RISK_DICTIONARY"]).RISK_DICTIONARY)
CLASSIFIED_PATH = LABELS_DIR / "classified_focus_points_10481.jsonl"

_TAXONOMY = None
_DICT = None


def load_taxonomy():
    """加载标签体系（A-H 一级 + 45 二级 + Q/P 属性）。"""
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return _TAXONOMY


def load_dictionary():
    """加载风险词典（规则层候选召回用）。"""
    global _DICT
    if _DICT is None:
        _DICT = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    return _DICT


def _split_keywords(name):
    """把"应收账款与坏账准备"拆成可命中的关键词：原串 + 顿号/与 分隔片段。"""
    if not name:
        return []
    out = [name]
    for sep in ("与", "、", "，"):
        if sep in name:
            for part in name.split(sep):
                part = part.strip()
                if part:
                    out.append(part)
    return out


def build_label_keywords():
    """标签 → 关键词：① 词典规则 keywords（强）② 二级主题名 ③ 一级主题名。"""
    kws = {}
    # ① 风险词典
    for cat in load_dictionary().get("categories", []):
        for rule in cat.get("rules", []):
            label = rule.get("label")
            kws.setdefault(label, [])
            for k in rule.get("keywords", []):
                if k and k not in kws[label]:
                    kws[label].append(k)
    # ②③ 分类体系（二级名 + 一级名）
    for cid, theme in load_taxonomy().get("risk_themes", {}).items():
        name = theme.get("name", "")
        for code, sub in theme.get("core_subthemes", {}).items():
            kws.setdefault(code, [])
            for k in _split_keywords(sub) + _split_keywords(name):
                if k and k not in kws[code]:
                    kws[code].append(k)
    return kws


def build_label_names():
    """二级编码 → 中文名。"""
    names = {}
    for cid, theme in load_taxonomy().get("risk_themes", {}).items():
        for code, sub in theme.get("core_subthemes", {}).items():
            names[code] = sub
    return names


# 我方俗称标签 → 官方二级编码（跨体系对齐的桥）
ALIAS_LABEL_MAP = {
    "盈利质量": ["D01", "C01"],      # 经营现金流与利润匹配 / 应收账款与坏账准备
    "盈利能力": ["A03"],             # 利润、扣非利润与业绩波动
    "商誉减值": ["C03"],             # 商誉与商誉减值
    "收入确认": ["B01"],             # 收入确认与截止性
    "内控": ["F06"],                 # 内部控制、三会治理与公司独立性
    "偿债能力": ["D03", "D04"],      # 债务结构 / 债务逾期
    "资金占用": ["E03"],             # 非经营性资金占用
    "担保": ["D05"],                 # 担保、抵质押与或有负债
    "信息披露": ["G04", "G05", "G06"],
    "关联交易": ["E02"],
    "持续经营": ["A06"],
    "财务异常": ["A03", "A01", "C01"],
    "行业对标": ["A01", "C01"],
    "市场异动": ["H01"],                 # 股价、成交量与市场交易异常
}


def expand_label_keywords(labels):
    """把标签集合（官方编码或我方俗称）展开为可命中的关键词集合。

    供 case_retriever 标签通道 / attributor 案例链接 统一使用：
    官方关注点是长句，必须用关键词做子串命中。
    """
    kws = set()
    for lab in labels or []:
        if not lab:
            continue
        codes = ALIAS_LABEL_MAP.get(lab, [lab])   # 俗称 → 官方编码；官方编码原样保留
        for code in codes:
            kws.update(LABEL_KEYWORDS.get(code, [code]))
    return kws


def load_classified_focus_points(path=None):
    """加载全量分类结果（10,481 条）→ 列表 dict（案例库丰富/标签样例用）。"""
    path = Path(path or CLASSIFIED_PATH)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ================= 模块级导出（加载一次） =================
TAXONOMY = load_taxonomy()
LABEL_NAMES = build_label_names()
LABEL_KEYWORDS = build_label_keywords()
QUESTION_TYPES = TAXONOMY.get("question_types", {})
STAGES = TAXONOMY.get("stages", {})


# ============================================================
# 自测入口（python -m backend.skills.risk_labels）
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.skills.risk_labels import (
        TAXONOMY, LABEL_NAMES, LABEL_KEYWORDS, QUESTION_TYPES, expand_label_keywords,
        load_classified_focus_points,
    )

    print("一级主题:", list(TAXONOMY["risk_themes"].keys()))
    print("二级主题数:", len(LABEL_NAMES), "| 质疑性质数:", len(QUESTION_TYPES))
    print("C03 关键词:", LABEL_KEYWORDS.get("C03", [])[:6])
    print("A03 关键词:", LABEL_KEYWORDS.get("A03", [])[:6])
    print("expand['盈利质量']:", list(expand_label_keywords(["盈利质量"]))[:6])
    rows = load_classified_focus_points()
    print("全量分类结果条数:", len(rows))
    if rows:
        print("样例:", {k: rows[0].get(k) for k in ("id", "primary_theme_l1", "primary_theme_l2",
                                                     "primary_question_type", "confidence")})
