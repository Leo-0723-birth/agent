#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: rule_risk_extract —— 规则风险抽取（读官方 risk_dictionary.yaml）
======================================================================
零成本、确定性、可解释的风险抽取通道：关键词 + 正则 + 否定词窗口。
与 FinBERT 门控、LLM 精细抽取构成"三通道"，规则层用于候选召回与交叉校验。

数据：backend/data/labels/risk_dictionary.yaml（任务1交付包，45类冻结版 v2.0）
用法：
    from backend.skills.rule_risk_extract import RuleRiskExtractor
    extractor = RuleRiskExtractor()
    hits = extractor.extract("公司预计2024年度亏损1.2亿元，商誉减值损失5亿元...")
    # hits = [{"rule_id","label","category_id","severity","matched_key","negated","evidence"}, ...]
"""
import json
import re
from pathlib import Path

from ..config import RISK_DICTIONARY
from .announcement_context_filter import contextual_suppression_reason

DICT_PATH = Path(RISK_DICTIONARY)


class RuleRiskExtractor:
    """规则风险抽取器：加载官方风险词典，对文本做规则匹配。"""

    def __init__(self, dict_path=None):
        self.dict_path = Path(dict_path or DICT_PATH)
        self.data = self._load()
        self.version = str(self.data.get("dictionary_version") or "unknown")
        self.negations = self.data.get("global_negation_patterns", [])
        self.rules = []
        for cat in self.data.get("categories", []):
            for rule in cat.get("rules", []):
                self.rules.append({**rule, "category_id": cat.get("category_id")})
        print(f"[rule_risk_extract] 加载词典 v{self.data.get('dictionary_version', '?')}："
              f"{len(self.rules)} 条规则 / {len(self.negations)} 个否定词")

    def _load(self):
        with open(self.dict_path, encoding="utf-8") as f:
            return json.load(f)

    # ---------- 主入口 ----------
    def extract(self, text, max_evidence=240):
        """返回可审计候选；否定/模板命中保留状态，但不得作为有效风险。"""
        text = text or ""
        hits = []
        for rule in self.rules:
            candidates = []
            for keyword in rule.get("keywords", []):
                candidates.extend(
                    (match.start(), match.end(), match.group(0))
                    for match in re.finditer(re.escape(keyword), text, re.IGNORECASE)
                )
            for expression in rule.get("regexes", []):
                try:
                    candidates.extend(
                        (match.start(), match.end(), match.group(0))
                        for match in re.finditer(expression, text, re.IGNORECASE)
                    )
                except re.error:
                    continue
            if not candidates:
                continue
            evaluated = []
            for pos, end, matched_key in sorted(
                set(candidates), key=lambda row: (row[0], -len(row[2]))
            ):
                negated = self._is_negated(text, pos, len(matched_key))
                rule_excluded = self._in_excluded_paragraph(rule, text, pos)
                context_reason = contextual_suppression_reason(
                    rule_id=str(rule.get("rule_id") or ""),
                    label=str(rule.get("label") or ""),
                    text=text,
                    start=pos,
                    end=end,
                )
                suppression_reason = (
                    "negated_context" if negated else
                    "rule_paragraph_exclusion" if rule_excluded else
                    context_reason
                )
                evidence, evidence_start, evidence_end = self._evidence(
                    text, pos, end, max_evidence
                )
                evaluated.append({
                    "rule_id": rule.get("rule_id"),
                    "label": rule.get("label"),              # 二级主题编码（如 C03）
                    "category_id": rule.get("category_id"),  # 一级主题（如 C）
                    "severity": rule.get("severity"),
                    "matched_key": matched_key,
                    "negated": negated,
                    "excluded": bool(rule_excluded or context_reason),
                    "suppression_reason": suppression_reason,
                    "evidence": evidence,
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "evidence_valid": bool(
                        matched_key and matched_key in text[evidence_start:evidence_end]
                    ),
                    "dictionary_version": self.version,
                })
                if not suppression_reason:
                    break
            accepted = next(
                (candidate for candidate in evaluated if not candidate["suppression_reason"]),
                None,
            )
            hits.append(accepted or evaluated[0])
        return hits

    def summarize(self, text):
        """按一级主题汇总命中（供 F 特征/报告使用）。"""
        hits = self.extract(text)
        summary = {}
        for h in hits:
            if h["negated"] or h["excluded"]:
                continue
            cid = h["category_id"]
            s = summary.setdefault(cid, {"count": 0, "labels": set(), "max_severity": "low"})
            s["count"] += 1
            s["labels"].add(h["label"])
            sev_rank = {"low": 1, "medium": 2, "high": 3}
            if sev_rank.get(h["severity"], 1) > sev_rank.get(s["max_severity"], 1):
                s["max_severity"] = h["severity"]
        for cid, s in summary.items():
            s["labels"] = sorted(s["labels"])
        return summary

    # ---------- 匹配 ----------
    def _match_rule(self, rule, text):
        """关键词/正则匹配，返回 (matched_key, pos)；未命中返回 (None, -1)。"""
        for kw in rule.get("keywords", []):
            pos = text.find(kw)
            if pos != -1:
                return kw, pos
        for rx in rule.get("regexes", []):
            try:
                m = re.search(rx, text)
            except re.error:
                continue
            if m:
                return rx, m.start()
        return None, -1

    def _is_negated(self, text, pos, length, window=16):
        """匹配位置前 window 字符内是否出现否定词（如"不存在""未发现"）。"""
        start = max(0, pos - window)
        before = text[start:pos]
        return any(neg in before for neg in self.negations)

    def _in_excluded_paragraph(self, rule, text, pos, radius=200):
        """命中附近是否匹配"段落排除正则"（模板话术不判风险）。"""
        excls = rule.get("paragraph_exclusion_regexes") or []
        if not excls:
            return False
        ctx = text[max(0, pos - radius): pos + radius]
        return any(re.search(rx, ctx) for rx in excls)

    @staticmethod
    def _evidence(text, pos, end, max_evidence):
        if pos < 0:
            return "", 0, 0
        sentence_start = max(text.rfind(mark, 0, pos) for mark in ("\n", "。", "！", "？", "；"))
        start = max(sentence_start + 1, pos - max_evidence // 2)
        candidates = [text.find(mark, end) for mark in ("\n", "。", "！", "？", "；")]
        positive = [value for value in candidates if value >= 0]
        sentence_end = min(positive) + 1 if positive else len(text)
        stop = min(sentence_end, start + max_evidence)
        return re.sub(r"\s+", " ", text[start:stop]).strip(), start, stop


# ============================================================
# 自测入口（python -m backend.skills.rule_risk_extract）
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.skills.rule_risk_extract import RuleRiskExtractor

    extractor = RuleRiskExtractor()
    samples = [
        ("正面命中", "公司预计2024年度亏损1.2亿元，主要系商誉减值损失所致，应收账款坏账准备计提不足。"),
        ("否定词过滤", "经核查，公司不存在商誉减值风险，未发现应收款项减值迹象。"),
        ("真实问询片段", "你公司2020年营业收入2.81亿元，同比增长159.16%；净利润381.60万元。"
                          "报告期末应收账款账面余额3.70亿元，占营业收入的比例为131.95%。"
                          "收购形成商誉9.87亿元，未计提商誉减值准备。"),
    ]
    for name, text in samples:
        print(f"\n===== {name} =====")
        for h in extractor.extract(text):
            flag = "（已否定）" if h["negated"] else ""
            print(f"  [{h['label']}/{h['severity']}] {h['matched_key']}{flag}")
            print(f"    证据: {h['evidence'][:60]}...")
