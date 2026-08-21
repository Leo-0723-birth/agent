#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给案例库补充"回复要点"（reply_excerpt）
========================================
从 inquiries.jsonl 的 inquiry_reply（5271 份回复）为每个案例（问询函）匹配回复，
提取回复原文段落前 800 字作为 reply_excerpt（证据溯源，原文引用）。

匹配规则：同公司、回复日期 ≥ 问询函日期 的最近一条回复；无则取该公司最早回复。
只改 case_db.json 元数据，不重新向量化（向量来自 类型+标题+关注点，与回复无关）。

用法：python -m backend.scripts.enrich_case_db_reply
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import CASE_DB_PATH, INQUIRY_JSONL

REPLY_EXCERPT_CHARS = 800


def _doc_text(doc):
    paras = doc.get("paragraphs") or []
    return "\n".join(p.get("text", "") for p in paras if p.get("text"))


def load_replies(path):
    """按公司索引回复：{stock_code: [(publish_date, text), ...]}。"""
    replies = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("doc_type") != "inquiry_reply":
                continue
            code = d.get("stock_code")
            date = str(d.get("publish_date") or "")[:10]
            text = _doc_text(d)
            replies.setdefault(code, []).append((date, text))
    for code in replies:
        replies[code].sort(key=lambda x: x[0])
    return replies


def match_reply(reply_list, letter_date):
    """找 回复日期 >= 函日期 的最近一条；无则取第一条。返回 (date, text)。"""
    if not reply_list:
        return None
    for date, text in reply_list:
        if date and letter_date and date >= letter_date:
            return date, text
    return reply_list[0]


def main():
    if not CASE_DB_PATH.exists():
        print(f"案例库不存在: {CASE_DB_PATH}")
        return 1
    entries = json.loads(CASE_DB_PATH.read_text(encoding="utf-8"))
    print(f"案例数: {len(entries)}")

    print(f"加载回复（{INQUIRY_JSONL}）...")
    replies = load_replies(INQUIRY_JSONL)
    print(f"有回复的公司数: {len(replies)}")

    matched = 0
    for e in entries:
        code = e.get("company")
        letter_date = str(e.get("publish_date") or "")[:10]
        r = match_reply(replies.get(code), letter_date)
        if r:
            e["reply_date"] = r[0]
            e["reply_excerpt"] = r[1][:REPLY_EXCERPT_CHARS]
            matched += 1
        else:
            e["reply_excerpt"] = ""

    CASE_DB_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"匹配到回复: {matched}/{len(entries)} 条")
    print(f"已写回: {CASE_DB_PATH}（向量未变，无需重建）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
