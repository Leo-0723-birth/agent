#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: announcement_search —— 公告本地仓库（检索近一年公告）
============================================================
从本地公告 PDF 目录（D:\\BaiduNetdiskDownload\\{公司代码}\\年份\\类型\\*.pdf）扫描、
解析日期/标题、建立索引缓存，按时间窗口检索。

迁移自：公告研读agents/announcement_store.py（原样保留核心逻辑）
数据位置：config.DATA_RAW（默认 D:\\BaiduNetdiskDownload）下的 {公司代码} 目录；
索引缓存：config.INDEX_DIR（backend/data/index/{code}_index.json）。

说明：当前数据源为【本地官方公告 PDF】；如需"公开平台实时拉取公告"
（巨潮/东财公告 API），可在本 Skill 上增加 fetcher 通道（注入 fetch 回调）。
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass


def _extract_text(pdf_path):
    """用 PyMuPDF 抽取 PDF 全文文本。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ASCII 数字日期: 2024-03-26 / 2024年3月26日 / 2024.03.26
_DATE_PATTERN = re.compile(r"(\d{4})\s*[年./\-]\s*(\d{1,2})\s*[月./\-]\s*(\d{1,2})\s*日?")

# 中文数字日期: 二〇二四年三月二十六日
_CN_NUM = {"〇": "0", "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
           "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_CN_DATE = re.compile(
    r"([〇零一二三四五六七八九]{4})\s*年\s*([〇零一二三四五六七八九十]{1,3})\s*月\s*"
    r"([〇零一二三四五六七八九十]{1,3})\s*日"
)

# 公告编号年份: 公告编号：2024-029 → 2024（发布年份的强先验）
_ANNO_NUM = re.compile(r"公告编号[:：]\s*(\d{4})\s*[-–—]\s*\d+")


def _cn_to_int(s):
    """中文数字转 int，支持 "二十六"、"十"、"三" 等。"""
    if not s:
        return None
    if "十" in s:
        parts = s.split("十")
        tens = int(_CN_NUM.get(parts[0], "1")) if parts[0] else 1
        ones = int(_CN_NUM.get(parts[1], "0")) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    out = "".join(_CN_NUM.get(c, c) for c in s)
    try:
        return int(out)
    except ValueError:
        return None


def _extract_date(text, fallback_year=None):
    """从正文抽取公告日期（落款日期），返回 'YYYY-MM-DD' 或 None。"""
    pub_year = fallback_year
    m = _ANNO_NUM.search(text)
    if m:
        pub_year = int(m.group(1))

    matches = []
    for m in _DATE_PATTERN.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2000 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
            matches.append((m.start(), y, mo, d))
    for m in _CN_DATE.finditer(text):
        y, mo, d = _cn_to_int(m.group(1)), _cn_to_int(m.group(2)), _cn_to_int(m.group(3))
        if y and mo and d and 2000 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
            matches.append((m.start(), y, mo, d))

    if not matches:
        return f"{pub_year}-01-01" if pub_year else None

    if pub_year:
        same_year = [(pos, y, mo, d) for pos, y, mo, d in matches if y == pub_year]
        if same_year:
            same_year.sort(key=lambda x: x[0])
            _, y, mo, d = same_year[-1]
            return f"{y:04d}-{mo:02d}-{d:02d}"

    matches.sort(key=lambda x: x[0])
    _, y, mo, d = matches[-1]
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _extract_title(text, fallback):
    """从正文顶部抽取标题（best-effort），找不到则用 fallback。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    keywords = ("公告", "报告", "提示", "说明", "更正", "回复", "预案", "决议")
    for l in lines[:40]:
        if ("证券代码" in l) or ("公告编号" in l):
            continue
        if re.match(r"^[一二三四五六七八九十]+、", l):
            continue
        if any(k in l for k in keywords) and 4 <= len(l) <= 80:
            return l
    return fallback


class AnnouncementStore:
    """公告本地仓库：扫描 + 索引 + 时间窗口检索。"""

    def __init__(self, data_root, cache_path=None):
        self.data_root = data_root
        self.cache_path = cache_path
        self.entries = self._load_index()

    def _load_index(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                return json.load(f)
        entries = self._build_index()
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        return entries

    def _scan_files(self):
        files = []
        if not os.path.isdir(self.data_root):
            return files
        for year_dir in sorted(os.listdir(self.data_root)):
            year_path = os.path.join(self.data_root, year_dir)
            if not os.path.isdir(year_path):
                continue
            m = re.match(r"(\d{4})", year_dir)
            year = int(m.group(1)) if m else None
            for type_dir in sorted(os.listdir(year_path)):
                type_path = os.path.join(year_path, type_dir)
                if not os.path.isdir(type_path):
                    continue
                for fn in sorted(os.listdir(type_path)):
                    if fn.lower().endswith(".pdf"):
                        files.append({
                            "path": os.path.join(type_path, fn),
                            "year": year,
                            "type": type_dir,
                            "id": fn[:-4],
                        })
        return files

    def _build_index(self):
        entries = []
        for f in self._scan_files():
            try:
                text = _extract_text(f["path"])
            except Exception as e:
                print(f"  [解析失败] {f['path']}: {e}")
                text = ""
            date = _extract_date(text, f["year"])
            title = _extract_title(text, f"{f['type']} {date or ''}")
            entries.append({
                "id": f["id"],
                "year": f["year"],
                "type": f["type"],
                "path": f["path"],
                "date": date,
                "title": title,
                "char_count": len(text),
            })
        entries.sort(key=lambda e: (e.get("date") or "", e["id"]))
        return entries

    def get_text(self, entry):
        """按需抽取某条公告的全文（带内存缓存）。"""
        if entry.get("text") is not None:
            return entry["text"]
        entry["text"] = _extract_text(entry["path"])
        return entry["text"]

    def search(self, days=365, as_of=None):
        """按时间窗口过滤公告，返回含全文的公告列表（按日期倒序）。"""
        if as_of is None:
            as_of = datetime.now().date()
        elif isinstance(as_of, str):
            as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
        start = as_of - timedelta(days=days)

        result = []
        for e in self.entries:
            d = e.get("date")
            if not d:
                continue
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start <= dt <= as_of:
                item = dict(e)
                item["text"] = self.get_text(e)
                result.append(item)
        result.sort(key=lambda e: e.get("date") or "", reverse=True)
        return result

    def date_range(self):
        dates = [e.get("date") for e in self.entries if e.get("date")]
        if not dates:
            return None, None
        return min(dates), max(dates)
