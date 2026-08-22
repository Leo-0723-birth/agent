#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: announcement_search —— 公告本地仓库（时间安全版）
============================================================
从本地公告 PDF 目录扫描、抽取正文、建立索引缓存，并按历史时点检索。

关键修复：
1. 不再把“报告期日期（如 2021-12-31）”当成公告披露日。
2. publication_date 只接受具有明确“发布/签署”语境的日期；
   若无法获得可靠精确日期，则使用文件夹年份的 12-31 作为保守可用日期，
   这样只会延迟文档进入历史样本，不会把未来信息提前泄漏。
3. 索引缓存带版本号和 data_root 校验，路径迁移后自动重建，避免旧 D 盘缓存继续生效。
4. 保留 date 字段作为 publication_date 的兼容别名，避免影响 AnnouncementReader 现有接口。
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

_INDEX_VERSION = 3
_DATE_PATTERN = re.compile(r"(\d{4})\s*[年./\-]\s*(\d{1,2})\s*[月./\-]\s*(\d{1,2})\s*日?")
_CN_NUM = {"〇": "0", "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
           "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_CN_DATE = re.compile(
    r"([〇零一二三四五六七八九]{4})\s*年\s*([〇零一二三四五六七八九十]{1,3})\s*月\s*"
    r"([〇零一二三四五六七八九十]{1,3})\s*日"
)
_ANNO_NUM = re.compile(r"公告编号[:：]\s*(\d{4})\s*[-–—]\s*\d+")
_POSITIVE_DATE_CUES = (
    "董事会", "监事会", "公告日期", "披露日期", "发布日期", "发布日",
    "公告日", "签署日期", "签署日", "报出日期", "批准报出",
)
_NEGATIVE_DATE_CUES = (
    "报告期", "报告期末", "截至", "期末", "季度", "半年度", "年度",
    "资产负债表", "利润表", "现金流量表", "会计期间",
)


def _extract_text(pdf_path):
    """用 PyMuPDF 抽取 PDF 全文文本。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n".join(pages)


def _cn_to_int(s):
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


def _collect_dates(text):
    matches = []
    for m in _DATE_PATTERN.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            datetime(y, mo, d)
        except ValueError:
            continue
        if 2000 <= y <= 2030:
            matches.append((m.start(), y, mo, d))
    for m in _CN_DATE.finditer(text):
        y, mo, d = _cn_to_int(m.group(1)), _cn_to_int(m.group(2)), _cn_to_int(m.group(3))
        if not (y and mo and d):
            continue
        try:
            datetime(y, mo, d)
        except ValueError:
            continue
        if 2000 <= y <= 2030:
            matches.append((m.start(), y, mo, d))
    matches.sort(key=lambda x: x[0])
    return matches


def _date_context(text, pos, radius=120):
    left = max(0, pos - radius)
    right = min(len(text), pos + radius)
    return text[left:right].replace("\n", " ")


def _extract_publication_date(text, folder_year=None):
    """
    返回 (publication_date, date_source, date_confidence)。

    v2 规则：
    1) 最优先识别文末正式落款日期，例如“XX股份有限公司董事会\\n2021年10月27日”。
       这是报告/公告最稳定的文档日期。
    2) 排除正文中的历史引用，例如“公司于2021年10月16日披露的……”。
    3) 若无法得到可靠精确日期，则保守使用年份-12-31。
    4) search() 仍采用 publication_date < as_of，因此落款当日不会进入当日回测，
       最早从次日开始可用，进一步降低同日披露时序不确定性。
    """
    year_hint = None
    m = _ANNO_NUM.search(text[:12000])
    if m:
        year_hint = int(m.group(1))
        year_source = "announcement_number"
    elif folder_year:
        year_hint = int(folder_year)
        year_source = "folder_year"
    else:
        year_source = None

    candidates = _collect_dates(text)

    def same_year(y):
        return (year_hint is None) or (y == year_hint)

    def is_historical_reference(pos):
        before = text[max(0, pos - 24):pos].replace("\n", " ")
        after = text[pos:min(len(text), pos + 80)].replace("\n", " ")
        # 典型：公司于2021年10月16日披露的《……》
        return (
            ("于" in before[-8:] and ("披露" in after or "发布" in after or "公告" in after))
            or ("详见" in before[-24:] and ("披露" in after or "公告" in after))
        )

    # ---- 优先级1：文末正式落款 ----
    tail_start = max(0, len(text) - 3500)
    signature_candidates = []
    for pos, y, mo, d in candidates:
        if pos < tail_start or not same_year(y):
            continue
        if is_historical_reference(pos):
            continue

        before = text[max(0, pos - 260):pos].replace("\n", " ")
        # 报告/公告常以“公司名 + 董事会/监事会 + 日期”收尾。
        if ("董事会" in before[-180:] or "监事会" in before[-180:]):
            signature_candidates.append((pos, y, mo, d))

    if signature_candidates:
        _, y, mo, d = signature_candidates[-1]
        return f"{y:04d}-{mo:02d}-{d:02d}", "tail_signature_date", "high"

    # ---- 优先级2：显式“公告日期/披露日期/发布日期”字段 ----
    explicit_candidates = []
    for pos, y, mo, d in candidates:
        if not same_year(y) or is_historical_reference(pos):
            continue

        before = text[max(0, pos - 80):pos].replace("\n", " ")
        if any(k in before[-40:] for k in ("公告日期", "披露日期", "发布日期", "发布日", "公告日")):
            explicit_candidates.append((pos, y, mo, d))

    if explicit_candidates:
        _, y, mo, d = explicit_candidates[-1]
        return f"{y:04d}-{mo:02d}-{d:02d}", "explicit_publication_field", "high"

    # ---- 优先级3：只有年份可靠时，保守到年末 ----
    if year_hint:
        return f"{year_hint:04d}-12-31", f"{year_source}_conservative", "medium"

    return None, "unresolved", "low"

def _extract_title(text, fallback):
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
    """公告本地仓库：扫描 + 时间安全索引 + 时间窗口检索。"""

    def __init__(self, data_root, cache_path=None):
        self.data_root = str(data_root)
        self.cache_path = cache_path
        self.entries = self._load_index()

    def _cache_payload_valid(self, payload):
        if not isinstance(payload, dict):
            return False
        if payload.get("index_version") != _INDEX_VERSION:
            return False
        cached_root = os.path.normcase(os.path.abspath(payload.get("data_root", "")))
        current_root = os.path.normcase(os.path.abspath(self.data_root))
        return cached_root == current_root and isinstance(payload.get("entries"), list)

    def _load_index(self):
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    payload = json.load(f)
                if self._cache_payload_valid(payload):
                    return payload["entries"]
                print("  [公告索引] 检测到旧版/旧路径缓存，自动重建")
            except Exception as e:
                print(f"  [公告索引] 缓存读取失败，自动重建: {e}")

        entries = self._build_index()
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            payload = {
                "index_version": _INDEX_VERSION,
                "data_root": os.path.abspath(self.data_root),
                "entries": entries,
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
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
            folder_year = int(m.group(1)) if m else None
            for type_dir in sorted(os.listdir(year_path)):
                type_path = os.path.join(year_path, type_dir)
                if not os.path.isdir(type_path):
                    continue
                for fn in sorted(os.listdir(type_path)):
                    if fn.lower().endswith(".pdf"):
                        files.append({
                            "path": os.path.join(type_path, fn),
                            "folder_year": folder_year,
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
            publication_date, date_source, date_confidence = _extract_publication_date(
                text, f["folder_year"]
            )
            title = _extract_title(text, f"{f['type']} {publication_date or ''}".strip())
            entries.append({
                "id": f["id"],
                "year": f["folder_year"],
                "folder_year": f["folder_year"],
                "type": f["type"],
                "path": f["path"],
                "publication_date": publication_date,
                "date": publication_date,
                "date_source": date_source,
                "date_confidence": date_confidence,
                "title": title,
                "char_count": len(text),
            })
        entries.sort(key=lambda e: (e.get("publication_date") or "", e["id"]))
        return entries

    def get_text(self, entry):
        if entry.get("text") is not None:
            return entry["text"]
        entry["text"] = _extract_text(entry["path"])
        return entry["text"]

    def search(self, days=365, as_of=None):
        if as_of is None:
            as_of = datetime.now().date()
        elif isinstance(as_of, str):
            as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
        start = as_of - timedelta(days=days)

        result = []
        for e in self.entries:
            d = e.get("publication_date") or e.get("date")
            if not d:
                continue
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start <= dt < as_of:
                item = dict(e)
                item["text"] = self.get_text(e)
                result.append(item)
        result.sort(key=lambda e: e.get("publication_date") or e.get("date") or "", reverse=True)
        return result

    def date_range(self):
        dates = [
            e.get("publication_date") or e.get("date")
            for e in self.entries
            if (e.get("publication_date") or e.get("date"))
        ]
        if not dates:
            return None, None
        return min(dates), max(dates)
