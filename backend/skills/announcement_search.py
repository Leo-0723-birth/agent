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
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import (
    ANNOUNCE_MAX_DOCUMENTS,
    ANNOUNCE_PDF_CACHE,
    OCR_DPI,
    OCR_ENABLED,
    OCR_MAX_PAGES_PER_DOCUMENT,
    OCR_MIN_CONFIDENCE,
    OCR_MIN_PAGE_CHARS,
)
from .announcement_context_filter import apply_title_policy, is_analysis_eligible
from .ocr_extract import OCR_PIPELINE_VERSION, RapidOCRPageEngine, extract_pdf_text


CNINFO_HOME = "https://www.cninfo.com.cn"
CNINFO_STATIC = "https://static.cninfo.com.cn/"
CNINFO_COMPANY_QUERY = f"{CNINFO_HOME}/new/information/topSearch/query"
CNINFO_ANNOUNCEMENT_QUERY = f"{CNINFO_HOME}/new/hisAnnouncement/query"


def _build_http_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": random.choice(
                (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                )
            )
        }
    )
    return session


def _request_json(session, method, url, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() in {"iso-8859-1", "latin-1"}:
        response.encoding = "utf-8"
    return response.json()


def _six_digit_code(value):
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", str(value or ""))
    return match.group(1) if match else ""


def _market(code):
    if code.startswith(("6", "9")) and not code.startswith("92"):
        return "SSE", "SH", "sse", "sh"
    if code.startswith(("0", "3")):
        return "SZSE", "SZ", "szse", "sz"
    if code.startswith(("4", "8", "92")):
        return "BSE", "BJ", "bj", "bj"
    raise ValueError(f"无法从证券代码判断交易所：{code}")


def _clean_title(value):
    return re.sub(r"<[^>]+>|\s+", " ", str(value or "")).strip()


def _official_date(value):
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds).date().isoformat()
    return date.fromisoformat(str(value or "")[:10].replace("/", "-")).isoformat()


class CninfoAnnouncementSource:
    """巨潮在线事实源：公司解析、公告分页、官方 PDF 下载和证据哈希。"""

    def __init__(
        self,
        session=None,
        cache_dir=None,
        max_documents=ANNOUNCE_MAX_DOCUMENTS,
        ocr_enabled=OCR_ENABLED,
        ocr_engine=None,
    ):
        self.session = session or _build_http_session()
        self.cache_dir = Path(cache_dir or ANNOUNCE_PDF_CACHE)
        self.max_documents = int(max_documents)
        self.ocr_enabled = bool(ocr_enabled)
        self.ocr_engine = ocr_engine or RapidOCRPageEngine()

    def resolve_company(self, user_input):
        raw = str(user_input or "").strip()
        if not raw:
            raise ValueError("公司代码或名称不能为空")
        requested_code = _six_digit_code(raw)
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Origin": CNINFO_HOME,
            "Referer": f"{CNINFO_HOME}/",
            "X-Requested-With": "XMLHttpRequest",
        }
        hits = _request_json(
            self.session,
            "POST",
            CNINFO_COMPANY_QUERY,
            data={"keyWord": requested_code or raw, "maxNum": 20},
            headers=headers,
        )
        hits = hits if isinstance(hits, list) else []
        if not hits:
            raise LookupError(f"巨潮资讯未找到公司：{raw}")

        def code_of(item):
            return str(item.get("code") or item.get("secCode") or "").strip()

        def name_of(item):
            return str(
                item.get("zwjc") or item.get("secName") or item.get("companyName")
                or item.get("orgName") or ""
            ).strip()

        selected = None
        if requested_code:
            selected = next((item for item in hits if code_of(item) == requested_code), None)
        else:
            normalized = re.sub(r"\s+", "", raw).lower()
            exact = [
                item for item in hits
                if re.sub(r"\s+", "", name_of(item)).lower() == normalized
            ]
            if len(exact) == 1:
                selected = exact[0]
            elif len(hits) == 1:
                selected = hits[0]
            else:
                candidates = "、".join(f"{code_of(i)} {name_of(i)}" for i in hits[:8])
                raise LookupError(f"公司名称存在多个匹配，请输入代码：{candidates}")
        if selected is None:
            raise LookupError(f"巨潮资讯未找到精确公司：{raw}")
        code = code_of(selected)
        exchange, suffix, _, _ = _market(code)
        return {
            "code": code,
            "secucode": f"{code}.{suffix}",
            "company_name": name_of(selected) or code,
            "exchange": exchange,
            "org_id": str(selected.get("orgId") or selected.get("orgid") or ""),
            "resolved_from": raw,
            "source_url": CNINFO_COMPANY_QUERY,
        }

    def _list_metadata(self, company, as_of, days):
        end = date.fromisoformat(str(as_of)[:10])
        start = end - timedelta(days=int(days) - 1)
        _, _, column, plate = _market(company["code"])
        stock = f"{company['code']},{company['org_id']}" if company.get("org_id") else ""
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": CNINFO_HOME,
            "Referer": f"{CNINFO_HOME}/new/commonUrl?url=disclosure/list/notice",
            "X-Requested-With": "XMLHttpRequest",
        }
        raw_items = []
        for page in range(1, 101):
            payload = {
                "pageNum": page,
                "pageSize": 30,
                "column": column,
                "tabName": "fulltext",
                "plate": plate,
                "stock": stock,
                "searchkey": "" if stock else company["code"],
                "secid": "",
                "category": "",
                "trade": "",
                "seDate": f"{start.isoformat()}~{end.isoformat()}",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            if page > 1:
                time.sleep(0.12)
            data = _request_json(
                self.session, "POST", CNINFO_ANNOUNCEMENT_QUERY,
                data=payload, headers=headers,
            )
            page_items = (data or {}).get("announcements") or []
            if not page_items:
                break
            raw_items.extend(page_items)
            if not (data or {}).get("hasMore") and len(page_items) < 30:
                break

        output, seen = [], set()
        for item in raw_items:
            if item.get("secCode") and str(item["secCode"]) != company["code"]:
                continue
            published = _official_date(item.get("announcementTime"))
            published_date = date.fromisoformat(published)
            if not start <= published_date <= end:
                continue
            adjunct = str(item.get("adjunctUrl") or "").lstrip("/")
            announcement_id = str(item.get("announcementId") or adjunct)
            identity = (announcement_id, adjunct)
            if identity in seen:
                continue
            seen.add(identity)
            output.append(
                {
                    "id": announcement_id,
                    "announcement_id": announcement_id,
                    "secucode": company["secucode"],
                    "company_name": str(item.get("secName") or company["company_name"]),
                    "title": _clean_title(item.get("announcementTitle")),
                    "date": published,
                    "published_at": published,
                    "type": str(item.get("announcementTypeName") or "公告"),
                    "source_name": "巨潮资讯网",
                    "source_tier": "official_current",
                    "source_url": (
                        f"{CNINFO_HOME}/new/disclosure/detail?stockCode={company['code']}"
                        f"&announcementId={announcement_id}"
                    ),
                    "pdf_url": CNINFO_STATIC + adjunct if adjunct else "",
                    "official": True,
                    "text": "",
                    "text_status": "not_fetched",
                    "char_count": 0,
                    "content_sha256": "",
                    "cache_path": "",
                    "ocr_status": "not_fetched",
                    "ocr_engine": "",
                    "ocr_candidate_pages": 0,
                    "ocr_attempted_pages": 0,
                    "ocr_succeeded_pages": 0,
                    "ocr_failed_pages": 0,
                }
            )
        output.sort(key=lambda row: (row["date"], row["id"]), reverse=True)
        return output

    def _process_pdf(self, item):
        if not item.get("pdf_url"):
            item["text_status"] = "no_pdf_url"
            return item
        cache_key = hashlib.sha256(item["pdf_url"].encode("utf-8")).hexdigest()[:20]
        safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", item["id"])[:80] or "announcement"
        path = self.cache_dir / f"{safe_id}_{cache_key}.pdf"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                content = path.read_bytes()
                status = "cached"
            else:
                response = self.session.get(item["pdf_url"], timeout=20)
                response.raise_for_status()
                content = response.content
                if len(content) > 50 * 1024 * 1024:
                    raise ValueError("PDF超过50MB安全上限")
                if not content.lstrip().startswith(b"%PDF"):
                    raise ValueError("下载内容不是PDF")
                path.write_bytes(content)
                status = "downloaded"
            item["content_sha256"] = hashlib.sha256(content).hexdigest()
            item["cache_path"] = str(path.resolve())
            extraction_path = path.with_suffix(".extract.json")
            extraction_signature = {
                "pipeline_version": OCR_PIPELINE_VERSION,
                "content_sha256": item["content_sha256"],
                "ocr_enabled": self.ocr_enabled,
                "ocr_dpi": OCR_DPI,
                "ocr_min_page_chars": OCR_MIN_PAGE_CHARS,
                "ocr_min_confidence": OCR_MIN_CONFIDENCE,
                "ocr_max_pages": OCR_MAX_PAGES_PER_DOCUMENT,
            }
            text, ocr_metadata = "", {}
            extraction_cache_hit = False
            try:
                cached_extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
                if (
                    cached_extraction.get("signature") == extraction_signature
                    and isinstance(cached_extraction.get("text"), str)
                    and isinstance(cached_extraction.get("metadata"), dict)
                ):
                    text = cached_extraction["text"]
                    ocr_metadata = cached_extraction["metadata"]
                    extraction_cache_hit = True
            except (OSError, ValueError, TypeError):
                pass
            if not extraction_cache_hit:
                import pymupdf
                with pymupdf.open(stream=content, filetype="pdf") as document:
                    text, ocr_metadata = extract_pdf_text(
                        document,
                        engine=self.ocr_engine,
                        enabled=self.ocr_enabled,
                        dpi=OCR_DPI,
                        min_page_chars=OCR_MIN_PAGE_CHARS,
                        min_confidence=OCR_MIN_CONFIDENCE,
                        max_pages=OCR_MAX_PAGES_PER_DOCUMENT,
                    )
                try:
                    extraction_path.write_text(
                        json.dumps(
                            {
                                "signature": extraction_signature,
                                "text": text,
                                "metadata": ocr_metadata,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            item.update(ocr_metadata)
            item["extraction_cache_hit"] = extraction_cache_hit
            item["extraction_cache_path"] = str(extraction_path.resolve())
            item["text"] = text
            item["char_count"] = len(text)
            if text and item.get("ocr_succeeded_pages", 0):
                item["text_status"] = f"{status}_ocr_parsed"
            elif text:
                item["text_status"] = f"{status}_native_parsed"
            else:
                item["text_status"] = f"{status}_empty_text"
        except Exception as exc:
            item["text_status"] = f"failed:{type(exc).__name__}:{str(exc)[:160]}"
        return item

    def search(self, user_input, days=365, as_of=None, pdf_budget_seconds=180):
        cutoff = str(as_of or date.today().isoformat())[:10]
        company = self.resolve_company(user_input)
        announcements = self._list_metadata(company, cutoff, days)
        for item in announcements:
            apply_title_policy(item, mark_unfetched=True)
        eligible = [item for item in announcements if is_analysis_eligible(item)]
        # PDF 下载/解析总预算：超过预算的公告标记为未获取并继续（不阻塞流水线；
        # F6 问询特征只需公告元数据 title+date，不受影响）
        deadline = time.time() + float(pdf_budget_seconds)
        for item in eligible[: self.max_documents]:
            if time.time() > deadline:
                item["text_status"] = "not_fetched_budget"
                continue
            self._process_pdf(item)
        return company, announcements

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
