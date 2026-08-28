"""联网公司名→代码搜索（合并本地 manifest + 新浪 suggest + 腾讯 smartbox）。

供前端主控按公司名称查代码，传给后续 Agent 使用。

策略：
- 输入是 6 位数字代码 → 直接标准化为 ``000004.SZ`` 返回。
- 否则依次查：本地 manifest → 本地缓存(a_stock_list.json) → 新浪 suggest → 腾讯 smartbox。
- 网络命中后写入本地缓存，减少后续搜索对外部接口的依赖。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests

from ..config import DATA_DIR, OUTPUT_DIR
from .stock_code import StockCodeError, infer_exchange, normalize_stock_code

_logger = logging.getLogger(__name__)

_SINA_URL = "https://suggest3.sinajs.cn/suggest/type=11,12&key={q}"
_SINA_TIMEOUT = 8
_TENCENT_URL = "https://smartbox.gtimg.cn/s3/"
_TENCENT_TIMEOUT = 5
_LOCAL_MANIFEST = OUTPUT_DIR / "reports" / "manifest.json"
_LOCAL_CACHE = DATA_DIR / "a_stock_list.json"


# ---------------------------------------------------------------------------
# 本地数据源
# ---------------------------------------------------------------------------
def _load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _load_local_manifest() -> list[dict]:
    data = _load_json_list(_LOCAL_MANIFEST)
    out: list[dict] = []
    for e in data:
        code = e.get("company", "")
        name = e.get("name", "")
        if not code or not name:
            continue
        mkt = code.rsplit(".", 1)[-1] if "." in code else ""
        out.append({"code": code, "name": name, "market": mkt or "", "source": "local"})
    return out


def _load_cache() -> list[dict]:
    """本地持久化缓存：来自网络搜索的成功结果。"""
    return _load_json_list(_LOCAL_CACHE)


def _save_to_cache(items: list[dict]) -> None:
    """把新搜索命中的结果追加到本地缓存，供后续离线搜索使用。"""
    if not items:
        return
    existing = {it["code"]: it for it in _load_cache()}
    for it in items:
        existing[it["code"]] = it
    merged = list(existing.values())
    try:
        _LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _LOCAL_CACHE.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        _logger.warning("保存公司搜索缓存失败: %s", e)


# ---------------------------------------------------------------------------
# 网络数据源
# ---------------------------------------------------------------------------
def _parse_sina_body(body: str) -> list[dict]:
    """解析新浪 suggest 返回体。

    样例：``九安医疗,11,002432,sz002432,九安医疗,,九安医疗,99,1,ESG,,``
    字段：name, type, code6, full_code(name+code) ...
    type=11 为 A 股；full_code 前缀 sh/sz/bj 可推断交易所。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for seg in body.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(",")
        if len(parts) < 4:
            continue
        name = parts[0].strip()
        type_ = parts[1].strip()
        code6 = parts[2].strip()
        full = parts[3].strip().lower()
        if type_ not in ("11", "12"):  # 11=A股, 12=B股；其余跳过
            continue
        if not name or not code6.isdigit() or len(code6) != 6:
            continue
        # 优先用 full 前缀判断交易所
        if full.startswith(("sh", "sz", "bj")) and len(full) >= 8:
            market = full[:2].upper()
            code6 = full[2:]
        else:
            try:
                market = infer_exchange(code6)
            except StockCodeError:
                continue
        if not code6.isdigit() or len(code6) != 6:
            continue
        code = f"{code6}.{market}"
        if code in seen:
            continue
        seen.add(code)
        out.append({"code": code, "name": name, "market": market, "source": "sina"})
    return out


def _search_sina(q: str, limit: int) -> list[dict]:
    try:
        resp = requests.get(
            _SINA_URL.format(q=requests.utils.quote(q)),
            timeout=_SINA_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        )
        resp.encoding = "gbk"
        text = resp.text.strip()
    except Exception as e:
        _logger.debug("新浪 suggest 请求失败: %s", e)
        return []
    m = re.search(r'var suggestvalue="(.*?)";?\s*$', text, re.S)
    if not m:
        _logger.debug("新浪 suggest 返回格式异常: %s", text[:120])
        return []
    return _parse_sina_body(m.group(1))[:limit]


def _search_tencent(q: str, limit: int) -> list[dict]:
    """腾讯 smartbox 联网搜索 A 股（备用源）。"""
    try:
        resp = requests.get(_TENCENT_URL, params={"v": 2, "q": q}, timeout=_TENCENT_TIMEOUT)
        text = resp.text.strip()
    except Exception as e:
        _logger.debug("腾讯 smartbox 请求失败: %s", e)
        return []
    m = re.search(r'v="(.*?)";?\s*$', text, re.S)
    if not m:
        _logger.debug("腾讯 smartbox 返回格式异常: %s", text[:120])
        return []
    body = m.group(1)
    out: list[dict] = []
    seen: set[str] = set()
    for seg in body.split("^"):
        parts = seg.split("~")
        if len(parts) < 2:
            continue
        code6 = parts[0].strip()
        name = parts[1].strip()
        if not (code6.isdigit() and len(code6) == 6) or not name:
            continue
        try:
            market = infer_exchange(code6)
        except StockCodeError:
            continue
        code = f"{code6}.{market}"
        if code in seen:
            continue
        seen.add(code)
        out.append({"code": code, "name": name, "market": market, "source": "tencent"})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 搜索入口
# ---------------------------------------------------------------------------
def _code_only_search(q: str) -> list[dict]:
    """如果输入是 6 位数字代码，直接标准化返回。"""
    digits = re.sub(r"\D", "", q)
    if len(digits) == 6:
        try:
            code = normalize_stock_code(digits)
            # 名称从本地查，查不到用代码占位
            name = digits
            for it in _load_local_manifest() + _load_cache():
                if it["code"] == code:
                    name = it["name"]
                    break
            return [{"code": code, "name": name, "market": code.rsplit(".", 1)[-1], "source": "normalize"}]
        except StockCodeError:
            pass
    return []


def _search_local_and_cache(q: str, limit: int) -> list[dict]:
    q_low = q.lower()
    results: dict[str, dict] = {}
    for it in _load_local_manifest() + _load_cache():
        code = it.get("code", "")
        name = it.get("name", "")
        if not code:
            continue
        if q_low in name.lower() or q_low in code.lower():
            results[code] = it
            if len(results) >= limit:
                break
    return list(results.values())


def search_companies(q: str, limit: int = 10) -> list[dict]:
    """按公司名称/代码搜索公司，合并本地 + 新浪 suggest + 腾讯 smartbox。

    返回示例：
        [{"code": "002432.SZ", "name": "九安医疗", "market": "SZ", "source": "sina"}, ...]
    """
    q = (q or "").strip()
    if not q:
        return []

    # 1) 纯代码直接标准化
    code_results = _code_only_search(q)
    if code_results:
        return code_results[:limit]

    # 2) 本地 manifest + 本地缓存
    results: dict[str, dict] = {}
    for it in _search_local_and_cache(q, limit):
        results[it["code"]] = it

    if len(results) >= limit:
        return list(results.values())[:limit]

    # 3) 新浪 suggest（主源）
    try:
        for it in _search_sina(q, limit):
            if it["code"] not in results:
                results[it["code"]] = it
        # 新浪命中则写入缓存
        if results:
            _save_to_cache(list(results.values()))
    except Exception as e:
        _logger.warning("新浪搜索异常: %s", e)

    if len(results) >= limit:
        return list(results.values())[:limit]

    # 4) 腾讯 smartbox（备用）
    try:
        for it in _search_tencent(q, limit):
            if it["code"] not in results:
                results[it["code"]] = it
        if results:
            _save_to_cache(list(results.values()))
    except Exception as e:
        _logger.warning("腾讯搜索异常: %s", e)

    return list(results.values())[:limit]
