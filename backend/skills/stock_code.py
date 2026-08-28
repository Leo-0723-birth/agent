"""上市公司股票代码的统一解析与校验。"""
from __future__ import annotations

import re
import unicodedata


class StockCodeError(ValueError):
    """股票代码为空、格式错误或交易所后缀冲突。"""


_EXCHANGE_ALIASES = {
    "SH": "SH",
    "SSE": "SH",
    "XSHG": "SH",
    "SZ": "SZ",
    "SZSE": "SZ",
    "XSHE": "SZ",
    "BJ": "BJ",
    "BSE": "BJ",
    "XBSE": "BJ",
}


def infer_exchange(code6: str) -> str:
    """按 A/B 股及北交所号段推断交易所。"""
    if code6.startswith("92") or code6.startswith(("4", "8")):
        return "BJ"
    if code6.startswith(("6", "9")):
        return "SH"
    if code6.startswith(("0", "2", "3")):
        return "SZ"
    raise StockCodeError(
        f"暂不支持代码 {code6}：请输入沪深北上市公司股票代码，例如 600000.SH、000001.SZ、920000.BJ。"
    )


def normalize_stock_code(value) -> str:
    """将常见输入统一为 ``000004.SZ``，并拒绝后缀冲突。

    支持 ``000004``、``000004.SZ``、``000004SZ``、``SZ000004``、
    大小写、全角字符及 1—6 位纯数字输入。
    """
    # 不用 ``value or ""``：pandas.NA 的布尔值未定义，会抛 TypeError。
    raw_value = "" if value is None else str(value)
    raw = unicodedata.normalize("NFKC", raw_value).strip().upper()
    if not raw:
        raise StockCodeError("公司股票代码不能为空。")

    # Excel/CSV 常见的数值字符串，如 4.0。
    if re.fullmatch(r"\d+\.0+", raw):
        raw = raw.split(".", 1)[0]

    compact = re.sub(r"[\s._/\\-]+", "", raw)
    aliases = "|".join(sorted(_EXCHANGE_ALIASES, key=len, reverse=True))
    prefix = re.fullmatch(fr"({aliases})(\d{{1,6}})", compact)
    suffix = re.fullmatch(fr"(\d{{1,6}})({aliases})", compact)
    digits_only = re.fullmatch(r"\d{1,6}", compact)

    explicit_exchange = None
    if prefix:
        explicit_exchange = _EXCHANGE_ALIASES[prefix.group(1)]
        digits = prefix.group(2)
    elif suffix:
        digits = suffix.group(1)
        explicit_exchange = _EXCHANGE_ALIASES[suffix.group(2)]
    elif digits_only:
        digits = digits_only.group(0)
    else:
        raise StockCodeError(
            f"无法识别股票代码“{value}”。支持示例：000001、000001.SZ、SZ000001、600000.SH、920000.BJ。"
        )

    code6 = digits.zfill(6)
    inferred = infer_exchange(code6)
    if explicit_exchange and explicit_exchange != inferred:
        raise StockCodeError(
            f"股票代码与交易所后缀不一致：{code6} 应使用 .{inferred}，不是 .{explicit_exchange}。"
        )
    return f"{code6}.{inferred}"


def normalize_company_input(value, *, allow_name: bool = False) -> str:
    """代码走严格归一化；允许名称时，仅无数字文本可作为公司名称透传。"""
    raw_value = "" if value is None else str(value)
    raw = unicodedata.normalize("NFKC", raw_value).strip()
    try:
        return normalize_stock_code(raw)
    except StockCodeError:
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", raw))
        if allow_name and raw and (has_cjk or not any(character.isdigit() for character in raw)):
            return raw
        raise
