"""Streamlit 展示层的数据整形工具。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date

from backend.skills.risk_labels import LABEL_NAMES, TAXONOMY


def risk_theme_name(code: str) -> str:
    """把风险主题编码转换为可读中文名，并保留未精分类状态。"""
    normalized = str(code or "").strip().upper()
    if not normalized:
        return "未标注主题"

    label_name = LABEL_NAMES.get(normalized)
    if label_name:
        return label_name

    if normalized.endswith("-CANDIDATE"):
        l1_code = normalized.split("-", 1)[0]
        l1_name = TAXONOMY.get("risk_themes", {}).get(l1_code, {}).get("name")
        if l1_name:
            return f"{l1_name}（待精分类）"

    return "未映射主题"


def risk_theme_distribution_rows(
    category_event_counts: Mapping | None,
    window_days: int,
) -> list[dict]:
    """从 30d/60d/90d 嵌套计数中生成已排序的展示行。"""
    window_counts = (category_event_counts or {}).get(f"{window_days}d", {})
    if not isinstance(window_counts, Mapping):
        return []

    parsed: list[tuple[str, int]] = []
    for raw_code, raw_count in window_counts.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            parsed.append((str(raw_code).strip().upper(), count))

    total = sum(count for _, count in parsed)
    rows = [
        {
            "主题代码": code,
            "风险主题": risk_theme_name(code),
            "图表标签": f"{risk_theme_name(code)} · {code}",
            "事件数": count,
            "占比": count / total if total else 0.0,
        }
        for code, count in parsed
    ]
    return sorted(rows, key=lambda row: (-row["事件数"], row["主题代码"]))


def risk_window_comparison_rows(scalar_features: Mapping | None) -> list[dict]:
    """生成 30/60/90 天公告量与风险事件数量对比。"""
    scalar = scalar_features or {}
    rows = []
    for days in (30, 60, 90):
        announcement_count = int(scalar.get(f"announcement_count_{days}d", 0) or 0)
        risk_count = int(scalar.get(f"risk_event_count_{days}d", 0) or 0)
        high_risk_count = int(scalar.get(f"high_risk_event_count_{days}d", 0) or 0)
        rows.append(
            {
                "时间窗口": f"最近 {days} 天",
                "公告总数": announcement_count,
                "风险事件": risk_count,
                "高风险事件": high_risk_count,
                "每份公告风险事件": (
                    risk_count / announcement_count if announcement_count else 0.0
                ),
            }
        )
    return rows


def _parse_date(value) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _month_keys(as_of: str, periods: int = 12) -> list[str]:
    cutoff = _parse_date(as_of)
    if cutoff is None:
        return []
    absolute_month = cutoff.year * 12 + cutoff.month - 1
    output = []
    for offset in range(periods - 1, -1, -1):
        value = absolute_month - offset
        year, month_index = divmod(value, 12)
        output.append(f"{year:04d}-{month_index + 1:02d}")
    return output


def risk_interval_comparison_rows(
    announcements: list[Mapping] | None,
    risk_factors: list[Mapping] | None,
    as_of: str,
) -> list[dict]:
    """生成互不重叠的最近 90 天三段统计，避免累计窗口天然递增。"""
    cutoff = _parse_date(as_of)
    if cutoff is None:
        return []
    intervals = (
        (0, 30, "最近 1–30 天"),
        (30, 60, "此前 31–60 天"),
        (60, 90, "此前 61–90 天"),
    )
    rows = []
    for lower, upper, label in intervals:
        interval_announcements = []
        for item in announcements or []:
            published = _parse_date(item.get("date"))
            if published is None:
                continue
            delta = (cutoff - published).days
            if lower <= delta < upper:
                interval_announcements.append(item)

        interval_factors = []
        for item in risk_factors or []:
            published = _parse_date(item.get("announcement_date"))
            if published is None:
                continue
            delta = (cutoff - published).days
            if lower <= delta < upper and item.get("evidence_valid", True):
                interval_factors.append(item)
        event_keys = {
            str(item.get("event_key") or item.get("risk_id") or index)
            for index, item in enumerate(interval_factors)
        }
        high_keys = {
            str(item.get("event_key") or item.get("risk_id") or index)
            for index, item in enumerate(interval_factors)
            if int(item.get("severity") or 1) >= 4
        }
        announcement_count = len(interval_announcements)
        rows.append(
            {
                "时间区间": label,
                "公告总数": announcement_count,
                "风险事件": len(event_keys),
                "高风险事件": len(high_keys),
                "中低风险事件": len(event_keys - high_keys),
                "风险事件/公告": (
                    len(event_keys) / announcement_count if announcement_count else 0.0
                ),
            }
        )
    return rows


def risk_monthly_severity_rows(
    risk_factors: list[Mapping] | None,
    as_of: str,
) -> list[dict]:
    """按最近 12 个自然月汇总去重风险事件，并拆分高/中/低严重度。"""
    months = _month_keys(as_of, 12)
    if not months:
        return []
    buckets = {
        month: {"高风险": set(), "中风险": set(), "低风险": set()}
        for month in months
    }
    for index, item in enumerate(risk_factors or []):
        published = _parse_date(item.get("announcement_date"))
        if published is None:
            continue
        month = published.strftime("%Y-%m")
        if month not in buckets or not item.get("evidence_valid", True):
            continue
        severity = int(item.get("severity") or 1)
        level = "高风险" if severity >= 4 else "中风险" if severity == 3 else "低风险"
        event_key = str(item.get("event_key") or item.get("risk_id") or index)
        buckets[month][level].add(event_key)
    return [
        {
            "月份": month,
            "高风险": len(buckets[month]["高风险"]),
            "中风险": len(buckets[month]["中风险"]),
            "低风险": len(buckets[month]["低风险"]),
            "事件总数": sum(len(values) for values in buckets[month].values()),
        }
        for month in months
    ]


def risk_theme_heatmap_rows(
    risk_factors: list[Mapping] | None,
    as_of: str,
    max_themes: int = 8,
) -> list[dict]:
    """生成最近 12 月 Top-N 风险主题矩阵；缺失组合显式补零。"""
    months = _month_keys(as_of, 12)
    if not months:
        return []
    event_rows = []
    seen = set()
    for index, item in enumerate(risk_factors or []):
        published = _parse_date(item.get("announcement_date"))
        if published is None or published.strftime("%Y-%m") not in months:
            continue
        if not item.get("evidence_valid", True):
            continue
        code = str(item.get("taxonomy_l2") or item.get("label") or "OTHER").upper()
        event_key = str(item.get("event_key") or item.get("risk_id") or index)
        unique_key = (published.strftime("%Y-%m"), code, event_key)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        event_rows.append(unique_key)
    theme_totals = Counter(code for _, code, _ in event_rows)
    top_codes = [code for code, _ in theme_totals.most_common(max(1, int(max_themes)))]
    if not top_codes:
        return []
    counts = Counter((month, code) for month, code, _ in event_rows if code in top_codes)
    output = []
    for rank, code in enumerate(top_codes):
        theme_label = f"{risk_theme_name(code)} · {code}"
        for month in months:
            output.append(
                {
                    "月份": month,
                    "主题代码": code,
                    "风险主题": theme_label,
                    "事件数": counts[(month, code)],
                    "主题全年合计": theme_totals[code],
                    "主题排序": rank,
                }
            )
    return output
