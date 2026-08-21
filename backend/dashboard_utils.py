"""Streamlit 展示层的数据整形工具。"""
from __future__ import annotations

from collections.abc import Mapping

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
