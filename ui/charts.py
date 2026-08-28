from __future__ import annotations

from collections import Counter

import altair as alt
import pandas as pd

from ui.theme import PALETTE


def probability_chart(prediction: dict) -> alt.Chart:
    rows = []
    for window in (30, 60, 90):
        value = prediction.get(f"probability_{window}d")
        if value is not None:
            rows.append({"预测窗口": f"{window} 天", "问询概率": float(value), "顺序": window})
    frame = pd.DataFrame(rows)
    bars = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("预测窗口:N", sort=["30 天", "60 天", "90 天"], title=None),
        y=alt.Y("问询概率:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("预测窗口:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=None),
        tooltip=["预测窗口:N", alt.Tooltip("问询概率:Q", format=".2%")],
    )
    labels = alt.Chart(frame).mark_text(dy=-9, fontWeight=700, color=PALETTE["ink"]).encode(
        x=alt.X("预测窗口:N", sort=["30 天", "60 天", "90 天"]),
        y="问询概率:Q",
        text=alt.Text("问询概率:Q", format=".2%"),
    )
    return (bars + labels).properties(height=240)


def shap_chart(shap_features: list, limit: int = 10) -> alt.Chart:
    rows = []
    for item in (shap_features or [])[:limit]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name, value = item[0], item[1]
        elif isinstance(item, dict):
            name = item.get("feature") or item.get("name")
            value = item.get("shap") if item.get("shap") is not None else item.get("value")
        else:
            continue
        if name and value is not None:
            rows.append({"特征": str(name), "SHAP 贡献": float(value), "方向": "推升风险" if float(value) >= 0 else "降低风险"})
    frame = pd.DataFrame(rows)
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("特征:N", sort="-x", title=None),
        x=alt.X("SHAP 贡献:Q", title="SHAP 贡献值"),
        color=alt.Color("方向:N", scale=alt.Scale(domain=["推升风险", "降低风险"], range=[PALETTE["coral"], PALETTE["blue"]]), legend=alt.Legend(orient="bottom")),
        tooltip=["特征:N", "方向:N", alt.Tooltip("SHAP 贡献:Q", format="+.5f")],
    ).properties(height=max(220, len(frame) * 30))


def model_performance_chart(summary: dict) -> alt.Chart:
    rows = []
    for window in (30, 60, 90):
        metrics = summary.get("windows", {}).get(str(window), {}).get("Ensemble", {})
        rows.extend(
            [
                {"窗口": f"{window} 天", "指标": "AUC", "数值": metrics.get("AUC")},
                {"窗口": f"{window} 天", "指标": "Top 10% 召回", "数值": metrics.get("Top10%Recall")},
                {"窗口": f"{window} 天", "指标": "F1", "数值": metrics.get("F1")},
            ]
        )
    frame = pd.DataFrame(rows).dropna()
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("窗口:N", sort=["30 天", "60 天", "90 天"], title=None),
        xOffset="指标:N",
        y=alt.Y("数值:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        color=alt.Color("指标:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=alt.Legend(orient="bottom")),
        tooltip=["窗口:N", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
    ).properties(height=245)


def ensemble_weights_chart(summary: dict) -> alt.Chart:
    labels = {"rf": "随机森林", "lgb": "LightGBM", "xgb": "XGBoost"}
    frame = pd.DataFrame(
        [{"模型": labels.get(name, name), "集成权重": value} for name, value in summary.get("ensemble_weights", {}).items()]
    )
    bars = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("模型:N", title=None),
        y=alt.Y("集成权重:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.5])),
        color=alt.Color("模型:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=None),
        tooltip=["模型:N", alt.Tooltip("集成权重:Q", format=".0%")],
    )
    labels_chart = alt.Chart(frame).mark_text(dy=-9, color=PALETTE["ink"], fontWeight=700).encode(
        x="模型:N", y="集成权重:Q", text=alt.Text("集成权重:Q", format=".0%")
    )
    return (bars + labels_chart).properties(height=245)


def risk_ranking_chart(frame: pd.DataFrame) -> alt.Chart:
    data = frame.copy()
    data["公司"] = data["company_code"].astype(str)
    data["风险概率"] = pd.to_numeric(data["risk_probability"], errors="coerce")
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("公司:N", sort="-x", title=None),
        x=alt.X("风险概率:Q", axis=alt.Axis(format="%"), title="60 天风险概率"),
        color=alt.Color("风险概率:Q", scale=alt.Scale(range=[PALETTE["gold"], PALETTE["coral"]]), legend=None),
        tooltip=["公司:N", alt.Tooltip("风险概率:Q", format=".2%"), "risk_rank:Q", "risk_level:N"],
    ).properties(height=max(240, len(data) * 26))


def case_score_chart(cases: list[dict]) -> alt.Chart:
    rows = [
        {
            "案例": f"{item.get('company', '')} · {item.get('publish_date', '')}",
            "RRF 融合得分": item.get("rrf_score", item.get("similarity")),
            "问询类型": item.get("inquiry_type", ""),
        }
        for item in (cases or [])[:5]
    ]
    frame = pd.DataFrame(rows).dropna(subset=["RRF 融合得分"])
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("案例:N", sort="-x", title=None),
        x=alt.X("RRF 融合得分:Q", title="RRF 融合得分"),
        color=alt.value(PALETTE["teal"]),
        tooltip=["案例:N", "问询类型:N", alt.Tooltip("RRF 融合得分:Q", format=".6f")],
    ).properties(height=230)


def risk_severity_chart(factors: list[dict]) -> alt.Chart:
    counts = Counter(int(item.get("severity") or 0) for item in (factors or []))
    frame = pd.DataFrame(
        [
            {
                "严重度": str(level),
                "风险等级": "高风险" if level >= 4 else ("中风险" if level >= 3 else "低风险"),
                "风险要素数": count,
            }
            for level, count in sorted(counts.items())
        ]
    )
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("严重度:N", title="严重度等级"),
        y=alt.Y("风险要素数:Q", title="风险要素数"),
        color=alt.Color(
            "风险等级:N",
            scale=alt.Scale(
                domain=["高风险", "中风险", "低风险"],
                range=[PALETTE["coral"], PALETTE["gold"], PALETTE["blue"]],
            ),
            legend=None,
        ),
        tooltip=["风险等级:N", "严重度:N", "风险要素数:Q"],
    ).properties(height=230)

