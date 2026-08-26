from __future__ import annotations

import altair as alt
import pandas as pd

from ui.theme import PALETTE


def probability_chart(prediction: dict) -> alt.Chart:
    rows = [{"窗口": f"{days} 天", "概率": prediction.get(f"probability_{days}d")} for days in (30, 60, 90)]
    frame = pd.DataFrame([row for row in rows if row["概率"] is not None])
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("窗口:N", sort=["30 天", "60 天", "90 天"], title=None),
        y=alt.Y("概率:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("窗口:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=None),
        tooltip=["窗口:N", alt.Tooltip("概率:Q", format=".2%")],
    ).properties(height=240)


def shap_chart(features: list, limit: int = 8) -> alt.Chart:
    rows = []
    for item in (features or [])[:limit]:
        name, value = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item.get("feature"), item.get("shap"))
        if name and value is not None:
            rows.append({"特征": str(name), "贡献": float(value), "方向": "推升风险" if float(value) >= 0 else "降低风险"})
    frame = pd.DataFrame(rows)
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("特征:N", sort="-x", title=None), x=alt.X("贡献:Q", title="SHAP 贡献"),
        color=alt.Color("方向:N", scale=alt.Scale(domain=["推升风险", "降低风险"], range=[PALETTE["coral"], PALETTE["blue"]]), legend=None),
        tooltip=["特征:N", "方向:N", alt.Tooltip("贡献:Q", format="+.5f")],
    ).properties(height=max(220, len(frame) * 28))


def case_score_chart(cases: list) -> alt.Chart:
    rows = []
    for item in (cases or [])[:5]:
        rows.append({
            "案例": f'{item.get("company", "")} · {item.get("publish_date", "")}',
            "得分": item.get("rrf_score", item.get("similarity")),
        })
    frame = pd.DataFrame(rows).dropna()
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(y=alt.Y("案例:N", sort="-x", title=None), x=alt.X("得分:Q", title="RRF 融合得分"), color=alt.value(PALETTE["teal"]), tooltip=["案例:N", alt.Tooltip("得分:Q", format=".6f")]).properties(height=230)


def risk_severity_chart(factors: list) -> alt.Chart:
    frame = pd.DataFrame([{"严重度": str(level), "要素数": sum(int(item.get("severity") or 0) == level for item in (factors or []))} for level in range(1, 6)])
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(x=alt.X("严重度:N", title="严重度"), y=alt.Y("要素数:Q", title="风险要素数"), color=alt.Color("严重度:N", legend=None), tooltip=["严重度:N", "要素数:Q"]).properties(height=230)


def model_performance_chart(summary: dict) -> alt.Chart:
    rows = []
    for window in (30, 60, 90):
        metrics = summary.get("windows", {}).get(str(window), {}).get("Ensemble", {})
        for key, label in (("AUC", "AUC"), ("F1", "F1"), ("Top10%Recall", "Top 10% 覆盖率")):
            if metrics.get(key) is not None:
                rows.append({"窗口": f"{window} 天", "指标": label, "数值": metrics[key]})
    frame = pd.DataFrame(rows)
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("窗口:N", sort=["30 天", "60 天", "90 天"], title=None),
        xOffset="指标:N", y=alt.Y("数值:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("指标:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=alt.Legend(orient="bottom")),
        tooltip=["窗口:N", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
    ).properties(height=245)


def ensemble_weights_chart(summary: dict) -> alt.Chart:
    labels = {"rf": "随机森林", "lgb": "LightGBM", "xgb": "XGBoost"}
    frame = pd.DataFrame([{"模型": labels.get(k, k), "权重": v} for k, v in summary.get("ensemble_weights", {}).items()])
    return alt.Chart(frame).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("模型:N", title=None), y=alt.Y("权重:Q", scale=alt.Scale(domain=[0, .5]), axis=alt.Axis(format="%")),
        color=alt.Color("模型:N", scale=alt.Scale(range=[PALETTE["blue"], PALETTE["teal"], PALETTE["gold"]]), legend=None),
        tooltip=["模型:N", alt.Tooltip("权重:Q", format=".0%")],
    ).properties(height=245)


def risk_ranking_chart(frame: pd.DataFrame) -> alt.Chart:
    data = frame.copy()
    data["公司"] = data["company_code"].astype(str)
    data["风险概率"] = pd.to_numeric(data["risk_probability"], errors="coerce")
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("公司:N", sort="-x", title=None), x=alt.X("风险概率:Q", axis=alt.Axis(format="%"), title="60 天风险概率"),
        color=alt.Color("风险概率:Q", scale=alt.Scale(range=[PALETTE["gold"], PALETTE["coral"]]), legend=None),
        tooltip=["公司:N", alt.Tooltip("风险概率:Q", format=".2%"), "risk_rank:Q", "risk_level:N"],
    ).properties(height=max(240, len(data) * 26))
