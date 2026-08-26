#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测建模 Agent 的可审计 Streamlit 展示入口。
运行：streamlit run 预测建模agent.py --server.port 8504
"""
from __future__ import annotations

import html
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.predictor import PredictorAgent
from backend.context import Context
from backend.skills.stock_code import normalize_stock_code
from ui.charts import ensemble_weights_chart, model_performance_chart, risk_ranking_chart
from ui.data import load_model_summary, load_risk_ranking
from ui.theme import apply_page_style
from ui.session import active_as_of, active_company, hydrate_page_state, shared_context_caption

st.set_page_config(
    page_title="预测建模 Agent",
    page_icon=":material/model_training:",
    layout="wide",
)
apply_page_style()


@st.cache_data(ttl="6h", max_entries=20, show_spinner=False)
def analyze_company(company: str, as_of: str, window: int) -> dict:
    """查表推理：建模数据集最新一行特征 → RF/LGB/XGB 三模型集成 → 概率 + SHAP。"""
    agent = PredictorAgent(horizons=["30d", "60d", "90d"])
    ctx = Context(company=company, as_of=as_of, window=window)
    agent.run(company, ctx)
    payload = asdict(ctx)
    return payload


# ---------------------------------------------------------------
# 展示层：主题感知配色 + 卡片 / 图表渲染
# ---------------------------------------------------------------
def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = (h or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return (1.0, 1.0, 1.0)


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    def _f(ch: float) -> float:
        return ch / 12.92 if ch <= 0.03928 else ((ch + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _f(r) + 0.7152 * _f(g) + 0.0722 * _f(b)


def _tint(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"


def _theme() -> dict:
    """解析 Streamlit 当前主题，返回语义化配色（浅/深两套，取自 dataviz 参考色板）。"""
    try:
        bg = st.get_option("theme.backgroundColor") or "#ffffff"
        card = st.get_option("theme.secondaryBackgroundColor") or "#f0f2f6"
        ink = st.get_option("theme.textColor") or "#0b0b0b"
    except Exception:
        bg, card, ink = "#ffffff", "#f0f2f6", "#0b0b0b"
    dark = _relative_luminance(_hex_to_rgb(bg)) < 0.5
    if dark:
        return {
            "dark": True,
            "pos": "#e29c9c",  # 正向贡献（推升问询概率）→ 柔和红（降饱和）
            "neg": "#8ab1df",  # 负向贡献（降低问询概率）→ 柔和蓝（降饱和）
            "bar": "#3987e5",  # 概率量条 / 主窗口强调边（顺序色，单蓝）
            "risk_low": "#0ca30c", "risk_mid": "#ec835a",
            "risk_high": "#e66767", "risk_none": "#898781",
            "ink": ink, "ink2": "#c3c2b7", "muted": "#898781",
            "card_bg": card, "grid": "#2c2c2a", "border": "rgba(255,255,255,0.12)",
            "axis_grid": "rgba(255,255,255,0.07)",
        }
    return {
        "dark": False,
        "pos": "#E58C82", "neg": "#6F9FD8", "bar": "#70B8AA",
        "risk_low": "#4F9587", "risk_mid": "#B88A32",
        "risk_high": "#D76055", "risk_none": "#71808D",
        "ink": ink, "ink2": "#52514e", "muted": "#898781",
        "card_bg": card, "grid": "#e1e0d9", "border": "rgba(11,11,11,0.12)",
        "axis_grid": "rgba(11,11,11,0.06)",
    }


_RISK_META = {
    "高": ("🔴", "risk_high"),
    "中": ("🟠", "risk_mid"),
    "低": ("🟢", "risk_low"),
    "未预测": ("⚪", "risk_none"),
}


# ---------------------------------------------------------------
# SHAP 特征中文业务释义（hover 提示用）
# 结构化特征按 F2-F6 前缀家族逐条解释；F1 语义特征运行时从描述表读取。
# ---------------------------------------------------------------
_FEATURE_CATEGORY = {
    "mkt": "市场特征",
    "sent": "舆情特征",
    "gov": "治理结构",
    "f2": "财务异常",
    "f6": "问询历史",
    "governance": "治理数据",
}

_FEATURE_HINTS = {
    # F2 财务异常
    "f2_trend_deterioration": "财务指标趋势恶化（如 ROE 连续下滑）",
    "f2_benford_flag": "财务数据本福特定律检验异常",
    "f2_p_roa": "ROA 异常的统计显著性（造假/异常概率）",
    "f2_industry_outlier_count": "相对行业偏离的异常指标数量",
    "f2_p_cfo": "经营现金流异常的统计显著性",
    "f2_neg_accruals_flag": "存在异常负应计项目",
    "f2_loss_flag": "当期是否亏损",
    "f2_profit_ocf_diverge": "净利润与经营现金流背离程度",
    "f2_ocf_to_profit_extreme": "现金流/净利润处于极端区间",
    "f2_high_debt_flag": "是否高资产负债率",
    "f2_neg_pe_flag": "市盈率为负（亏损）",
    # F6 监管问询历史
    "f6_last_inquiry_interval_days": "距上次被问询的天数",
    "f6_first_inquiry_interval_days": "距首次被问询的天数",
    "f6_inquiry_count_12m": "近 12 个月被问询次数",
    "f6_inquiry_count_60m": "近 60 个月被问询次数",
    "f6_attention_letter_count": "收到关注函的数量",
    "f6_avg_inquiry_interval_days": "历次被问询的平均间隔天数",
    "f6_inquiry_count_24m": "近 24 个月被问询次数",
    "f6_annual_report_inquiry_count": "年报被问询次数",
    "f6_inquiry_interval_cv": "被问询间隔的波动程度",
    "f6_unreplied_count": "尚未回复的问询函数量",
    # F5 治理结构
    "gov_top10_holder_ratio": "前十大股东合计持股比例",
    "gov_independent_director_ratio": "独立董事占比",
    "gov_top1_top2_gap": "第一大与第二大股东持股差距",
    "gov_top1_top10_ratio": "第一大股东占前十大股东比例",
    "gov_top1_holder_ratio": "第一大股东持股比例",
    "gov_board_size": "董事会规模",
    "gov_independent_director_count": "独立董事人数",
    "gov_nonstandard_audit_opinion": "是否被出具非标准审计意见",
    "gov_auditor_change": "是否更换会计师事务所",
    "gov_big4_auditor": "是否由四大会计师事务所审计",
    "gov_top10_holder_count": "前十大股东人数",
    "governance_year": "治理数据的报告年份",
    # F3 市场特征
    "mkt_pe_ratio": "市盈率（估值水平）",
    "mkt_max_drawdown_60d": "近 60 日最大回撤",
    "mkt_pb_industry_zscore": "市净率相对行业的偏离（Z 值）",
    "mkt_volume_ratio_20d": "近 20 日量比",
    "mkt_institutional_holding_ratio": "机构持股比例",
    "mkt_volume_cv_20d": "近 20 日成交量波动",
    "mkt_institutional_holding_change": "机构持股变动",
    "mkt_return_5d": "近 5 日收益率",
    "mkt_pe_change_qoq": "市盈率环比变动",
    "mkt_pb_ratio": "市净率",
    "mkt_return_20d": "近 20 日收益率",
    "mkt_excess_return_20d": "近 20 日超额收益（相对大盘）",
    "mkt_volatility_60d": "近 60 日股价波动率",
    "mkt_cap_industry_zscore": "总市值相对行业的偏离（Z 值）",
    "mkt_amihud_illiquidity_20d": "近 20 日非流动性（Amihud 指标）",
    "mkt_return_60d": "近 60 日收益率",
    "mkt_volatility_20d": "近 20 日股价波动率",
    "mkt_market_cap": "总市值",
    "mkt_pb_change_qoq": "市净率环比变动",
    "mkt_days_since_last_risk_warning": "距上次风险警示公告的天数",
    "mkt_institutional_holder_count": "机构持股家数",
    "mkt_market_cap_qoq": "总市值环比变动",
    "mkt_financing_balance_change": "融资余额变动",
    "mkt_financing_balance": "融资余额",
    "mkt_securities_balance": "融券余额",
    "mkt_extreme_down_days_20d": "近 20 日极端下跌天数",
    "mkt_abnormal_volume_days_20d": "近 20 日异常放量天数",
    "mkt_log_market_cap": "总市值（对数）",
    "mkt_risk_warning_count_90d": "近 90 日风险警示公告次数",
    "mkt_risk_warning_count_30d": "近 30 日风险警示公告次数",
    # F4 舆情特征
    "sent_read_count_30d": "近 30 日公告阅读量",
    "sent_guba_negative_ratio_30d": "近 30 日股吧负面情绪占比",
    "sent_guba_positive_ratio_30d": "近 30 日股吧正面情绪占比",
    "sent_guba_sentiment_volatility_30d": "近 30 日股吧情绪波动",
    "sent_post_count_5d": "近 5 日发帖量",
    "sent_post_daily_peak_30d": "近 30 日单日发帖峰值",
    "sent_guba_sentiment_mean_30d": "近 30 日股吧情绪均值",
    "sent_sentiment_mean_30d": "近 30 日情绪均值",
    "sent_sentiment_volatility_30d": "近 30 日情绪波动",
    "sent_post_count_30d": "近 30 日发帖量",
    "sent_negative_ratio_30d": "近 30 日负面情绪占比",
    "sent_news_title_count_30d": "近 30 日新闻标题数量",
    "sent_news_count_30d": "近 30 日新闻数量",
    "sent_news_count_10d": "近 10 日新闻数量",
    "sent_news_count_5d": "近 5 日新闻数量",
    "sent_news_daily_peak_30d": "近 30 日单日新闻峰值",
    "sent_negative_news_count_30d": "近 30 日负面新闻数量",
    "sent_negative_peak_30d": "近 30 日负面情绪峰值",
}


@st.cache_data(show_spinner=False)
def _semantic_hints() -> dict:
    """F1 语义特征 business_name（semantic_NNN → 中文业务含义）。"""
    path = (PROJECT_ROOT / "backend" / "data" / "modeling" / "raw" / "f1_selection"
            / "semantic_feature_descriptions_原版.csv")
    out: dict = {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, r in df.iterrows():
            out[str(r["feature_name"]).strip()] = str(r.get("business_name", "")).strip()
    except Exception:
        pass
    return out


def _feature_category(name: str) -> str:
    if name.startswith("regulatory_inquiry_semantic_"):
        return "监管问询语义(F1)"
    return _FEATURE_CATEGORY.get(name.split("_")[0], "其他")


def _feature_hint(name: str) -> str:
    if name.startswith("regulatory_inquiry_semantic_"):
        key = name.replace("regulatory_inquiry_semantic_", "semantic_")
        hint = _semantic_hints().get(key)
        return f"语义对比：{hint}" if hint else "监管问询语义主成分特征"
    return _FEATURE_HINTS.get(name, "结构化特征")


def _inject_styles(c: dict) -> None:
    st.markdown(
        f"""
        <style>
        .pv-row {{ display:flex; flex-wrap:wrap; gap:12px; }}
        .pv-card {{
            flex:1 1 150px; min-width:150px; box-sizing:border-box;
            border:1px solid {c['border']}; border-radius:12px;
            padding:14px 16px; background:{c['card_bg']};
        }}
        .pv-label {{ font-size:12px; color:{c['muted']}; margin-bottom:8px; letter-spacing:0.02em; }}
        .pv-label .tag {{ font-size:10px; color:{c['ink2']}; border:1px solid {c['border']};
                          border-radius:6px; padding:1px 5px; margin-left:6px; }}
        .pv-value {{ font-size:24px; font-weight:650; line-height:1.25; color:{c['ink']};
                     font-variant-numeric:tabular-nums; }}
        .pv-badge {{ display:inline-flex; align-items:center; gap:6px; padding:5px 12px;
                     border-radius:999px; font-size:16px; font-weight:650; line-height:1.4; }}
        .pv-table {{ border-collapse:collapse; width:100%; margin-top:6px; }}
        .pv-table th, .pv-table td {{ text-align:left; padding:6px 10px; border-bottom:1px solid {c['grid']}; }}
        .pv-table th {{ color:{c['muted']}; font-size:12px; font-weight:600; }}
        .pv-table td.rank {{ color:{c['muted']}; width:32px; }}
        .pv-table td.feat {{ color:{c['ink']}; }}
        .pv-table td.val {{ font-variant-numeric:tabular-nums; font-weight:600; text-align:right; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pct(v, nd: int = 2) -> str:
    return f"{v * 100:.{nd}f}%" if v is not None else "—"


def _metric_card(c: dict, label: str, value: str, tag: str | None = None, accent: bool = False) -> str:
    style = f' style="border-left:4px solid {c["bar"]};"' if accent else ""
    tag_html = f'<span class="tag">{html.escape(tag)}</span>' if tag else ""
    return (f'<div class="pv-card"{style}>'
            f'<div class="pv-label">{html.escape(label)}{tag_html}</div>'
            f'<div class="pv-value">{value}</div></div>')


def _render_probability(pred: dict, c: dict) -> None:
    _inject_styles(c)

    p30 = pred.get("probability_30d")
    p60 = pred.get("probability_60d")
    p90 = pred.get("probability_90d")
    risk = pred.get("risk_level") or "未预测"
    conf = pred.get("confidence")

    cards = [
        _metric_card(c, "30 天问询概率", _pct(p30)),
        _metric_card(c, "60 天问询概率", _pct(p60), tag="主窗口", accent=True),
        _metric_card(c, "90 天问询概率", _pct(p90)),
    ]

    icon, key = _RISK_META.get(risk, _RISK_META["未预测"])
    color = c[key]
    badge = (f'<span class="pv-badge" style="color:{color};'
             f'background:{_tint(color, 0.16 if c["dark"] else 0.12)};'
             f'border:1px solid {_tint(color, 0.5)};">{icon} {html.escape(risk)}</span>')
    cards.append(
        f'<div class="pv-card" style="border-left:4px solid {color};">'
        f'<div class="pv-label">风险等级</div><div class="pv-value">{badge}</div></div>'
    )

    cards.append(_metric_card(c, "模型置信度", _pct(conf, 1)))

    st.markdown(f'<div class="pv-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_shap(shap, c: dict) -> None:
    df = pd.DataFrame(shap, columns=["特征", "贡献值"])
    df["贡献值"] = df["贡献值"].astype(float)
    # 按贡献绝对值升序 → Altair 纵轴首个在最下，绝对值最大的特征落在最上
    df = (df.assign(_abs=df["贡献值"].abs())
            .sort_values("_abs", ascending=True)
            .reset_index(drop=True))
    df["方向"] = df["贡献值"].map(lambda v: "正向贡献" if v >= 0 else "负向贡献")
    df["类别"] = df["特征"].map(_feature_category)
    df["释义"] = df["特征"].map(_feature_hint)

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("特征:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=300, labelColor=c["muted"], titleColor=c["muted"]),
                    scale=alt.Scale(paddingInner=0.45)),
            x=alt.X("贡献值:Q",
                    title="SHAP 贡献值（正向推升 ↑ / 负向降低 ↓）",
                    axis=alt.Axis(gridColor=c["axis_grid"], labelColor=c["muted"], titleColor=c["muted"])),
            color=alt.Color(
                "方向:N",
                scale=alt.Scale(domain=["正向贡献", "负向贡献"], range=[c["pos"], c["neg"]]),
                legend=alt.Legend(title=None, orient="top", labelColor=c["muted"]),
            ),
            tooltip=[
                alt.Tooltip("特征:N", title="特征"),
                alt.Tooltip("类别:N", title="类别"),
                alt.Tooltip("释义:N", title="业务释义"),
                alt.Tooltip("贡献值:Q", title="贡献值", format="+.5f"),
                alt.Tooltip("方向:N", title="方向"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color=c["muted"], strokeDash=[3, 3])
        .encode(x=alt.X("zero:Q", title=None))
    )
    chart = (bars + rule).properties(height=max(160, 30 * len(df) + 50))
    st.altair_chart(chart, width="stretch")

    # 明细表：按绝对值降序 + 正负着色
    rows = []
    n = len(df)
    for rank, idx in enumerate(range(n - 1, -1, -1), start=1):
        r = df.iloc[idx]
        v = float(r["贡献值"])
        color = c["pos"] if v >= 0 else c["neg"]
        arrow = "↑" if v >= 0 else "↓"
        rows.append(
            f'<tr><td class="rank">{rank}</td>'
            f'<td class="feat">{html.escape(str(r["特征"]))}</td>'
            f'<td class="val" style="color:{color};">{arrow} {v:+.5f}</td></tr>'
        )
    table = (
        '<table class="pv-table"><thead><tr><th>#</th><th>特征</th><th>贡献值</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    st.markdown(table, unsafe_allow_html=True)


st.title("预测建模 Agent")
st.caption("按 (company, as_of) 从建模数据集取最新一行特征 → 三模型集成（RF+LightGBM+XGBoost）→ 30/60/90 天问询概率 + SHAP 归因。")
if shared_context_caption():
    st.info(shared_context_caption(), icon=":material/sync:")

with st.sidebar:
    st.subheader("运行设置")
    as_of_value = st.date_input("特征锚点 T", value=active_as_of(), max_value=date.today(),
                                help="取该日前最新一期特征（建模数据集的最近报告期）进行推理。")
    st.caption("公司需在建模数据集 backend/data/modeling/processed_dataset.csv 内。")
    st.caption("端口约定：8504（独立运行：streamlit run 预测建模agent.py --server.port 8504）")

with st.form("company_query_form", border=True):
    company_input = st.text_input(
        "公司代码（带交易所后缀）",
        value=active_company(),
        placeholder="例如：000004.SZ",
    )
    submitted = st.form_submit_button("开始预测", type="primary", icon=":material/search:")

if submitted:
    normalized = company_input.strip()
    if not normalized:
        st.error("请输入公司代码。", icon=":material/error:")
    else:
        try:
            normalized = normalize_stock_code(normalized)
            with st.status("正在查表并执行三模型集成推理……", expanded=True) as status:
                result = analyze_company(normalized, as_of_value.isoformat(), 60)
                st.session_state["prediction_analysis"] = result
                status.update(label="预测完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("prediction_analysis", None)
            st.error(f"预测失败：{type(exc).__name__}: {exc}", icon=":material/error:")

model_summary = load_model_summary()
st.subheader("模型质量与业务解释")
quality_left, quality_right = st.columns(2)
with quality_left:
    with st.container(border=True):
        st.markdown("#### 三个预测窗口的模型质量")
        st.altair_chart(model_performance_chart(model_summary), width="stretch")
        st.caption("雾蓝表示 AUC，青绿表示 F1，柔金表示 Top 10% 召回；悬停可查看精确数值。")
with quality_right:
    with st.container(border=True):
        st.markdown("#### 三模型集成权重")
        st.altair_chart(ensemble_weights_chart(model_summary), width="stretch")
        st.caption("柱顶直接标注权重；颜色只区分模型，不表示风险高低。")

with st.container(border=True):
    st.markdown("#### 60 天批量公司风险排名 · Top 10")
    st.altair_chart(risk_ranking_chart(load_risk_ranking(60, 10)), width="stretch")
    st.caption("柔金表示需要关注，珊瑚红表示更高风险；横轴与悬停均保留概率数值。")

result = hydrate_page_state("prediction_analysis")
if result:
    pred = result.get("prediction", {})
    trace = result.get("trace_log", [])
    c = _theme()

    st.subheader(f"问询概率（T={pred.get('feature_anchor') or '—'} 特征锚点）")
    _render_probability(pred, c)

    if pred.get("reason"):
        st.warning(pred["reason"], icon=":material/warning:")

    shap = pred.get("shap_features", []) or []
    if shap:
        st.subheader("SHAP 归因 Top 特征（集成贡献代理）")
        _render_shap(shap, c)
    else:
        st.info("无 SHAP 特征（模型未加载或公司不在建模数据集内）。")

    with st.expander("🔍 审计追踪"):
        st.json(trace, expanded=False)

    with st.expander("查看完整预测结果"):
        st.json(pred, expanded=False)
else:
    with st.container(border=True):
        st.subheader("页面会展示什么")
        st.write("30/60/90 天监管问询概率、风险等级、置信度、SHAP 特征贡献图与推理追踪。")
        st.caption("先使用默认示例 000004.SZ，点击“开始预测”即可。若显示“未找到该股票特征”，说明该公司不在建模数据集内。")
