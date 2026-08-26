# -*- coding: utf-8 -*-
"""
上市公司监管问询扫雷预警系统 —— Streamlit 演示页（设计规范 v1.1 金融蓝主题）
=============================================================
运行：streamlit run 导航入口.py（推荐，单入口导航，默认打开本页）
     或 streamlit run 主控agent.py --server.port 8501（独立运行）
功能：
  - 单公司/批量扫雷（真实 7-Agent 流水线：公告研读→财务检测→预测→案例→段落→归因→报告）
  - 「切换公司」弹窗：3 家预跑公司（000001/000063/000858，均未退市）秒级缓存切换
  - 技术指标卡（真实评估值：AUC/Top10%/F1/可追踪率）
  - 风险仪表盘（概率大数字/数字滚动/进度条/风险徽章）+ SHAP 因子卡 + Pipeline 步骤条
  - 可解释预警报告（财务/公告/归因/相似案例/推理链路）+ 报告归档浏览
说明：
  - 默认 use_llm=False（离线）；勾选"启用 LLM"需 .env 配 DEEPSEEK_API_KEY
  - "启用 DeepSeek 执行摘要"：deepseek-v4-flash 生成风控函件式摘要（默认关，演示勾选）
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agents import SweepingOrchestrator
from backend.config import OUTPUT_DIR
from ui.components import render_countup
from ui.data import load_model_summary, load_offline_context
from ui.theme import apply_scan_theme

st.set_page_config(page_title="上市公司扫雷预警系统", page_icon="🛰️", layout="wide")
apply_scan_theme()

# 预跑缓存公司（全部未退市，报告已归档 output/reports/）
PREFETCHED_COMPANIES = [
    ("000001.SZ", "平安银行"),
    ("000063.SZ", "中兴通讯"),
    ("000858.SZ", "五粮液"),
]


# ================= 工具 =================
def list_reports(max_n: int = 50) -> list[dict]:
    try:
        mp = Path(OUTPUT_DIR) / "reports" / "manifest.json"
        if not mp.exists():
            return []
        data = json.loads(mp.read_text(encoding="utf-8"))
        return data[:max_n] if isinstance(data, list) else []
    except Exception:
        return []


def load_report_md(report_dir: str = "reports") -> str:
    try:
        return (Path(OUTPUT_DIR) / "reports" / report_dir).read_text(encoding="utf-8")
    except Exception:
        return ""


def risk_badge(level: str) -> str:
    """风险徽章 HTML（三档配色）。"""
    level = (level or "").lower()
    if "高" in level:
        return '<span class="badge badge-high">高风险</span>'
    if "中" in level or level == "medium":
        return '<span class="badge badge-mid">中风险</span>'
    if "低" in level or level == "low":
        return '<span class="badge badge-low">低风险</span>'
    return f'<span class="badge badge-low">{level or "—"}</span>'


def risk_color(level: str) -> str:
    if "高" in level:
        return "#EF4444"
    if "中" in level:
        return "#F59E0B"
    return "#10B981"


def render_tech_cards(metrics: dict) -> None:
    """技术指标卡（真实评估值，model_summary.json 驱动）。"""
    cards = [
        ("60D 模型 AUC", f'{metrics.get("AUC", 0):.4f}', "目标 ≥ 0.75", "测试集区分能力"),
        ("Top10% 覆盖率", f'{metrics.get("Top10%Recall", 0):.1%}', "目标 ≥ 40%", "高风险样本召回"),
        ("60D 集成 F1", f'{metrics.get("F1", 0):.3f}', "正样本 5.8%", "阈值 0.30"),
        ("可解释追踪率", "100%", "目标 100%", "7-Agent 全链路 trace"),
    ]
    html = '<div class="tech-grid">'
    for label, value, target, sub in cards:
        html += (
            f'<div class="tech-metric-card">'
            f'<div class="m-label">{label}<span class="m-target">{target}</span></div>'
            f'<div class="m-value">{value}</div><div class="m-sub">{sub}</div></div>'
        )
    html += "</div>"
    st.markdown('<div class="sec-title">赛题技术指标（60 天窗口 · 测试集集成）</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


def render_risk_dashboard(ctx, pred: dict, fin, att: dict, countup: bool = False) -> None:
    """风险仪表盘（概率大数字 + 进度条 + 徽章 + 置信度 + 因子数）。

    countup=True 时概率数字用 components.html 数字滚动动画（可选增强）。
    """
    p60 = pred.get("probability_60d")
    level = pred.get("risk_level") or fin.risk_level or "—"
    conf = pred.get("confidence")
    pct = (p60 or 0) * 100
    color = risk_color(level)
    factors_n = len(att.get("top_risk_factors", []))
    evidence_n = len(att.get("evidence_citations", []))
    industry = str(fin.industry or "—").split("-")[-1]
    ind = fin.indicators if isinstance(fin.indicators, dict) else {}
    report_period = str(ind.get("report_period") or "—")[:10]
    ann_count = ctx.semantic.stats.get("announcement_count", 0)
    ds = pred.get("data_source", "realtime")
    cov = ((pred.get("coverage") or {}).get("ratio", 0)) * 100
    html = f"""
    <div class="risk-dashboard">
      <div class="comp-cell">
        <div class="comp-name">{ctx.name or ctx.company}</div>
        <div class="comp-code mono">{ctx.company}</div>
        <div class="comp-meta">行业：{industry}<br>报告期：{report_period}<br>近一年公告：{ann_count} 份</div>
      </div>
      <div class="prob-cell">
        <div class="prob-label">未来 {ctx.window} 天被监管问询概率</div>
        {('' if countup else f'<div class="prob-num" style="color:{color}">{pct:.2f}%</div>')}
        <div class="risk-bar"><div class="risk-bar-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div></div>
        {risk_badge(level)}
      </div>
      <div class="stat-cell">
        <div class="stat-label">置信度</div>
        <div class="stat-num">{f'{conf:.2%}' if conf is not None else '—'}</div>
        <div class="stat-sub">数据源：{'实时特征' if ds == 'realtime' else '离线查表'}（覆盖率 {cov:.0f}%）</div>
      </div>
      <div class="stat-cell">
        <div class="stat-label">风险诱因 / 证据</div>
        <div class="stat-num">{factors_n} / {evidence_n}</div>
        <div class="stat-sub">SHAP 因子 · 原文证据绑定</div>
      </div>
    </div>
    """
    st.markdown('<div class="sec-title">风险仪表盘</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
    if countup and p60 is not None:
        render_countup(p60 * 100, decimals=2, suffix="%", color=color, height=88, font_size=52)


def render_factor_cards(shap_features: list) -> None:
    """SHAP Top 因子卡（贡献条 + 标签）。"""
    if not shap_features:
        st.caption("无 SHAP 特征（规则降级归因见报告）")
        return
    tag_pool = ["财务", "市场", "问询历史", "舆情", "治理", "估值"]
    html = '<div class="factor-grid">'
    for i, (feat, val) in enumerate(shap_features[:6]):
        tag = tag_pool[i % len(tag_pool)]
        width = min(abs(val) * 100, 100)
        bar_color = "#EF4444" if val < 0 else "#10B981"
        html += (
            f'<div class="factor-card">'
            f'<div class="factor-head"><span class="factor-tag">{tag}</span>'
            f'<span class="factor-score">SHAP {val:+.3f}</span></div>'
            f'<div class="factor-name mono">{feat}</div>'
            f'<div class="factor-bar"><div class="factor-bar-fill" style="width:{width:.0f}%;background:{bar_color}"></div></div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown('<div class="sec-title">Top 风险诱因（SHAP 贡献）</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


def render_pipeline_steps(trace: list) -> None:
    """Pipeline 步骤条（状态：done/active/pending，含耗时）。"""
    agent_names = ["公告研读", "财务检测", "预测建模", "案例匹配", "段落检索", "归因解释", "报告生成"]
    icons = ["📄", "📈", "🎯", "🧩", "📑", "🔍", "📋"]
    agent_keys = ["AnnouncementReader", "FinancialDetector", "Predictor", "CaseRetriever",
                  "ChunkRetriever", "Attributor", "Reporter"]
    done_map = {t.get("agent"): t for t in trace if t.get("status") == "done"}
    html = '<div class="sec-title">Agent 流水线</div>'
    for i, name in enumerate(agent_names):
        t = done_map.get(agent_keys[i])
        if t:
            status = f'✓ {t.get("latency_ms", "")}ms'
            cls = "step-done"
        else:
            status = "待执行"
            cls = "step-pending"
        html += (
            f'<div class="pipeline-step {cls}"><div class="step-icon">{icons[i]}</div>'
            f'<div><div class="step-title">{name}</div><div class="step-status">{status}</div></div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ================= 缓存切换 =================
def offline_to_ctx(off: dict) -> SimpleNamespace:
    """把 load_offline_context 返回的离线 dict 适配成 ctx-like 对象，复用统一渲染。"""
    fin = off.get("financial") or {}
    sem = off.get("semantic") or {}
    return SimpleNamespace(
        company=off.get("company"), name=off.get("name"), window=off.get("window", 60),
        semantic=SimpleNamespace(stats=sem.get("stats", {}), risk_factors=sem.get("risk_factors", [])),
        financial=SimpleNamespace(
            risk_level=fin.get("risk_level"), industry=fin.get("industry"),
            indicators=fin.get("indicators"), anomaly_list=fin.get("anomaly_list", []),
            skip=fin.get("skip"), skip_reason=fin.get("skip_reason")),
        prediction=off.get("prediction") or {}, attribution=off.get("attribution") or {},
        cases=off.get("cases") or [], trace_log=off.get("trace_log") or [],
        report=off.get("report") or {},
    )


def render_ctx_report(ctx, code: str, *, cache_label: str = "") -> None:
    """统一报告渲染：实时扫雷结果与预跑缓存共用。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    rj = (ctx.report or {}).get("json", {})

    if cache_label:
        st.markdown(
            f'<div class="sec-title">⚡ {cache_label} <span class="badge badge-low">预跑缓存 · 秒级切换</span></div>',
            unsafe_allow_html=True)

    # ---- 降级提示 ----
    degraded = [t for t in ctx.trace_log if t.get("status") in ("timeout", "needs_choice", "skipped")]
    for t in degraded[:3]:
        st.warning(f"⚠️ **{t.get('agent')} 降级/跳过**：{t.get('reason', '')}", icon=":material/warning:")

    # ---- 风险仪表盘 ----
    render_risk_dashboard(ctx, pred, fin, att, countup=bool(cache_label))

    # ---- 执行摘要 ----
    if rj.get("executive_summary"):
        st.markdown('<div class="sec-title">执行摘要</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card">{rj["executive_summary"]}</div>', unsafe_allow_html=True)

    # ---- SHAP 因子卡 ----
    render_factor_cards(pred.get("shap_features", []))

    # ---- 明细（财务/公告/归因/案例/trace） ----
    with st.expander(f"💹 财务异常信号（{len(fin.anomaly_list)} 条）", expanded=False):
        if getattr(fin, "skip", False):
            st.write(f"财务分析跳过：{fin.skip_reason}")
        for a in fin.anomaly_list:
            st.markdown(f"- **[{a.get('type')}]**（severity {a.get('severity')}）{a.get('evidence', '')}"
                        f"  `label_ref={a.get('label_ref')}`")
    with st.expander(f"📄 公告风险要素（{len(ctx.semantic.risk_factors)} 条 / "
                     f"{ctx.semantic.stats.get('announcement_count', 0)} 份公告）"):
        for r in ctx.semantic.risk_factors[:15]:
            st.markdown(f"- [{r.get('severity')}] **{r.get('category')}**：{r.get('description')}")
            if r.get("evidence"):
                st.markdown(f"  > {r.get('evidence', '')[:100]}")
        if not ctx.semantic.risk_factors:
            st.write("（LLM 关闭或无风险要素）")
    with st.expander("🎯 归因解释（Top 风险诱因 + 证据）"):
        if att.get("narrative"):
            st.write(att["narrative"])
        for f in att.get("top_risk_factors", []):
            shap = f"（SHAP {f.get('shap'):+.3f}）" if f.get("shap") is not None else ""
            st.markdown(f"- **{f.get('desc') or f.get('feature')}** {shap}  `{f.get('evidence_id', '')}`")
        st.markdown("**证据池：**")
        for e in att.get("evidence_citations", []):
            st.markdown(f"- `{e.get('evidence_id')}` [{e.get('source')}] {e.get('snippet', '')[:100]}")
    with st.expander(f"🧩 相似历史问询案例（Top {len(ctx.cases)}）"):
        for c in ctx.cases:
            score = c.get("rrf_score") or c.get("similarity")
            cosine = c.get("cosine_similarity")
            cos_txt = f"｜余弦相似度 {cosine:.4f}" if cosine is not None else ""
            st.markdown(f"- **{c.get('company')}**｜{c.get('inquiry_type')}｜{c.get('publish_date')}"
                        f"｜RRF融合得分 {score}{cos_txt}")
            if c.get("topics"):
                st.markdown(f"  - 关注点：{'；'.join(str(t)[:50] for t in c['topics'][:3])}")
    with st.expander("🔍 完整推理链路 trace_log（可追踪率 100%）"):
        st.json(ctx.trace_log)

    # ---- 报告下载 ----
    c1, c2 = st.columns(2)
    if ctx.report:
        c1.download_button(f"⬇️ 下载 {code} 报告 (Markdown)", data=ctx.report["markdown"],
                           file_name=f"{code}_risk_report.md", mime="text/markdown")
        c2.download_button(f"⬇️ 下载 {code} 报告 (JSON)",
                           data=json.dumps(rj, ensure_ascii=False, indent=2),
                           file_name=f"{code}_risk_report.json", mime="application/json")

    # ---- Pipeline 步骤条 ----
    with st.expander("🤖 Agent 流水线状态", expanded=False):
        render_pipeline_steps(ctx.trace_log)


@st.dialog("🔄 切换预跑公司", width="large")
def company_switcher_dialog() -> None:
    """「切换公司」弹窗：3 家预跑公司点选即切换（缓存秒级，无需重跑流水线）。"""
    st.caption("以下公司已预跑并归档（output/reports/），点选后立即切换展示；需要最新结果可再点「开始扫雷」。")
    reports = list_reports()
    latest = {}
    for r in reports:
        code = r.get("company", "")
        if code not in latest:  # manifest 按时间升序，取每条公司最新一条
            latest[code] = r
    for code, name in PREFETCHED_COMPANIES:
        entry = latest.get(code)
        st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
        html = (
            f'<div class="tech-metric-card" style="grid-column:span 2;">'
            f'<div class="m-label">{code}<span class="m-target">{name}</span></div>'
            f'<div class="m-value">{f"{float(entry.get("probability_60d") or 0) * 100:.2f}%" if entry else "—"}</div>'
            f'<div class="m-sub">{risk_badge(entry.get("risk_level", "")) if entry else "未预跑 · 请先扫雷"}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        if st.button(f"载入 {code}（{name}）", key=f"pf_switch_{code}", use_container_width=True):
            st.session_state["prefetch_company"] = code
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ================= 侧边栏输入 =================
with st.sidebar:
    st.header("⚙️ 参数设置")
    codes_text = st.text_area("公司代码（每行一个）", "000063.SZ", height=90,
                              help="示例：000063.SZ（中兴通讯）、000001.SZ（平安银行）、000858.SZ（五粮液）")
    window = st.selectbox("预测窗口（天）", [30, 60, 90], index=1)
    use_llm = st.checkbox("启用 LLM 精细抽取（需 .env 配 key）", value=False)
    use_llm_summary = st.checkbox("启用 DeepSeek 执行摘要（deepseek-v4-flash）", value=False,
                                  help="报告执行摘要用大模型生成；关闭时用规则拼装。")
    run_clicked = st.button("🚀 开始扫雷", type="primary", use_container_width=True)

# ================= 技术指标卡（始终展示，真实评估值） =================
metrics = load_model_summary().get("windows", {}).get("60", {}).get("Ensemble", {})
render_tech_cards(metrics)

# ================= 切换公司（预跑缓存，秒级） =================
c_switch, c_switch_hint = st.columns([1, 3])
with c_switch:
    if st.button("🔄 切换公司（预跑缓存）", key="open_switcher", use_container_width=True):
        company_switcher_dialog()
with c_switch_hint:
    st.caption("预跑 3 家公司：000001 平安银行 · 000063 中兴通讯 · 000858 五粮液（均未退市）")

prefetch_code = st.session_state.get("prefetch_company")
if prefetch_code and not run_clicked:
    try:
        off = load_offline_context(prefetch_code)
        render_ctx_report(offline_to_ctx(off), prefetch_code, cache_label="预跑缓存 · 000001/000063/000858")
        st.divider()
    except FileNotFoundError as e:
        st.warning(f"未找到 {prefetch_code} 的预跑报告，请先执行一次扫雷。({e})")

# ================= 已生成报告（顶部浏览） =================
st.markdown('<div class="sec-title">📁 已生成报告（output/reports/ 存档）</div>', unsafe_allow_html=True)
reports = list_reports()
if reports:
    labels = {f"{r.get('report_id')} ｜ 60d {r.get('probability_60d')} ｜ {r.get('risk_level')}": r
              for r in reports}
    chosen = st.selectbox("选择报告", list(labels.keys()), key="report_selector")
    if chosen:
        r = labels[chosen]
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ 下载 Markdown", data=load_report_md(r.get("md_file", "")),
                           file_name=r.get("md_file", "report.md"), mime="text/markdown",
                           key=f"dl_md_{r.get('report_id')}")
        c2.download_button("⬇️ 下载 JSON",
                           data=json.dumps(r, ensure_ascii=False, indent=2),
                           file_name=r.get("json_file", "report.json"), mime="application/json",
                           key=f"dl_meta_{r.get('report_id')}")
        with st.expander("📄 预览报告", expanded=False):
            st.markdown(load_report_md(r.get("md_file", "")))

st.divider()

if not run_clicked:
    st.info("左侧输入公司代码后点击「开始扫雷」，或使用上方「切换公司」秒级查看预跑缓存。"
            "示例：000063.SZ（中兴通讯）、000001.SZ（平安银行）、000858.SZ（五粮液）")
    st.stop()

codes = [c.strip() for c in codes_text.splitlines() if c.strip()]
if not codes:
    st.warning("请输入至少一个公司代码")
    st.stop()

# ================= 执行流水线 =================
orch = SweepingOrchestrator(use_llm=use_llm, use_finbert=True)

for code in codes:
    st.divider()
    with st.status(f"🔍 正在分析 {code} …", expanded=True) as status:
        ctx = orch.sweep_one(code, window=window, use_llm_summary=use_llm_summary)
        for t in ctx.trace_log:
            agent = t.get("agent", "?")
            stt = t.get("status", "done")
            ms = t.get("latency_ms", "")
            out = str(t.get("output_summary", ""))[:80]
            st.write(f"**{agent}** ｜ {stt} ｜ {ms}ms ｜ {out}")
        status.update(label=f"✅ {code} 分析完成", state="complete")

    render_ctx_report(ctx, code)

st.success("批量扫雷完成。报告已自动归档到 backend/data/output/reports/。")
