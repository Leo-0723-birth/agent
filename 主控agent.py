# -*- coding: utf-8 -*-
"""
上市公司监管问询扫雷预警系统 —— Streamlit 演示页（设计规范 v1.1 金融蓝主题）
=============================================================
布局对齐 docs/ui_preview_主控页.html：
  - 顶栏：品牌 + Agent 在线状态 + 「⇄ 切换公司」按钮（弹窗含参数设置）
  - 左栏：Agent 流水线（7 步状态）
  - 右栏：技术指标卡 → 风险仪表盘 → 执行摘要 → SHAP 因子卡 → 风险报告
功能：
  - 单公司/批量扫雷（真实 7-Agent 流水线）
  - 「切换公司」弹窗：3 家预跑公司缓存秒级切换 + 运行参数（窗口/LLM/BGE/摘要）
  - 技术指标卡（真实评估值：AUC 0.8312 / Top10% 46.3% / F1 0.337 / 可追踪率 100%）
  - 风险仪表盘（概率大数字/数字滚动/进度条/风险徽章）+ SHAP 因子卡 + Pipeline 步骤条
说明：
  - 默认 use_llm=False（离线）；勾选"启用 LLM"需 .env 配 DEEPSEEK_API_KEY
"""
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agents import SweepingOrchestrator
from backend.config import OUTPUT_DIR
from ui.data import load_model_summary, load_offline_context
from ui.theme import apply_scan_theme

st.set_page_config(page_title="上市公司扫雷预警系统", page_icon="🛰️", layout="wide")
apply_scan_theme()

# 主控页：隐藏 Streamlit 原生页面导航侧栏，使顶栏真正通栏、
# 形成「单一流水线侧栏」布局（对齐设计稿 ui_preview_主控页.html）。
# 页面跳转改由侧栏内的 st.page_link 承接，原生侧栏在其他 Agent 页会恢复。
st.markdown(
    "<style>[data-testid='stSidebar']{display:none !important;}</style>",
    unsafe_allow_html=True,
)

# 预跑缓存公司（全部未退市，报告已归档 output/reports/）
PREFETCHED_COMPANIES = [
    ("000001.SZ", "平安银行"),
    ("000063.SZ", "中兴通讯"),
    ("000858.SZ", "五粮液"),
]
AGENT_STEPS = [
    ("公告研读", "📄"), ("财务检测", "📈"), ("预测建模", "🎯"), ("案例匹配", "🧩"),
    ("段落检索", "📑"), ("归因解释", "🔍"), ("报告生成", "📋"),
]
AGENT_KEYS = ["AnnouncementReader", "FinancialDetector", "Predictor", "CaseRetriever",
              "ChunkRetriever", "Attributor", "Reporter"]


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


# ================= 顶栏 / 左栏 =================
def render_header(agent_online: int = 7) -> None:
    """顶栏：通栏吸顶 60px —— 品牌 + Agent 在线状态 + 切换公司按钮（对齐设计稿 .app-header）。"""
    h_l, h_r = st.columns([5, 1], vertical_alignment="center", gap="small")
    with h_l:
        st.html(
            '<div class="app-header"><div class="header-left">'
            '<div class="logo">警</div>'
            '<div class="brand-name"><span>监管问询扫雷预警系统</span></div></div>'
            f'<div class="header-right"><div class="agent-status">'
            f'<span class="status-dot"></span>{agent_online} 个 Agent 在线</div></div></div>'
        )
    with h_r:
        if st.button("⇄ 切换公司", key="open_switcher"):
            company_switcher_dialog()


def render_sidebar_pipeline(trace: list) -> None:
    """左栏：Agent 流水线 7 步状态（对齐设计稿 .sidebar）。"""
    done_map = {t.get("agent"): t for t in trace or [] if t.get("status") == "done"}
    html = '<div class="section-title" style="margin-top:0">Agent 流水线</div>'
    for i, (name, icon) in enumerate(AGENT_STEPS):
        t = done_map.get(AGENT_KEYS[i])
        if t:
            detail = t.get("output_summary") or ""
            status = f'✓ {detail[:24] or f"{t.get("latency_ms", "")}ms"}'
            cls = "step-done"
        else:
            status = "待执行"
            cls = "step-pending"
        html += (
            f'<div class="pipeline-step {cls}"><div class="step-icon">{icon}</div>'
            f'<div><div class="step-title">{name}</div><div class="step-status">{status}</div></div></div>'
        )
    st.html(f'<aside class="app-sidebar">{html}</aside>')

    # 页面导航（隐藏原生侧栏后，这里承接页面跳转，对齐设计稿「单一侧栏」）
    st.markdown('<div class="sidebar-nav-title">页面导航</div>', unsafe_allow_html=True)
    _nav = [
        ("主控 Agent", "主控agent.py", ":material/dashboard:"),
        ("公告研读 Agent", "公告研读agent.py", ":material/article:"),
        ("财务异常 Agent", "财务异常agent.py", ":material/account_balance:"),
        ("预测建模 Agent", "预测建模agent.py", ":material/model_training:"),
        ("案例匹配 Agent", "案例匹配agent.py", ":material/compare_arrows:"),
        ("归因分析 Agent", "归因分析agent.py", ":material/psychology:"),
        ("报告生成 Agent", "报告生成agent.py", ":material/description:"),
    ]
    for _label, _script, _icon in _nav:
        st.page_link(_script, label=_label, icon=_icon)


# ================= 技术指标 / 仪表盘 / 因子 =================
def render_tech_cards(metrics: dict) -> None:
    cards = [
        ("AUC", f'{metrics.get("AUC", 0):.4f}', "目标 ≥ 0.75", "RF+LGB+XGB 集成 · 60d", ""),
        ("Top10% 覆盖率", f'{metrics.get("Top10%Recall", 0):.1%}', "目标 ≥ 40%", "60d 问询召回", "unit"),
        ("F1-Score", f'{metrics.get("F1", 0):.3f}', "正样本 5.8%", "60d 集成 · 阈值 0.30", ""),
        ("可解释追踪率", "100%", "目标 100%", "7-Agent 全链路 trace", "unit"),
    ]
    html = '<div class="tech-grid">'
    for label, value, target, sub, _u in cards:
        html += (
            f'<div class="tech-metric-card"><div class="m-label">{label}'
            f'<span class="m-target">{target}</span></div>'
            f'<div class="m-value">{value}</div><div class="m-sub">{sub}</div></div>'
        )
    html += "</div>"
    st.markdown('<div class="sec-title">赛题技术指标（60 天窗口 · 测试集集成）</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


def render_risk_dashboard(ctx, pred: dict, fin, att: dict, countup: bool = False) -> None:
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
        <div class="comp-meta">行业：{industry}<br>最新报告期：{report_period}<br>近一年公告：{ann_count} 份</div>
      </div>
      <div class="prob-cell">
        <div class="prob-label">未来 {ctx.window} 天被监管问询概率</div>
        <div class="prob-num" style="color:{color}">{pct:.2f}%</div>
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
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def render_factor_cards(shap_features: list) -> None:
    if not shap_features:
        st.caption("无 SHAP 特征（规则降级归因见报告）")
        return
    tag_pool = ["市场异动", "问询历史", "问询历史", "舆情", "市值", "财务"]
    name_pool = {
        "mkt_volume_ratio_20d": "20 日量比",
        "f6_last_inquiry_interval_days": "最近问询间隔",
        "f6_inquiry_count_60m": "60 月问询次数",
        "sent_guba_negative_ratio_30d": "股吧负面占比 30d",
        "mkt_market_cap": "总市值",
        "mkt_log_market_cap": "总市值（对数）",
    }
    desc_pool = {
        "mkt_volume_ratio_20d": "当日成交量 / 前 19 日均量异常",
        "f6_last_inquiry_interval_days": "距最近一次监管问询天数",
        "f6_inquiry_count_60m": "历史监管问询频次",
        "sent_guba_negative_ratio_30d": "近 30 天负面舆情比例",
        "mkt_market_cap": "当前总市值水平",
        "mkt_log_market_cap": "当前总市值水平（对数）",
    }
    html = '<div class="factor-grid">'
    for i, (feat, val) in enumerate(shap_features[:6]):
        tag = tag_pool[i % len(tag_pool)]
        width = min(abs(val) * 100, 100)
        mag = abs(val)
        if mag > 0.3:
            bar_color = "#EF4444"
        elif mag > 0.15:
            bar_color = "#F59E0B"
        else:
            bar_color = "#10B981"
        desc = desc_pool.get(feat, "SHAP 特征贡献")
        html += (
            f'<div class="factor-card"><div class="factor-head">'
            f'<span class="factor-tag">{tag}</span>'
            f'<span class="factor-score">SHAP {val:+.3f}</span></div>'
            f'<div class="factor-name">{name_pool.get(feat, feat)}</div>'
            f'<div class="factor-desc">{desc}</div>'
            f'<div class="factor-bar"><div class="factor-bar-fill" style="width:{width:.0f}%;background:{bar_color}"></div></div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown('<div class="sec-title">Top 风险诱因（SHAP 贡献）</div>', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


# ================= 缓存切换 =================
def offline_to_ctx(off: dict) -> SimpleNamespace:
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
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    rj = (ctx.report or {}).get("json", {})

    if cache_label:
        st.markdown(
            f'<div class="sec-title">⚡ {cache_label} <span class="badge badge-low">预跑缓存 · 秒级切换</span></div>',
            unsafe_allow_html=True)

    degraded = [t for t in ctx.trace_log if t.get("status") in ("timeout", "needs_choice", "skipped")]
    for t in degraded[:3]:
        st.warning(f"⚠️ **{t.get('agent')} 降级/跳过**：{t.get('reason', '')}", icon=":material/warning:")

    render_risk_dashboard(ctx, pred, fin, att, countup=bool(cache_label))

    if rj.get("executive_summary"):
        st.markdown('<div class="sec-title">执行摘要</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card">{rj["executive_summary"]}</div>', unsafe_allow_html=True)

    render_factor_cards(pred.get("shap_features", []))

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

    # 底部操作行（对齐设计稿：⬇ 下载 Markdown / ⬇ 导出 JSON / 🔍 查看推理链路）
    if ctx.report:
        md = ctx.report.get("markdown") or ""
        rj = ctx.report.get("json") or {}
        render_action_row(
            md, json.dumps(rj, ensure_ascii=False, indent=2),
            f"{code}_risk_report.md", f"{code}_risk_report.json",
            trace=ctx.trace_log or [], key=code,
        )


def render_action_row(md: str, data_json: str, md_name: str, json_name: str,
                      trace: list, key: str) -> None:
    """报告底部操作行（对齐设计稿）：⬇ 下载 Markdown（主）/ ⬇ 导出 JSON（次）/ 🔍 查看推理链路（幽灵）。
    采用自包含 HTML（data-URI 下载 + <details> 展开 trace），避免 :has 选择器误匹配外层容器。"""
    import base64
    md_b64 = base64.b64encode((md or "").encode("utf-8")).decode()
    js_b64 = base64.b64encode((data_json or "").encode("utf-8")).decode()
    trace_txt = json.dumps(trace or [], ensure_ascii=False, indent=2)
    trace_html = trace_txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <div class="action-row">
      <a class="act-btn act-primary" href="data:text/markdown;base64,{md_b64}" download="{md_name}">⬇ 下载 Markdown</a>
      <a class="act-btn act-secondary" href="data:application/json;base64,{js_b64}" download="{json_name}">⬇ 导出 JSON</a>
      <details class="act-details">
        <summary class="act-btn act-ghost">🔍 查看推理链路</summary>
        <pre class="trace-pre">{trace_html}</pre>
      </details>
    </div>
    """
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def render_report_cards(current_code: str = "") -> None:
    """风险报告卡（对齐设计稿：featured + 普通卡，卡内右侧「预览」按钮）。"""
    st.markdown('<div class="sec-title">风险报告</div>', unsafe_allow_html=True)
    reports = list_reports()
    if not reports:
        st.caption("暂无已生成报告。扫雷完成后自动归档。")
        return
    # 去重：每公司仅保留最新一份（manifest 可能累积多轮扫雷的重复归档）
    _latest = {}
    for r in reports:
        c = r.get("company")
        if c not in _latest or str(r.get("generated_at", "")) > str(_latest[c].get("generated_at", "")):
            _latest[c] = r
    ordered = sorted(_latest.values(), key=lambda r: str(r.get("generated_at", "")), reverse=True)
    current = next((r for r in ordered if r.get("company") == current_code), ordered[0]) if current_code else ordered[0]
    others = [r for r in ordered if r.get("report_id") != current.get("report_id")][:2]
    for r in [current] + others:
        featured = " featured" if r == current else ""
        rid = r.get("report_id") or r.get("company")
        card_html = (
            f'<div class="report-card{featured}">'
            f'<div class="report-title">{r.get("name") or r.get("company")} · 风险提示函 '
            f'{risk_badge(r.get("risk_level", ""))}</div>'
            f'<div class="report-meta mono">{r.get("company")} · {str(r.get("generated_at", ""))[:10]} · 八章风控函件式</div>'
            f'</div>'
        )
        c_card, c_btn = st.columns([4, 1], vertical_alignment="center", gap="small")
        with c_card:
            st.markdown(card_html, unsafe_allow_html=True)
        with c_btn:
            if st.button("预览", key=f"preview_{rid}"):
                st.session_state[f"show_preview_{rid}"] = not st.session_state.get(f"show_preview_{rid}", False)
        if st.session_state.get(f"show_preview_{rid}"):
            with st.expander(f"📄 {r.get('name') or r.get('company')} · 报告预览", expanded=True):
                st.markdown(load_report_md(r.get("md_file", "")))


# ================= 切换公司弹窗（含参数设置） =================
@st.dialog("🎯 选择目标公司", width="large")
def company_switcher_dialog() -> None:
    """「切换公司」弹窗：预跑 3 家快捷切换 + 运行参数（对齐设计稿 modal + 参数抽屉）。"""
    st.caption("预跑缓存秒级切换；也可以直接输入代码执行新扫雷。参数设置已合并到此弹窗。")

    # --- 预跑快捷切换 ---
    reports = list_reports()
    latest = {}
    for r in reports:
        code = r.get("company", "")
        if code not in latest:
            latest[code] = r
    st.markdown("**⚡ 预跑公司（缓存秒级切换）**")
    for code, name in PREFETCHED_COMPANIES:
        entry = latest.get(code)
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.markdown(f"**{code}** {name}")
        c2.markdown(f'<div style="text-align:right">{f"{float(entry.get("probability_60d") or 0) * 100:.2f}%" if entry else "—"}</div>',
                    unsafe_allow_html=True)
        c3.markdown(risk_badge(entry.get("risk_level", "")) if entry else "未预跑", unsafe_allow_html=True)
        if st.button(f"载入 {code}", key=f"pf_switch_{code}", use_container_width=True):
            st.session_state["prefetch_company"] = code
            st.session_state["sweep_code"] = code
            st.rerun()
        st.divider()

    # --- 参数设置（原侧边栏移入） ---
    st.markdown("**⚙️ 运行参数**")
    code_input = st.text_input("公司代码（单选）", value=st.session_state.get("sweep_code", "000063.SZ"),
                               placeholder="例如：000063.SZ、中兴通讯")
    batch_input = st.text_area("批量扫雷名单（每行一个）", value=st.session_state.get("sweep_batch", ""),
                               height=70, placeholder="多个代码每行一个，留空则扫雷上方单选代码")
    window = st.selectbox("预测窗口（天）", [30, 60, 90],
                          index=[30, 60, 90].index(st.session_state.get("sweep_window", 60)))
    use_llm = st.checkbox("启用 LLM 精细抽取（需 .env 配 key）", value=st.session_state.get("sweep_llm", False))
    use_semantic = st.checkbox("BGE 语义检索（案例匹配）", value=st.session_state.get("sweep_semantic", True))
    use_llm_summary = st.checkbox("DeepSeek 执行摘要（deepseek-v4-flash）",
                                  value=st.session_state.get("sweep_summary", False))
    c_ok, c_run = st.columns(2)
    with c_ok:
        if st.button("✓ 确认切换", key="dlg_confirm", type="primary", use_container_width=True):
            st.session_state["sweep_code"] = code_input.strip()
            st.session_state["sweep_batch"] = batch_input
            st.session_state["sweep_window"] = window
            st.session_state["sweep_llm"] = use_llm
            st.session_state["sweep_semantic"] = use_semantic
            st.session_state["sweep_summary"] = use_llm_summary
            st.session_state["prefetch_company"] = code_input.strip()
            st.session_state["run_clicked"] = False
            st.rerun()
    with c_run:
        if st.button("🚀 开始扫雷", key="dlg_run", use_container_width=True):
            st.session_state["sweep_code"] = code_input.strip()
            st.session_state["sweep_batch"] = batch_input
            st.session_state["sweep_window"] = window
            st.session_state["sweep_llm"] = use_llm
            st.session_state["sweep_semantic"] = use_semantic
            st.session_state["sweep_summary"] = use_llm_summary
            st.session_state["run_clicked"] = True
            st.rerun()


# ================= 页面主体 =================
render_header()

# 读取运行参数（默认值）
run_clicked = st.session_state.get("run_clicked", False)
window = st.session_state.get("sweep_window", 60)
use_llm = st.session_state.get("sweep_llm", False)
use_semantic = st.session_state.get("sweep_semantic", True)
use_llm_summary = st.session_state.get("sweep_summary", False)
sweep_code = st.session_state.get("sweep_code", "000063.SZ")
sweep_batch = st.session_state.get("sweep_batch", "")

# 预跑缓存展示（未点开始扫雷时，默认中兴通讯 —— 对齐设计稿样例）
prefetch_code = st.session_state.get("prefetch_company") or "000063.SZ"
prefetch_off = None
if not run_clicked:
    try:
        prefetch_off = load_offline_context(prefetch_code)
    except Exception:
        prefetch_off = None

# ===== 两栏布局（设计稿 .layout：左 ~260px Agent 流水线 + 右主内容） =====
left_col, right_col = st.columns([1, 4], gap="medium")
with left_col:
    if prefetch_off is not None:
        render_sidebar_pipeline(prefetch_off.get("trace_log", []))
    else:
        render_sidebar_pipeline([])

with right_col:
    # 技术指标卡（真实评估值，始终展示）
    metrics = load_model_summary().get("windows", {}).get("60", {}).get("Ensemble", {})
    render_tech_cards(metrics)

    # 预跑缓存报告 + 风险报告卡（含底部操作行）
    if prefetch_off is not None:
        render_ctx_report(offline_to_ctx(prefetch_off), prefetch_code,
                          cache_label="预跑缓存 · 000001/000063/000858")
        render_report_cards(current_code=prefetch_code)

    st.divider()

    if not run_clicked:
        st.info("点右上角「⇄ 切换公司」选择预跑缓存或输入公司代码；或直接点弹窗内「🚀 开始扫雷」执行实时流水线。"
                "示例：000063.SZ（中兴通讯）、000001.SZ（平安银行）、000858.SZ（五粮液）")
        st.stop()

    # ===== 执行流水线 =====
    codes = [c.strip() for c in sweep_batch.splitlines() if c.strip()] or [sweep_code]
    if not codes or not codes[0]:
        st.warning("请输入至少一个公司代码")
        st.stop()

    orch = SweepingOrchestrator(use_llm=use_llm, use_finbert=True, use_semantic_cases=use_semantic)

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
        render_report_cards(current_code=code)

    st.success("批量扫雷完成。报告已自动归档到 backend/data/output/reports/。")
