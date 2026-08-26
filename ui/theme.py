from __future__ import annotations

import streamlit as st

PALETTE = {
    "canvas": "#F3F0EB", "paper": "#FFFDFA", "ink": "#314456",
    "muted": "#71808D", "wine": "#8F1D2C", "blue": "#6F9FD8",
    "teal": "#70B8AA", "gold": "#E2B964", "coral": "#E58C82",
    "line": "#DFE5E8",
}

# ============ 设计规范 v1.1（金融蓝） ============
BRAND = {
    "primary": "#1E40AF", "secondary": "#3B82F6", "light": "#EFF6FF", "lighter": "#DBEAFE",
    "risk_low": "#10B981", "risk_low_bg": "#D1FAE5", "risk_low_tx": "#065F46",
    "risk_mid": "#F59E0B", "risk_mid_bg": "#FEF3C7", "risk_mid_tx": "#92400E",
    "risk_high": "#EF4444", "risk_high_bg": "#FEE2E2", "risk_high_tx": "#991B1B",
    "text": "#0F172A", "text2": "#64748B", "muted": "#94A3B8",
    "page": "#F1F5F9", "card": "#FFFFFF", "border": "#E2E8F0",
}


def _scan_css() -> str:
    """设计规范 v1.1 主题 CSS（金融蓝 + 风险三档 + 等宽数字 + 卡片/动效）。"""
    b = BRAND
    return f"""
    <style>
    :root {{
      --brand-primary:{b['primary']}; --brand-secondary:{b['secondary']};
      --brand-light:{b['light']}; --brand-lighter:{b['lighter']};
      --risk-low:{b['risk_low']}; --risk-low-bg:{b['risk_low_bg']}; --risk-low-tx:{b['risk_low_tx']};
      --risk-mid:{b['risk_mid']}; --risk-mid-bg:{b['risk_mid_bg']}; --risk-mid-tx:{b['risk_mid_tx']};
      --risk-high:{b['risk_high']}; --risk-high-bg:{b['risk_high_bg']}; --risk-high-tx:{b['risk_high_tx']};
      --text-primary:{b['text']}; --text-secondary:{b['text2']}; --text-muted:{b['muted']};
      --bg-page:{b['page']}; --bg-card:{b['card']}; --border:{b['border']};
      --shadow-lg:0 12px 24px -8px rgba(30,64,175,0.15); --radius-lg:12px; --radius-xl:16px;
    }}
    .stApp {{ background:linear-gradient(180deg,#FAFBFD 0%,var(--bg-page) 100%); color:var(--text-primary); }}
    .block-container {{ padding-top:1.2rem; padding-bottom:3rem; max-width:1400px; }}
    [data-testid="stSidebar"] {{ background:#fff; border-right:1px solid var(--border); }}
    [data-testid="stMetricValue"] {{ font-family:"SF Mono",Consolas,monospace; font-variant-numeric:tabular-nums; color:var(--text-primary); }}
    /* 按钮三级 */
    .stButton > button {{ border-radius:8px; font-size:13px; }}
    .stButton > button[kind="primary"] {{ background:var(--brand-primary); border-color:var(--brand-primary); }}
    .stButton > button[kind="primary"]:hover {{ background:var(--brand-secondary); }}
    .stButton > button:not([kind]) {{ background:#fff; color:var(--brand-secondary); border:1px solid var(--brand-secondary); }}
    .stButton > button:not([kind]):hover {{ background:var(--brand-light); }}
    /* 区块标题 */
    .sec-title {{ font-size:16px; font-weight:600; margin:18px 0 12px; display:flex; align-items:center; gap:8px; }}
    .sec-title::before {{ content:""; width:4px; height:16px; background:var(--brand-primary); border-radius:2px; }}
    /* 技术指标卡 */
    .tech-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
    .tech-metric-card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; transition:transform .3s cubic-bezier(.4,0,.2,1),box-shadow .3s; }}
    .tech-metric-card:hover {{ transform:translateY(-2px); box-shadow:var(--shadow-lg); }}
    .tech-metric-card .m-label {{ font-size:12px; color:var(--text-secondary); display:flex; justify-content:space-between; align-items:center; }}
    .tech-metric-card .m-target {{ background:var(--brand-light); color:var(--brand-primary); padding:1px 6px; border-radius:4px; font-size:11px; }}
    .tech-metric-card .m-value {{ font-size:24px; font-weight:700; font-family:"SF Mono",Consolas,monospace; margin-top:4px; }}
    .tech-metric-card .m-sub {{ font-size:11px; color:var(--text-muted); margin-top:2px; }}
    /* 风险仪表盘 */
    .risk-dashboard {{ background:linear-gradient(135deg,var(--brand-light) 0%,#fff 100%); border:1px solid var(--brand-lighter); border-radius:16px; padding:22px 24px; display:grid; grid-template-columns:1.1fr 1.3fr 0.9fr 0.9fr; gap:20px; margin-top:16px; }}
    .risk-dashboard .comp-name {{ font-size:17px; font-weight:700; }}
    .risk-dashboard .comp-code {{ color:var(--text-muted); font-size:12px; margin-top:2px; }}
    .risk-dashboard .comp-meta {{ color:var(--text-secondary); font-size:12px; margin-top:8px; line-height:1.8; }}
    .risk-dashboard .prob-label {{ font-size:12px; color:var(--text-secondary); }}
    .prob-num {{ font-size:44px; font-weight:700; font-family:"SF Mono",Consolas,monospace; line-height:1.1; margin-top:2px; }}
    .risk-bar {{ height:10px; background:#E2E8F0; border-radius:999px; overflow:hidden; margin-top:8px; }}
    .risk-bar-fill {{ height:100%; border-radius:999px; animation:scan-fill 1.5s cubic-bezier(.4,0,.2,1); }}
    @keyframes scan-fill {{ from{{width:0;}} }}
    .badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-top:8px; }}
    .badge::before {{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }}
    .badge-low {{ background:var(--risk-low-bg); color:var(--risk-low-tx); }}
    .badge-mid {{ background:var(--risk-mid-bg); color:var(--risk-mid-tx); }}
    .badge-high {{ background:var(--risk-high-bg); color:var(--risk-high-tx); }}
    .risk-dashboard .stat-label {{ font-size:12px; color:var(--text-secondary); }}
    .risk-dashboard .stat-num {{ font-size:24px; font-weight:600; font-family:"SF Mono",Consolas,monospace; margin-top:4px; }}
    .risk-dashboard .stat-sub {{ font-size:11px; color:var(--text-muted); margin-top:2px; }}
    /* Pipeline 步骤 */
    .pipeline-step {{ display:flex; gap:12px; align-items:flex-start; padding:10px 12px; border-radius:10px; position:relative; }}
    .pipeline-step::before {{ content:""; position:absolute; left:27px; top:44px; bottom:-6px; width:2px; background:var(--border); }}
    .pipeline-step:last-child::before {{ display:none; }}
    .step-icon {{ width:32px; height:32px; border-radius:50%; display:grid; place-items:center; font-size:14px; flex-shrink:0; }}
    .step-done .step-icon {{ background:#ECFDF5; color:var(--risk-low); }}
    .step-active .step-icon {{ background:var(--brand-primary); color:#fff; box-shadow:0 0 0 4px rgba(30,64,175,0.15); animation:scan-pulse 1.5s infinite; }}
    .step-pending .step-icon {{ background:var(--bg-page); color:var(--text-muted); }}
    @keyframes scan-pulse {{ 0%,100%{{box-shadow:0 0 0 4px rgba(30,64,175,0.15);}} 50%{{box-shadow:0 0 0 8px rgba(30,64,175,0.05);}} }}
    .step-title {{ font-size:13px; font-weight:500; }}
    .step-status {{ font-size:11px; color:var(--text-muted); }}
    /* 因子卡 */
    .factor-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:8px; }}
    .factor-card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:12px 14px; }}
    .factor-head {{ display:flex; justify-content:space-between; font-size:12px; }}
    .factor-tag {{ background:var(--brand-light); color:var(--brand-primary); padding:1px 8px; border-radius:6px; font-size:11px; }}
    .factor-name {{ font-weight:600; font-size:13px; margin-top:6px; }}
    .factor-desc {{ color:var(--text-secondary); font-size:12px; margin-top:3px; }}
    .factor-bar {{ height:6px; background:#E2E8F0; border-radius:999px; margin-top:8px; overflow:hidden; }}
    .factor-bar-fill {{ height:100%; border-radius:999px; }}
    .factor-score {{ font-family:"SF Mono",Consolas,monospace; font-size:12px; }}
    /* 摘要 / 报告卡 */
    .summary-card {{ background:#fff; border:1px solid var(--border); border-left:4px solid var(--brand-primary); border-radius:12px; padding:14px 16px; margin-top:16px; }}
    .report-card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; transition:transform .3s,box-shadow .3s,border-color .3s; margin-top:12px; }}
    .report-card:hover {{ transform:translateY(-4px); box-shadow:var(--shadow-lg); border-color:var(--brand-lighter); }}
    .report-card.featured {{ border:2px solid #BFDBFE; background:linear-gradient(180deg,#F8FAFF,#fff); }}
    .report-head {{ display:flex; justify-content:space-between; align-items:center; }}
    .report-title {{ font-weight:600; font-size:14px; }}
    .report-meta {{ color:var(--text-muted); font-size:11px; margin-top:4px; }}
    .mono {{ font-family:"SF Mono",Consolas,monospace; font-variant-numeric:tabular-nums; }}
    .status-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--risk-low); box-shadow:0 0 0 4px rgba(16,185,129,0.15); animation:scan-pulse 2s infinite; }}
    @media (max-width:1024px) {{ .tech-grid{{grid-template-columns:repeat(2,1fr);}} .risk-dashboard{{grid-template-columns:1fr 1fr;}} }}
    @media (max-width:768px) {{ .risk-dashboard{{grid-template-columns:1fr;}} .factor-grid{{grid-template-columns:1fr;}} }}
    </style>
    """


def apply_scan_theme() -> None:
    """设计规范 v1.1 金融蓝主题（扫雷预警系统）。"""
    st.html(_scan_css())


def apply_page_style() -> None:
    """统一风险报告风格，保持页面内容和业务组件不变。"""
    st.html(f"""
    <style>
    :root {{ --risk-canvas:{PALETTE['canvas']}; --risk-paper:{PALETTE['paper']};
      --risk-ink:{PALETTE['ink']}; --risk-muted:{PALETTE['muted']};
      --risk-wine:{PALETTE['wine']}; --risk-blue:{PALETTE['blue']};
      --risk-teal:{PALETTE['teal']}; --risk-gold:{PALETTE['gold']};
      --risk-coral:{PALETTE['coral']}; --risk-line:{PALETTE['line']}; }}
    .stApp {{ color:var(--risk-ink); }}
    .block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1500px; }}
    [data-testid="stSidebar"] {{ border-right:1px solid var(--risk-line); }}
    [data-testid="stMetricValue"] {{ color:var(--risk-ink); font-variant-numeric:tabular-nums; }}
    .risk-header {{ background:var(--risk-paper); border-top:5px solid var(--risk-wine);
      border-bottom:1px solid #E3DBD2; padding:1.25rem 1.35rem 1.1rem;
      margin:0 0 1rem; border-radius:0 0 10px 10px; }}
    .risk-eyebrow {{ color:var(--risk-wine); font-size:.72rem; font-weight:750;
      letter-spacing:.12em; text-transform:uppercase; }}
    .risk-title {{ color:var(--risk-ink); font-family:Georgia,"Noto Serif SC",serif;
      font-size:clamp(1.55rem,2.4vw,2.35rem); font-weight:700; line-height:1.28; margin:.3rem 0 .4rem; }}
    .risk-subtitle {{ color:var(--risk-muted); font-size:.92rem; line-height:1.7; }}
    .risk-meta {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }}
    .risk-chip {{ display:inline-flex; border:1px solid var(--risk-line); border-radius:999px;
      padding:.28rem .62rem; color:var(--risk-muted); background:#F8FAFB; font-size:.73rem; }}
    .risk-chip.offline {{ background:#FFF3D7; border-color:#EDD598; color:#76530D; font-weight:700; }}
    .risk-chip.live {{ background:#EAF5F0; border-color:#BBD9CB; color:#35645B; font-weight:700; }}
    .risk-chip.danger {{ background:#FBECE9; border-color:#E9BDB6; color:#9B3B36; font-weight:700; }}
    .risk-metric-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.75rem; margin:.4rem 0 1.1rem; }}
    .risk-metric-card {{ min-width:0; min-height:116px; background:#F7F9FA; border:1px solid var(--risk-line); border-top:3px solid var(--risk-blue); border-radius:9px; padding:.82rem .9rem; display:grid; grid-template-rows:1.7rem 2.35rem auto; }}
    .risk-metric-card:nth-child(2) {{ border-top-color:var(--risk-teal); }} .risk-metric-card:nth-child(3) {{ border-top-color:var(--risk-gold); }}
    .risk-metric-label,.risk-metric-note {{ color:var(--risk-muted); font-size:.76rem; line-height:1.35; }}
    .risk-metric-value {{ color:var(--risk-ink); font-size:1.55rem; font-weight:750; line-height:1; align-self:center; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .risk-metric-unit {{ color:var(--risk-muted); font-size:.72rem; margin-left:.25rem; }}
    .risk-evidence {{ background:var(--risk-paper); border:1px solid var(--risk-line); border-left:4px solid var(--risk-gold); border-radius:8px; padding:.85rem 1rem; margin:.5rem 0; }}
    .risk-evidence.high {{ border-left-color:var(--risk-coral); }} .risk-evidence-title {{ display:flex; justify-content:space-between; gap:1rem; font-weight:700; }}
    .risk-evidence-meta {{ color:var(--risk-muted); font-size:.72rem; margin:.24rem 0 .45rem; }} .risk-evidence-quote {{ background:#F8F5F1; padding:.55rem .7rem; border-radius:5px; line-height:1.65; font-family:Georgia,"Noto Serif SC",serif; }}
    .risk-links a {{ color:var(--risk-wine); text-decoration:none; font-size:.78rem; font-weight:700; margin-right:.8rem; }}
    .risk-trace {{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:.55rem; margin:.5rem 0; }} .risk-trace-step {{ border:1px solid var(--risk-line); border-radius:8px; padding:.65rem; background:#F7F9FA; min-width:0; }}
    .risk-trace-step b {{ display:block; font-size:.75rem; }} .risk-trace-step span {{ display:block; color:var(--risk-muted); font-size:.65rem; margin-top:.2rem; overflow-wrap:anywhere; }}
    .risk-dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%; background:var(--risk-teal); margin-right:.35rem; }} .risk-dot.error {{ background:var(--risk-coral); }} .risk-dot.skip {{ background:var(--risk-gold); }}
    .risk-source-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; }} .risk-source-card {{ display:block; border:1px solid var(--risk-line); border-radius:8px; padding:.78rem; color:var(--risk-ink)!important; text-decoration:none!important; background:#fff; }} .risk-source-card b,.risk-source-card span {{ display:block; }} .risk-source-card span {{ color:var(--risk-muted); font-size:.68rem; margin-top:.2rem; }}
    .risk-note {{ color:var(--risk-muted); font-size:.74rem; line-height:1.6; }}
    @media (max-width:1100px) {{ .risk-metric-grid {{ grid-template-columns:repeat(3,1fr); }} .risk-trace {{ grid-template-columns:repeat(4,1fr); }} }}
    @media (max-width:700px) {{ .block-container {{ padding-top:1rem; }} .risk-metric-grid {{ grid-template-columns:repeat(2,1fr); }} .risk-trace,.risk-source-grid {{ grid-template-columns:repeat(2,1fr); }} }}
    </style>
    """)
