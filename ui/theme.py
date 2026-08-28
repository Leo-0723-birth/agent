from __future__ import annotations

import streamlit as st


PALETTE = {
    "canvas": "#F3F0EB",
    "paper": "#FFFDFA",
    "ink": "#314456",
    "muted": "#71808D",
    "wine": "#8F1D2C",
    "blue": "#6F9FD8",
    "teal": "#70B8AA",
    "gold": "#E2B964",
    "coral": "#E58C82",
    "line": "#DFE5E8",
}


def apply_page_style() -> None:
    """Apply the small amount of CSS needed for the approved report-like UI."""
    st.html(
        f"""
        <style>
        :root {{
          --risk-canvas: {PALETTE['canvas']};
          --risk-paper: {PALETTE['paper']};
          --risk-ink: {PALETTE['ink']};
          --risk-muted: {PALETTE['muted']};
          --risk-wine: {PALETTE['wine']};
          --risk-blue: {PALETTE['blue']};
          --risk-teal: {PALETTE['teal']};
          --risk-gold: {PALETTE['gold']};
          --risk-coral: {PALETTE['coral']};
          --risk-line: {PALETTE['line']};
        }}
        .stApp {{ color: var(--risk-ink); }}
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}
        [data-testid="stSidebar"] {{ border-right: 1px solid var(--risk-line); }}
        [data-testid="stMetric"] {{ min-width: 0; }}
        [data-testid="stMetricValue"] {{
          color: var(--risk-ink); font-variant-numeric: tabular-nums lining-nums;
          letter-spacing: -0.025em;
        }}
        .risk-header {{
          background: var(--risk-paper); border-top: 5px solid var(--risk-wine);
          border-bottom: 1px solid #E3DBD2; padding: 1.25rem 1.35rem 1.1rem;
          margin: 0 0 1rem; border-radius: 0 0 10px 10px;
        }}
        .risk-eyebrow {{ color: var(--risk-wine); font-size: .72rem; font-weight: 750;
          letter-spacing: .12em; text-transform: uppercase; }}
        .risk-title {{ color: var(--risk-ink); font-family: Georgia, "Noto Serif SC", serif;
          font-size: clamp(1.55rem, 2.4vw, 2.35rem); font-weight: 700;
          line-height: 1.28; margin: .3rem 0 .4rem; }}
        .risk-subtitle {{ color: var(--risk-muted); font-size: .92rem; line-height: 1.7; }}
        .risk-meta {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }}
        .risk-chip {{ display:inline-flex; align-items:center; border:1px solid var(--risk-line);
          border-radius:999px; padding:.28rem .62rem; color:var(--risk-muted);
          background:#F8FAFB; font-size:.73rem; }}
        .risk-chip.offline {{ background:#FFF3D7; border-color:#EDD598; color:#76530D; font-weight:700; }}
        .risk-chip.live {{ background:#EAF5F0; border-color:#BBD9CB; color:#35645B; font-weight:700; }}
        .risk-chip.danger {{ background:#FBECE9; border-color:#E9BDB6; color:#9B3B36; font-weight:700; }}
        .risk-metric-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:.75rem; margin:.4rem 0 1.1rem; }}
        .risk-metric-card {{ min-width:0; min-height:116px; box-sizing:border-box;
          background:#F7F9FA; border:1px solid var(--risk-line); border-top:3px solid var(--risk-blue);
          border-radius:9px; padding:.82rem .9rem; display:grid;
          grid-template-rows:1.7rem 2.35rem auto; }}
        .risk-metric-card:nth-child(2) {{ border-top-color:var(--risk-teal); }}
        .risk-metric-card:nth-child(3) {{ border-top-color:var(--risk-gold); }}
        .risk-metric-card:nth-child(4) {{ border-top-color:var(--risk-coral); }}
        .risk-metric-label {{ color:var(--risk-muted); font-size:.76rem; line-height:1.35; }}
        .risk-metric-value {{ color:var(--risk-ink); font-size:1.55rem; font-weight:750;
          line-height:1; align-self:center; font-variant-numeric:tabular-nums lining-nums; white-space:nowrap; }}
        .risk-metric-unit {{ color:var(--risk-muted); font-size:.72rem; font-weight:500; margin-left:.25rem; }}
        .risk-metric-note {{ color:#8A949C; font-size:.67rem; line-height:1.4; align-self:end; }}
        .risk-evidence {{ background:var(--risk-paper); border:1px solid var(--risk-line);
          border-left:4px solid var(--risk-gold); border-radius:8px; padding:.85rem 1rem; margin:.5rem 0; }}
        .risk-evidence.high {{ border-left-color:var(--risk-coral); }}
        .risk-evidence-title {{ display:flex; justify-content:space-between; gap:1rem; font-weight:700; }}
        .risk-evidence-meta {{ color:var(--risk-muted); font-size:.72rem; margin:.24rem 0 .45rem; }}
        .risk-evidence-quote {{ background:#F8F5F1; padding:.55rem .7rem; border-radius:5px;
          line-height:1.65; font-family:Georgia,"Noto Serif SC",serif; }}
        .risk-evidence-quote mark {{ background:#FFE4AA; color:#6D3E00; padding:0 .12rem; }}
        .risk-evidence-issue {{ color:#56636E; font-size:.82rem; margin-top:.48rem; }}
        .risk-links {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.55rem; }}
        .risk-links a {{ color:var(--risk-wine); text-decoration:none; border-bottom:1px dotted var(--risk-wine); font-size:.78rem; font-weight:700; }}
        .risk-links a:hover {{ border-bottom-style:solid; }}
        .risk-trace {{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:.55rem; margin:.5rem 0; }}
        .risk-trace-step {{ border:1px solid var(--risk-line); border-radius:8px; padding:.65rem;
          background:#F7F9FA; min-width:0; }}
        .risk-trace-step b {{ display:block; font-size:.75rem; }}
        .risk-trace-step span {{ display:block; color:var(--risk-muted); font-size:.65rem; margin-top:.2rem; overflow-wrap:anywhere; }}
        .risk-dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%; background:var(--risk-teal); margin-right:.35rem; }}
        .risk-dot.error {{ background:var(--risk-coral); }} .risk-dot.skip {{ background:var(--risk-gold); }}
        .risk-source-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; }}
        .risk-source-card {{ display:block; border:1px solid var(--risk-line); border-radius:8px;
          padding:.78rem; color:var(--risk-ink)!important; text-decoration:none!important; background:#fff; }}
        .risk-source-card b {{ display:block; font-size:.82rem; }}
        .risk-source-card span {{ display:block; color:var(--risk-muted); font-size:.68rem; margin-top:.2rem; }}
        .risk-note {{ color:var(--risk-muted); font-size:.74rem; line-height:1.6; }}
        .risk-color-legend {{ display:flex; flex-wrap:wrap; align-items:center; gap:.45rem 1rem;
          margin:.25rem 0 .1rem; color:var(--risk-muted); font-size:.76rem; }}
        .risk-color-legend-item {{ display:inline-flex; align-items:center; gap:.4rem; white-space:nowrap; }}
        .risk-color-swatch {{ width:.78rem; height:.78rem; border-radius:2px;
          background:var(--legend-color); box-shadow:inset 0 0 0 1px rgba(49,68,86,.08); }}
        .risk-color-legend-note {{ flex-basis:100%; color:#8A949C; font-size:.7rem; line-height:1.45; }}
        .risk-evidence-check {{ display:inline-flex; align-items:center; border:1px solid #BBD9CB;
          border-radius:999px; padding:.14rem .48rem; color:#35645B; background:#EAF5F0;
          font-size:.66rem; font-weight:700; margin-right:.45rem; }}
        @media (max-width: 1100px) {{ .risk-metric-grid {{ grid-template-columns:repeat(3,1fr); }} .risk-trace {{ grid-template-columns:repeat(4,1fr); }} }}
        @media (max-width: 700px) {{ .risk-metric-grid {{ grid-template-columns:repeat(2,1fr); }} .risk-trace {{ grid-template-columns:repeat(2,1fr); }} .risk-source-grid {{ grid-template-columns:repeat(2,1fr); }} }}
        </style>
        """
    )

