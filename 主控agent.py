# -*- coding: utf-8 -*-
"""
上市公司监管问询扫雷预警系统 —— Streamlit 演示页
================================================
运行：streamlit run 导航入口.py（推荐，单入口导航，默认打开本页）
     或 streamlit run 主控agent.py --server.port 8501（独立运行）
功能：
  - 单公司/批量扫雷（真实 6-Agent 流水线：公告研读 → 财务检测 → 案例检索 → 归因 → 报告）
  - 流水线实时状态（st.status 逐环节点亮）
  - 可解释预警报告（预测结论 / 财务异常 / 公告风险 / 归因 / 相似案例 / 推理链路）
  - 📁 已生成报告浏览（output/reports/ 存档：Markdown 预览 + JSON/文件下载）
说明：
  - 默认 use_llm=False（离线）；勾选"启用 LLM"需 .env 配 DEEPSEEK_API_KEY
  - "启用 DeepSeek 执行摘要"：用 deepseek-v4-flash 生成风控函件式摘要（默认关，演示时勾选）
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agents import SweepingOrchestrator
from backend.config import OUTPUT_DIR

st.set_page_config(page_title="上市公司扫雷预警系统", page_icon="🛰️", layout="wide")
st.title("🛰️ 上市公司监管问询扫雷预警系统")
st.caption("基于 Agentic AI · 6-Agent 流水线（公告研读→财务检测→案例检索→归因→报告）· 可解释预警")


def list_reports(max_n: int = 20) -> list[dict]:
    """读取 output/reports/manifest.json 报告索引（按时间倒序）。"""
    try:
        mp = Path(OUTPUT_DIR) / "reports" / "manifest.json"
        if not mp.exists():
            return []
        data = json.loads(mp.read_text(encoding="utf-8"))
        return data[:max_n] if isinstance(data, list) else []
    except Exception:
        return []


def load_report_md(report_dir: str = "reports") -> str:
    """按选中的文件名读取 Markdown 内容。"""
    try:
        p = Path(OUTPUT_DIR) / "reports" / report_dir
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


# ================= 侧边栏输入 =================
with st.sidebar:
    st.header("⚙️ 参数设置")
    codes_text = st.text_area("公司代码（每行一个）", "000004.SZ", height=90)
    window = st.selectbox("预测窗口（天）", [30, 60, 90], index=1)
    use_llm = st.checkbox("启用 LLM 精细抽取（需 .env 配 key）", value=False)
    use_llm_summary = st.checkbox("启用 DeepSeek 执行摘要（deepseek-v4-flash）", value=False,
                                  help="报告执行摘要用大模型生成；关闭时用规则拼装。")
    run_clicked = st.button("🚀 开始扫雷", type="primary", use_container_width=True)

# ================= 已生成报告浏览区（无论是否扫雷都显示） =================
st.subheader("📁 已生成报告（output/reports/ 存档）")
reports = list_reports()
if not reports:
    st.caption("暂无已生成报告。扫雷完成后自动归档，可在此预览/下载。")
else:
    labels = {f"{r.get('report_id')} ｜ {r.get('probability_60d')} ｜ {r.get('risk_level')}": r
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
        st.caption(f"生成时间 {r.get('generated_at')} ｜ 窗口 {r.get('window')} 天")
        with st.expander("📄 预览报告", expanded=True):
            st.markdown(load_report_md(r.get("md_file", "")))

st.divider()

if not run_clicked:
    st.info("左侧输入公司代码后点击「开始扫雷」。示例：000004.SZ（国华网安）")
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

    # ================= 报告展示 =================
    st.subheader(f"📋 {code} 扫雷预警报告")
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    rj = (ctx.report or {}).get("json", {})

    # 执行摘要（优先展示）
    if rj.get("executive_summary"):
        with st.container(border=True):
            st.markdown("**一、执行摘要**")
            st.write(rj["executive_summary"])

    c1, c2, c3 = st.columns(3)
    p60 = pred.get("probability_60d")
    c1.metric("60天问询概率", f"{p60:.4f}" if p60 is not None else "未预测")
    level = pred.get("risk_level") or fin.risk_level or "—"
    c2.metric("风险等级", level)
    conf = pred.get("confidence")
    c3.metric("置信度", f"{conf:.2f}" if conf is not None else "—")

    # 概率条
    probs = {f"{w}天": pred.get(f"probability_{w}d") for w in (30, 60, 90)}
    if any(v is not None for v in probs.values()):
        st.bar_chart({k: [v] for k, v in probs.items() if v is not None}, height=180)

    # 财务异常
    with st.expander(f"💹 财务异常信号（{len(fin.anomaly_list)} 条）", expanded=True):
        if fin.skip:
            st.write(f"财务分析跳过：{fin.skip_reason}")
        for a in fin.anomaly_list:
            st.markdown(f"- **[{a.get('type')}]**（severity {a.get('severity')}）{a.get('evidence', '')}"
                        f"  `label_ref={a.get('label_ref')}`")

    # 公告风险要素
    with st.expander(f"📄 公告风险要素（{len(ctx.semantic.risk_factors)} 条 / "
                     f"{ctx.semantic.stats.get('announcement_count', 0)} 份公告）"):
        for r in ctx.semantic.risk_factors[:15]:
            st.markdown(f"- [{r.get('severity')}] **{r.get('category')}**：{r.get('description')}")
            if r.get("evidence"):
                st.markdown(f"  > {r.get('evidence', '')[:100]}")
        if not ctx.semantic.risk_factors:
            st.write("（LLM 关闭或无风险要素）")

    # 归因
    with st.expander("🎯 归因解释（Top 风险诱因 + 证据）"):
        if att.get("narrative"):
            st.write(att["narrative"])
        for f in att.get("top_risk_factors", []):
            shap = f"（SHAP {f.get('shap'):+.3f}）" if f.get("shap") is not None else ""
            st.markdown(f"- **{f.get('desc') or f.get('feature')}** {shap}  `{f.get('evidence_id', '')}`")
        st.markdown("**证据池：**")
        for e in att.get("evidence_citations", []):
            st.markdown(f"- `{e.get('evidence_id')}` [{e.get('source')}] {e.get('snippet', '')[:100]}")

    # 相似案例
    with st.expander(f"🧩 相似历史问询案例（Top {len(ctx.cases)}）"):
        for c in ctx.cases:
            score = c.get("rrf_score") or c.get("similarity")
            st.markdown(f"- **{c.get('company')}**｜{c.get('inquiry_type')}｜{c.get('publish_date')}"
                        f"｜RRF融合得分 {score}")
            if c.get("topics"):
                st.markdown(f"  - 关注点：{'；'.join(str(t)[:50] for t in c['topics'][:3])}")

    # 推理链路
    with st.expander("🔍 完整推理链路 trace_log（可追踪率 100%）"):
        st.json(ctx.trace_log)

    # 报告下载
    if ctx.report:
        st.download_button(
            f"⬇️ 下载 {code} 报告 (Markdown)",
            data=ctx.report["markdown"],
            file_name=f"{code}_risk_report.md",
            mime="text/markdown",
        )

st.success("批量扫雷完成。报告已自动归档到 backend/data/output/reports/（可在上方「已生成报告」浏览/下载）。")
