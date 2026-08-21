#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: risk_report_render —— 生成可解释预警报告（Markdown / JSON）
==================================================================
输入：ctx 全量结果（prediction / financial / semantic / cases / attribution / trace_log）
输出：
  render_json(ctx)     → 结构化报告 dict（供 API / 前端 / 落盘）
  render_markdown(ctx) → 人类可读 Markdown 报告（供展示 / 导出）

报告结构（对齐赛题任务4：概率 + 风险标签 + 主要诱因 + 相似案例 + 原文证据 + 置信度 + 推理链路）。
"""
from datetime import datetime


# ================= JSON 报告 =================
def render_json(ctx):
    """把 ctx 全量转成结构化报告 dict。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    return {
        "company": ctx.company,
        "name": ctx.name,
        "window": ctx.window,
        "as_of": ctx.as_of,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction": {
            "probability_60d": pred.get("probability_60d"),
            "probability_90d": pred.get("probability_90d"),
            "risk_level": pred.get("risk_level"),
            "confidence": pred.get("confidence"),
            "shap_features": pred.get("shap_features", []),
        },
        "financial": {
            "risk_level": fin.risk_level,
            "skip": fin.skip,
            "skip_reason": fin.skip_reason,
            "anomaly_list": fin.anomaly_list,
            "features_count": len(fin.features),
            "llm_analysis": fin.llm_analysis,
        },
        "semantic": {
            "announcement_count": ctx.semantic.stats.get("announcement_count", 0),
            "risk_factor_count": len(ctx.semantic.risk_factors),
            "risk_factors": ctx.semantic.risk_factors,
            "f1_features": ctx.semantic.f1_features,
        },
        "similar_cases": ctx.cases,
        "attribution": {
            "top_risk_factors": ctx.attribution.get("top_risk_factors", []),
            "evidence_citations": ctx.attribution.get("evidence_citations", []),
            "case_links": ctx.attribution.get("case_links", []),
            "narrative": ctx.attribution.get("narrative", ""),
        },
        "trace_log": ctx.trace_log,
    }


# ================= Markdown 报告 =================
def render_markdown(ctx):
    """生成人类可读的 Markdown 预警报告。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}

    lines = []
    lines.append(f"# 上市公司监管问询扫雷预警报告")
    lines.append(f"")
    lines.append(f"- 公司：**{ctx.company}** {ctx.name or ''}")
    lines.append(f"- 预测时点：{ctx.as_of} ｜ 预测窗口：{ctx.window} 天")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # 1. 预测结论
    lines.append("## 一、预测结论")
    lines.append("")
    p60 = pred.get("probability_60d")
    if p60 is not None:
        lines.append(f"- **未来 {ctx.window} 天被监管问询概率：{p60:.4f}**")
        lines.append(f"- 风险等级：**{pred.get('risk_level', '—')}** ｜ 置信度：{pred.get('confidence', '—')}")
    else:
        lines.append(f"- 预测：未预测（{pred.get('reason', '预测模型未接入')}）")
    lines.append("")

    # 2. 财务异常
    lines.append("## 二、财务异常信号（{n} 条）".format(n=len(fin.anomaly_list)))
    lines.append("")
    if fin.skip:
        lines.append(f"- 财务分析跳过：{fin.skip_reason}")
    else:
        for a in fin.anomaly_list:
            sev = "●" * a.get("severity", 0) + "○" * (5 - a.get("severity", 0))
            lines.append(f"- **[{a.get('type')}]** 严重度 {a.get('severity')} {sev}")
            lines.append(f"  - {a.get('evidence', '')}  `(label_ref={a.get('label_ref')})`")
    lines.append("")

    # 3. 公告风险要素
    lines.append("## 三、公告风险要素（{n} 条，{m} 份公告）".format(
        n=len(ctx.semantic.risk_factors), m=ctx.semantic.stats.get("announcement_count", 0)))
    lines.append("")
    for r in ctx.semantic.risk_factors[:10]:
        lines.append(f"- [{r.get('severity')}] {r.get('category')}：{r.get('description')}")
        if r.get("evidence"):
            lines.append(f"  > 原文：{r.get('evidence', '')[:100]}")
    if not ctx.semantic.risk_factors:
        lines.append("- （LLM 关闭或无风险要素）")
    lines.append("")

    # 4. 归因解释
    lines.append("## 四、归因解释（为什么被问询）")
    lines.append("")
    if att.get("narrative"):
        lines.append(f"{att['narrative']}")
        lines.append("")
    lines.append("**Top 风险诱因（SHAP 贡献）：**")
    lines.append("")
    for f in att.get("top_risk_factors", []):
        shap = f"（SHAP {f.get('shap'):+.3f}）" if f.get("shap") is not None else ""
        ev = f" 证据 {f.get('evidence_id')}" if f.get("evidence_id") else ""
        lines.append(f"- {f.get('desc') or f.get('feature')} {shap}{ev}")
    lines.append("")

    # 5. 相似案例
    lines.append("## 五、相似历史问询案例（Top {n}）".format(n=len(ctx.cases)))
    lines.append("")
    for c in ctx.cases:
        lines.append(f"- **{c.get('company')}**｜{c.get('inquiry_type')}｜{c.get('publish_date')}"
                     f"｜相似度 {c.get('similarity')}")
        topics = c.get("topics", [])
        if topics:
            lines.append(f"  - 关注点：{'；'.join(str(t)[:60] for t in topics[:3])}")
    if not ctx.cases:
        lines.append("- （无相似案例）")
    lines.append("")

    # 6. 证据与推理链路
    lines.append("## 六、证据与推理链路")
    lines.append("")
    for e in att.get("evidence_citations", []):
        lines.append(f"- `{e.get('evidence_id')}` [{e.get('source')}] {e.get('snippet', '')[:120]}")
    lines.append("")
    lines.append("**Agent 执行链路（可追踪率 100%）：**")
    for t in ctx.trace_log:
        lines.append(f"- {t.get('agent')} ｜ {t.get('status', 'done')}"
                     f"｜ {t.get('latency_ms', '')}ms ｜ {t.get('output_summary', '')}")
    lines.append("")

    return "\n".join(lines)
