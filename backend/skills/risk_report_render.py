#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill: risk_report_render —— 生成可解释预警报告（Markdown / JSON）
==================================================================
输入：ctx 全量结果（prediction / financial / semantic / cases / attribution / trace_log）
输出：
  render_json(ctx, executive_summary="")     → 结构化报告 dict（供 API / 前端 / 落盘）
  render_markdown(ctx, executive_summary="") → 人类可读 Markdown 报告（供展示 / 导出）

报告结构（八章 · 风控函件式 · 对齐赛题任务4）：
  ① 函件头（公司/时点/窗口/数据源与新鲜度声明/报告编号）
  ② 执行摘要（LLM 叙事或规则拼装，见 reporter._executive_summary）
  ③ 公司画像与数据快照（行业/报告期/实时覆盖率）
  ④ 风险评分卡（30/60/90d 概率 + 等级 + 置信度 + SHAP Top 诱因）
  ⑤ 财务异常信号（异常清单 + 关键指标）
  ⑥ 公告风险要素（taxonomy 分类 + 原文证据）
  ⑦ 相似历史问询案例（Top-N + 关注点 + 时间穿越说明）
  ⑧ 证据与推理链路 + 局限性 + 免责声明
"""
from datetime import datetime


# ================= 报告编号 =================
def report_id(ctx, generated_at=None) -> str:
    """报告编号：公司_时点_生成时间戳。"""
    ts = (generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    code = str(ctx.company).replace(".", "_")
    return f"{code}_{str(ctx.as_of or 'NA')[:10]}_{ts}"


# ================= 规则版执行摘要（LLM 关闭/失败时兜底） =================
def rules_summary(ctx) -> str:
    """规则拼装执行摘要：只使用结构化事实，不编造。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    p60 = pred.get("probability_60d")
    level = pred.get("risk_level") or fin.risk_level or "—"
    conf = pred.get("confidence")

    parts = []
    if p60 is not None:
        parts.append(f"未来 {ctx.window} 日被监管问询概率 {p60:.4f}（风险等级：{level}，置信度：{conf or '—'}）")
    else:
        parts.append(f"预测未完成（{pred.get('reason', '模型未接入')}），以下依据实时规则信号。")

    # Top 诱因
    top = att.get("top_risk_factors", [])[:3]
    if top:
        descs = [str(t.get("desc") or t.get("feature")) for t in top]
        parts.append("主要风险诱因：" + "；".join(descs))

    # 财务异常
    anomalies = fin.anomaly_list[:3]
    if anomalies:
        parts.append(f"财务异常信号 {len(fin.anomaly_list)} 条（如 {anomalies[0].get('type')}）")

    # 公告风险
    rf = ctx.semantic.risk_factors[:2]
    if rf:
        cats = [str(r.get("category")) for r in rf if r.get("category")]
        if cats:
            parts.append("公告风险主题：" + "、".join(cats))

    # 相似案例
    if ctx.cases:
        c0 = ctx.cases[0]
        parts.append(f"相似历史案例 Top1：{c0.get('company')}（{c0.get('inquiry_type')}）")

    # 数据源
    ds = pred.get("data_source", "realtime")
    cov = (pred.get("coverage") or {}).get("ratio")
    parts.append(f"数据来源：{'实时特征' if ds == 'realtime' else '离线查表'}"
                 f"{'，实时覆盖率 ' + format(cov, '.0%') if cov else ''}")

    return "；".join(parts) + "。"


# ================= LLM 执行摘要输入（防幻觉：只给事实） =================
def build_summary_facts(ctx) -> str:
    """把 ctx 结构化事实压缩成紧凑文本，供 DeepSeek 生成摘要（禁止超出这些事实）。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}

    lines = []
    lines.append(f"公司：{ctx.company} {ctx.name or ''}；预测时点 {ctx.as_of}；窗口 {ctx.window} 天")
    p30, p60, p90 = pred.get("probability_30d"), pred.get("probability_60d"), pred.get("probability_90d")
    lines.append(f"概率：30d={p30} 60d={p60} 90d={p90}；等级={pred.get('risk_level')}；置信度={pred.get('confidence')}")

    top = att.get("top_risk_factors", [])[:5]
    if top:
        items = []
        for t in top:
            name = t.get("desc") or t.get("feature")
            shap = t.get("shap")
            items.append(f"{name}" + (f"(SHAP {shap:+.3f})" if shap is not None else ""))
        lines.append("Top 风险诱因：" + "；".join(items))

    if fin.anomaly_list:
        lines.append("财务异常：" + "；".join(
            f"{a.get('type')}({a.get('severity')}级)" for a in fin.anomaly_list[:5]))
    if ctx.semantic.risk_factors:
        lines.append("公告风险主题：" + "、".join(
            str(r.get("category")) for r in ctx.semantic.risk_factors[:5] if r.get("category")))
    if ctx.cases:
        lines.append("相似案例：" + "；".join(
            f"{c.get('company')}({c.get('inquiry_type')},{c.get('publish_date')})" for c in ctx.cases[:3]))
    ds = pred.get("data_source", "realtime")
    lines.append(f"数据来源={ds}；实时覆盖率={((pred.get('coverage') or {}).get('ratio'))}")
    return "\n".join(lines)


SUMMARY_SYSTEM = (
    "你是证券监管风控分析师。请根据【仅提供的事实】撰写一段约 3-4 句的《执行摘要》，"
    "口吻为监管风险提示函：先给结论（概率/等级），再列主要风险诱因，"
    "可提及同类历史案例，最后给一句核查建议。"
    "要求：只使用提供的事实，禁止编造数字或事件；不使用 markdown 符号；不超过 180 字。"
)


# ================= JSON 报告 =================
def render_json(ctx, executive_summary=""):
    """把 ctx 全量转成结构化报告 dict（八章）。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    generated_at = datetime.now().isoformat(timespec="seconds")
    return {
        "report_id": report_id(ctx),
        "company": ctx.company,
        "name": ctx.name,
        "window": ctx.window,
        "as_of": ctx.as_of,
        "generated_at": generated_at,
        "data_source": pred.get("data_source", "realtime"),
        "executive_summary": executive_summary or rules_summary(ctx),
        "profile": {
            "industry": fin.industry,
            "report_period": (fin.indicators or {}).get("report_period"),
            "coverage": pred.get("coverage"),
            "announcement_count": ctx.semantic.stats.get("announcement_count", 0),
        },
        "scorecard": {
            "probability_30d": pred.get("probability_30d"),
            "probability_60d": pred.get("probability_60d"),
            "probability_90d": pred.get("probability_90d"),
            "risk_level": pred.get("risk_level"),
            "confidence": pred.get("confidence"),
            "feature_anchor": pred.get("feature_anchor"),
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
            "top_risk_factors": att.get("top_risk_factors", []),
            "evidence_citations": att.get("evidence_citations", []),
            "case_links": att.get("case_links", []),
            "narrative": att.get("narrative", ""),
        },
        "trace_log": ctx.trace_log,
        "disclaimer": "本报告由自动化 Agent 流水线生成，仅供研究演示，不构成任何投资建议或事实认定。",
    }


# ================= Markdown 报告（八章） =================
def render_markdown(ctx, executive_summary=""):
    """生成风控函件式 Markdown 预警报告。"""
    pred = ctx.prediction or {}
    fin = ctx.financial
    att = ctx.attribution or {}
    generated_at = datetime.now()

    L = []
    add = L.append

    # ---------- ① 函件头 ----------
    add("# 上市公司监管问询风险提示函")
    add("")
    add(f"**报告编号**：{report_id(ctx, generated_at)}")
    add(f"**致**：{ctx.company} {ctx.name or ''}（监管风险扫描）")
    add(f"**预测时点**：{ctx.as_of} ｜ **预测窗口**：{ctx.window} 天 ｜ **生成时间**：{generated_at.isoformat(timespec='seconds')}")
    ds = pred.get("data_source", "realtime")
    add(f"**数据源**：{'实时特征（公告研读 + 财务/行情/舆情实时爬取）' if ds == 'realtime' else '离线建模数据查表'}"
        f"（实时覆盖率 {((pred.get('coverage') or {}).get('ratio', 0)):.0%}）")
    add("")
    add("---")
    add("")

    # ---------- ② 执行摘要 ----------
    add("## 一、执行摘要")
    add("")
    add(executive_summary or rules_summary(ctx))
    add("")
    add("---")
    add("")

    # ---------- ③ 公司画像与数据快照 ----------
    add("## 二、公司画像与数据快照")
    add("")
    add(f"- 行业：**{fin.industry or '—'}**")
    add(f"- 最新报告期：{((fin.indicators or {}).get('report_period')) or '—'}")
    add(f"- 近一年公告：{ctx.semantic.stats.get('announcement_count', 0)} 份 ｜ "
        f"识别风险要素：{len(ctx.semantic.risk_factors)} 条")
    add(f"- 财务异常信号：{len(fin.anomaly_list)} 条 ｜ 相似案例命中：{len(ctx.cases)} 条")
    if fin.skip:
        add(f"- ⚠️ 财务分析跳过：{fin.skip_reason}")
    add("")
    add("---")
    add("")

    # ---------- ④ 风险评分卡 ----------
    add("## 三、风险评分卡（问询概率）")
    add("")
    add("| 窗口 | 概率 | 等级 | 置信度 |")
    add("|------|------|------|--------|")
    for w, key in (("30 天", "probability_30d"), ("60 天", "probability_60d"), ("90 天", "probability_90d")):
        p = pred.get(key)
        add(f"| {w} | {f'{p:.4f}' if p is not None else '—'} |"
            f" {pred.get('risk_level') if key == 'probability_60d' else '—'} |"
            f" {pred.get('confidence') if key == 'probability_60d' else '—'} |")
    add("")
    add("**Top 风险诱因（SHAP 贡献）**：")
    add("")
    for f in att.get("top_risk_factors", []):
        shap = f"（SHAP {f.get('shap'):+.3f}）" if f.get("shap") is not None else ""
        src = "｜降级归因" if f.get("is_fallback") else ""
        add(f"- {f.get('desc') or f.get('feature')} {shap}{src}")
    if not att.get("top_risk_factors"):
        add("- （无 SHAP/降级诱因）")
    add("")
    add("---")
    add("")

    # ---------- ⑤ 财务异常信号 ----------
    add(f"## 四、财务异常信号（{len(fin.anomaly_list)} 条）")
    add("")
    if fin.skip:
        add(f"- 财务分析跳过：{fin.skip_reason}")
    else:
        for a in fin.anomaly_list:
            sev = "●" * int(a.get("severity", 0)) + "○" * (5 - int(a.get("severity", 0)))
            add(f"- **[{a.get('type')}]** 严重度 {a.get('severity')} {sev}")
            add(f"  - {a.get('evidence', '')}  `(label_ref={a.get('label_ref')})`")
        if not fin.anomaly_list:
            add("- 未发现规则异常信号。")
    add("")
    add("---")
    add("")

    # ---------- ⑥ 公告风险要素 ----------
    add(f"## 五、公告风险要素（{len(ctx.semantic.risk_factors)} 条）")
    add("")
    for r in ctx.semantic.risk_factors[:12]:
        add(f"- [{r.get('severity')}] **{r.get('category')}**：{r.get('description')}")
        if r.get("evidence"):
            add(f"  > 原文：{str(r.get('evidence', ''))[:120]}")
    if not ctx.semantic.risk_factors:
        add("- （无风险要素）")
    add("")
    add("---")
    add("")

    # ---------- ⑦ 相似历史问询案例 ----------
    add(f"## 六、相似历史问询案例（Top {len(ctx.cases)}，含时间穿越控制）")
    add("")
    for c in ctx.cases:
        score = c.get("rrf_score") or c.get("similarity")
        cosine = c.get("cosine_similarity")
        cos_txt = f"｜余弦相似度 {cosine:.4f}" if cosine is not None else ""
        add(f"- **{c.get('company')}**｜{c.get('inquiry_type')}｜{c.get('publish_date')}"
            f"｜RRF融合得分 {score}{cos_txt}")
        topics = c.get("topics") or []
        if topics:
            add(f"  - 关注点：{'；'.join(str(t)[:60] for t in topics[:3])}")
    if not ctx.cases:
        add("- （无相似案例）")
    add("")
    add("---")
    add("")

    # ---------- ⑧ 证据与推理链路 + 局限性 + 免责 ----------
    add("## 七、证据与推理链路")
    add("")
    for e in att.get("evidence_citations", []):
        add(f"- `{e.get('evidence_id')}` [{e.get('source')}] {str(e.get('snippet', ''))[:120]}")
    add("")
    add("**Agent 执行链路（可追踪率 100%）：**")
    for t in ctx.trace_log:
        add(f"- {t.get('agent')} ｜ {t.get('status', 'done')}｜ {t.get('latency_ms', '')}ms ｜ {str(t.get('output_summary', ''))[:60]}")
    add("")
    add("## 八、局限性说明")
    add("")
    add("- 概率为模型输出，非事实认定；实时特征覆盖率见报告头，缺失列以训练集中位数兜底。")
    add("- 相似案例基于历史问询函语义/标签匹配，仅供参照，不构成同因推断。")
    add("- 公告风险信号必须结合原文复核；扫描件 OCR 存在识别误差。")
    add("")
    add("---")
    add("")
    add("> 免责声明：本报告由自动化 Agent 流水线生成，仅供研究演示，不构成任何投资建议或事实认定。")
    add("")
    return "\n".join(L)
