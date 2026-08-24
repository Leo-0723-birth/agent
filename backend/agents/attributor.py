#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归因解释 Agent (AttributorAgent) —— 任务4 的可解释性
====================================================
职责：读取预测 Agent 写下的 SHAP 特征贡献（Step1 已前移到预测 Agent）
      → 特征映射为可读风险因素 → 证据定位 → 案例链接
      → LLM 归因叙事（证据白名单防幻觉）→ 输出 ctx.attribution 三元组。

设计说明（重要）：
    - SHAP 计算前移到 PredictorAgent：预测时同时输出概率与特征贡献
      （写入 ctx.prediction.shap_features = [(feature_name, value), ...]），
      本 Agent 只读取，不再持有预测模型对象
    - 若 ctx.prediction 无 shap_features（模型不支持等），优雅降级：
      用 ctx.financial.anomaly_list + ctx.semantic.risk_factors 作为诱因来源
    - 防幻觉：narrative 生成只允许引用证据池（evidence_pool）中的 evidence_id

输入：company、ctx（prediction / semantic / financial / cases）
输出：ctx.attribution
      {top_risk_factors, evidence_citations, case_links, narrative, confidence, validation}

防幻觉（已落地）：
    - narrative 生成只允许引用证据池（evidence_pool）中的 evidence_id；
    - 生成后 validate_narrative() 校验引用，池外引用（幻觉）重试一次，再不行记录进
      ctx.attribution.validation.hallucination_refs；
    - evidence_citations 只输出被叙事实际引用的证据（白名单过滤）。

TODO（后续优化）：
    - FEATURE_MAP 扩充：与 scripts/build_concern_dict.py 的关注点↔指标词典打通
    - evidence_id 统一注册表：与公告研读/案例解析共用一套证据 ID 体系（当前为
      本 Agent 本地生成 fin_XXX / sem_XXX，上游尚未写 evidence_id）
    - 上游依赖：PredictorAgent 需产出 ctx.prediction.shap_features（当前为占位
      NotImplementedError，归因只能走降级路径）
"""
import re

from ..llm import chat
from .base import AgentBase

# 标签 → 关键词：官方标签体系（任务1交付包）；缺失时退回 case_retriever 内置版
try:
    from ..skills.risk_labels import expand_label_keywords
except Exception:  # pragma: no cover - 仅当 risk_labels 不可用时兜底
    from .case_retriever import label_keywords as _fallback_label_keywords

    def expand_label_keywords(labels):
        kws = set()
        lk = _fallback_label_keywords()
        for lab in labels or []:
            if not lab:
                continue
            mapped = lk.get(lab)
            if mapped:
                kws.update(mapped)
            elif not str(lab).isascii():
                kws.add(lab)
        return kws

# 证据编号正则：只识别 fin_000 / sem_000 这类统一编号（防幻觉校验用）
EVIDENCE_ID_RE = re.compile(r"\b(?:fin|sem)_\d{3}\b")

# 特征 → 可读风险因素映射表（覆盖 models_manifest.json 的全部 135 个建模特征）
FEATURE_MAP = {
    # —— 基础指标（financial_detector 原始指标，实时路径用）——
    "cf_income_ratio":      {"desc": "经营现金流/净利润（盈利质量）", "label_ref": "盈利质量", "source": "financial"},
    "debt_to_assets_ratio": {"desc": "资产负债率",                   "label_ref": "偿债能力", "source": "financial"},
    "roe":                  {"desc": "净资产收益率",                 "label_ref": "盈利能力", "source": "financial"},
    "roe_trend_4q":         {"desc": "ROE 近4期趋势（斜率）",        "label_ref": "盈利能力", "source": "financial"},
    "revenue_yoy_growth":   {"desc": "营收同比增速",                 "label_ref": "收入确认", "source": "financial"},
    "net_profit_yoy_growth":{"desc": "净利润同比增速",               "label_ref": "盈利能力", "source": "financial"},
    "anomaly_count":        {"desc": "财务异常信号数量",             "label_ref": "综合",     "source": "financial"},
    "max_severity":         {"desc": "最高异常严重度",               "label_ref": "综合",     "source": "financial"},
    # —— F2 财务异常特征（f2_calc 产出）——
    "f2_roe":               {"desc": "净资产收益率",                 "label_ref": "盈利能力", "source": "financial"},
    "f2_roa":               {"desc": "总资产收益率",                 "label_ref": "盈利能力", "source": "financial"},
    "f2_net_margin":        {"desc": "净利率",                       "label_ref": "盈利能力", "source": "financial"},
    "f2_debt_ratio":        {"desc": "资产负债率",                   "label_ref": "偿债能力", "source": "financial"},
    "f2_loss_flag":         {"desc": "亏损标志",                     "label_ref": "盈利能力", "source": "financial"},
    "f2_high_debt_flag":    {"desc": "高负债标志（>70%）",           "label_ref": "偿债能力", "source": "financial"},
    "f2_neg_pe_flag":       {"desc": "负市盈率标志",                 "label_ref": "盈利能力", "source": "financial"},
    "f2_neg_pb_flag":       {"desc": "负市净率标志",                 "label_ref": "市场估值", "source": "financial"},
    "f2_neg_accruals_flag": {"desc": "负应计项标志",                 "label_ref": "盈利质量", "source": "financial"},
    "f2_ocf_to_profit":     {"desc": "经营现金流/净利润",            "label_ref": "盈利质量", "source": "financial"},
    "f2_ocf_to_profit_extreme": {"desc": "现金流/利润极端值标志",    "label_ref": "盈利质量", "source": "financial"},
    "f2_roe_industry_rank": {"desc": "ROE 行业百分位",               "label_ref": "行业对标", "source": "financial"},
    "f2_beneish_m":         {"desc": "Beneish M-Score（盈余操纵）",  "label_ref": "盈利能力", "source": "financial"},
    "f2_benford_flag":      {"desc": "Benford 数字分布异常标志",     "label_ref": "财务异常", "source": "financial"},
    "f2_benford_max_dev":   {"desc": "Benford 最大偏离卡方",         "label_ref": "财务异常", "source": "financial"},
    "f2_trend_deterioration":{"desc": "趋势恶化综合指标（0-5）",      "label_ref": "盈利能力", "source": "financial"},
    "f2_accrual_quality_zscore": {"desc": "应计质量 Z-Score",        "label_ref": "盈利质量", "source": "financial"},
    "f2_profit_ocf_diverge":{"desc": "利润-现金流背离标志",          "label_ref": "盈利质量", "source": "financial"},
    "f2_industry_outlier_count": {"desc": "行业离群指标计数（|Z|>1.96）", "label_ref": "行业对标", "source": "financial"},
    "f2_p_roa":             {"desc": "ROA 异常分位（F-Score）",      "label_ref": "盈利能力", "source": "financial"},
    "f2_p_cfo":             {"desc": "现金流异常分位（F-Score）",    "label_ref": "盈利质量", "source": "financial"},
    "f2_z_roe":             {"desc": "ROE 行业 Z-Score",             "label_ref": "行业对标", "source": "financial"},
    "f2_z_debt_ratio":      {"desc": "负债率行业 Z-Score",           "label_ref": "行业对标", "source": "financial"},
    # —— F6 问询历史特征 ——
    "f6_inquiry_count_12m": {"desc": "近12个月被问询次数",           "label_ref": "历史问询", "source": "financial"},
    "f6_inquiry_count_24m": {"desc": "近24个月被问询次数",           "label_ref": "历史问询", "source": "financial"},
    "f6_inquiry_count_60m": {"desc": "近60个月被问询次数",           "label_ref": "历史问询", "source": "financial"},
    "f6_annual_report_inquiry_count": {"desc": "年报问询函次数",     "label_ref": "历史问询", "source": "financial"},
    "f6_attention_letter_count": {"desc": "关注函数量",              "label_ref": "历史问询", "source": "financial"},
    "f6_restructuring_inquiry_count": {"desc": "重组问询次数",       "label_ref": "历史问询", "source": "financial"},
    "f6_first_inquiry_interval_days": {"desc": "距首次问询天数",     "label_ref": "历史问询", "source": "financial"},
    "f6_last_inquiry_interval_days": {"desc": "距最近问询天数",      "label_ref": "历史问询", "source": "financial"},
    "f6_avg_inquiry_interval_days": {"desc": "平均问询间隔天数",     "label_ref": "历史问询", "source": "financial"},
    "f6_inquiry_interval_cv": {"desc": "问询间隔变异系数",           "label_ref": "历史问询", "source": "financial"},
    "f6_unreplied_count":   {"desc": "未回复问询次数",               "label_ref": "信息披露", "source": "financial"},
    # —— F5 公司治理特征 ——
    "gov_audit_firm_change": {"desc": "审计机构变更",               "label_ref": "审计质量", "source": "financial"},
    "gov_auditor_change":    {"desc": "签字会计师变更",             "label_ref": "审计质量", "source": "financial"},
    "gov_big4_auditor":      {"desc": "是否四大会计师事务所审计",   "label_ref": "审计质量", "source": "financial"},
    "gov_nonstandard_audit_opinion": {"desc": "非标准审计意见",      "label_ref": "审计质量", "source": "financial"},
    "gov_board_size":        {"desc": "董事会规模",                 "label_ref": "公司治理", "source": "financial"},
    "gov_independent_director_count": {"desc": "独立董事人数",      "label_ref": "公司治理", "source": "financial"},
    "gov_independent_director_ratio": {"desc": "独立董事占比",      "label_ref": "公司治理", "source": "financial"},
    "gov_top1_holder_ratio": {"desc": "第一大股东持股比例",         "label_ref": "股权结构", "source": "financial"},
    "gov_top10_holder_ratio": {"desc": "前十大股东持股比例",        "label_ref": "股权结构", "source": "financial"},
    "gov_top10_holder_count": {"desc": "前十大股东户数",            "label_ref": "股权结构", "source": "financial"},
    "gov_top1_top10_ratio":  {"desc": "第一大/前十大持股比",        "label_ref": "股权结构", "source": "financial"},
    "gov_top1_top2_gap":     {"desc": "第一二大股东持股差距",       "label_ref": "股权结构", "source": "financial"},
    "governance_year":       {"desc": "治理数据年份",               "label_ref": "公司治理", "source": "financial"},
    # —— F3 市场特征 ——
    "mkt_market_cap":        {"desc": "总市值",                     "label_ref": "市场估值", "source": "financial"},
    "mkt_log_market_cap":    {"desc": "对数市值",                   "label_ref": "市场估值", "source": "financial"},
    "mkt_pe_ratio":          {"desc": "市盈率（TTM）",              "label_ref": "市场估值", "source": "financial"},
    "mkt_pb_ratio":          {"desc": "市净率",                     "label_ref": "市场估值", "source": "financial"},
    "mkt_market_cap_qoq":    {"desc": "市值环比变化",               "label_ref": "市场异动", "source": "financial"},
    "mkt_pe_change_qoq":     {"desc": "市盈率环比变化",             "label_ref": "市场估值", "source": "financial"},
    "mkt_pb_change_qoq":     {"desc": "市净率环比变化",             "label_ref": "市场估值", "source": "financial"},
    "mkt_cap_industry_zscore": {"desc": "市值行业 Z-Score",         "label_ref": "行业对标", "source": "financial"},
    "mkt_pb_industry_zscore": {"desc": "市净率行业 Z-Score",        "label_ref": "行业对标", "source": "financial"},
    "mkt_return_5d":         {"desc": "5日收益率",                  "label_ref": "市场异动", "source": "financial"},
    "mkt_return_20d":        {"desc": "20日收益率",                 "label_ref": "市场异动", "source": "financial"},
    "mkt_return_60d":        {"desc": "60日收益率",                 "label_ref": "市场异动", "source": "financial"},
    "mkt_excess_return_20d": {"desc": "20日超额收益",               "label_ref": "市场异动", "source": "financial"},
    "mkt_volatility_20d":    {"desc": "20日波动率",                 "label_ref": "市场异动", "source": "financial"},
    "mkt_volatility_60d":    {"desc": "60日波动率",                 "label_ref": "市场异动", "source": "financial"},
    "mkt_max_drawdown_60d":  {"desc": "60日最大回撤",               "label_ref": "市场异动", "source": "financial"},
    "mkt_extreme_down_days_20d": {"desc": "20日暴跌天数",           "label_ref": "市场异动", "source": "financial"},
    "mkt_volume_ratio_20d":  {"desc": "20日量比",                   "label_ref": "市场异动", "source": "financial"},
    "mkt_volume_cv_20d":     {"desc": "20日成交量变异系数",         "label_ref": "市场异动", "source": "financial"},
    "mkt_abnormal_volume_days_20d": {"desc": "异常放量天数",        "label_ref": "市场异动", "source": "financial"},
    "mkt_amihud_illiquidity_20d": {"desc": "Amihud 非流动性",       "label_ref": "市场异动", "source": "financial"},
    "mkt_financing_balance": {"desc": "融资余额",                   "label_ref": "杠杆资金", "source": "financial"},
    "mkt_securities_balance": {"desc": "融券余额",                  "label_ref": "杠杆资金", "source": "financial"},
    "mkt_financing_balance_change": {"desc": "融资余额变化",        "label_ref": "杠杆资金", "source": "financial"},
    "mkt_institutional_holding_ratio": {"desc": "机构持股比例",      "label_ref": "机构行为", "source": "financial"},
    "mkt_institutional_holding_change": {"desc": "机构持股变化",     "label_ref": "机构行为", "source": "financial"},
    "mkt_institutional_holder_count": {"desc": "机构持仓家数",       "label_ref": "机构行为", "source": "financial"},
    "mkt_risk_warning_count_30d": {"desc": "近30日风险警示次数",    "label_ref": "风险警示", "source": "financial"},
    "mkt_risk_warning_count_90d": {"desc": "近90日风险警示次数",    "label_ref": "风险警示", "source": "financial"},
    "mkt_days_since_last_risk_warning": {"desc": "距上次风险警示天数", "label_ref": "风险警示", "source": "financial"},
    # —— F4 情绪特征 ——
    "sent_news_count_5d":    {"desc": "近5日新闻数量",              "label_ref": "市场情绪", "source": "financial"},
    "sent_news_count_10d":   {"desc": "近10日新闻数量",             "label_ref": "市场情绪", "source": "financial"},
    "sent_news_count_30d":   {"desc": "近30日新闻数量",             "label_ref": "市场情绪", "source": "financial"},
    "sent_news_title_count_30d": {"desc": "近30日新闻标题数",       "label_ref": "市场情绪", "source": "financial"},
    "sent_news_daily_peak_30d": {"desc": "近30日新闻日峰值",        "label_ref": "市场情绪", "source": "financial"},
    "sent_negative_news_count_30d": {"desc": "近30日负面新闻数",    "label_ref": "市场情绪", "source": "financial"},
    "sent_negative_ratio_30d": {"desc": "近30日负面新闻占比",       "label_ref": "市场情绪", "source": "financial"},
    "sent_sentiment_mean_30d": {"desc": "近30日情绪均值",           "label_ref": "市场情绪", "source": "financial"},
    "sent_sentiment_volatility_30d": {"desc": "近30日情绪波动",     "label_ref": "市场情绪", "source": "financial"},
    "sent_negative_peak_30d": {"desc": "近30日负面情绪峰值",        "label_ref": "市场情绪", "source": "financial"},
    "sent_post_count_5d":    {"desc": "近5日股吧帖子数",            "label_ref": "市场情绪", "source": "financial"},
    "sent_post_count_30d":   {"desc": "近30日股吧帖子数",           "label_ref": "市场情绪", "source": "financial"},
    "sent_post_daily_peak_30d": {"desc": "近30日帖子日峰值",        "label_ref": "市场情绪", "source": "financial"},
    "sent_comment_count_30d": {"desc": "近30日评论数",              "label_ref": "市场情绪", "source": "financial"},
    "sent_read_count_30d":   {"desc": "近30日阅读量",               "label_ref": "市场情绪", "source": "financial"},
    "sent_guba_sentiment_mean_30d": {"desc": "近30日股吧情绪均值",  "label_ref": "市场情绪", "source": "financial"},
    "sent_guba_sentiment_volatility_30d": {"desc": "近30日股吧情绪波动", "label_ref": "市场情绪", "source": "financial"},
    "sent_guba_positive_ratio_30d": {"desc": "近30日股吧正面占比",  "label_ref": "市场情绪", "source": "financial"},
    "sent_guba_negative_ratio_30d": {"desc": "近30日股吧负面占比",  "label_ref": "市场情绪", "source": "financial"},
}

# 前缀降级映射：未命中 FEATURE_MAP 的特征按前缀归类为可读组（尤其 50 个语义嵌入维度）
PREFIX_MAP = [
    ("regulatory_inquiry_semantic", "问询函语义特征（文本嵌入维度）", "语义信号", "semantic"),
    ("f2_", "财务异常特征", "财务异常", "financial"),
    ("f6_", "问询历史特征", "历史问询", "financial"),
    ("gov_", "公司治理特征", "公司治理", "financial"),
    ("mkt_", "市场特征", "市场异动", "financial"),
    ("sent_", "情绪特征", "市场情绪", "financial"),
]


class AttributorAgent(AgentBase):
    name = "Attributor"

    def __init__(self, top_k=5, shap_threshold=0.05, use_llm=True):
        super().__init__()
        self.top_k = top_k
        self.shap_threshold = shap_threshold
        self.use_llm = use_llm

    # ============ Step 1: 读取特征贡献（已前移到预测 Agent） ============
    def read_shap(self, ctx):
        """读取 ctx.prediction.shap_features = [(feature_name, value), ...]。
        缺失时返回 []（触发降级路径）。"""
        shap_list = ctx.prediction.get("shap_features", []) or []
        ranked = sorted(shap_list, key=lambda x: -abs(x[1]))
        return [(n, v) for n, v in ranked[:self.top_k] if abs(v) >= self.shap_threshold]

    # ============ Step 1b: 降级路径（无 SHAP 时用异常+风险标签） ============
    def fallback_factors(self, ctx):
        """无 SHAP 时，用财务异常 + 公告风险标签作为诱因来源。

        返回结构与 SHAP 路径一致（统一 factor 字段），并标记 is_fallback=True，
        供下游（报告/前端）区分"模型归因"与"规则降级归因"。
        """
        factors = []
        # 财务异常按严重度降序，取 top_k
        anomalies = sorted(
            ctx.financial.anomaly_list,
            key=lambda a: -int(a.get("severity", 0)),
        )
        for a in anomalies[:self.top_k]:
            factors.append({
                "feature": a.get("indicator", a.get("type")),
                "shap": None,
                "description": a.get("evidence", ""),
                "label_ref": a.get("label_ref", "综合"),
                "source": "financial",
                "evidence_id": None,
                "is_fallback": True,
            })
        # 公告风险要素按严重度降序，取 top_k
        risks = sorted(
            ctx.semantic.risk_factors,
            key=lambda r: -int(r.get("severity", 0)),
        )
        for r in risks[:self.top_k]:
            label = r.get("taxonomy_l2") or r.get("category") or "其他"
            factors.append({
                "feature": "label_" + str(label),
                "shap": None,
                "description": r.get("description", ""),
                "label_ref": str(label),
                "source": "semantic",
                "evidence_id": None,
                "is_fallback": True,
            })
        return factors

    # ============ Step 2: 特征 → 可读风险因素映射 ============
    @staticmethod
    def _resolve_feature(name):
        """特征名 → {desc, label_ref, source}：先精确匹配 FEATURE_MAP，
        未命中再按前缀归类（覆盖 50 个语义嵌入维度等无法逐条命名的情况），
        最后才退回未知。"""
        if name in FEATURE_MAP:
            return FEATURE_MAP[name]
        for prefix, desc, label, source in PREFIX_MAP:
            if name.startswith(prefix):
                return {"desc": f"{desc}（{name}）", "label_ref": label, "source": source}
        return {"desc": name, "label_ref": "其他", "source": "unknown"}

    def map_factors(self, top_features):
        """黑盒特征名 → 统一 factor 结构 {feature, shap, description, label_ref, source}。

        FEATURE_MAP 中的 "desc" 统一归一为 "description"，与降级路径字段对齐。
        """
        mapped = []
        for n, v in top_features:
            m = self._resolve_feature(n)
            mapped.append({
                "feature": n,
                "shap": v,
                "description": m.get("desc", n),
                "label_ref": m.get("label_ref", "其他"),
                "source": m.get("source", "unknown"),
                "evidence_id": None,
                "is_fallback": False,
            })
        return mapped

    # ============ Step 3: 证据池 + 证据定位 ============
    def _build_evidence_pool(self, ctx):
        """把财务异常 + 公告证据片段统一编号为证据池（evidence_id 白名单）。"""
        pool = []
        for i, a in enumerate(ctx.financial.anomaly_list):
            pool.append({
                "evidence_id": f"fin_{i:03d}",
                "source": "财务异常检测",
                "label_ref": a.get("label_ref", ""),
                "indicator": a.get("indicator"),
                "snippet": a.get("evidence", ""),
            })
        for i, s in enumerate(ctx.semantic.evidence_snippets):
            pool.append({
                "evidence_id": f"sem_{i:03d}",
                "source": "公告研读",
                "label_ref": s.get("category", ""),
                "snippet": s.get("text", ""),
            })
        return pool

    def evidence_locate(self, factors, pool):
        """按 label_ref / indicator 把因素绑定到证据池中的 evidence_id。

        找不到证据的因素不剔除，而是标记 no_evidence=True（降级为"模型侧信号"，
        保留完整 top-K 供前端展示），只有全部无证据时才由 execute 走降级路径。
        """
        for f in factors:
            ev = None
            if f.get("source") == "financial":
                ev = next((e for e in pool if e.get("indicator") == f.get("feature")), None)
            if ev is None:
                ev = next((e for e in pool if e.get("label_ref") == f.get("label_ref")), None)
            if ev:
                f["evidence_id"] = ev["evidence_id"]
                f["no_evidence"] = False
            else:
                f["evidence_id"] = None
                f["no_evidence"] = True
        return factors

    # ============ Step 4: 案例链接 ============
    def case_link(self, factors, cases):
        """取 top 因素的 label_ref（官方编码或俗称）→ 关键词，与相似案例 focus_points 重合。"""
        labels = [f.get("label_ref") for f in factors if f.get("label_ref")]
        kws = expand_label_keywords(labels)
        scored = []
        for c in cases:
            fps = " ".join(c.get("topics", []))
            hit = sum(1 for kw in kws if kw and kw in fps)
            if hit > 0:
                scored.append((c, hit))
        scored.sort(key=lambda x: (-x[1], -x[0].get("similarity", 0)))
        return [c for c, _ in scored[:3]]

    @staticmethod
    def _factor_tag(f):
        """因素展示标签：来源（SHAP/规则降级）+ 证据状态（有证据/模型侧信号）。"""
        if f.get("is_fallback"):
            src = "规则降级"
        elif f.get("shap") is not None:
            src = f"SHAP={f.get('shap')}"
        else:
            src = "模型信号"
        if f.get("evidence_id"):
            return f"{src}，证据={f['evidence_id']}"
        return f"{src}，模型侧信号（无直接证据）"

    # ============ Step 5: LLM 归因叙事（证据白名单防幻觉） ============
    def narrative_generate(self, company, ctx, factors, pool, links, reject_ids=None):
        """生成自然语言归因。只允许引用证据池中的 evidence_id，禁止编造。

        reject_ids：上一次生成校验出的池外引用（幻觉编号），本次显式禁止。
        """
        factor_text = "\n".join(
            [f"- {f.get('description', f.get('feature'))}"
             f"（{self._factor_tag(f)}）"
             for f in factors])
        pool_text = "\n".join(
            [f"[{e['evidence_id']}] {e['snippet']}" for e in pool if e.get("snippet")])
        case_text = "\n".join(
            [f"- {c.get('company')}｜{c.get('inquiry_type')}｜相似度{c.get('similarity')}｜关注点{c.get('topics')}"
             for c in links]) or "- 无高度重合的相似案例"
        reject_text = ""
        if reject_ids:
            reject_text = (
                f"\n【硬性约束】以下编号不在证据池中，本次禁止引用："
                f"{', '.join(reject_ids)}"
            )
        prompt = (
            f"公司 {company} 被问询概率 {ctx.prediction.get('probability_60d')}，"
            f"风险等级 {ctx.prediction.get('risk_level')}。\n\n"
            f"Top 风险因素（含证据编号）：\n{factor_text}\n\n"
            f"证据池（只能引用以下 evidence_id，禁止编造任何证据）：\n{pool_text}"
            f"{reject_text}\n\n"
            f"相似历史案例：\n{case_text}\n\n"
            "请生成 3-5 句话的归因解释：先讲最可能触发监管问询的诱因（引用证据编号），"
            "再讲历史先例佐证。必须引用证据池中的 evidence_id，禁止编造数字或证据。"
        )
        try:
            return chat("你是资深投研/合规专家，生成可复核的归因解释。",
                        prompt, temperature=0.1, json_mode=False)
        except Exception as e:
            return f"[归因叙事生成失败] {e}"

    def validate_narrative(self, narrative, pool):
        """校验叙事引用的 evidence_id 是否都在证据池（防幻觉白名单）。

        返回 (cited_ids, invalid_ids)：cited_ids 为叙事中出现的全部证据编号，
        invalid_ids 为其中不在证据池的部分（幻觉引用，应剔除或重试）。
        """
        cited = set(EVIDENCE_ID_RE.findall(narrative or ""))
        valid_ids = {e.get("evidence_id") for e in pool if e.get("evidence_id")}
        return cited & valid_ids, cited - valid_ids

    # ============ 主入口 ============
    def execute(self, company, ctx):
        # Step 1: 读 SHAP（缺失则降级）
        shap = self.read_shap(ctx)
        if shap:
            factors = self.map_factors(shap)
        else:
            factors = self.fallback_factors(ctx)

        # Step 3: 证据池 + 定位
        pool = self._build_evidence_pool(ctx)
        factors = self.evidence_locate(factors, pool)
        if not factors:                       # 全被剔除时保留降级因素
            factors = self.fallback_factors(ctx)

        # Step 4: 案例链接
        links = self.case_link(factors, ctx.cases)

        # Step 5: 叙事（含防幻觉校验：池外引用重试一次，再不行记录）
        narrative = ""
        hallucination_refs = []
        if self.use_llm:
            narrative = self.narrative_generate(company, ctx, factors, pool, links)
            _cited, invalid = self.validate_narrative(narrative, pool)
            if invalid:
                narrative = self.narrative_generate(
                    company, ctx, factors, pool, links, reject_ids=sorted(invalid)
                )
                _cited, invalid = self.validate_narrative(narrative, pool)
                hallucination_refs = sorted(invalid)

        # Step 6: 聚合输出（evidence_citations = 诱因绑定的证据 + 叙事额外引用的证据，白名单过滤）
        cited_ids = {f.get("evidence_id") for f in factors if f.get("evidence_id")}
        cited_ids |= set(EVIDENCE_ID_RE.findall(narrative or ""))
        evidence_citations = [
            {k: e.get(k) for k in ("evidence_id", "source", "label_ref", "snippet") if e.get(k)}
            for e in pool
            if e.get("evidence_id") and e.get("evidence_id") in cited_ids
        ]
        ctx.attribution = {
            "top_risk_factors": factors,
            "evidence_citations": evidence_citations,
            "case_links": links,
            "narrative": narrative,
            "confidence": ctx.prediction.get("confidence", 0),
            "validation": {
                "factors_with_evidence": len(factors),
                "cited_evidence_count": len(evidence_citations),
                "case_link_count": len(links),
                "hallucination_refs": hallucination_refs,
                "narrative_validated": not hallucination_refs,
            },
        }
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.attributor）
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context
    # AttributorAgent 已在本模块定义，直接使用（避免重复导入）

    ctx = Context(company="000004.SZ")
    # 模拟上游产物（预测/财务/公告/案例）
    ctx.prediction = {"probability_60d": 0.72, "risk_level": "高",
                      "confidence": 0.87,
                      "shap_features": [("cf_income_ratio", -0.21),
                                        ("roe", -0.18), ("anomaly_count", 0.12)]}
    ctx.financial.anomaly_list = [
        {"type": "双负信号", "severity": 4, "indicator": "cf_income_ratio",
         "evidence": "净利润-639万与经营现金流-2799万均为负", "label_ref": "盈利质量"},
        {"type": "亏损", "severity": 4, "indicator": "roe",
         "evidence": "ROE=-7.05%<0", "label_ref": "盈利能力"},
    ]
    ctx.semantic.evidence_snippets = [
        {"category": "商誉减值", "text": "本年计提商誉减值损失45,057.53万元"},
    ]
    ctx.cases = [
        {"case_id": "IC-000005-2023", "company": "ST星源", "inquiry_type": "年报问询函",
         "similarity": 0.81, "topics": ["盈利质量", "资金占用"]},
    ]
    agent = AttributorAgent(use_llm=False)
    agent.execute("000004.SZ", ctx)
    print("归因因素:", ctx.attribution["top_risk_factors"])
    print("案例链接:", ctx.attribution["case_links"])
