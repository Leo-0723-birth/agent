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
      {top_risk_factors, evidence_citations, case_links, narrative, confidence}

TODO（后续优化）：
    - FEATURE_MAP 扩充：与 scripts/build_concern_dict.py 的关注点↔指标词典打通
    - evidence_id 统一注册表：与公告研读/案例解析共用一套证据 ID 体系
    - narrative 结构校验：LLM 输出引用必须都在证据池中，否则剔除/重试
"""
from ..llm import chat
from .base import AgentBase
from .case_retriever import LABEL_KEYWORDS   # 标签→关键词映射（与检索共用）

# 特征 → 可读风险因素映射表（种子；build_concern_dict.py 产物可扩充）
FEATURE_MAP = {
    "cf_income_ratio":      {"desc": "经营现金流/净利润（盈利质量）", "label_ref": "盈利质量", "source": "financial"},
    "debt_to_assets_ratio": {"desc": "资产负债率",                   "label_ref": "偿债能力", "source": "financial"},
    "roe":                  {"desc": "净资产收益率",                 "label_ref": "盈利能力", "source": "financial"},
    "roe_trend_4q":         {"desc": "ROE 近4期趋势（斜率）",        "label_ref": "盈利能力", "source": "financial"},
    "revenue_yoy_growth":   {"desc": "营收同比增速",                 "label_ref": "收入确认", "source": "financial"},
    "net_profit_yoy_growth":{"desc": "净利润同比增速",               "label_ref": "盈利能力", "source": "financial"},
    "anomaly_count":        {"desc": "财务异常信号数量",             "label_ref": "综合",     "source": "financial"},
    "max_severity":         {"desc": "最高异常严重度",               "label_ref": "综合",     "source": "financial"},
    # 语义侧（公告风险标签 one-hot 后的特征名，前缀 label_）
    # "label_收入确认":      {"desc": "公告中收入确认类风险信号",      "label_ref": "收入确认", "source": "semantic"},
    # "label_商誉减值":      {"desc": "公告中商誉减值类风险信号",      "label_ref": "商誉减值", "source": "semantic"},
}


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
        """无 SHAP 时，用财务异常 + 公告风险标签作为诱因来源。"""
        factors = []
        for a in ctx.financial.anomaly_list[:self.top_k]:
            factors.append({
                "feature": a.get("indicator", a.get("type")),
                "shap": None,
                "description": a.get("evidence", ""),
                "label_ref": a.get("label_ref", "综合"),
                "source": "financial",
                "evidence_id": None,
            })
        for r in ctx.semantic.risk_factors[:self.top_k]:
            factors.append({
                "feature": "label_" + r.get("category", "其他"),
                "shap": None,
                "description": r.get("description", ""),
                "label_ref": r.get("category", "其他"),
                "source": "semantic",
                "evidence_id": None,
            })
        return factors

    # ============ Step 2: 特征 → 可读风险因素映射 ============
    def map_factors(self, top_features):
        """黑盒特征名 → {description, label_ref, source}。"""
        return [{"feature": n, "shap": v, **FEATURE_MAP.get(n,
                {"desc": n, "label_ref": "其他", "source": "unknown"})}
                for n, v in top_features]

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
        """按 label_ref / indicator 把因素绑定到证据池中的 evidence_id。"""
        located = []
        for f in factors:
            ev = None
            if f.get("source") == "financial":
                ev = next((e for e in pool if e.get("indicator") == f.get("feature")), None)
            if ev is None:
                ev = next((e for e in pool if e.get("label_ref") == f.get("label_ref")), None)
            if ev:
                f["evidence_id"] = ev["evidence_id"]
                located.append(f)      # 有证据的因素才进归因（无证据剔除）
        return located

    # ============ Step 4: 案例链接 ============
    def case_link(self, factors, cases):
        """取 top 因素的 label_ref 关键词 与相似案例 focus_points 重合度最高的 1-3 个。"""
        kws = set()
        for f in factors:
            lab = f.get("label_ref")
            if lab:
                kws.update(LABEL_KEYWORDS.get(lab, [lab]))
        scored = []
        for c in cases:
            fps = " ".join(c.get("topics", []))
            hit = sum(1 for kw in kws if kw and kw in fps)
            if hit > 0:
                scored.append((c, hit))
        scored.sort(key=lambda x: (-x[1], -x[0].get("similarity", 0)))
        return [c for c, _ in scored[:3]]

    # ============ Step 5: LLM 归因叙事（证据白名单防幻觉） ============
    def narrative_generate(self, company, ctx, factors, pool, links):
        """生成自然语言归因。只允许引用证据池中的 evidence_id，禁止编造。"""
        factor_text = "\n".join(
            [f"- {f.get('description', f.get('feature'))}（SHAP={f.get('shap')}，证据={f.get('evidence_id')}）"
             for f in factors])
        pool_text = "\n".join(
            [f"[{e['evidence_id']}] {e['snippet']}" for e in pool if e.get("snippet")])
        case_text = "\n".join(
            [f"- {c.get('company')}｜{c.get('inquiry_type')}｜相似度{c.get('similarity')}｜关注点{c.get('topics')}"
             for c in links]) or "- 无高度重合的相似案例"
        prompt = (
            f"公司 {company} 被问询概率 {ctx.prediction.get('probability_60d')}，"
            f"风险等级 {ctx.prediction.get('risk_level')}。\n\n"
            f"Top 风险因素（含证据编号）：\n{factor_text}\n\n"
            f"证据池（只能引用以下 evidence_id，禁止编造任何证据）：\n{pool_text}\n\n"
            f"相似历史案例：\n{case_text}\n\n"
            "请生成 3-5 句话的归因解释：先讲最可能触发监管问询的诱因（引用证据编号），"
            "再讲历史先例佐证。必须引用证据池中的 evidence_id，禁止编造数字或证据。"
        )
        try:
            return chat("你是资深投研/合规专家，生成可复核的归因解释。",
                        prompt, temperature=0.1, json_mode=False)
        except Exception as e:
            return f"[归因叙事生成失败] {e}"

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

        # Step 5: 叙事
        narrative = ""
        if self.use_llm:
            narrative = self.narrative_generate(company, ctx, factors, pool, links)

        # Step 6: 聚合输出
        ctx.attribution = {
            "top_risk_factors": factors,
            "evidence_citations": [e for e in pool if e.get("evidence_id")],
            "case_links": links,
            "narrative": narrative,
            "confidence": ctx.prediction.get("confidence", 0),
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
