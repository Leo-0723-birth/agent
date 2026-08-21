#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公告研读 Agent (AnnouncementReaderAgent) —— 任务1 的核心 Agent
================================================================
职责：输入公司代码 → 检索近一年公告 → FinBERT 粗分类（门控）→ LLM 精细抽取风险要素
      → 输出 F1 语义特征（标签统计 + 向量化）+ 初步风险异常因素（供案例检索/归因使用）。

流程：
  resolve 公司 → announcement_search(近一年) → FinBERT 粗分类(门控)
  → LLM 风险要素抽取 → F1 特征构建 → 写回 ctx.semantic

输出（写回 ctx.semantic）：
    announcements       公告元数据（不含全文，全文走 store 按需取）
    finbert_signals     FinBERT 粗分类信号（含门控分数）
    risk_factors        风险要素（跨公告汇总：category/description/evidence/severity/announcement_id）
    evidence_snippets   证据片段（原文引用）
    per_announcement    每份公告的抽取结果
    f1_features         F1 标签统计特征（label_*_count / label_*_max_severity / ...）
    f1_vector           F1 语义向量（风险要素 embedding 均值）
    stats               统计（公告数/送LLM数/门控数/要素数/高危数）

调用的 Skill：
    skills/announcement_search  （AnnouncementStore：本地公告 PDF 检索）
    skills/finbert_classify     （FinBERT2-base：风险粗分类门控）
    skills/embedding            （F1 语义向量化；bge / fallback）
    llm.chat / llm.chat_json    （DeepSeek-v4-flash：输入解析 + 风险要素抽取）

使用的模型：
    - FinBERT2-base（valuesimplex-ai-lab，本地权重，中文金融预训练）
    - DeepSeek-v4-flash（API，.env 配置 DEEPSEEK_API_KEY）
    - Embedding：config.EMBEDDING_BACKEND（bge 或 fallback）

数据位置：
    - 公告数据：config.DATA_RAW / {公司代码} / 年份 / 类型 / *.pdf（本地官方数据）
    - 公告索引缓存：config.INDEX_DIR / {code}_index.json
"""
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass

from ..config import ANNOUNCE_WINDOW_DAYS, DATA_RAW, FINBERT_GATE, INDEX_DIR, MAX_TEXT_CHARS
from ..llm import chat_json
from .base import AgentBase


class AnnouncementReaderAgent(AgentBase):
    name = "AnnouncementReader"

    def __init__(self, data_root=None, use_finbert=True, use_llm=True, use_rule=True,
                 max_text_chars=MAX_TEXT_CHARS, gate_threshold=FINBERT_GATE):
        super().__init__()
        self.data_root = data_root or str(DATA_RAW)
        self.use_llm = use_llm
        self.use_rule = use_rule          # 规则抽取通道（官方 risk_dictionary，零成本兜底）
        self.max_text_chars = max_text_chars
        self.gate_threshold = gate_threshold
        self.finbert = None
        self.rule_extractor = None        # 懒加载
        if use_finbert:
            try:
                from ..skills.finbert_classify import FinBERTClient
                self.finbert = FinBERTClient()
            except Exception as e:
                print(f"[公告研读] FinBERT 加载失败，跳过粗分类（改用全量送 LLM）: {e}")

    # ================= 数据访问 =================
    def _store(self, company):
        """公告仓库：数据目录 = DATA_RAW/{code}，索引缓存到 data/index/。"""
        from ..skills.announcement_search import AnnouncementStore
        cache = Path(INDEX_DIR) / f"{company.replace('.', '_')}_index.json"
        return AnnouncementStore(Path(self.data_root) / company, cache_path=str(cache))

    # ================= Skill 1: 输入解析（LLM 可选） =================
    def resolve_company(self, user_input):
        """把名称/简称/代码解析为标准 secucode。失败返回空 dict。"""
        prompt = f"""你是A股上市公司识别助手。请识别用户输入所指的公司，输出 JSON：
{{"secucode": "6位代码.交易所后缀", "company_name": "公司简称", "matched": true/false}}
规则：6/9开头→.SH，0/3开头→.SZ，4/8开头→.BJ；无法识别 matched=false。
用户输入：{user_input}"""
        try:
            return chat_json("", prompt, max_tokens=200)
        except Exception:
            return {}

    # ================= Skill 2: 公告检索（近一年） =================
    def announcement_search(self, store, days=ANNOUNCE_WINDOW_DAYS, as_of=None):
        return store.search(days=days, as_of=as_of)

    # ================= Skill 3: FinBERT 粗分类（门控信号） =================
    def finbert_classify(self, announcements):
        signals = []
        for a in announcements:
            if self.finbert is None:
                signals.append({"announcement_id": a["id"], "categories": [], "max_score": 0.0})
                continue
            try:
                cls = self.finbert.classify(a["text"][:self.max_text_chars])
            except Exception:
                cls = {"categories": [], "top_category": None, "max_score": 0.0}
            signals.append({"announcement_id": a["id"], **cls})
        return signals

    # ================= Skill 4: LLM 风险要素抽取（evidence 原文强制） =================
    def llm_extract(self, company_name, announcements):
        """对单份公告做风险要素抽取（防幻觉：evidence 必须是正文原话）。"""
        risk_factors, evidence_snippets, per_ann = [], [], {}
        for a in announcements:
            aid = a["id"]
            prompt = f"""你是资深投研/合规风控专家，正在扫描A股上市公司公告，识别可能触发监管问询的风险要素。

公司：{company_name}
公告标题：{a.get('title', '')}
公告日期：{a.get('date', '')}
公告类型：{a.get('type', '')}

公告正文（可能被截断）：
\"\"\"
{(a.get('text') or '')[:self.max_text_chars]}
\"\"\"

请从这份公告中抽取可能引发监管问询的风险关注点，只抽取【真实存在、有原文依据】的风险，不要臆测。请以 json 格式输出（不要多余文字）：
{{
  "risk_factors": [
    {{"category": "财务异常|披露矛盾|关联交易|担保事项|资金占用|业绩预告偏差|会计处理争议|公司治理|并购重组|其他",
      "description": "一句话描述该风险点",
      "evidence": "从正文原文引用的关键句子",
      "severity": 1到5的整数}}
  ],
  "risk_level": "high|medium|low|none",
  "summary": "一句话总结该公告的整体风险"
}}
规则：
- evidence 必须是公告正文中的原话，若找不到原文依据，不要输出这条 risk_factor。
- 若该公告没有明显风险，risk_factors 输出空数组 []，risk_level 为 "none"。
- 只输出 JSON 对象本身。"""
            try:
                result = chat_json("", prompt, max_tokens=2000)
            except Exception as e:
                result = {"risk_factors": [], "risk_level": "none", "summary": "", "error": str(e)}
            rfs = result.get("risk_factors", [])
            for rf in rfs:
                rf["announcement_id"] = aid
                rf["announcement_date"] = a.get("date")
                rf["announcement_type"] = a.get("type")
                risk_factors.append(rf)
                if rf.get("evidence"):
                    evidence_snippets.append({
                        "announcement_id": aid,
                        "category": rf.get("category"),
                        "text": rf["evidence"],
                    })
            per_ann[aid] = {
                "risk_factors": rfs,
                "risk_level": result.get("risk_level", "none"),
                "summary": result.get("summary", ""),
            }
        return risk_factors, evidence_snippets, per_ann

    # ================= Skill 5: F1 特征构建（标签统计 + 向量化） =================
    def build_f1(self, risk_factors):
        """F1 语义特征：① 标签统计（one-hot/计数/最高严重度）② 语义向量（embedding 均值）。"""
        counts = Counter(r.get("category", "其他") for r in risk_factors)
        max_sev = {}
        for r in risk_factors:
            c = r.get("category", "其他")
            max_sev[c] = max(max_sev.get(c, 0), r.get("severity", 0))

        f1_features = {f"label_{c}_count": n for c, n in counts.items()}
        f1_features.update({f"label_{c}_max_severity": s for c, s in max_sev.items()})
        f1_features["total_risk_factors"] = len(risk_factors)
        f1_features["high_severity_count"] = sum(
            1 for r in risk_factors if r.get("severity", 0) >= 4)

        # 语义向量：风险要素描述 embedding 的均值（无要素时返回 None）
        f1_vector = None
        texts = [f"{r.get('category', '')}:{r.get('description', '')}" for r in risk_factors]
        if texts:
            from ..skills.embedding import embed
            vecs = embed(texts)
            f1_vector = vecs.mean(axis=0).tolist()
        return f1_features, f1_vector

    # ================= Skill 4b: 规则风险抽取（三通道之一，官方词典） =================
    def rule_extract(self, announcements):
        """用官方 risk_dictionary 做规则抽取（关键词+正则+否定词，evidence 原文强制）。
        返回的风险要素 category = 官方二级编码（如 C03/A03），与官方标签体系对齐。"""
        if not self.use_rule:
            return []
        if self.rule_extractor is None:
            try:
                from ..skills.rule_risk_extract import RuleRiskExtractor
                self.rule_extractor = RuleRiskExtractor()
            except Exception as e:
                print(f"[公告研读] 规则抽取器加载失败（跳过规则通道）: {e}")
                self.rule_extractor = False
        if not self.rule_extractor:
            return []

        sev_map = {"low": 2, "medium": 3, "high": 4}
        factors = []
        for a in announcements:
            try:
                hits = self.rule_extractor.extract((a.get("text") or "")[:self.max_text_chars])
            except Exception:
                continue
            for h in hits:
                if h.get("negated"):
                    continue
                factors.append({
                    "category": h.get("label"),              # 官方二级编码（如 C03）
                    "category_id": h.get("category_id"),     # 官方一级（如 C）
                    "description": h.get("matched_key", ""),
                    "evidence": h.get("evidence", ""),
                    "severity": sev_map.get(h.get("severity"), 3),
                    "source": "rule",
                    "announcement_id": a["id"],
                    "announcement_date": a.get("date"),
                    "announcement_type": a.get("type"),
                })
        return factors

    # ================= 主入口 =================
    def execute(self, company, ctx):
        # 1. 输入解析（名称 → 代码）
        code = company
        if self.use_llm and not re.match(r"^\d{6}", code):
            resolved = self.resolve_company(code)
            code = resolved.get("secucode") or code
            ctx.name = resolved.get("company_name") or ctx.name
        ctx.company = code

        # 2. 公告检索（近一年，截至 ctx.as_of）
        store = self._store(code)
        as_of = ctx.as_of or str(date.today())
        announcements = self.announcement_search(store, days=ANNOUNCE_WINDOW_DAYS, as_of=as_of)

        # 3. FinBERT 粗分类 + 门控（低于阈值的不送 LLM）
        finbert_signals = self.finbert_classify(announcements)
        signal_map = {s["announcement_id"]: s for s in finbert_signals}
        if self.use_llm and self.gate_threshold is not None:
            llm_candidates = [
                a for a in announcements
                if signal_map.get(a["id"], {}).get("max_score", 0.0) >= self.gate_threshold
            ]
        else:
            llm_candidates = announcements

        # 4. LLM 精细抽取（关闭 LLM 时为空，流程不断）
        if self.use_llm:
            risk_factors, evidence_snippets, per_ann = self.llm_extract(
                ctx.name or code, llm_candidates)
        else:
            risk_factors, evidence_snippets = [], []
            per_ann = {a["id"]: {"risk_factors": [], "risk_level": "none", "summary": ""}
                       for a in llm_candidates}

        # 4.5 被门控过滤的公告补标记
        gated_ids = {a["id"] for a in announcements} - {a["id"] for a in llm_candidates}
        for aid in gated_ids:
            per_ann.setdefault(aid, {"risk_factors": [], "risk_level": "none",
                                     "summary": "", "gated_by_finbert": True})

        # 4.6 规则抽取通道（官方词典，确定性兜底 + 交叉校验；LLM 与规则并集）
        rule_factors = self.rule_extract(announcements)
        if rule_factors:
            risk_factors = rule_factors + risk_factors

        # 5. F1 特征
        f1_features, f1_vector = self.build_f1(risk_factors)

        # 6. 统计 + 写回 ctx.semantic
        ctx.semantic.announcements = [{k: v for k, v in a.items() if k != "text"}
                                      for a in announcements]
        ctx.semantic.finbert_signals = finbert_signals
        ctx.semantic.risk_factors = risk_factors
        ctx.semantic.evidence_snippets = evidence_snippets
        ctx.semantic.per_announcement = per_ann
        ctx.semantic.f1_features = f1_features
        ctx.semantic.f1_vector = f1_vector
        ctx.semantic.stats = {
            "announcement_count": len(announcements),
            "llm_processed_count": len(llm_candidates),
            "gated_count": len(gated_ids),
            "risk_factor_count": len(risk_factors),
            "high_severity_count": sum(1 for r in risk_factors if r.get("severity", 0) >= 4),
            "window_days": ANNOUNCE_WINDOW_DAYS,
            "as_of": as_of,
        }
        return ctx


# ============================================================
# 自测入口（python -m backend.agents.announcement_reader）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.context import Context

    agent = AnnouncementReaderAgent(use_finbert=False, use_llm=False)
    ctx = Context(company="000004.SZ", window=60, as_of="2025-12-02")
    agent.execute("000004.SZ", ctx)
    print(f"公告数: {ctx.semantic.stats['announcement_count']}")
    print(f"风险要素: {len(ctx.semantic.risk_factors)} | F1特征: {ctx.semantic.f1_features}")
    print(f"F1向量维度: {len(ctx.semantic.f1_vector) if ctx.semantic.f1_vector else '空'}")
