#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主控调度 Agent (SweepingOrchestrator) —— 任务3 的核心
=====================================================
职责：把 7-Agent 流水线串成完整 Pipeline，批量扫雷，全程记录 trace。
流程：
  公告研读 → 财务异常检测 → (预测建模) → 监管案例检索 → 归因解释

关于 ReAct（Reasoning + Acting）机制：
  - 本编排器采用【固定 Plan 模板】的确定性编排（方案 5.4 推荐）：
    Plan（固定步骤）→ Dispatch（调用各 Agent）→ Observe（读共享 ctx）→ 下一环节
    —— 即 ReAct 循环的"确定性"版本，保证稳定、可复现、可追踪（赛题硬要求）。
  - 可选升级（ReAct 动态版）：主控用 LLM（backend.llm.chat，低温度 + JSON Schema）
    动态规划每家公司要调哪些 Skill；代价是推理不稳定，需加约束与校验。
  - 本实现默认走确定性版本；后续如需"动态规划"，在 _plan() 中替换即可。

调度顺序：
  Phase1  公告研读 Agent（→ ctx.semantic：风险要素/证据/F1 特征）
  Phase2  财务异常检测 Agent（→ ctx.financial：异常清单/F2 特征）
  Phase3  预测建模 Agent（→ ctx.prediction：概率/SHAP；未填充则占位）
  Phase4  监管案例检索 Agent（→ ctx.cases：RRF 融合 Top-5）
  Phase5  归因解释 Agent（→ ctx.attribution：诱因/证据/案例/叙事）
  （Phase1-2 与 Phase3-4 各自无依赖，可并行——当前为同步串行，后续可改线程池）

用法：
    orchestrator = SweepingOrchestrator()
    ctx = orchestrator.sweep_one("000004.SZ", window=60)   # 单公司
    reports = orchestrator.sweep_batch(["000004.SZ", "000005.SZ"])  # 批量
"""
import logging
import sys
from datetime import date
from pathlib import Path

_logger = logging.getLogger(__name__)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass

from ..context import Context
from ..skills.stock_code import normalize_company_input
from .base import AgentBase


class SweepingOrchestrator(AgentBase):
    name = "SweepingOrchestrator"

    def __init__(self, use_llm=True, use_finbert=True, use_rule=True, rate_limit=0.5,
                 use_checkpointer=False, use_semantic_cases=True, max_documents=20,
                 progress_callback=None):
        super().__init__(progress_callback=progress_callback)
        self.use_llm = use_llm          # 公告研读/归因的 LLM 开关
        self.use_finbert = use_finbert  # FinBERT 门控开关
        self.use_rule = use_rule        # 规则抽取通道（官方词典）
        self.rate_limit = rate_limit    # 财务爬虫限速
        self.use_checkpointer = use_checkpointer
        self.use_semantic_cases = bool(use_semantic_cases)
        self.max_documents = None if max_documents is None else int(max_documents)
        self._graph = None
        try:
            if progress_callback is not None:
                raise RuntimeError("细粒度进度回调使用确定性串行编排")
            # LangGraph 编排（首选）：7 节点图，见 backend/agents/graph.py
            from .graph import build_graph, memory_checkpointer
            checkpointer = memory_checkpointer() if use_checkpointer else None
            self._graph = build_graph(
                use_llm=use_llm, use_finbert=use_finbert, use_rule=use_rule,
                rate_limit=rate_limit, checkpointer=checkpointer,
                use_semantic_cases=self.use_semantic_cases,
                max_documents=self.max_documents,
            )
        except Exception as e:
            self._graph = None
            _logger.warning("LangGraph 不可用，回落确定性串行编排: %s: %s", type(e).__name__, e)

    def _agent_callback(self, agent_key):
        if self.progress_callback is None:
            return None

        def callback(payload):
            self.progress_callback({"agent_key": agent_key, **(payload or {})})
        return callback

    # ============ Plan：固定流水线（确定性 ReAct） ============
    def execute(self, company, ctx):
        """统一入口：对单家公司执行完整流水线（写回同一个 ctx）。"""
        if self._graph is not None:
            # LangGraph 编排（首选）：节点内部复用 AgentBase.run()，
            # trace 仍追加进 ctx.trace_log，审计格式与旧版一致。
            self._graph.invoke({
                "ctx": ctx,
                "company": company,
                "window": ctx.window,
                "as_of": ctx.as_of,
            })
            return ctx
        # 兜底：确定性串行编排（langgraph 未安装/导入失败时）
        self._run_announcement(company, ctx)   # Phase 1
        company = ctx.company or company       # 公告研读解析名称后向下游传播标准代码
        self._run_financial(company, ctx)      # Phase 2
        self._run_predict(company, ctx)        # Phase 3（占位，待填充）
        self._run_cases(company, ctx)          # Phase 4
        self._run_chunks(company, ctx)         # Phase 4.5（段落级证据召回，可选）
        self._run_attribution(company, ctx)    # Phase 5
        self._run_report(company, ctx)         # Phase 6
        return ctx

    def sweep_one(self, company, window=60, as_of=None, use_llm_summary=False):
        """单公司扫雷：新建 Context 并跑完整流水线。返回 ctx。

        use_llm_summary：报告执行摘要是否用 DeepSeek 生成（默认关，演示时勾选）。
        """
        company = normalize_company_input(company, allow_name=True)
        ctx = Context(company=company, window=window,
                      as_of=as_of or str(date.today()))
        ctx.use_llm_summary = bool(use_llm_summary)
        return self.execute(company, ctx)

    def sweep_batch(self, companies, window=60):
        """批量扫雷：逐家执行，返回摘要列表（供排序/演示）。"""
        reports = []
        for c in companies:
            ctx = self.sweep_one(c, window)
            reports.append(self._summarize(ctx))
        return reports

    # ============ Dispatch：各环节 ============
    def _run_announcement(self, company, ctx):
        from .announcement_reader import AnnouncementReaderAgent
        agent = AnnouncementReaderAgent(
            max_documents=self.max_documents,
            use_finbert=self.use_finbert,
            use_llm=self.use_llm,
            use_rule=self.use_rule,
            progress_callback=self._agent_callback("announcement"),
        )
        agent.run(company, ctx)      # base.run 自动把 trace 追加进 ctx.trace_log

    def _run_financial(self, company, ctx):
        from .financial_detector import FinancialDetectorAgent
        agent = FinancialDetectorAgent(use_llm=False, rate_limit=self.rate_limit)
        agent.progress_callback = self._agent_callback("financial")
        agent.run(company, ctx)

    def _run_predict(self, company, ctx):
        """预测建模（待填充）：已实现则调用；未实现则占位并记录 trace。"""
        try:
            from .predictor import PredictorAgent
            agent = PredictorAgent()
            agent.progress_callback = self._agent_callback("prediction")
            agent.run(company, ctx)
        except Exception as e:
            ctx.prediction = {"probability_60d": None, "risk_level": "未预测",
                              "confidence": None, "shap_features": []}
            ctx.trace_log.append({"agent": "Predictor", "status": "skipped",
                                  "reason": f"预测建模 Agent 未填充: {e}",
                                  "trace_complete": True})

    def _run_cases(self, company, ctx):
        try:
            from .case_retriever import CaseRetrieverAgent
            agent = CaseRetrieverAgent(use_semantic=self.use_semantic_cases)
            agent.progress_callback = self._agent_callback("case")
            agent.run(company, ctx)
        except Exception as e:
            ctx.cases = []
            ctx.trace_log.append({"agent": "CaseRetriever", "status": "skipped",
                                  "reason": f"案例检索不可用: {type(e).__name__}: {e}",
                                  "trace_complete": True})

    def _run_attribution(self, company, ctx):
        from .attributor import AttributorAgent
        agent = AttributorAgent(use_llm=self.use_llm)
        agent.progress_callback = self._agent_callback("attribution")
        agent.run(company, ctx)

    def _run_chunks(self, company, ctx):
        """chunk 级段落检索（可选）：chunk 索引缺失时自动跳过，不打断流水线。"""
        try:
            from .chunk_retriever import ChunkRetrieverAgent
            agent = ChunkRetrieverAgent()
            agent.progress_callback = self._agent_callback("chunk")
            agent.run(company, ctx)
        except Exception as e:
            ctx.trace_log.append({"agent": "ChunkRetriever", "status": "skipped",
                                  "reason": f"chunk 索引不可用: {e}",
                                  "trace_complete": True})

    def _run_report(self, company, ctx):
        from .reporter import ReporterAgent
        agent = ReporterAgent()
        agent.progress_callback = self._agent_callback("report")
        agent.run(company, ctx)

    # ============ Aggregate：摘要 ============
    @staticmethod
    def _summarize(ctx):
        return {
            "company": ctx.company,
            "name": ctx.name,
            "window": ctx.window,
            "announcements": ctx.semantic.stats.get("announcement_count", 0),
            "risk_factors": len(ctx.semantic.risk_factors),
            "financial_level": ctx.financial.risk_level,
            "financial_anomalies": len(ctx.financial.anomaly_list),
            "prediction": ctx.prediction.get("probability_60d"),
            "similar_cases": len(ctx.cases),
            "attribution_factors": len(ctx.attribution.get("top_risk_factors", [])),
            "trace_steps": len(ctx.trace_log),
        }


# ============================================================
# 自测入口（python -m backend.agents.orchestrator）
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    orch = SweepingOrchestrator(use_llm=False, use_finbert=False)
    ctx = orch.sweep_one("000004.SZ", window=60, as_of="2025-12-02")
    print("===== 流水线执行摘要 =====")
    print(f"公司: {ctx.company} | 公告: {ctx.semantic.stats.get('announcement_count')} 份 | "
          f"风险要素: {len(ctx.semantic.risk_factors)}")
    print(f"财务: {ctx.financial.risk_level} | 异常: {len(ctx.financial.anomaly_list)} 条")
    print(f"预测: {ctx.prediction.get('probability_60d')} | "
          f"相似案例: {len(ctx.cases)} | 归因诱因: {len(ctx.attribution.get('top_risk_factors', []))}")
    print(f"trace 步数: {len(ctx.trace_log)}")
    for t in ctx.trace_log:
        print(f"  - {t.get('agent')} | {t.get('status', 'done')}")
