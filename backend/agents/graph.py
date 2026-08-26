#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 编排图（7-Agent 流水线）
=================================
用 LangGraph StateGraph 替代 Orchestrator 的硬编码线性链：
  State = {ctx, company, window, as_of}，ctx 即共享 Context（Agent 间唯一通信）。

节点（与旧 orchestrator 顺序一致）：
  announcement → financial → predictor → case → chunk → attribution → report

特性：
- 节点容错：predictor / chunk 与旧实现一致，异常时标记 skipped 不打断流水线；
  其余节点异常向上抛出（与旧 orchestrator 行为一致）。
- checkpointer：可选（MemorySaver），启用后支持断点续跑 / 逐步回放：
      graph.invoke(state, config={"configurable": {"thread_id": "audit-001"}})
- 图内节点复用 AgentBase.run() → trace 继续追加进 ctx.trace_log，审计格式不变。

依赖：langgraph>=1,<2（已锁定，见 requirements.txt）。
"""
from typing import TypedDict

from backend.context import Context


class SweepState(TypedDict):
    """LangGraph 共享状态：ctx 承载全部业务数据，其余为调用参数。"""
    ctx: Context
    company: str
    window: int
    as_of: str


# ================= 节点构造 =================

def _run_with_deadline(agent_name, fn, state, timeout):
    """节点级看门狗：daemon 线程 + join(timeout)。

    - 超时：记录 trace 后立即返回（不打断图，流水线永不永久挂起）；
    - 异常：向上抛出（供容错节点捕获）；
    - daemon 线程保证进程退出不被挂起任务拖住。
    """
    import threading
    box = {}

    def worker():
        try:
            box["value"] = fn(state)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        ctx = state["ctx"]
        ctx.trace_log.append({
            "agent": agent_name,
            "status": "timeout",
            "reason": f"节点执行超过 {timeout}s（网络异常），后续环节继续",
            "trace_complete": True,
        })
        return {"ctx": ctx}
    if "error" in box:
        raise box["error"]
    return box["value"]


def _node_factory(agent_factory, timeout=240, name="?"):
    """把「构造 agent 并 agent.run()」包装成 LangGraph 节点（带看门狗）。

    agent_factory: 无参可调用，返回一个 AgentBase 实例。
    timeout: 节点执行上限（秒），超时记录 trace 后继续。
    name: 节点名（用于超时/错误 trace）。
    """
    def node(state: SweepState) -> dict:
        return _run_with_deadline(name, _run_agent, state, timeout)

    def _run_agent(state: SweepState) -> dict:
        agent = agent_factory()
        company = state["ctx"].company or state["company"]
        agent.run(company, state["ctx"])
        return {"ctx": state["ctx"]}
    return node


def _tolerant_node(agent_factory, timeout=120):
    """容错节点：异常/超时时记录 skipped/timeout trace，不打断图（用于 predictor/chunk）。"""
    def node(state: SweepState) -> dict:
        agent = agent_factory()
        try:
            return _run_with_deadline(
                getattr(agent, "name", "?"), _run_tolerant, state, timeout)
        except Exception as e:
            state["ctx"].trace_log.append({
                "agent": getattr(agent, "name", "?"),
                "status": "skipped",
                "reason": f"{type(e).__name__}: {e}",
                "trace_complete": True,
            })
            return {"ctx": state["ctx"]}

    def _run_tolerant(state: SweepState) -> dict:
        agent = agent_factory()
        try:
            company = state["ctx"].company or state["company"]
            agent.run(company, state["ctx"])
        except Exception as e:
            state["ctx"].trace_log.append({
                "agent": getattr(agent, "name", "?"),
                "status": "skipped",
                "reason": f"{type(e).__name__}: {e}",
                "trace_complete": True,
            })
        return {"ctx": state["ctx"]}
    return node


# ================= 图构建 =================

def build_graph(use_llm=False, use_finbert=False, use_rule=True,
                rate_limit=0.5, checkpointer=None):
    """构建并编译 7 节点流水线图。

    参数：
        use_llm / use_finbert / use_rule / rate_limit：透传给各 Agent（与旧 orchestrator 一致）
        checkpointer：langgraph 持久化实例（如 MemorySaver()），None 则不启用
    返回：编译后的 CompiledStateGraph，用法：
        state = graph.invoke({"ctx": Context(...), "company": "...", "window": 60, "as_of": "..."})
        ctx = state["ctx"]
    """
    from langgraph.graph import END, StateGraph

    # Agent 类惰性导入（与 backend/agents/__init__ 的懒加载风格一致，避免循环依赖）
    from .announcement_reader import AnnouncementReaderAgent
    from .attributor import AttributorAgent
    from .case_retriever import CaseRetrieverAgent
    from .chunk_retriever import ChunkRetrieverAgent
    from .financial_detector import FinancialDetectorAgent
    from .predictor import PredictorAgent
    from .reporter import ReporterAgent

    g = StateGraph(SweepState)

    g.add_node("announcement", _node_factory(
        lambda: AnnouncementReaderAgent(use_finbert=use_finbert, use_llm=use_llm, use_rule=use_rule),
        timeout=420, name="AnnouncementReader"))
    g.add_node("financial", _node_factory(
        lambda: FinancialDetectorAgent(use_llm=False, rate_limit=rate_limit),
        timeout=240, name="FinancialDetector"))
    g.add_node("predictor", _tolerant_node(PredictorAgent, timeout=120))
    g.add_node("case", _node_factory(CaseRetrieverAgent, timeout=180, name="CaseRetriever"))
    g.add_node("chunk", _tolerant_node(ChunkRetrieverAgent, timeout=60))
    g.add_node("attribution", _node_factory(
        lambda: AttributorAgent(use_llm=use_llm), timeout=60, name="Attributor"))
    g.add_node("report", _node_factory(ReporterAgent, timeout=60, name="Reporter"))

    g.set_entry_point("announcement")
    g.add_edge("announcement", "financial")
    g.add_edge("financial", "predictor")
    g.add_edge("predictor", "case")
    g.add_edge("case", "chunk")
    g.add_edge("chunk", "attribution")
    g.add_edge("attribution", "report")
    g.add_edge("report", END)

    return g.compile(checkpointer=checkpointer)


def memory_checkpointer():
    """MemorySaver 检查点（进程内，演示/审计用；重启即清空）。"""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


# ============================================================
# 自测入口（python -m backend.agents.graph）
# ============================================================
if __name__ == "__main__":
    import sys
    from datetime import date
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from backend.context import Context

    graph = build_graph(use_llm=False, use_finbert=False)
    state = graph.invoke({
        "ctx": Context(company="000004.SZ", window=60, as_of=str(date(2025, 12, 2))),
        "company": "000004.SZ", "window": 60, "as_of": "2025-12-02",
    })
    ctx = state["ctx"]
    print("===== LangGraph 流水线执行摘要 =====")
    print(f"公司: {ctx.company} | 公告: {ctx.semantic.stats.get('announcement_count')} 份 | "
          f"风险要素: {len(ctx.semantic.risk_factors)}")
    print(f"财务: {ctx.financial.risk_level} | 异常: {len(ctx.financial.anomaly_list)} 条")
    print(f"预测: {ctx.prediction.get('probability_60d')} | "
          f"相似案例: {len(ctx.cases)} | 归因诱因: {len(ctx.attribution.get('top_risk_factors', []))}")
    print(f"trace 步数: {len(ctx.trace_log)}")
    for t in ctx.trace_log:
        print(f"  - {t.get('agent')} | {t.get('status', 'done')}")
