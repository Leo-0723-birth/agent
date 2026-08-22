# LangGraph 迁移方案（评估稿 · 暂未实施）

> 状态：**方案文档**，代码未动。决策入口：`SweepingOrchestrator` 改为 LangGraph 薄封装，7 个 Agent 业务代码零改动。
> 更新：2026-08-22

## 1. 现状架构（无需大改的部分）

```
用户 → 主控agent.py / 报告生成agent.py / 自测入口
        └─ SweepingOrchestrator.sweep_one(company, window, as_of)
             └─ 硬编码线性链（7 环节）：
                公告研读 → 财务检测 → 预测建模 → 案例检索 → chunk 检索 → 归因解释 → 报告生成
                全程共享一个 Context（dataclass），trace 追加进 ctx.trace_log
```

关键事实：
- 7 个 Agent 统一接口 `execute(company, ctx)`，只读写共享 `Context`（`backend/context.py`）；
- 编排逻辑集中在 `SweepingOrchestrator._run_*`（`backend/agents/orchestrator.py`）；
- 对外入口（`主控agent.py`、`报告生成agent.py`、`python -m backend.agents.orchestrator`）都只调 `sweep_one()`。

**结论：业务层（skills/特征/模型/检索/归因/报告）与编排层解耦良好，迁移只动编排层。**

## 2. 目标架构

```
用户 → 同上一模一样的入口
        └─ SweepingOrchestrator.sweep_one()   ← 薄封装，内部调 graph.invoke()
             └─ LangGraph StateGraph(SweepState)
                 State = { "ctx": Context, "company": str, "window": int, "as_of": str }
                 节点：announcement → financial → predictor → case → chunk → attribution → report
                 可选：条件边（财务跳过/预测降级）、并行（announcement ∥ financial）、
                      checkpointer（断点续跑/回放）、human-in-the-loop（确认后继续）
```

## 3. 迁移步骤（预计 0.5~1 个工作日）

### 步骤 1：加依赖
```bash
pip install "langgraph>=0.2,<1" "langchain-core>=0.3"
```
> LangGraph 版本间 API 差异大，**锁定 0.2.x 大版本**，升级需单独评审。

### 步骤 2：新增 `backend/agents/graph.py`（核心新文件，约 100 行）

```python
# -*- coding: utf-8 -*-
"""LangGraph 编排图：7 节点流水线（替代 Orchestrator 的硬编码链）。"""
from typing import TypedDict
from langgraph.graph import StateGraph, END

from backend.context import Context

class SweepState(TypedDict):
    ctx: Context
    company: str
    window: int
    as_of: str

def _mk_node(agent_factory):
    """把「构造 agent 并 run」包装成 LangGraph 节点。"""
    def node(state: SweepState) -> dict:
        agent = agent_factory()
        agent.run(state["company"], state["ctx"])
        return {"ctx": state["ctx"]}
    return node

from backend.agents.announcement_reader import AnnouncementReaderAgent
from backend.agents.financial_detector import FinancialDetectorAgent
from backend.agents.predictor import PredictorAgent
from backend.agents.case_retriever import CaseRetrieverAgent
from backend.agents.chunk_retriever import ChunkRetrieverAgent
from backend.agents.attributor import AttributorAgent
from backend.agents.reporter import ReporterAgent

def build_graph(checkpointer=None):
    g = StateGraph(SweepState)
    g.add_node("announcement", _mk_node(lambda: AnnouncementReaderAgent(use_finbert=False, use_llm=False)))
    g.add_node("financial",    _mk_node(lambda: FinancialDetectorAgent(use_llm=False)))
    g.add_node("predictor",    _mk_node(PredictorAgent))
    g.add_node("case",         _mk_node(CaseRetrieverAgent))
    g.add_node("chunk",        _mk_node(ChunkRetrieverAgent))
    g.add_node("attribution",  _mk_node(lambda: AttributorAgent(use_llm=False)))
    g.add_node("report",       _mk_node(ReporterAgent))
    # 线性主链
    g.set_entry_point("announcement")
    g.add_edge("announcement", "financial")
    g.add_edge("financial", "predictor")
    g.add_edge("predictor", "case")
    g.add_edge("case", "chunk")
    g.add_edge("chunk", "attribution")
    g.add_edge("attribution", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
```

### 步骤 3：`orchestrator.py` 改薄封装（对外接口不变）

```python
class SweepingOrchestrator(AgentBase):
    def __init__(self, use_llm=False, use_finbert=False, use_rule=True, rate_limit=0.5,
                 use_checkpointer=False):
        super().__init__()
        self.graph = build_graph(checkpointer=MemorySaver() if use_checkpointer else None)

    def sweep_one(self, company, window=60, as_of=None):
        ctx = Context(company=company, window=window, as_of=as_of or str(date.today()))
        self.graph.invoke({"ctx": ctx, "company": company, "window": window, "as_of": ctx.as_of})
        return ctx
```

**好处：`主控agent.py`、`报告生成agent.py`、`python -m backend.agents.orchestrator` 全部零改动。**

### 步骤 4：测试
- 保留 `python -m backend.agents.orchestrator` 自测（000004.SZ 全流程，Predictor 应显示 done）；
- 新增 `backend/tests/test_graph.py`：节点顺序正确、财务跳过分支、predictor 降级分支、trace 完整性。

### 步骤 5：可选增值（按评审需求取舍）
1. **条件路由**：财务检测 `skip=True`（金融/地产）→ 跳过预测仍走后续；预测查不到特征 → 走规则降级分支；
2. **并行**：`announcement` 与 `financial` 无依赖 → 用 `add_edge` 双入口 + fan-in（注意两者写 ctx 不同字段，无冲突）；
3. **Checkpoint**：`MemorySaver()`（演示）/ `SqliteSaver`（持久化），支持断点续跑与逐步回放；
4. **Human-in-the-loop**：`interrupt_before` / `interrupt_after`，答辩时可"评委确认后继续"；
5. **流式输出**：`graph.stream()` 逐节点推送，替代现在前端轮询 trace_log。

## 4. 工作量与风险

| 项 | 评估 |
|---|---|
| 代码改动 | 新增 `graph.py`（~100 行）+ 改 `orchestrator.py`（~30 行）+ requirements 一行；7 个 Agent 业务代码 0 改动 |
| 风险 | LangGraph API 版本漂移（锁定 0.2.x）；checkpointer 需持久化后端（演示用 Memory 即可）；并行后 trace 顺序需重跑验证 |
| 验收标准 | 自测 7 环节全 done 且预测概率非空；AppTest 主控页/报告页渲染正常；新增图单测通过 |

## 5. 结论

- 现编排层已足够支撑演示；LangGraph 是**叙事加分项**（图编排/状态机/条件路由/断点续跑，正好呼应赛题"Agentic AI"）。
- 建议在答辩定稿前按"步骤 1-4"落地轻量版，增值项（并行/HITL/流式）按评审反馈再加。
