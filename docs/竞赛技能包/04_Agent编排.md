# 技能 04 · Agent 与 LangGraph 编排

## 7 个 Agent（backend/agents/）
| Agent | 文件 | 写回 ctx | 说明 |
|---|---|---|---|
| 公告研读 | announcement_reader.py | ctx.semantic | 巨潮公告 + 规则/FinBERT/LLM 三通道 + F1 |
| 财务检测 | financial_detector.py | ctx.financial | F2 67 维 + F3 35 维 + 双负兜底；金融/地产不跳过 |
| 预测建模 | predictor.py | ctx.prediction | 三模型集成 30/60/90d + SHAP 查表推理 |
| 案例检索 | case_retriever.py | ctx.cases | 4785 库 + RRF + 三源标签 + 时间穿越控制 |
| chunk 检索 | chunk_retriever.py | ctx.chunks | 段落级证据召回（可选，容错） |
| 归因解释 | attributor.py | ctx.attribution | SHAP + 证据白名单 + validate_narrative 防幻觉 |
| 报告生成 | reporter.py | ctx.report | Markdown/JSON 六章报告 |

**统一接口**：`execute(company, ctx) -> ctx`；`AgentBase.run()` 自动记录 trace（run_id/耗时/摘要）追加进 `ctx.trace_log`。

## LangGraph 图（backend/agents/graph.py）
```python
State = {"ctx": Context, "company": str, "window": int, "as_of": str}
节点: announcement → financial → predictor → case → chunk → attribution → report
```
- `build_graph(use_llm, use_finbert, use_rule, rate_limit, checkpointer)` → 编译图
- predictor/chunk 用容错节点（异常标记 skipped 不打断）
- `memory_checkpointer()` = MemorySaver（断点续跑/回放：`graph.get_state_history(config)`）

## 使用方式
```python
from backend.agents.graph import build_graph
from backend.context import Context
graph = build_graph(use_llm=False, use_finbert=False)
state = graph.invoke({"ctx": Context(company="000004.SZ", window=60, as_of="2026-08-22"),
                      "company": "000004.SZ", "window": 60, "as_of": "2026-08-22"})
ctx = state["ctx"]
```
- `orchestrator.sweep_one()` = 薄封装（graph 首选，langgraph 缺失回落旧串行链）
- 单 Agent 独立：`FinancialDetectorAgent(use_llm=False).run("000001.SZ", ctx)`

## 未来可做（未实施）
并行（announcement∥financial）、条件路由（财务跳过→降级分支）、human-in-the-loop（interrupt_before）、前端流式（graph.stream）
