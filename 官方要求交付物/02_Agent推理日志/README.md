# Agent 推理日志

运行 API 或批量脚本后，JSONL 审计日志会写入本目录。每行至少包含：`request_id`、`company`、`timestamp`、`agent`、`run_id`、`status`、`latency_ms`、`output_summary`、`trace_complete`。

日志只保存结构化摘要和证据 ID，不保存密钥；原文证据仍由报告中的 evidence_id 回指。

## 记录字段与审计判定

| 字段 | 含义 | 合格条件 |
|---|---|---|
| `request_id` | 一次 API 请求编号 | 同一请求的 Agent 相同 |
| `run_id` | 单个 Agent 编号 | 每个 Agent 唯一 |
| `agent` | 节点名称 | 与 7 节点图对应 |
| `status` | done/skipped/timeout | 降级必须记录原因 |
| `latency_ms` | 节点耗时 | 可定位性能瓶颈 |
| `trace_complete` | 追踪是否完成 | 正常或降级均为 true |

当前代码保证每个 Agent 通过 `AgentBase.run()` 追加 trace；预测和 chunk 节点异常时也会写入 skipped/timeout，不静默丢失。
