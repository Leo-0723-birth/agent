# 关键证据与典型案例

证据遵循“原文白名单”约定：报告中的 `fin_*`、`sem_*`、`chunk_*` ID 只能引用 Context 中的原文片段；归因校验会识别叙事中的池外证据 ID。

推荐展示顺序：预测概率 → Top 风险诱因 → 原文证据 → 相似历史问询案例 → 完整 trace。

## 评委可核查的证据链

一次 000004.SZ 演示应能从 `prediction.probability_60d` 进入归因 Top 风险因素，再通过 `evidence_id` 定位公告或财务原文，最后查看案例的公司、问询类型、日期和关注点。报告层只保存原文片段或其 ID，不把 LLM 改写文本当作证据。

目录中的典型案例讲解稿提供展示顺序；实际运行产生的 Markdown/JSON 报告归档在 `backend/data/output/reports/`，可交叉核对。
