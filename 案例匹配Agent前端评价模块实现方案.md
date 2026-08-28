# 案例匹配 Agent 前端评价模块实现方案

> 依据：第五届中国研究生金融科技创新大赛“揭榜挂帅”赛题《基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警算法探索》技术指标口径、官方 `标签与评测数据集`、项目现有 `api/` + `backend/` 代码结构。

---

## 1. 目标与范围

### 1.1 目标
在现有扫雷预警系统前端，注入并持续评价三项“风险语义抽取与归因”层指标：

| 指标 | 官方阈值 | 当前基线（参考） |
|---|---|---|
| 监管关注点分类准确率 | ≥ 80% | 待接入前端评价 |
| 关键证据片段召回率 | ≥ 85% | 待接入前端评价 |
| 相似历史问询案例匹配 Top-5 命中率 | ≥ 70% | Hybrid RRF 已可达 1.0（内部 proxy） |

### 1.2 评价对象
- **AnnouncementReader** 输出：`semantic.risk_factors`（含 `taxonomy_l1/l2`、`evidence`、`description`）。
- **CaseRetriever** 输出：`similar_cases`（Top-K 历史问询案例）。
- **ChunkRetriever** 输出：`chunks`（段落级证据召回，可选）。

### 1.3 Ground Truth 来源
- `D:/BaiduNetdiskDownload/标签与评测数据集/evaluation_ground_truth.csv`
  - `secucode`：公司代码
  - `publish_date`：问询函下发日期
  - `announcement_title`：官方标题
  - `regulatory_focus_points_json`：标准关注点/风险诱因/证据片段 JSON

---

## 2. 三项指标的计算公式、输入输出与阈值判断

### 2.1 监管关注点分类准确率（Concern Classification Accuracy）

#### 定义
模型对“本次问询会涉及哪些二级监管主题（45 类编码 A01-H02）”的预测，与 Ground Truth 中人工/Teacher 标注的关注点分类之间的吻合程度。

#### 计算公式
采用**多标签 Jaccard 准确率**作为单样本指标，避免单标签掩盖多主题问询：

```
Accuracy_i = |PredLabels_i ∩ TrueLabels_i| / |PredLabels_i ∪ TrueLabels_i|

MacroAccuracy = (1/N) * Σ Accuracy_i
```

其中：
- `PredLabels_i`：第 i 个样本的预测 45 类二级编码集合（来自 `risk_factors[].taxonomy_l2` 经 `expand_labels` 展开）。
- `TrueLabels_i`：第 i 个样本 GT 关注点经 NLP/关键词映射到的 45 类二级编码集合。

#### 输入
```json
{
  "prediction": {
    "secucode": "002055.SZ",
    "publish_date": "2020-07-07",
    "predicted_labels": ["A03", "B01", "C01", "D03"]
  },
  "ground_truth": {
    "secucode": "002055.SZ",
    "publish_date": "2020-07-07",
    "true_labels": ["A03", "B01", "C01", "C02", "D03"]
  }
}
```

#### 输出
```json
{
  "sample_accuracy": 0.6667,
  "tp": 3, "fp": 1, "fn": 2,
  "threshold_pass": false,
  "missing_labels": ["C02"],
  "extra_labels": []
}
```

#### 阈值判断
- 单样本 `sample_accuracy ≥ 0.8` 视为命中。
- 数据集级别 `MacroAccuracy ≥ 0.8` 视为达标。
- 若单样本跌破 0.8，标记为“分类异常”，进入看板“待复核”列表。

---

### 2.2 关键证据片段召回率（Evidence Recall）

#### 定义
模型抽取的风险证据片段（原文句子或短语）与 GT 中列出的监管关注点文本之间的覆盖程度。使用**语义召回**而非字符完全匹配，避免同义改写导致漏判。

#### 计算公式

**步骤 1：片段向量化**
- 将预测证据 `pred_evidence_j` 与 GT 证据 `gt_evidence_k` 分别用 BGE-large-zh-v1.5 编码为向量。

**步骤 2：相似度匹配**
```
sim(j,k) = cosine(embed(pred_evidence_j), embed(gt_evidence_k))
matched(j,k) = 1 if sim(j,k) ≥ τ else 0
```
建议阈值 `τ = 0.72`（可在线调参）。

**步骤 3：召回率**
```
Recall_i = (1/|GT_i|) * Σ_k max_j matched(j,k)

MacroRecall = (1/N) * Σ Recall_i
```

即：每条 GT 证据若被至少一条预测证据语义命中，则该 GT 证据被召回。

#### 输入
```json
{
  "prediction": {
    "evidences": [
      "2019 年大额亏损 4.5 亿元",
      "流动负债偿还存在逾期"
    ]
  },
  "ground_truth": {
    "evidences": [
      "详细说明 2019 年大额亏损的原因",
      "补充披露截至问询函发出日流动负债偿还情况和逾期债务情况"
    ]
  }
}
```

#### 输出
```json
{
  "recall": 1.0,
  "precision": 1.0,
  "f1": 1.0,
  "matched_pairs": [
    {"pred_idx": 0, "gt_idx": 0, "sim": 0.81},
    {"pred_idx": 1, "gt_idx": 1, "sim": 0.76}
  ],
  "threshold_pass": true
}
```

#### 阈值判断
- 单样本 `Recall_i ≥ 0.85` 视为命中。
- 数据集级别 `MacroRecall ≥ 0.85` 视为达标。
- 低于阈值时，输出“未召回 GT 证据”列表，触发人工补标。

---

### 2.3 相似历史问询案例匹配 Top-5 命中率（Case Hit@5）

#### 定义
针对一个目标问询事件，模型返回的 Top-5 相似历史案例中，至少存在 1 条“相关”案例的比例。相关性采用**45 类二级主题重合**作为 proxy（与 `backend/scripts/evaluate_case_retrieval.py` 对齐）。

#### 计算公式

**弱命中（Loose Hit）—— 满足即可计入 Top-5 命中：**
```
LooseHit@5_i = 1 if ∃ candidate ∈ Top5_i, |CandidateLabels ∩ TrueLabels_i| ≥ 1
```

**强命中（Strict Hit）—— 用于质量分层：**
```
StrictHit@5_i = 1 if ∃ candidate ∈ Top5_i, relevance_grade ≥ 2
```

其中 `relevance_grade` 复用现有脚本定义：
- grade 3：单标签目标精确命中该 L2；或多标签目标重合 ≥2 且覆盖率 ≥0.5。
- grade 2：多标签目标重合 ≥2 或覆盖率 ≥0.5。
- grade 1：至少共享 1 个二级主题。

**数据集级别：**
```
Hit@5 = (1/N) * Σ LooseHit@5_i
StrictHit@5 = (1/N) * Σ StrictHit@5_i
```

#### 输入
```json
{
  "prediction": {
    "top5_cases": [
      {"case_id": "case_123", "taxonomy_labels": ["A03", "B01"]},
      {"case_id": "case_456", "taxonomy_labels": ["C01"]}
    ]
  },
  "ground_truth": {
    "true_labels": ["A03", "B01", "C01"]
  }
}
```

#### 输出
```json
{
  "hit@5": 1.0,
  "strict_hit@5": 1.0,
  "label_recall@5": 0.6667,
  "mrr": 1.0,
  "threshold_pass": true,
  "matched_case_id": "case_123"
}
```

#### 阈值判断
- 官方口径要求 `Hit@5 ≥ 0.7`，使用 **Loose Hit** 作为达标判定。
- 内部质量监控额外跟踪 **StrictHit@5** 与 **LabelRecall@5**，用于模型迭代。

---

## 3. 前端埋点与后端接口设计

### 3.1 总体数据流

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ 前端扫雷页面 │────▶│ /api/evaluate│────▶│ 评价任务队列     │
│ (index.html) │     │   (FastAPI)  │     │ (后台异步计算)   │
└─────────────┘     └──────────────┘     └─────────────────┘
       │                                            │
       │  埋点事件：predict / case / evidence       │
       │                                            ▼
       │                                    ┌─────────────────┐
       └──────────────────────────────────│ 评价结果库       │
                                            │ (SQLite/JSONL)  │
                                            └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ 评价看板 API     │
                                            │ /api/metrics/... │
                                            └─────────────────┘
```

### 3.2 新增 API 端点

#### 3.2.1 `POST /api/evaluate/submit`
前端在每次扫雷完成后，将预测结果与可见的 GT 对齐信息提交给后端。

**请求体：**
```json
{
  "secucode": "002055.SZ",
  "publish_date": "2020-07-07",
  "window": 60,
  "mode": "realtime",
  "prediction": {
    "risk_factors": [
      {
        "taxonomy_l1": "A",
        "taxonomy_l2": "A03",
        "description": "2019 年大额亏损",
        "evidence": "2019 年度归属于上市公司股东的净利润为 -4.5 亿元",
        "severity": 5,
        "confidence": 0.92
      }
    ],
    "similar_cases": [
      {
        "case_id": "case_xxx",
        "company": "000005.SZ",
        "publish_date": "2023-06-14",
        "topics": ["营业收入下滑", "持续经营能力"],
        "taxonomy_labels": ["A03", "A06"],
        "similarity": 0.88
      }
    ]
  },
  "client_info": {
    "session_id": "sess_abc123",
    "user_agent": "...",
    "timestamp": "2026-08-28T18:24:00+08:00"
  }
}
```

**响应：**
```json
{
  "eval_id": "eval_20260828_001",
  "status": "queued",
  "message": "评价任务已入队，异步计算中"
}
```

#### 3.2.2 `GET /api/evaluate/{eval_id}`
查询单次评价结果。

**响应：**
```json
{
  "eval_id": "eval_20260828_001",
  "secucode": "002055.SZ",
  "publish_date": "2020-07-07",
  "status": "completed",
  "metrics": {
    "concern_accuracy": 0.6667,
    "evidence_recall": 1.0,
    "case_hit@5": 1.0,
    "case_strict_hit@5": 1.0
  },
  "threshold_status": {
    "concern_accuracy": false,
    "evidence_recall": true,
    "case_hit@5": true
  },
  "details": { ... }
}
```

#### 3.2.3 `GET /api/metrics/dashboard`
返回看板聚合数据。

**响应：**
```json
{
  "window": "all",
  "sample_count": 128,
  "concern_accuracy": {
    "value": 0.8125,
    "threshold": 0.8,
    "pass": true,
    "trend": [0.79, 0.80, 0.81, 0.8125]
  },
  "evidence_recall": {
    "value": 0.871,
    "threshold": 0.85,
    "pass": true,
    "trend": [0.82, 0.85, 0.86, 0.871]
  },
  "case_hit@5": {
    "value": 0.734,
    "threshold": 0.7,
    "pass": true,
    "trend": [0.68, 0.71, 0.72, 0.734]
  },
  "alerts": [
    {
      "level": "warning",
      "metric": "concern_accuracy",
      "message": "近 24h 单样本准确率均值 0.76，低于阈值 0.80",
      "triggered_at": "2026-08-28T18:00:00+08:00"
    }
  ]
}
```

#### 3.2.4 `POST /api/evaluate/feedback`
人工标注回流接口。

**请求体：**
```json
{
  "eval_id": "eval_20260828_001",
  "annotator": "reviewer_01",
  "feedback": {
    "risk_factors": [
      {
        "risk_id": "risk_xxx",
        "correct": false,
        "corrected_labels": ["A03", "C01"],
        "comment": "应同时标记资产减值"
      }
    ],
    "similar_cases": [
      {
        "case_id": "case_xxx",
        "relevant": true,
        "relevance_grade": 3,
        "comment": "同一行业同类亏损"
      }
    ]
  }
}
```

### 3.3 前端埋点设计

在前端 `index.html` 的 `AppState` 中新增 `evaluationQueue`：

```javascript
// 扫雷结果返回后自动触发
function submitEvaluation(payload) {
  return apiFetch('/api/evaluate/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

// 在 renderResult 或 WebSocket onComplete 后调用
AppState.subscribe('currentResult', (result) => {
  if (result && result.code) {
    submitEvaluation(buildEvalPayload(result));
  }
});
```

埋点字段（最小集合）：

| 字段 | 含义 | 示例 |
|---|---|---|
| `event` | 事件类型 | `scan_complete`, `case_feedback`, `evidence_feedback` |
| `secucode` | 公司代码 | `002055.SZ` |
| `publish_date` | GT 问询日期 | `2020-07-07` |
| `pred_labels` | 预测 45 类编码 | `["A03", "B01"]` |
| `top5_case_ids` | 返回案例 ID | `["case_1", "case_2", ...]` |
| `session_id` | 前端会话 | `sess_abc123` |
| `timestamp` | 埋点时间 | ISO 8601 |

---

## 4. Top-5 案例匹配命中判定规则与人工标注回流

### 4.1 命中判定规则（双档）

| 档位 | 判定条件 | 用途 |
|---|---|---|
| **弱命中 Loose Hit** | Top-5 中任一案例与 GT 共享 ≥1 个 45 类二级主题 | 官方 Hit@5 计算 |
| **强命中 Strict Hit** | 满足 relevance_grade ≥ 2（多主题重合或覆盖率 ≥50%） | 内部质量分层、人工复核优先级 |

### 4.2 命中示例

**GT：** `A03`（利润业绩波动）、`B01`（收入确认）、`C01`（应收账款坏账）

| Top-5 案例标签 | Loose Hit? | Strict Hit? | 说明 |
|---|---|---|---|
| `["A03", "B01"]` | ✅ | ✅ | 重合 2 个且覆盖率 2/3 |
| `["A03"]` | ✅ | ❌ | 仅 1 个重合 |
| `["C02", "D03"]` | ❌ | ❌ | 无重合 |

### 4.3 人工标注回流机制

#### 4.3.1 回流入口
在前端“相似监管案例”卡片上增加“👍 命中 / 👎 不相关”按钮：

```html
<div class="case-card" data-case-id="${c.caseId}">
  ...
  <div class="case-actions">
    <button onclick="feedbackCase('${c.caseId}', true)">命中</button>
    <button onclick="feedbackCase('${c.caseId}', false)">不相关</button>
  </div>
</div>
```

#### 4.3.2 回流数据存储
创建 `evaluation_feedback` 表（或 JSONL）：

```sql
CREATE TABLE evaluation_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  eval_id TEXT NOT NULL,
  case_id TEXT,
  risk_id TEXT,
  annotator TEXT,
  relevant INTEGER,           -- 1/0/null
  corrected_labels TEXT,      -- JSON 数组
  comment TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 4.3.3 回流应用
1. **热更新 GT：** 人工修正后的标签/证据可覆盖原 GT，用于后续重算指标。
2. **训练数据增强：** 经审核的 feedback 写入 `backend/data/feedback/` 作为微调样本。
3. **模型迭代：** 每月根据 feedback 统计误判标签 Top-K，触发 `build_concern_dict.py` 与 `train_models.py` 重跑。

---

## 5. 评价看板原型

### 5.1 看板入口
在现有 `index.html` 顶部导航或侧边栏新增“评价中心”入口：

```html
<div class="sidebar-section">质量运营</div>
<div class="pipeline-step" onclick="showEvalDashboard()">
  <div class="step-icon">📈</div>
  <div>
    <div class="step-title">评价中心</div>
    <div class="step-status">指标监控与人工复核</div>
  </div>
</div>
```

### 5.2 看板布局

```
┌────────────────────────────────────────────────────────────┐
│  监管关注点分类准确率   关键证据片段召回率   案例 Top-5 命中率  │
│      81.3% ✅            87.1% ✅            73.4% ✅         │
├────────────────────────────────────────────────────────────┤
│  [趋势折线图：近 7 日三项指标变化]                            │
├────────────────────────────────────────────────────────────┤
│  样本分布  │  标签混淆矩阵  │  证据召回热力图                   │
├────────────────────────────────────────────────────────────┤
│  异常告警列表                                  待复核样本列表  │
└────────────────────────────────────────────────────────────┘
```

### 5.3 关键组件

| 组件 | 数据来源 | 交互 |
|---|---|---|
| **指标卡** | `/api/metrics/dashboard` | 点击下钻到样本列表 |
| **趋势图** | `trend` 数组 | 支持 24h/7d/30d 切换 |
| **混淆矩阵** | 标签级 TP/FP/FN | 点击单元格查看误判样本 |
| **证据召回热力图** | GT 证据 × 预测证据 相似度 | 红色=未召回，绿色=命中 |
| **待复核列表** | 阈值未通过或人工反馈未一致的样本 | 支持一键标注 |

### 5.4 异常样本详情页
对于未达标样本，展示：

```
公司：002055.SZ  问询日：2020-07-07  准确率：0.67 ❌

预测标签：A03, B01, C01
真实标签：A03, B01, C01, C02, D03
缺失标签：C02, D03

关键证据召回：
  ✅ GT-1 “2019 年大额亏损的原因” ← 预测证据-1
  ❌ GT-2 “流动负债偿还情况和逾期债务” ← 未召回

相似案例 Top-5：
  1. case_123 [A03,B01] ✅ 强命中
  2. case_456 [C01]    ✅ 弱命中
  ...

[人工修正] [忽略] [加入训练集]
```

---

## 6. 异常告警策略

### 6.1 阈值跌破告警

| 触发条件 | 级别 | 通知方式 |
|---|---|---|
| 单日 `concern_accuracy < 0.8` | warning | 看板红点 + 日志 |
| 连续 3 日 `evidence_recall < 0.85` | critical | 邮件/企业微信 |
| `case_hit@5 < 0.7` 且样本数 ≥30 | critical | 邮件 + 看板置顶 |

### 6.2 数据漂移告警

监控预测标签分布与 GT 标签分布的 KL 散度：

```
KL(PredLabelDist || GTLabelDist) > 0.3  → 告警
```

常见于监管主题季节性变化（如年报季后减值类问题激增）。

### 6.3 证据质量告警

- 预测证据平均长度 < 10 字 → “证据过短”告警
- 预测证据与原文 overlap < 0.5 → “幻觉证据”告警
- GT 证据长期未被召回的标签 Top-3 → 触发规则/LLM 提示词优化

### 6.4 案例检索告警

- Top-5 案例平均相似度 < 0.5 → 检索质量下降
- 同一 case_id 高频出现 → 案例库分布偏差
- 实时检索超时率 > 5% → 兜底离线库告警

---

## 7. 落地路线图

### Phase 1：数据层（1-2 天）
- [ ] 将 `evaluation_ground_truth.csv` 加载为 SQLite/内存索引，按 `(secucode, publish_date)` 对齐。
- [ ] 实现 GT 关注点 → 45 类编码映射（复用 `label_keywords_v2.py` + LLM 辅助）。
- [ ] 扩展 `api/models.py` 增加 `EvaluationRequest/EvaluationResponse/FeedbackRequest`。

### Phase 2：后端评价服务（2-3 天）
- [ ] 新增 `api/evaluate.py`：实现 `/api/evaluate/submit`、`/api/evaluate/{eval_id}`、`/api/metrics/dashboard`、`/api/evaluate/feedback`。
- [ ] 新增 `backend/skills/evaluator.py`：封装三项指标计算逻辑（分类、证据召回、案例命中）。
- [ ] 接入 BGE embedding 做证据语义匹配。
- [ ] 评价任务异步执行（使用 `asyncio` 或后台线程，避免阻塞实时扫雷）。

### Phase 3：前端看板（2-3 天）
- [ ] 在 `index.html` 增加“评价中心”入口与页面。
- [ ] 实现指标卡、趋势图、混淆矩阵、待复核列表（可用 Chart.js 或 ECharts）。
- [ ] 在“相似监管案例”卡片增加人工命中/不相关反馈按钮。
- [ ] 扫雷完成后自动提交 `/api/evaluate/submit`。

### Phase 4：告警与回流（1-2 天）
- [ ] 实现告警规则引擎与日志/邮件通知。
- [ ] 建立 feedback 数据表与人工审核流程。
- [ ] 将审核后的 feedback 写入训练数据目录，定期触发模型重训。

---

## 8. 关键设计决策

| 决策点 | 方案 | 理由 |
|---|---|---|
| GT 对齐键 | `(secucode, publish_date)` | 同一公司可能有多次问询，需精确到日期 |
| 标签体系 | 45 类二级编码（A01-H02） | 与案例检索标签通道、官方评测口径一致 |
| 证据匹配 | BGE 语义相似度 ≥ 0.72 | 避免同义改写导致漏判，阈值可在线调参 |
| 案例命中 | Loose Hit 为达标口径 | 对齐官方 Top-5 命中率 ≥ 70% 要求 |
| 评价时机 | 实时扫雷完成后异步评价 | 不阻塞主流程，但保证数据新鲜 |
| 回流机制 | 前端一键反馈 + 后端审核入库 | 低成本启动，逐步积累人工标注 |

---

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| GT 标签覆盖不全 | 用 keyword + LLM 映射补全，并标记 confidence；人工 feedback 逐步修正 |
| 实时评价耗时长 | 证据匹配走 BGE 批量编码；评价任务异步执行 |
| 人工反馈量不足 | 优先对未达标样本弹窗引导标注；运营期设置标注任务 |
| 指标波动大 | 看板同时展示滑动窗口均值（7d）与单日值 |

---

## 10. 附录：核心代码模块建议

```
api/
  evaluate.py              # 新增：评价 API
  models.py                # 新增：Eval* Pydantic 模型
backend/
  skills/
    evaluator.py            # 新增：三项指标计算
    ground_truth_store.py  # 新增：GT 加载与对齐
frontend (index.html)
  新增：评价中心页面 + 埋点 + 反馈按钮
scripts/
  run_eval_dashboard.py    # 新增：本地一键启动评价看板
```
