# 财务异常检测 Agent 设计方案（FinancialDetector）

> 依据：底层建模方案 2.3/2.4（F2/F3 特征）+ 现有实现（桌面 `财务异常agent/` 已跑通）+ 你的确认
> 决策摘要：
> ① 数据源：本 Agent 面向【评委演示】→ 保留东方财富免费接口（实时真实数据）；比赛模型训练用官方数据集（另走 scripts/train_predictor.py），二者分离；
> ② 特殊行业（金融/地产/建筑）现阶段**以跳过为主**，后续全面开发后再优化行业专属规则；
> ③ 异常输出格式统一为"归因 Agent 可直接使用"的结构，写入共享 context（ctx.financial.anomaly_list）。

---

## 一、定位与数据源

```
┌──────────────────────────────────────────────────────────┐
│  FinancialDetectorAgent（backend/agents/financial_detector.py）│
│  输入：company 代码（000004.SZ）                            │
│  数据源：skills/financial_data_fetch.py（东方财富免费接口）   │
│  输出：ctx.financial.{features, indicators, benchmarks,     │
│                      anomaly_list, risk_level, skip, ...}  │
└──────────────────────────────────────────────────────────┘
```

| 用途 | 数据源 | 说明 |
|------|--------|------|
| 评委演示（本 Agent） | 东方财富免费接口（实时） | 无需登录/token，已跑通（000004 21 期数据） |
| 比赛模型训练/测试 | 官方数据集（另走 `scripts/train_predictor.py`） | 特征按披露日 ≤ T 对齐，与本 Agent 解耦 |
| 行业对标（可选） | `config.FIN_WIND_CSV` 指向 wind 特征 CSV | 配置后启用 Z-Score；未配置则优雅降级 |

---

## 二、分层架构

```
层1 数据获取   DataFetcher.fetch_company_profile / fetch_financials
层2 指标计算   最新一期 indicators（~12 项）+ 派生（cf_to_profit / roe_trend_4q）
层3 行业对标   industry_benchmark（同报告期同行 Z-Score，可选）
层4 异常检测   anomaly_detect（规则 + 双负兜底 + 行业偏离）
层5 双输出     ctx.financial.features（→ 预测模型）+ anomaly_list（→ 归因/报告）
```

---

## 三、异常规则表（阈值全部进 config.py）

| 异常类型 | 触发条件 | severity | label_ref | 说明 |
|---------|---------|:---:|---------|------|
| 现金流背离 | 经营现金流/净利润 < FIN_CF_TO_PROFIT(1.0) | 3 | 盈利质量 | 利润质量存疑 |
| **双负信号（新增兜底）** | **净利润<0 且 经营现金流<0** | 4 | 盈利质量 | 比值负/负失真为正，直接预警 |
| 高负债 | 资产负债率 > FIN_DEBT_RATIO_MAX(70%) | 3 | 偿债能力 | |
| 亏损 | ROE < FIN_ROE_NEGATIVE(0) | 4 | 盈利能力 | |
| 盈利持续恶化 | ROE 近4期趋势斜率 < FIN_ROE_TREND_SLOPE(-5) | 4 | 盈利能力 | |
| 行业偏离 | 任一指标 \|Z\| > FIN_Z_SCORE(2) | 2 | 行业对标 | |

> **双负兜底原理**（修复现有 bug）：000004 案例 2026Q1 净利润 -639 万、经营现金流 -2798 万，`cf_to_profit = (-2798)/(-639) = 4.38 > 1`，旧规则"现金流背离"不触发——但公司其实深度恶化。兜底：**净利润与现金流双负时，比值失真，直接给"双负信号"**。

---

## 四、异常输出格式（归因 Agent 直接可用）

```json
{
  "type": "现金流背离",
  "severity": 3,
  "indicator": "cf_income_ratio",
  "value": 0.31,
  "threshold": "< 1.0",
  "evidence": "经营现金流/净利润 = 0.31 < 1.0（2024年报）",
  "label_ref": "盈利质量"
}
```

- `indicator`：特征名（与 F2 特征对齐，供 SHAP 归因对照）
- `evidence`：带原文数字的说明（供报告展示）
- `label_ref`：监管关注点标签（与公告研读的风险标签体系对齐 → 案例检索可做"财务模式+语义标签"混合对标）

**存放位置：写入共享 context**（`ctx.financial.anomaly_list`）——归因 Agent 读取它做证据引用，报告 Agent 读取它做诱因展示，预测 Agent 读取 features 做模型输入。

---

## 五、特殊行业跳过（现阶段策略）

| 行业 | 策略 |
|------|------|
| 金融业 / 房地产业 | 跳过常规财务分析（高杠杆属行业常态），输出行业特点说明 |
| 建筑业 | 不跳过（垫资模式需看回款） |
| 无数据/网络失败 | 跳过，记录 skip_reason |

> 后续优化方向（非现阶段）：金融看不良率/拨备覆盖率、地产看现金流/有息负债结构——用行业专属规则替代"一刀切跳过"。

---

## 六、迁移映射（现有代码 → backend）

| 现有文件 | 迁到 | 改动 |
|---------|------|------|
| fin_detector_agent.py | backend/agents/financial_detector.py | 签名改 execute(company, ctx)；异常输出改新格式；加双负兜底；阈值走 config |
| data_fetcher.py | backend/skills/financial_data_fetch.py | 基本原样 |
| llm_client.py（analyze/resolve） | 并入 backend/llm.py 调用（chat/chat_json） | 提示词内联在 agent |
| sweep_pipeline.py | 参考 → orchestrator | 拆 Phase |

---

## 七、待办（后续全面开发）

- [ ] 补齐 F2 特征族（Benford/M-Score/F-Score、应收/存货/应计、Q4收入占比、毛利率波动率）
- [ ] F3 市场异动（超额收益/放量/波动率）、F5 治理、F6 历史监管
- [ ] 官方数据集接入（披露日对齐）
- [ ] 行业专属规则替代"跳过"
- [ ] 用比赛标签校准阈值（哪些阈值组合真能预测问询）
