# 归因解释 Agent（Attributor）设计方案与内部框架

> 用途：回答"**为什么预测这家公司会被问询**"——把黑盒概率转成"诱因 + 证据 + 案例"三元组。
> 定位：7-Agent 流水线第 6 环（公告研读 → 财务检测 → 预测 → 案例检索 → **归因** → 报告）。
> 对应赛题任务 4：可解释性（逻辑解释有效性 ≥ 85 分、证据片段召回 ≥ 85%）。

---

## 一、归因 Agent 在流水线中的位置

```
公告研读Agent ──→ ctx.semantic.{risk_factors, evidence_snippets}  ─┐
财务检测Agent ──→ ctx.financial.{anomaly_list, features}         ──┤
预测建模Agent ──→ ctx.prediction.{probability, confidence}        ──┼─→ 归因Agent ──→ 报告Agent
案例检索Agent ──→ ctx.cases (Top-5 相似案例)                      ──┘      │
                                                                         └─→ ctx.attribution
```

**归因 Agent 是"汇合点"**：它同时消费预测概率、财务异常、公告风险标签/证据、相似案例——等前面全部完成后才执行。

---

## 二、输入 / 输出

### 输入（全部来自共享 ctx）

| 数据 | 来源 | 用途 |
|------|------|------|
| `ctx.prediction`（含 **shap_features** 特征贡献） | 预测 Agent | 概率/等级/置信度 + SHAP 特征贡献（Step1 已前移到预测 Agent） |
| `ctx.financial.anomaly_list` | 财务检测 | 财务侧证据（indicator/label_ref/evidence） |
| `ctx.financial.anomaly_list` | 财务检测 | 财务侧证据（indicator/label_ref/evidence） |
| `ctx.semantic.evidence_snippets` + `risk_factors` | 公告研读 | 文本侧证据 |
| `ctx.cases` | 案例检索 | 相似案例对照 |

### 输出（写入 ctx.attribution）

```json
{
  "top_risk_factors": [
    {"feature": "cf_income_ratio", "shap": -0.21,
     "description": "经营现金流/净利润仅0.31，盈利质量差",
     "label_ref": "盈利质量", "evidence_id": "fin_001"}
  ],
  "evidence_citations": [
    {"evidence_id": "fin_001", "source": "2024年报",
     "snippet": "经营现金流 1,234 万元，同比下降 71.2%..."}
  ],
  "case_links": [
    {"case_id": "IC-000005-2023-0614", "company": "ST星源",
     "similarity": 0.81, "inquiry_type": "年报问询函",
     "topics": ["非经营性资金占用", "保留意见"]}
  ],
  "narrative": "该公司被问询概率 0.72。关键诱因是经营现金流/净利润仅 0.31（SHAP 贡献最高）...",
  "confidence": 0.87
}
```

---

## 三、内部框架（六步流水线）

```
AttributorAgent.execute(company, ctx)
│
├─ Step 1  读取特征贡献（已前移到预测 Agent）
│     PredictorAgent 推理时同时计算 SHAP（TreeExplainer）写入 ctx.prediction.shap_features
│     本 Agent 直接读取 → 取 Top-K（按 |shap|）；缺失时降级用财务异常+风险标签
│
├─ Step 2  特征 → 风险因素映射（FEATURE_MAP 映射表）
│     黑盒特征名 → {中文描述, label_ref, 证据来源}
│     cf_income_ratio → {"经营现金流/净利润", "盈利质量", "financial"}
│     → Top-K 可读风险因素
│
├─ Step 3  证据定位（evidence_locate）
│     按 indicator / label_ref 匹配：
│       财务因素 → ctx.financial.anomaly_list（已带 evidence）
│       语义因素 → ctx.semantic.evidence_snippets（公告原文片段）
│     → 每个因素绑定 evidence_id（无证据的因素降级/剔除）
│
├─ Step 4  案例链接（case_link）
│     取 top 因素的 label_ref 集合 ∩ 相似案例 topics 重合度最高的 1-3 个
│     → "当前公司 ↔ 历史被问询公司" 对照
│
├─ Step 5  归因叙事生成（narrative_generate，LLM）
│     输入：结构化因素 + 证据池（白名单）+ 案例对照
│     约束：只允许引用证据池中的 evidence_id，禁止编造
│     → 自然语言归因报告
│
└─ Step 6  聚合校验输出
     置信度 = 模型置信度；结构完整性校验（缺证据/缺案例则标记）
     → ctx.attribution
```

---

## 四、关键组件：特征映射表（SHAP 黑盒 → 人话的桥梁）

SHAP 输出的是特征名（`cf_income_ratio`），必须映射成"能读的风险因素"：

```python
FEATURE_MAP = {
    "cf_income_ratio":      {"desc": "经营现金流/净利润（盈利质量）", "label_ref": "盈利质量", "source": "financial"},
    "debt_to_assets_ratio": {"desc": "资产负债率",                   "label_ref": "偿债能力", "source": "financial"},
    "roe":                  {"desc": "净资产收益率",                 "label_ref": "盈利能力", "source": "financial"},
    "roe_trend_4q":         {"desc": "ROE 近4期趋势（斜率）",        "label_ref": "盈利能力", "source": "financial"},
    "anomaly_count":        {"desc": "财务异常信号数量",             "label_ref": "综合",     "source": "financial"},
    "max_severity":         {"desc": "最高异常严重度",               "label_ref": "综合",     "source": "financial"},
    # 语义侧（公告风险标签 one-hot 后的特征名）
    "label_收入确认":        {"desc": "公告中收入确认类风险信号",      "label_ref": "收入确认", "source": "semantic"},
    "label_商誉减值":        {"desc": "公告中商誉减值类风险信号",      "label_ref": "商誉减值", "source": "semantic"},
}
```

> 这份映射表正是 `scripts/build_concern_dict.py`（关注点↔指标词典）的产物之一——归因 Agent 与词典打通。

---

## 五、防幻觉三件套（评委最关心）

1. **证据白名单**：LLM prompt 只提供"证据池"（evidence_id + 片段），强制"只能引用池中 evidence_id，禁止编造"；
2. **阈值过滤**：`|shap| < 阈值` 的因素不进叙事；找不到证据的因素降级为"模型侧信号"，不写进归因；
3. **结构校验**：LLM 输出的 narrative 里引用的 evidence_id 必须都在池中——不在则剔除或重试一次。

---

## 六、代码骨架（backend/agents/attributor.py 填充方向）

```python
from ..llm import chat_json
from .base import AgentBase

class AttributorAgent(AgentBase):
    name = "Attributor"

    def __init__(self, top_k=5, shap_threshold=0.05, use_llm=True):
        super().__init__()
        self.top_k = top_k
        self.shap_threshold = shap_threshold
        self.use_llm = use_llm

    # Step 1（前移到预测 Agent）：本 Agent 只读取
    def read_shap(self, ctx):
        shap_list = ctx.prediction.get("shap_features", [])   # [(feature, value), ...]
        ranked = sorted(shap_list, key=lambda x: -abs(x[1]))
        return [(n, v) for n, v in ranked[:self.top_k] if abs(v) >= self.shap_threshold]

    # Step 2
    def map_factors(self, top_features):
        return [{"feature": n, "shap": v, **FEATURE_MAP.get(n,
                {"desc": n, "label_ref": "其他", "source": "unknown"})}
                for n, v in top_features]

    # Step 3
    def evidence_locate(self, factors, ctx):
        fin_map = {a.get("indicator"): a for a in ctx.financial.anomaly_list}
        sem_map = {s.get("label_ref"): s for s in ctx.semantic.evidence_snippets}
        for f in factors:
            if f["source"] == "financial" and f["feature"] in fin_map:
                f["evidence_id"] = fin_map[f["feature"]].get("evidence_id", "fin_" + f["feature"])
            elif f["source"] == "semantic" and f["label_ref"] in sem_map:
                f["evidence_id"] = sem_map[f["label_ref"]].get("evidence_id")
        return [f for f in factors if f.get("evidence_id")]   # 无证据的剔除

    # Step 4
    def case_link(self, factors, cases):
        labels = {f["label_ref"] for f in factors}
        return [c for c in cases if set(c.get("topics", [])) & labels][:3]

    # Step 5（防幻觉：证据白名单）
    def narrative_generate(self, company, prediction, factors, evidence_pool, case_links):
        prompt = (
            f"公司 {company} 被问询概率 {prediction.get('probability_60d')}，风险等级 {prediction.get('risk_level')}。\n"
            f"Top 风险因素：{factors}\n证据池（只能引用以下 evidence_id）：{evidence_pool}\n"
            f"相似案例：{case_links}\n"
            "请生成 3-5 句话的归因解释，必须引用证据池中的 evidence_id，禁止编造。"
        )
        return chat_json("你是资深投研/合规专家，生成可复核的归因解释。", prompt, temperature=0.1)

    # 主入口
    def execute(self, company, ctx):
        factors = self.map_factors(self.read_shap(ctx))
        factors = self.evidence_locate(factors, ctx)
        links = self.case_link(factors, ctx.cases)
        evidence_pool = [{"evidence_id": f["evidence_id"], "snippet": f.get("evidence")}
                         for f in factors]
        ctx.attribution = {
            "top_risk_factors": factors,
            "evidence_citations": evidence_pool,
            "case_links": links,
            "narrative": self.narrative_generate(company, ctx.prediction, factors,
                                                 evidence_pool, links) if self.use_llm else "",
            "confidence": ctx.prediction.get("confidence", 0),
        }
        return ctx
```

> 前置依赖：`ctx.features` 必须含 `feature_names`（预测 Agent 组装时写入）；预测模型由 orchestrator 注入。

---

## 七、三个案例（000005/000009/000014）与归因 Agent 的关系

你的 `05_09_14.csv` 是**问询事件索引**（secucode + 发函日 + 标题 + 监管关注点 JSON），
三个 rar 里的 PDF 是**函件原文证据**。它们的用途是**建历史案例库**（供案例检索 Agent 用），
归因 Agent 只消费"检索到的相似案例"。

案例入库 Schema：

```json
{
  "case_id": "IC-000005-2023-0614",
  "company": "000005.SZ",
  "company_name": "ST星源",
  "publish_date": "2023-06-14",
  "inquiry_type": "年报问询函",
  "focus_points": ["营业收入、扣非前后净利润及经营性现金流下滑原因",
                    "持续经营能力不确定性", "保留意见事项", "违规担保", "资金占用"],
  "focus_vector": [0.01, 0.23, ...],       // embedding（BGE/stella）
  "letter_excerpt": "...",                  // 函原文片段（PDF 解析）
  "reply_excerpt": "..."                    // 回复要点（PDF 解析，可选）
}
```

> ⚠️ 注意：CSV 是 GBK 编码，Python 读取用 `encoding="gbk"`；rar 需先解压（本地无 7-Zip/WinRAR）。

---

## 八、三个 Agent 的编排关系（公告研读 / 案例检索 / 归因）

### 8.1 各司其职

| Agent | 分析对象 | 核心产出 | 服务谁 |
|-------|---------|---------|--------|
| 公告研读 | **目标公司**的公告 | 目标公司风险标签 + 证据片段 | 预测（特征）、案例检索（画像）、归因（证据） |
| 案例检索 | **历史问询案例库**（RAG） | 相似案例 Top-5（含相似度/关注点） | 归因（案例对照）、报告 |
| 归因 | 前面所有产出 | "诱因+证据+案例"三元组 + 叙事 | 报告 |

### 8.2 调度顺序（归因是汇合点）

```
Phase A（并行）  公告研读 ∥ 财务检测
Phase B          特征组装
Phase C（并行）  预测（含 SHAP）∥ 案例检索      ← 案例检索需要公告研读的风险标签（画像）
Phase D          归因（依赖 C 的两个输出）       ← 汇合点
Phase E          报告
```

### 8.3 数据流转

| 数据 | 生产者 | 消费者 |
|------|--------|--------|
| 目标公司风险标签 + 证据片段 | 公告研读 | 预测(特征)、案例检索(画像)、归因(证据) |
| 财务特征 + 异常清单 | 财务检测 | 预测、案例检索(画像)、归因(证据) |
| 概率 + SHAP 特征贡献 | 预测 | 归因、报告 |
| 相似案例 Top-5 | 案例检索 | 归因(案例对照)、报告 |

---

## 九、证据白名单（防幻觉）应在哪些地方落地

**原则：所有"LLM 生成内容"必须绑定原文证据 ID；所有入库证据必须是原文引用。** 落地位置：

| 位置 | 防幻觉措施 | 状态 |
|------|-----------|------|
| ① 公告研读抽取 | Prompt 强制 `evidence` 必须是公告正文原话，找不到原文就不输出该条 | ✅ 已实现 |
| ② 案例解析抽取 | 同上（evidence = 问询函/回复原句 verbatim） | ⏳ 待实现（build_case_vector_db） |
| ③ 案例检索（RAG） | 检索是向量相似度，非 LLM 生成，天然有据；但案例摘录必须存**原文**（不存 LLM 转述） | ⏳ 待实现 |
| ④ 归因叙事生成 | 只允许引用证据池 evidence_id（白名单） | ✅ 本 Agent 已实现 |
| ⑤ 报告叙事 | 同上（引用归因给出的 evidence_id） | ⏳ 待实现 |

**RAG 的"防幻觉"本质**：检索本身不生成，所以不会编造——但前提是**库里存的是原文摘录**（而非 LLM 摘要），否则检索结果是"二手转述"，溯源就断了。案例库 Schema 里 `letter_excerpt` 必须存 PDF 原句。
