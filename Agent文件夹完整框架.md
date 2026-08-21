# Agent 文件夹完整框架（backend/ 全量设计）

> 用途：扫雷预警系统后端的完整目录规划——每个文件放什么、每个 Agent 文件里有什么、共享数据怎么定义。
> 与现有代码的关系：公告研读 agents 的 6 个文件（announcement_reader 等）整体迁入本框架的 `backend/agents/` 与 `backend/skills/`。

---

## 一、完整目录树

```
competition_agent/
├── backend/
│   ├── __init__.py                # 空（包标记）
│   ├── config.py                  # ★ 所有配置（路径/参数/模型/阈值）集中于此
│   ├── llm.py                     # ★ 共享 LLM 客户端（全项目唯一模型访问点）
│   ├── context.py                 # ★ 共享 Context（dataclass，Agent 间唯一通信）
│   ├── agents/                    # ★ 7 个 Agent + 基类 + 主控
│   │   ├── __init__.py            # 导出全部 Agent 与 Orchestrator
│   │   ├── base.py                # AgentBase（统一 execute 接口）+ TraceLogger
│   │   ├── orchestrator.py        # 主控：Plan→Dispatch→Aggregate（路线A自研版）
│   │   ├── announcement_reader.py # 公告研读（需 LLM）【现有代码迁入】
│   │   ├── financial_detector.py  # 财务异常检测（纯计算）【新建】
│   │   ├── case_retriever.py      # 监管案例检索（向量检索）【新建】
│   │   ├── predictor.py           # 预测建模（ML 推理）【新建】
│   │   ├── attributor.py          # 归因解释（需 LLM）【新建】
│   │   └── reporter.py            # 报告生成（需 LLM）【新建】
│   ├── skills/                    # ★ 原子能力（Agent 调用，不直接含业务编排）
│   │   ├── __init__.py
│   │   ├── announcement_search.py # 公告检索（现有 announcement_store 迁入/封装）
│   │   ├── financial_indicator_calc.py  # 财务指标计算+行业Z-Score
│   │   ├── benford_check.py       # Benford 定律检验 + M-Score
│   │   ├── contradiction_detect.py# 跨公告矛盾检测（需 LLM）
│   │   ├── embedding.py           # ★ 统一 Embedding 入口（BGE/stella 配置切换）
│   │   ├── vector_store.py        # ★ 向量库封装（ChromaDB/Milvus）
│   │   ├── inquiry_case_match.py  # 相似问询案例检索（embedding + 向量库）
│   │   ├── shap_explain.py        # SHAP 归因（预测模型解释）
│   │   └── risk_report_render.py  # 报告渲染（Markdown/JSON）
│   ├── models/                    # ★ 模型与权重（不放代码，只放产物）
│   │   ├── embedding/             # BGE-large-zh-v1.5 / stella-base-zh-v3 权重
│   │   ├── finbert/               # FinBERT2-base 权重（或引用 HF 缓存）
│   │   └── predictor/             # xgb_model.json / lgb_model.txt / stacking.pkl / scaler.pkl
│   ├── data/                      # ★ 数据（运行时产物）
│   │   ├── raw/                   # 原始公告 PDF（或 config 指向外部盘，避免复制）
│   │   ├── index/                 # 公告索引缓存（announcement_index.json）
│   │   ├── vector_db/             # ChromaDB 持久化（历史案例库 + 证据片段库）
│   │   └── output/                # context 快照 / 报告 / trace.log
│   ├── scripts/                   # ★ 离线任务（不进流水线，手动跑）
│   │   ├── build_case_vector_db.py   # 离线：解析历史问询函 → 建案例向量库
│   │   ├── train_predictor.py        # 离线：训练 XGBoost/集成预测模型
│   │   └── build_concern_dict.py     # 离线：抽取"关注点↔指标"映射词典
│   └── tests/
│       ├── test_announcement_store.py  # 日期抽取/窗口检索单测
│       └── test_context.py             # Context 读写单测
├── app.py                        # ★ Streamlit 入口（与 backend/ 同级）
├── requirements.txt              # 重写：streamlit/pymupdf/torch/transformers/...
├── .env / .env.example           # LLM 与模型配置（密钥不进代码）
└── README.md                     # 项目说明 + 运行方式 + 答辩要点
```

---

## 二、各层职责

### config.py —— 一切配置的单一来源
```python
# backend/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent      # competition_agent/
DATA_RAW = Path(os.getenv("DATA_RAW", r"D:\BaiduNetdiskDownload"))  # 原始公告根目录
OUTPUT_DIR = BASE_DIR / "backend" / "data" / "output"
INDEX_DIR = BASE_DIR / "backend" / "data" / "index"
VECTOR_DB_DIR = BASE_DIR / "backend" / "data" / "vector_db"

# LLM
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.1                # 低温度：稳定性（方案 5.4）
# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
# 预测
PREDICT_WINDOW = 60                  # 默认预测窗口（天）
RISK_THRESHOLDS = {"high": 0.6, "medium": 0.3}   # 风险等级阈值
# FinBERT 门控
FINBERT_GATE = 0.5                   # 粗分类分数低于此阈值的公告不送 LLM
```

### llm.py —— 共享 LLM（见《项目结构搭建与LLM按需调用.md》，一行 import 即用）

### context.py —— 共享数据（见下文第四节）

### agents/base.py —— 统一接口 + 追踪
现有 `agent_base.py` 直接迁入（AgentBase + TraceLogger），唯一改动：`execute` 签名统一为 `execute(company, ctx)`，并把 `trace` 追加进 `ctx.trace_log`。

### agents/__init__.py —— 统一导出
```python
# backend/agents/__init__.py
from .base import AgentBase, TraceLogger
from .announcement_reader import AnnouncementReaderAgent
from .financial_detector import FinancialDetectorAgent
from .case_retriever import CaseRetrieverAgent
from .predictor import PredictorAgent
from .attributor import AttributorAgent
from .reporter import ReporterAgent
from .orchestrator import SweepingOrchestrator

__all__ = [...]
```

---

## 三、每个 Agent 文件里应该包含什么（标准模板）

每个 Agent 文件固定五段结构（以 financial_detector 为例，纯计算型）：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务异常检测 Agent (FinancialDetector) —— 任务2的特征工程
========================================================
职责：计算 F2-F6 量化特征 + 异常科目清单（纯计算，不调用 LLM）。
输入：company（代码），ctx（读取原始财务/行情数据）
输出：写入 ctx.financial.{features, anomaly_list}
"""

# 第一段：import（需要 LLM 的 Agent 才有 `from ..llm import chat`）
import numpy as np
from ..config import DATA_RAW
from .base import AgentBase

# 第二段：Skill 实现（方法级原子能力）
class FinancialDetectorAgent(AgentBase):
    name = "FinancialDetector"

    def __init__(self, data_root=None):
        super().__init__()
        self.data_root = data_root or DATA_RAW

    # --- Skill 1: 指标计算 ---
    def calc_indicators(self, company):
        """返回 F2 财务特征 dict"""
        ...

    # --- Skill 2: Benford/M-Score ---
    def benford_check(self, report_numbers):
        """返回异常得分 + 可疑科目"""
        ...

    # 第三段：主入口（统一签名）
    def execute(self, company, ctx):
        """读 ctx 需要的数据 → 计算 → 写回 ctx.financial"""
        ctx.financial.features = self.calc_indicators(company)
        ctx.financial.anomaly_list = self.benford_check(...)
        return ctx

# 第四段：自测入口（每个 Agent 都要有，能独立运行验证）
if __name__ == "__main__":
    from ..context import Context
    ctx = Context(company="000004.SZ")
    agent = FinancialDetectorAgent()
    agent.execute("000004.SZ", ctx)
    print(ctx.financial.anomaly_list)
```

**模板约定**：
1. 类文档字符串必须写清：职责 / 输入 / 输出（写回 ctx 哪个字段）；
2. Skill = 方法级函数，一个方法一个原子能力（可被其他 Agent 复用）；
3. `execute(company, ctx)` 统一签名：**读 ctx 需要的、写自己的字段**；
4. 需要 LLM 的 Agent 才有 `from ..llm import chat`；纯计算 Agent 零 LLM 依赖；
5. 每个文件带 `if __name__ == "__main__"` 自测入口；
6. 所有失败必须兜底（try/except + 返回空结构 + 记入 ctx.trace_log），演示不许断。

---

## 四、共享 Context 完整字段定义

```python
# backend/context.py
from dataclasses import dataclass, field

@dataclass
class Semantic:                        # ← 公告研读产出
    announcements: list = field(default_factory=list)      # 公告元数据（不含全文，全文走索引）
    finbert_signals: list = field(default_factory=list)    # FinBERT 粗分类信号
    risk_factors: list = field(default_factory=list)       # LLM 抽取风险要素（跨公告汇总）
    evidence_snippets: list = field(default_factory=list)  # 证据片段（原文引用）
    per_announcement: dict = field(default_factory=dict)   # 每份公告的抽取结果

@dataclass
class Financial:                       # ← 财务异常检测产出
    features: dict = field(default_factory=dict)           # F2-F6 特征
    anomaly_list: list = field(default_factory=list)       # 异常科目清单

@dataclass
class Context:
    company: str = ""                  # 公司代码
    name: str = ""                     # 公司名称
    window: int = 60                   # 预测窗口（天）
    as_of: str = ""                    # 预测时点 T（主时钟锚点）

    semantic: Semantic = field(default_factory=Semantic)
    financial: Financial = field(default_factory=Financial)
    features: dict = field(default_factory=dict)           # 特征组装后（F1-F6 拼接）
    prediction: dict = field(default_factory=dict)         # 概率/风险等级/置信度
    cases: list = field(default_factory=list)              # 相似案例 Top-5
    attribution: dict = field(default_factory=dict)        # 归因三元组
    report: dict = field(default_factory=dict)             # 报告
    trace_log: list = field(default_factory=list)          # 全链路追踪
```

---

## 五、Skill 接口清单

| Skill | 输入 | 输出 | 被谁调用 |
|-------|------|------|---------|
| `announcement_search` | 公司代码+窗口 | 公告列表（含元数据） | 公告研读 |
| `financial_indicator_calc` | 公司代码 | F2 财务特征 dict | 财务异常检测 |
| `benford_check` | 报表数字 | 异常得分+可疑科目 | 财务异常检测 |
| `contradiction_detect` | 多份公告文本 | 矛盾点列表+证据 | 公告研读 |
| `embedding` | 文本列表 | 归一化向量 | 案例检索/证据向量化 |
| `vector_store` | 向量+元数据 | 写入/查询向量库 | 案例检索/公告研读 |
| `inquiry_case_match` | 特征向量+风险标签 | Top-K 案例+相似度 | 案例检索 |
| `shap_explain` | 模型+特征 | 特征贡献度排序 | 归因解释 |
| `risk_report_render` | 全量结果 | Markdown/JSON 报告 | 报告生成 |

---

## 六、models/ 与 data/ 的放置规则

| 内容 | 放哪里 | 说明 |
|------|--------|------|
| Embedding 权重（BGE/stella） | `models/embedding/` | 首次下载后本地化，离线可用 |
| FinBERT2-base 权重 | `models/finbert/`（或 HF 缓存 `HF_HOME`） | 现有 finbert_client 已用 |
| 预测模型产物 | `models/predictor/` | xgb/lgb/stacking/scaler，PredictorAgent 加载 |
| 历史问询案例向量库 | `data/vector_db/` | 由 `scripts/build_case_vector_db.py` 离线建 |
| 公告索引 | `data/index/` | announcement_index.json（现有已生成） |
| context 快照/报告/trace | `data/output/` | 演示产物与交付物 |

---

## 七、scripts/ 离线任务（不进流水线）

| 脚本 | 任务 | 何时跑 |
|------|------|--------|
| `build_case_vector_db.py` | 解析历史问询函 → 抽取关注点 → embedding → 入库 | 开发期一次；数据更新时重跑 |
| `train_predictor.py` | 特征工程 + 训练 XGBoost/集成 + 校准 + SHAP 基线 | 开发期；模型迭代时重跑 |
| `build_concern_dict.py` | 从问询函反解"关注点↔指标"映射词典 | 开发期一次 |

---

## 八、编码与协作规范

1. 全部文件 **UTF-8**，Windows 下脚本开头保留 `sys.stdout.reconfigure(encoding="utf-8")`（现有代码已有）；
2. **路径一律走 config.py**，禁止硬编码（现有 llm_client/run.py 的硬编码路径是待改点）；
3. `execute(company, ctx)` 统一签名；Agent 间**不直接互相调用**，只通过 ctx 通信；
4. trace：每个 Agent 执行完把 trace 追加进 `ctx.trace_log`（Orchestrator 最后汇总落盘）；
5. 降级：任何 Agent 失败返回空结构 + 错误记录，不抛断主流程；
6. 每个 Agent 自带 `__main__` 自测，改完先单跑验证再进流水线。

---

## 九、分阶段落地顺序

| 阶段 | 内容 | 对应现状 |
|------|------|---------|
| D0 | 建 backend 骨架：config/llm/context/base + 目录 | 1 天 |
| D1 | 公告研读迁入 backend/agents/（改 Context 化 + FinBERT 门控） | 现有代码改造，0.5-1 天 |
| D2 | 新建 financial_detector + predictor（接你训练的模型） | 1-2 天 |
| D3 | orchestrator 串起来 + Streamlit app.py（缺的 Agent 用假数据占位） | 1 天 |
| D4 | 新建 case_retriever/attributor/reporter + 离线建库脚本 | 2 天 |
| D5 | 演示打磨 + 单测 + requirements 重写 | 1 天 |

**原则**：先"纵向打通"（输入→概率→报告），再"横向补全"（7 Agent 全接上）。
