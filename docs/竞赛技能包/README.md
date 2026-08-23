# 竞赛技能包 · 智能风控与量化建模赛道 · 东吴证券

> 基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警算法探索
> 本目录是项目的"技能包"：任何开发会话先读本 README，再按需取用对应技能文件。

## 技能索引

| 文件 | 技能 | 解决什么问题 |
|---|---|---|
| `01_领域知识.md` | 赛题与监管问询领域知识 | 赛题要求、问询函机制、关注点体系 A-H/45 二级、数据源与交付物 |
| `02_数据与特征.md` | 数据资产与特征工程 | F1-F6 特征、建模数据集、4785 案例库、向量库、索引/缓存 |
| `03_建模与评估.md` | 建模与评估 | 三模型集成(RF/LGB/XGB)、SHAP、指标口径、正样本率提升方案 |
| `04_Agent编排.md` | Agent 与 LangGraph 编排 | 7 Agent 接口、LangGraph 图、checkpointer、trace/防幻觉约定 |
| `05_开发与验收.md` | 开发规范与验收清单 | 自测入口、测试、页面/端口约定、git 规范、答辩演示清单 |
| `06_提示词模板.md` | LLM 提示词与结构化输出 | 抽取/归因/报告 prompt 模板、chat_structured 用法 |

## 快速上手（新会话三分钟）

```bash
# 1. 看架构总览（README.md 根文档）+ 本技能包 README
# 2. 跑通自测
.venv\Scripts\Activate.ps1
python -m backend.agents.orchestrator          # 全流程（LangGraph 图编排）
python -m backend.agents.graph                 # 直接跑图（含 trace 摘要）
# 3. 打开演示
streamlit run 导航入口.py                       # http://localhost:8501
```

## 关键约定（改代码前必读）

1. **Agent 接口**：所有 Agent 实现 `execute(company, ctx)`，只读写共享 `Context`；
2. **防幻觉**：LLM 生成内容必须绑定原文证据 ID；证据取原文，不存 LLM 转述；
3. **编排**：首选 LangGraph（`backend/agents/graph.py`），`orchestrator` 是薄封装；
4. **LLM**：统一走 `backend/llm.py`（LangChain 双通道），不要直接调 API；
5. **端口**：主控 8501 起，各 Agent 审计页 8502-8507；
6. **依赖**：LangChain/LangGraph 锁 1.x（见 requirements.txt），升级需单独评估；
7. **git**：本地提交后经确认再推送（此前有过"未推送"写进提交说明的教训）。
