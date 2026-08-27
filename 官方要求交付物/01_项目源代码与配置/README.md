# 项目源代码与配置索引

完整源代码位于仓库根目录：

- `backend/agents/`：公告研读、财务检测、预测、案例、chunk、归因、报告和 LangGraph 编排。
- `backend/skills/`：公告检索、OCR、风险标签、特征计算、向量检索和报告渲染。
- `backend/models/`：预测模型和 Embedding 资产。
- `backend/data/`：标签、案例库、建模数据、离线公告和报告输出。
- `requirements.txt`：完整运行依赖。
- `导航入口.py`：Streamlit 单入口演示系统。
- `主控agent.py`：单公司/批量扫雷主页面。

推荐入口：`streamlit run 导航入口.py`。

## 已核对的核心资产

当前仓库已存在 7 个 Agent、LangGraph 图编排、预测模型 9 个窗口权重文件、`models_manifest.json`、4785 份案例库、官方风险标签资产和离线公告快照。模型侧建模数据记录为 37,222 条、205 列；预测结果和评估产物已复制到 `../03_测试集结果/`，便于评委先看成果再查源码。

默认关闭 LLM，使用离线资产即可完成可审计演示；实时与离线兜底状态会由预测结果的 `data_source` 和 `coverage` 字段区分。
