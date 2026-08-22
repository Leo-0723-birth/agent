# 基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警系统

## 项目简介

本项目面向上市公司监管问询风险识别场景，构建基于 Agentic AI
架构的智能风险预警系统。

系统实现： - 公告自动解析 - 风险因素抽取 - 监管关注点标签映射 -
财务异常检测 - 历史监管案例检索 - 风险归因解释

## 当前版本状态

当前版本为 Agent 流水线完善交接版本。

相比初始版本已完成：

  模块             状态
  ---------------- ------
  公告解析 Agent   完成
  风险因素抽取     完成
  风险标签映射     完成
  财务检测         完成
  案例检索 Agent   完成
  案例检索评估     完成
  风险归因         完成

## 主要改动

### CaseRetriever Agent

初始版本： - 仅基于文本Embedding检索 - 缺少监管标签约束 - 无匹配解释

当前版本： - 引入 BGE-large-zh-v1.5 - 融合监管风险标签 - Hybrid
Retrieval - RRF融合排序 - 输出匹配原因

流程：

风险因素 → Embedding检索 + 标签匹配 → RRF排序 → Top-K历史案例

## 案例检索实验结果

测试样本：

Samples: 300

Semantic Only: - Hit@5: 0.9933 - MRR: 0.9808

Label Only: - Hit@5: 0.81 - MRR: 0.7078

Hybrid RRF: - Hit@5: 1.0 - MRR: 0.9769 - Label Recall@5: 1.0

## 运行方式

Python 3.10

安装：

``` bash
pip install -r requirements.txt
```

运行：

``` bash
python run_demo.py
```

## 提交说明

GitHub版本不包含： - 模型权重 - API Key - **pycache**

保留： - 核心代码 - 必要数据 - 实验结果 - 文档
