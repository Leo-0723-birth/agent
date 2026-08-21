# -*- coding: utf-8 -*-
"""backend —— 扫雷预警系统后端包。

结构：
  config.py    全局配置（路径/参数/模型）
  llm.py       共享 LLM 客户端（一行 import 即用）
  context.py   共享 Context（Agent 间唯一通信）
  agents/      7 个 Agent + 基类 + 主控编排
  skills/      原子能力（Skill）
  models/      模型权重与训练产物
  data/        运行时数据（raw/index/vector_db/output）
  scripts/     离线任务（建库/训练/词典）
  tests/       单元测试
"""
