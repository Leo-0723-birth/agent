#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享 Context（Agent 间唯一通信方式）
====================================
所有 Agent 不直接互相调用，只读写同一个 Context 对象：
    - 每个 Agent 实现 execute(company, ctx)：读 ctx 需要的字段，写自己的字段
    - 主控 Orchestrator 控制读写顺序与并行

用法：
    from backend.context import Context
    ctx = Context(company="000004.SZ", window=60)
    AnnouncementReaderAgent().execute("000004.SZ", ctx)
    print(ctx.semantic.risk_factors)
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Semantic:
    """← 公告研读 Agent 产出"""
    announcements: list = field(default_factory=list)       # 公告元数据（不含全文）
    finbert_signals: list = field(default_factory=list)     # FinBERT 粗分类信号
    risk_factors: list = field(default_factory=list)        # LLM 抽取风险要素（跨公告汇总）
    risk_labels: list = field(default_factory=list)         # RiskMapper Agent 输出
    evidence_snippets: list = field(default_factory=list)   # 证据片段（原文引用）
    per_announcement: dict = field(default_factory=dict)    # 每份公告的抽取结果
    stats: dict = field(default_factory=dict)               # 统计（公告数/要素数/门控数）


@dataclass
class Financial:
    """← 财务异常检测 Agent 产出"""
    features: dict = field(default_factory=dict)            # F2-F6 特征
    anomaly_list: list = field(default_factory=list)        # 异常清单（type/severity/indicator/value/threshold/evidence/label_ref）
    indicators: dict = field(default_factory=dict)          # 原始指标（含 report_period）
    benchmarks: dict = field(default_factory=dict)          # 行业对标 Z-Score
    industry: str = ""                                      # 行业
    risk_level: str = ""                                    # 低/中/高/跳过
    skip: bool = False                                      # 是否跳过财务分析（特殊行业等）
    skip_reason: str = ""                                   # 跳过原因
    llm_analysis: str = ""                                  # LLM 财务解读（可选）


@dataclass
class Context:
    """共享上下文（主时钟 = 公司 + 预测时点）"""
    company: str = ""                # 公司代码（000004.SZ）
    name: str = ""                   # 公司名称
    window: int = 60                 # 预测窗口（天）
    as_of: str = ""                  # 预测时点 T（Y-m-d，特征锚定时刻）

    semantic: Semantic = field(default_factory=Semantic)    # 公告研读
    financial: Financial = field(default_factory=Financial)  # 财务检测
    features: dict = field(default_factory=dict)            # 特征组装后（F1-F6 拼接）
    prediction: dict = field(default_factory=dict)          # 预测：概率/等级/置信度
    cases: list = field(default_factory=list)               # 案例检索：相似案例 Top-5
    attribution: dict = field(default_factory=dict)         # 归因：诱因/证据/案例对照
    report: dict = field(default_factory=dict)              # 报告
    trace_log: list = field(default_factory=list)           # 全链路追踪

    def to_dict(self) -> dict:
        """转 dict（供 JSON 序列化/持久化）。"""
        return asdict(self)
