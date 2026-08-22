# -*- coding: utf-8 -*-
"""agents 包：7 个 Agent + 基类 + 主控编排。

填充说明：各 Agent 类实现后，取消下方注释即可统一导出。
当前已实现：base（AgentBase/TraceLogger）。
"""
from .base import AgentBase, TraceLogger
from .announcement_reader import AnnouncementReaderAgent
from .financial_detector import FinancialDetectorAgent
from .case_retriever import CaseRetrieverAgent
from .attributor import AttributorAgent
from .orchestrator import SweepingOrchestrator

# from .predictor import PredictorAgent
# from .reporter import ReporterAgent

__all__ = ["AgentBase", "TraceLogger", "AnnouncementReaderAgent",
           "FinancialDetectorAgent", "CaseRetrieverAgent",
           "AttributorAgent", "SweepingOrchestrator"]
