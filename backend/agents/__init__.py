# -*- coding: utf-8 -*-
"""Agent 公共导出；具体实现按需加载，避免无关可选依赖互相阻塞。"""
from .base import AgentBase, TraceLogger

__all__ = ["AgentBase", "TraceLogger", "AnnouncementReaderAgent",
           "FinancialDetectorAgent", "CaseRetrieverAgent", "ChunkRetrieverAgent",
           "AttributorAgent", "ReporterAgent", "SweepingOrchestrator"]

_LAZY_EXPORTS = {
    "AnnouncementReaderAgent": (".announcement_reader", "AnnouncementReaderAgent"),
    "FinancialDetectorAgent": (".financial_detector", "FinancialDetectorAgent"),
    "CaseRetrieverAgent": (".case_retriever", "CaseRetrieverAgent"),
    "ChunkRetrieverAgent": (".chunk_retriever", "ChunkRetrieverAgent"),
    "AttributorAgent": (".attributor", "AttributorAgent"),
    "ReporterAgent": (".reporter", "ReporterAgent"),
    "SweepingOrchestrator": (".orchestrator", "SweepingOrchestrator"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
