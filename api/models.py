"""Pydantic 数据模型 —— 与前端 index.html 的 JS 数据结构对齐。"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.config import SCAN_MAX_DOCUMENTS

# 模型实际训练的预测窗口；任意 window 吸附到最近的支持窗口，
# 避免"请求 45 天却静默按 60 天计算"且响应不透明的参数不一致。
SUPPORTED_WINDOWS = (30, 60, 90)


def _snap_window(value: int) -> int:
    if value in SUPPORTED_WINDOWS:
        return value
    return min(SUPPORTED_WINDOWS, key=lambda w: (abs(w - value), -w))


class FactorItem(BaseModel):
    name: str
    tag: str            # finance / text / market
    tagText: str
    desc: str
    score: float        # 0~1
    color: str
    evidenceId: Optional[str] = ""      # 关联的风险证据 ID，用于前端跳转
    evidenceAnchor: Optional[str] = ""  # 前端锚点 id


class AnnouncementRiskItem(BaseModel):
    date: str                           # 公告日期
    level: str                          # 风险等级（高/中/低）
    l1: str                             # 一级分类
    l2: str                             # 二级分类
    description: str                    # 风险描述
    evidence: str                       # 原文证据
    title: str                          # 公告标题
    sourceUrl: Optional[str] = ""       # 巨潮详情页
    pdfUrl: Optional[str] = ""          # PDF 原文


class AttributionEvidenceItem(BaseModel):
    factor: str                         # 因子名称
    evidence: str                       # 归因原文/证据
    source: Optional[str] = ""          # 来源说明
    anchor: Optional[str] = ""          # 前端锚点


class FinancialAnomalyItem(BaseModel):
    type: str                           # 异常类型
    severity: int                       # 严重度 1-5
    indicator: str                      # 指标名
    value: Any                          # 本期值
    threshold: str                      # 阈值说明
    evidence: str                       # 证据描述
    label_ref: str                      # 标签体系归类


class SimilarCaseItem(BaseModel):
    caseId: str
    company: str
    publishDate: str
    inquiryType: str
    topics: List[str]
    similarity: float
    matchReason: List[str]


class AnalyzeResponse(BaseModel):
    code: str
    name: str
    risk: float                         # 问询概率 %
    level: str                          # low / mid / high
    levelText: str                      # 低风险 / 中风险 / 高风险
    confidence: float                   # 0~1
    confidenceMeaning: str = "predicted_class_score"  # 非统计置信区间
    dataSource: str = "unknown"         # realtime / offline_lookup / offline_snapshot
    dataCoverage: dict = Field(default_factory=dict)
    announcementPdfs: dict = Field(default_factory=dict)  # 公告 PDF 获取情况：fetched/parsed/analyzed/metadata_total/window_days
    degradedReasons: List[str] = Field(default_factory=list)
    featureAnchor: str = ""
    modelVersion: str = ""
    riskByWindow: Optional[dict] = Field(default_factory=dict)
    windowPredictions: Optional[List[dict]] = Field(default_factory=list)  # 30/60/90 各自完整预测（risk+confidence+factors+metrics），供前端窗口切换
    modelMetrics: Optional[dict] = Field(default_factory=dict)              # 三窗口模型评估指标（AUC/F1/Top10/Threshold）
    factors: int                        # 风险因子总数
    summary: str                        # HTML 摘要
    factorsList: List[FactorItem]
    financialTable: List[List[str]]     # [[指标, 本期值, 行业均值, 偏离度], ...]
    textSummary: str
    caseMatch: str
    attribution: str
    conclusion: str
    advice: str
    # 扩展字段（真实数据补充）
    reportMarkdown: Optional[str] = ""
    traceLog: Optional[list] = Field(default_factory=list)
    similarCases: Optional[List[SimilarCaseItem]] = Field(default_factory=list)
    generatedAt: Optional[str] = ""
    # 风险证据与公告研读明细（前端仪表盘卡片数据）
    announcementRisks: Optional[List[AnnouncementRiskItem]] = Field(default_factory=list)
    announcementReview: Optional[dict] = Field(default_factory=dict)
    attributionEvidence: Optional[List[AttributionEvidenceItem]] = Field(default_factory=list)
    riskFactorDetails: Optional[List[FactorItem]] = Field(default_factory=list)
    financialAnomalies: Optional[List[FinancialAnomalyItem]] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    codes: List[str] = Field(..., description="公司代码列表，如 ['000063.SZ']")
    window: int = Field(60, description="预测窗口天数（非 30/60/90 时吸附到最近窗口）")
    use_llm: bool = Field(True, description="是否启用LLM精细抽取（默认开启，逐份深度抽取更全面）")
    use_bge: bool = Field(True, description="是否启用BGE语义检索")

    @field_validator("window")
    @classmethod
    def snap_window(cls, value: int) -> int:
        return _snap_window(value)


# ==================== 方案 C：实时扫雷 ====================

class ScanRequest(BaseModel):
    code: str = Field(..., description="公司代码，如 000001.SZ")
    window: int = Field(60, ge=1, le=365, description="预测窗口天数（非 30/60/90 时吸附到最近窗口）")
    as_of: Optional[str] = Field(None, description="分析截止日期（YYYY-MM-DD），空则默认今天")
    use_llm: bool = Field(True, description="是否启用LLM精细抽取（默认开启，与FinBERT门控配合聚焦候选）")
    use_bge: bool = Field(True, description="是否启用BGE语义检索")
    max_documents: Optional[int] = Field(
        SCAN_MAX_DOCUMENTS, ge=1, le=150,
        description="公告研读深读文档数：从近一年公告中按时间最近取前 N 份 PDF（50/100/150）",
    )
    realtime: bool = Field(False, description="默认返回离线快照；为 true 时执行实时 Agent 流水线")
    force: bool = Field(False, description="是否取消当前任务并切换到新公司")

    @field_validator("window")
    @classmethod
    def snap_window(cls, value: int) -> int:
        return _snap_window(value)

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        from datetime import date

        normalized = str(value).strip()
        try:
            return date.fromisoformat(normalized).isoformat()
        except ValueError as exc:
            raise ValueError("as_of 必须是有效的 YYYY-MM-DD 日期") from exc


class ScanResponse(BaseModel):
    task_id: str
    code: str
    status: str
    message: str
    cached: bool = False
    result: Optional[AnalyzeResponse] = None


class ProgressMessage(BaseModel):
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    type: Literal["progress", "complete", "error", "fallback", "cancelled"] = "progress"
    step: int = 0
    total: int = 0
    agent: str = ""
    agent_key: str = ""
    status: str = "running"             # pending / running / done / skipped / error / cancelled
    progress_percent: int = Field(0, ge=0, le=100)
    message: str = ""
    elapsed_ms: int = 0
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
    fatal: bool = False
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
