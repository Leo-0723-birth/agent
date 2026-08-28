"""Pydantic 数据模型 —— 与前端 index.html 的 JS 数据结构对齐。"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


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
    riskByWindow: Optional[dict] = {}   # {"30d": x, "60d": y, "90d": z}（单位 %，前端切换窗口用）
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
    traceLog: Optional[list] = []
    similarCases: Optional[List[SimilarCaseItem]] = []
    generatedAt: Optional[str] = ""
    # 新增：融合 Streamlit 细节
    announcementRisks: Optional[List[AnnouncementRiskItem]] = []    # 公告研读风险证据表
    attributionEvidence: Optional[List[AttributionEvidenceItem]] = []  # 风险归因原文
    riskFactorDetails: Optional[List[FactorItem]] = []              # 全部风险因子（查看全部用）
    financialAnomalies: Optional[List[FinancialAnomalyItem]] = []   # 财务异常信号清单


class AnalyzeRequest(BaseModel):
    codes: List[str] = Field(..., description="公司代码列表，如 ['000063.SZ']")
    window: int = Field(60, description="预测窗口天数")
    use_llm: bool = Field(False, description="是否启用LLM精细抽取")
    use_bge: bool = Field(True, description="是否启用BGE语义检索")


# ==================== 方案 C：实时扫雷 ====================

class ScanRequest(BaseModel):
    code: str = Field(..., description="公司代码，如 000001.SZ")
    window: int = Field(60, ge=1, le=365, description="预测窗口天数")
    as_of: Optional[str] = Field(None, description="分析截止日期（YYYY-MM-DD），空则默认今天")
    use_llm: bool = Field(False, description="是否启用LLM精细抽取（演示建议关闭，可大幅提速）")
    use_bge: bool = Field(True, description="是否启用BGE语义检索")
    max_documents: Optional[int] = Field(5, ge=1, le=100, description="公告研读最大文档数（越小越快）")
    realtime: bool = Field(False, description="默认返回离线快照；为 true 时执行实时 Agent 流水线")
    force: bool = Field(False, description="是否取消当前任务并切换到新公司")


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
