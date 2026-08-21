#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置（单一来源）
====================
所有路径 / 参数 / 模型配置集中于此，业务代码禁止硬编码路径。
用法：
    from backend.config import DATA_RAW, FINBERT_GATE, RISK_THRESHOLDS
"""
import os
from pathlib import Path

# 尽量加载项目根 .env（密钥）；无 python-dotenv 时静默跳过，靠系统环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent                 # competition_agent/
DATA_RAW = Path(os.getenv("DATA_RAW", r"D:\BaiduNetdiskDownload"))  # 原始公告/问询函根目录
DATA_DIR = BASE_DIR / "backend" / "data"
INDEX_DIR = DATA_DIR / "index"          # 公告索引缓存（announcement_index.json）
VECTOR_DB_DIR = DATA_DIR / "vector_db"  # 向量库持久化（案例库/证据库）
OUTPUT_DIR = DATA_DIR / "output"        # context 快照 / 报告 / trace.log
MODEL_DIR = BASE_DIR / "backend" / "models"

# ---------- LLM（backend/llm.py 使用） ----------
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = 0.1        # 低温度：推理稳定（方案 5.4）
LLM_MAX_TOKENS = 2000

# ---------- Embedding ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")  # 或 stella-base-zh-v3
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fallback")  # bge 需显式启用；默认零下载且不混称真实模型

# ---------- 预测 ----------
PREDICT_WINDOW = 60                       # 默认预测窗口（天）
RISK_THRESHOLDS = {"high": 0.6, "medium": 0.3}   # 风险等级阈值

# ---------- 公告研读 ----------
ANNOUNCE_WINDOW_DAYS = 365    # 公告检索窗口（天）
ANNOUNCE_MAX_DOCUMENTS = int(os.getenv("ANNOUNCE_MAX_DOCUMENTS", "120"))
ANNOUNCE_PDF_CACHE = Path(
    os.getenv("ANNOUNCE_PDF_CACHE", str(DATA_DIR / "cache" / "pdfs"))
)
ANNOUNCE_SOURCE = os.getenv("ANNOUNCE_SOURCE", "cninfo").lower()
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
OCR_DPI = int(os.getenv("OCR_DPI", "180"))
OCR_MIN_PAGE_CHARS = int(os.getenv("OCR_MIN_PAGE_CHARS", "40"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.50"))
OCR_MAX_PAGES_PER_DOCUMENT = int(os.getenv("OCR_MAX_PAGES_PER_DOCUMENT", "80"))
FINBERT_GATE = float(os.getenv("FINBERT_GATE", "0.5"))
FINBERT_ENABLED = os.getenv("FINBERT_ENABLED", "false").lower() in {
    "1", "true", "yes", "on"
}
FINBERT_GATE_ENABLED = os.getenv("FINBERT_GATE_ENABLED", "false").lower() in {
    "1", "true", "yes", "on"
}  # 未经公告标注集校准前默认不启用门控
MAX_TEXT_CHARS = 8000         # 送 LLM 的公告正文截断长度

# ---------- 财务异常检测（backend/agents/financial_detector.py 使用） ----------
FIN_WIND_CSV = os.getenv("FIN_WIND_CSV", "")   # 行业对标样本（wind 特征 CSV 路径，可空）
FIN_CF_TO_PROFIT = 1.0        # 现金流/净利润阈值
FIN_DEBT_RATIO_MAX = 70.0     # 资产负债率阈值（%）
FIN_ROE_NEGATIVE = 0.0        # ROE 低于此值视为亏损（%）
FIN_ROE_TREND_SLOPE = -5.0    # ROE 近4期趋势斜率阈值（个百分点/季）
FIN_Z_SCORE = 2.0             # 行业偏离 |Z| 阈值

# ---------- 案例检索 / 案例库（case_retriever / build_case_db） ----------
INQUIRY_DATA_DIR = Path(os.getenv("INQUIRY_DATA_DIR", r"D:\BaiduNetdiskDownload\监管问询函及回复数据集"))
EVAL_GT_CSV = Path(os.getenv("EVAL_GT_CSV", r"D:\BaiduNetdiskDownload\标签与评测数据集\evaluation_ground_truth.csv"))
CASE_DB_PATH = VECTOR_DB_DIR / "case_db.json"       # 案例元数据（含官方关注点 + 原文摘录）
CASE_VEC_PATH = VECTOR_DB_DIR / "case_vectors.npy"  # 案例向量
CASE_META_PATH = VECTOR_DB_DIR / "case_meta.json"   # 案例库构建元数据（embedding 后端/维度校验）
CASE_TOP_K = 5                  # 案例检索返回 Top-K
RRF_K = 60                      # RRF 融合常数（k）

# ---------- 关注点词典 / 案例重建源头（02_监管问询 数据产物，队友交付） ----------
CONCERN_DICT_PATH = Path(BASE_DIR) / "backend" / "data" / "labels" / "concern_dict.json"
INQUIRY_JSONL = Path(os.getenv("INQUIRY_JSONL", r"D:\新建文件夹\02_监管问询\01_数据清单与结构化文本\inquiries.jsonl"))
RULE_RISKS_JSONL = Path(os.getenv("RULE_RISKS_JSONL", r"D:\新建文件夹\02_监管问询\02_风险标签\inquiry_rule_risks.jsonl"))
EVAL_GT_NORMALIZED_CSV = Path(os.getenv("EVAL_GT_NORMALIZED_CSV", r"D:\新建文件夹\02_监管问询\05_标签评测与报告\evaluation_ground_truth_normalized.csv"))
CASE_EXCERPT_CHARS = 1200      # 案例原文摘录截断长度

# ---------- chunk 级段落检索（chunk_retriever / build_chunk_index） ----------
CHUNK_DB_PATH = VECTOR_DB_DIR / "chunk_db.json"        # chunk 元数据（段落级）
CHUNK_VEC_PATH = VECTOR_DB_DIR / "chunk_vectors.npy"   # chunk 向量（BGE 1024 维）
CHUNK_TOP_K = 8                 # chunk 检索返回 Top-K
CHUNK_INDEX_JSONL = Path(os.getenv("CHUNK_INDEX_JSONL",
    r"D:\新建文件夹\02_监管问询\03_向量索引与Chroma\inquiry_embedding_index.jsonl"))
