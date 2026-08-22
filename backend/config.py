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
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fallback")  # bge(需下载权重) | fallback(零依赖) | 加载失败自动退回 fallback

# ---------- 预测 ----------
PREDICT_WINDOW = 60                       # 默认预测窗口（天）
RISK_THRESHOLDS = {"high": 0.6, "medium": 0.3}   # 风险等级阈值

# ---------- 公告研读 ----------
ANNOUNCE_WINDOW_DAYS = 365    # 公告检索窗口（天）
FINBERT_GATE = 0.5            # FinBERT 粗分类门控：低于该相似度的公告不送 LLM
MAX_TEXT_CHARS = 8000         # 送 LLM 的公告正文截断长度

# ---------- 财务异常检测（backend/agents/financial_detector.py 使用） ----------
FIN_WIND_CSV = os.getenv("FIN_WIND_CSV", "")   # 行业对标样本（wind 特征 CSV 路径，可空）
FIN_CF_TO_PROFIT = 1.0        # 现金流/净利润阈值
FIN_DEBT_RATIO_MAX = 70.0     # 资产负债率阈值（%）
FIN_ROE_NEGATIVE = 0.0        # ROE 低于此值视为亏损（%）
FIN_ROE_TREND_SLOPE = -5.0    # ROE 近4期趋势斜率阈值（个百分点/季）
FIN_Z_SCORE = 2.0             # 行业偏离 |Z| 阈值

# ---------- 案例检索 / 案例库（case_retriever / build_case_vector_db） ----------
INQUIRY_DATA_DIR = Path(os.getenv("INQUIRY_DATA_DIR", r"D:\BaiduNetdiskDownload\监管问询函及回复数据集"))
EVAL_GT_CSV = Path(os.getenv("EVAL_GT_CSV", r"D:\BaiduNetdiskDownload\标签与评测数据集\evaluation_ground_truth.csv"))
CASE_DB_PATH = VECTOR_DB_DIR / "case_db.json"       # 案例元数据（含官方关注点 + 原文摘录）
CASE_VEC_PATH = VECTOR_DB_DIR / "case_vectors.npy"  # 案例向量
CASE_TOP_K = 5                  # 案例检索返回 Top-K
RRF_K = 60                      # RRF 融合常数（k）
