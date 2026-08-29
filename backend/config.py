#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置（单一来源）
====================
所有路径 / 参数 / 模型配置集中于此，业务代码禁止硬编码路径。
用法：
    from backend.config import DATA_RAW, FINBERT_GATE, RISK_THRESHOLDS
"""
import logging
import os
from pathlib import Path

# 统一日志输出（各模块用 logging.getLogger(__name__)，此处配置一次格式与级别）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)

# 尽量加载项目根 .env（密钥）；无 python-dotenv 时静默跳过，靠系统环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent                 # competition_agent/
# 本机外部数据根目录（队员各自本地盘；通过 .env 覆盖 DATA_ROOT / INQUIRY_ROOT，
# 无需逐个改下面的绝对路径）
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(BASE_DIR / "data")))
INQUIRY_ROOT = Path(os.getenv("INQUIRY_ROOT", str(BASE_DIR / "data" / "inquiry")))
DATA_RAW = Path(os.getenv("DATA_RAW", str(DATA_ROOT)))  # 原始公告/问询函根目录
DATA_DIR = BASE_DIR / "backend" / "data"
INDEX_DIR = DATA_DIR / "index"          # 公告索引缓存（announcement_index.json）
VECTOR_DB_DIR = DATA_DIR / "vector_db"  # 向量库持久化（案例库/证据库）
OUTPUT_DIR = DATA_DIR / "output"        # context 快照 / 报告 / trace.log
TRACE_DIR = OUTPUT_DIR / "traces"       # 实时流水线 trace 落盘（JSONL，赛后复盘审计）
MODEL_DIR = BASE_DIR / "backend" / "models"

# 比赛历史库（2020—2024 历史研究数据；只作历史证据，不替代当前公告事实）
# 默认使用仓库内可移植数据包；需要完整外部交付物时可用环境变量覆盖。
COMPETITION_DATA_ROOT = Path(os.getenv(
    "COMPETITION_DATA_ROOT",
    str(BASE_DIR / "公告解析"),
)).expanduser()

_COMPETITION_RULE_RISKS_JSONL = (
    COMPETITION_DATA_ROOT
    / "02_风险标签抽取模块"
    / "规则风险结果"
    / "announcement_rule_risks.jsonl"
)
_COMPETITION_RULE_RISKS_GZIP = _COMPETITION_RULE_RISKS_JSONL.with_suffix(".jsonl.gz")
_DEFAULT_COMPETITION_RULE_RISKS = (
    _COMPETITION_RULE_RISKS_JSONL
    if _COMPETITION_RULE_RISKS_JSONL.is_file()
    else _COMPETITION_RULE_RISKS_GZIP
)
COMPETITION_RULE_RISKS = Path(os.getenv(
    "COMPETITION_RULE_RISKS",
    str(_DEFAULT_COMPETITION_RULE_RISKS),
)).expanduser()
COMPETITION_SEMANTIC_FEATURES = Path(os.getenv(
    "COMPETITION_SEMANTIC_FEATURES",
    str(COMPETITION_DATA_ROOT / "04_语义特征生成模块" / "核心Parquet" / "semantic_features.parquet"),
)).expanduser()

# ---------- LLM（backend/llm.py 使用） ----------
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = 0.1        # 低温度：推理稳定（方案 5.4）
LLM_MAX_TOKENS = 2000
LLM_THINKING = os.getenv("DEEPSEEK_THINKING", "0") == "1"

# ---------- Embedding ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")  # 或 stella-base-zh-v3
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "bge")  # bge=BGE-large-zh-v1.5(1024维,权重已在项目内,自动回落)；fallback=零依赖兜底(65536维,与案例库不兼容)

# ---------- 预测 ----------
PREDICT_WINDOW = 60                       # 默认预测窗口（天）
RISK_THRESHOLDS = {"high": 0.6, "medium": 0.3}   # 风险等级阈值

# ---------- 公告研读 ----------
ANNOUNCE_WINDOW_DAYS = 365    # 公告检索窗口（天）
F1_DECAY_HALF_LIFE_DAYS = int(os.getenv("F1_DECAY_HALF_LIFE_DAYS", "180"))  # F1 时间衰减半衰期（天）：age=180d → 权重0.5
ANNOUNCE_MAX_DOCUMENTS = int(os.getenv("ANNOUNCE_MAX_DOCUMENTS", "120"))
# 实时扫雷（方案C）默认深读公告数：API 请求与扫描路径编排器统一取此值，
# 避免各层默认值不一致导致"请求 5 份实际深读 N 份"的参数漂移。
# 批量/全量处理仍用 ANNOUNCE_MAX_DOCUMENTS。
SCAN_MAX_DOCUMENTS = int(os.getenv("SCAN_MAX_DOCUMENTS", "50"))
# 实时扫雷是否启用 FinBERT 通道（前端未暴露该开关，由后端配置统一控制）。
SCAN_USE_FINBERT = os.getenv("SCAN_USE_FINBERT", "true").lower() in {
    "1", "true", "yes", "on"
}
ANNOUNCE_PDF_CACHE = Path(
    os.getenv("ANNOUNCE_PDF_CACHE", str(DATA_DIR / "cache" / "pdfs"))
)
ANNOUNCE_OFFLINE_SNAPSHOT_DIR = Path(
    os.getenv(
        "ANNOUNCE_OFFLINE_SNAPSHOT_DIR",
        str(DATA_DIR / "offline_announcements"),
    )
)
ANNOUNCE_OFFLINE_ENABLED = os.getenv("ANNOUNCE_OFFLINE_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
ANNOUNCE_SOURCE = os.getenv("ANNOUNCE_SOURCE", "cninfo").lower()
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
OCR_DPI = int(os.getenv("OCR_DPI", "180"))
OCR_MIN_PAGE_CHARS = int(os.getenv("OCR_MIN_PAGE_CHARS", "40"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.50"))
OCR_MAX_PAGES_PER_DOCUMENT = int(os.getenv("OCR_MAX_PAGES_PER_DOCUMENT", "80"))
FINBERT_GATE = float(os.getenv("FINBERT_GATE", "0.5"))
# 三通道全开为默认（权重已随项目缓存，进程级单例只加载一次）；
# 显式设 FINBERT_ENABLED=0/false 可关闭。
FINBERT_ENABLED = os.getenv("FINBERT_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
FINBERT_GATE_ENABLED = os.getenv("FINBERT_GATE_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}  # FinBERT 门控：仅规则命中或 FinBERT 高分公告进入 LLM 精细抽取
MAX_TEXT_CHARS = 8000         # 送 LLM 的公告正文截断长度

# ---------- 财务异常检测（backend/agents/financial_detector.py 使用） ----------
FIN_WIND_CSV = os.getenv("FIN_WIND_CSV", "")   # 行业对标样本（wind 特征 CSV 路径，可空）
FIN_CF_TO_PROFIT = 1.0        # 现金流/净利润阈值
FIN_DEBT_RATIO_MAX = 70.0     # 资产负债率阈值（%）
FIN_ROE_NEGATIVE = 0.0        # ROE 低于此值视为亏损（%）
FIN_ROE_TREND_SLOPE = -5.0    # ROE 近4期趋势斜率阈值（个百分点/季）
FIN_Z_SCORE = 2.0             # 行业偏离 |Z| 阈值

# ---------- 案例检索 / 案例库（case_retriever / build_case_db） ----------
INQUIRY_DATA_DIR = Path(os.getenv("INQUIRY_DATA_DIR", str(DATA_ROOT / "监管问询函及回复数据集")))
EVAL_GT_CSV = Path(os.getenv("EVAL_GT_CSV", str(DATA_ROOT / "标签与评测数据集" / "evaluation_ground_truth.csv")))
CASE_DB_PATH = VECTOR_DB_DIR / "case_db.json"       # 案例元数据（含官方关注点 + 原文摘录）
CASE_VEC_PATH = VECTOR_DB_DIR / "case_vectors.npy"  # 案例向量
CASE_META_PATH = VECTOR_DB_DIR / "case_meta.json"   # 案例库构建元数据（embedding 后端/维度校验）
CASE_TOP_K = 5                  # 案例检索返回 Top-K
RRF_K = 60                      # RRF 融合常数（k）

# ---------- 关注点词典 / 案例重建源头（02_监管问询 数据产物，队友交付） ----------
CONCERN_DICT_PATH = Path(BASE_DIR) / "backend" / "data" / "labels" / "concern_dict.json"
# 风险词典：默认 v2.1（45/45 主题规则全覆盖，扩展自冻结版 v2.0.0）；环境变量可覆盖回退
RISK_DICTIONARY = Path(os.getenv("RISK_DICTIONARY",
    str(Path(BASE_DIR) / "backend" / "data" / "labels" / "risk_dictionary_v2.1-taxonomy-v1.1.yaml")))
INQUIRY_JSONL = Path(os.getenv("INQUIRY_JSONL", str(INQUIRY_ROOT / "02_监管问询" / "01_数据清单与结构化文本" / "inquiries.jsonl")))
RULE_RISKS_JSONL = Path(os.getenv("RULE_RISKS_JSONL", str(INQUIRY_ROOT / "02_监管问询" / "02_风险标签" / "inquiry_rule_risks.jsonl")))
EVAL_GT_NORMALIZED_CSV = Path(os.getenv("EVAL_GT_NORMALIZED_CSV", str(INQUIRY_ROOT / "02_监管问询" / "05_标签评测与报告" / "evaluation_ground_truth_normalized.csv")))
CASE_EXCERPT_CHARS = 1200      # 案例原文摘录截断长度

# ---------- 预测建模（PredictorAgent / train_models） ----------
PREDICTOR_MODEL_DIR = Path(BASE_DIR) / "backend" / "models" / "predictor"   # 模型+清单
MODELING_DATASET = Path(BASE_DIR) / "backend" / "data" / "modeling" / "processed_dataset.csv"  # 查表推理用
PREDICTOR_HORIZONS = ("30d", "60d", "90d")   # 推理输出窗口
PREDICTOR_TOP_SHAP = 8                        # SHAP 输出 Top-K 特征

# 特征家族前缀（白名单）：训练/推理只使用这些前缀的列作为模型输入
# 注意：F1 前缀会随数据源变更而调整；修改后须重新训练模型。
FEATURE_FAMILY_PREFIXES = (
    "announcement_semantic_",  # F1：公告语义特征
    "f2_",                     # F2：财务异常特征
    "mkt_",                    # F3：市场特征
    "sent_",                   # F4：舆情特征
    "gov_",                    # F5：治理特征
    "f6_",                     # F6：问询历史特征
)

# 训练标签事件类型（build_modeling_dataset.py 中过滤 inquiry_events 的 kind 字段）
TARGET_INQUIRY_KIND = os.getenv("TARGET_INQUIRY_KIND", "letter")

# 样本切分名称（支持大小写归一化）
TRAIN_SPLIT_NAMES = ("Train", "train", "TRAIN")
VALIDATION_SPLIT_NAMES = ("Validation", "validation", "VALIDATION", "Val", "val", "VAL")
TEST_SPLIT_NAMES = ("Test", "test", "TEST")

# 训练脚本中的特征筛选阈值
FEATURE_VARIANCE_THRESHOLD = float(os.getenv("FEATURE_VARIANCE_THRESHOLD", "1e-12"))
FEATURE_CORR_THRESHOLD = float(os.getenv("FEATURE_CORR_THRESHOLD", "0.95"))
FEATURE_CORR_SAMPLE_SIZE = int(os.getenv("FEATURE_CORR_SAMPLE_SIZE", "5000"))
FEATURE_IMPORTANCE_THRESHOLD = float(os.getenv("FEATURE_IMPORTANCE_THRESHOLD", "0"))
FEATURE_FILTER_MIN_FEATURES = int(os.getenv("FEATURE_FILTER_MIN_FEATURES", "50"))

# ---------- XGBoost-Cox 生存模型接口（可选，预留） ----------
# 训练后放置：model_survival_xgb.json（survival:cox Booster）
#            survival_baseline_hazard.json（Breslow 累积基线风险 H0(t)）
#            survival_features.json（训练特征清单）
# 缺失时 PredictorAgent 自动回退三模型集成（现有路径不变）
PREDICTOR_SURVIVAL_XGB = Path(BASE_DIR) / "backend" / "models" / "predictor" / "model_survival_xgb.json"
PREDICTOR_SURVIVAL_BASELINE = Path(BASE_DIR) / "backend" / "models" / "predictor" / "survival_baseline_hazard.json"
PREDICTOR_SURVIVAL_FEATURES = Path(BASE_DIR) / "backend" / "models" / "predictor" / "survival_features.json"

# ---------- 财务异常 F4/F5/F6 离线特征表（feature_loader / crawl_* 使用） ----------
PREPROCESSED_DIR = Path(BASE_DIR) / "backend" / "data" / "modeling" / "preprocessed"
FEATURE_TABLE_CONFIG = {
    "F2": {"files": [PREPROCESSED_DIR / "F2_financial_anomaly.csv"],
           "key": "company_code", "period": "report_period"},
    "F3": {"files": [PREPROCESSED_DIR / "F3_market_features.csv"],
           "key": "company_code", "period": "report_period"},
    "F4": {"files": [PREPROCESSED_DIR / "F4_sentiment_features.csv"],
           "key": "company_code", "period": "report_period"},
    "F5": {"files": [PREPROCESSED_DIR / "F5_ownership_governance.csv"],
           "key": "company_code", "period": "report_period"},
    "F6": {"files": [PREPROCESSED_DIR / "F6_inquiry_history.csv"],
           "key": "company_code", "period": "report_period"},
}
META_COLS = {"company_code", "stock_code", "report_period", "T_date", "split",
             "industry", "matched_stat_date", "governance_year", "audit_year"}
# 在线爬虫用（F5 年报 PDF 目录 / F6 问询事件缓存）
ANNUAL_REPORT_DIR = Path(os.getenv("ANNUAL_REPORT_DIR",
    str(DATA_ROOT / "上市公司公告与定期报告数据集" / "上市公司公告与定期报告数据集")))
INQUIRY_EVENTS_CSV = Path(os.getenv("INQUIRY_EVENTS_CSV",
    str(INQUIRY_ROOT / "02_监管问询" / "F2-F6" / "中间产物及特征来源" / "中间产物10,056份问询回复函的日期+类型解析结果(下次重跑免解析)" / "inquiry_events.csv")))
