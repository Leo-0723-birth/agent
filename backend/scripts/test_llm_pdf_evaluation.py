#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试：用 LLM 直接对 PDF 原文分类，计算中层指标。

流程：
1. 根据 risk_extractions_long.csv 将 case_db.json 中的案例映射到本地 PDF；
2. 用 pdfplumber 提取 PDF 文本；
3. 复用 AnnouncementReaderAgent._llm_extract 对原文做 45 类风险分类与证据抽取；
4. 与 case_db.json 的 taxonomy_labels / focus_points 对比，计算：
   - 监管关注点分类准确率（多标签 Jaccard）
   - 关键证据片段召回率（预测 evidence 在 focus_points 中的覆盖度）
   - 相似历史问询案例 Top-5 命中率（沿用 case retriever）
5. 输出指标并保存预测结果，供 evaluator.py 切换预测源。

注意：
- 为控制时间与成本，默认随机采样 SAMPLE_SIZE 个案例；
- 同一个 (company, year) 下有多条 case_db 记录时，会共享同一批 PDF；
- PDF 文本截断到 MAX_TEXT_CHARS（与 AnnouncementReader 一致）。
"""
from __future__ import annotations

import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.label_keywords_v2 import TAXONOMY_NAMES
from backend.llm import chat_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

CASE_DB_PATH = PROJECT_ROOT / "backend" / "data" / "vector_db" / "case_db.json"
RISK_EXTRACT_PATH = Path("D:/BaiduNetdiskDownload/监管问询函及回复数据集/extraction_output/risk_extractions_long.csv")
PDF_BASE_DIR = Path("D:/BaiduNetdiskDownload/监管问询函及回复数据集")
OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "eval_cache" / "llm_pdf_predictions.json"
SAMPLE_SIZE = 50  # 默认采样数，可通过命令行覆盖
MAX_TEXT_CHARS = 12000  # PDF 文本截断长度


def _load_case_db():
    return json.loads(CASE_DB_PATH.read_text(encoding="utf-8"))


def _load_pdf_mapping():
    df = pd.read_csv(RISK_EXTRACT_PATH, encoding="utf-8-sig")
    df = df[df["doc_type"] == "inquiry_letter"]
    mapping = {}
    for _, row in df.iterrows():
        key = (str(row["company_code"]), str(row["year"]))
        pdf_path = Path(str(row["pdf_path"]))
        mapping.setdefault(key, set()).add(pdf_path)
    return mapping


def _extract_pdf_text(pdf_path: Path, max_chars: int = MAX_TEXT_CHARS) -> str:
    """用 pdfplumber 提取文本并截断。"""
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 pdfplumber，请先安装") from exc

    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                full_text += txt + "\n"
                if len(full_text) >= max_chars:
                    break
    except Exception as exc:
        _logger.warning("PDF 提取失败 %s: %s", pdf_path, exc)
        return ""
    return full_text[:max_chars].strip()


def _map_case_to_pdfs(case, pdf_mapping):
    company = case.get("company", "")
    year = str(case.get("publish_date", ""))[:4]
    if not company or not year:
        return []
    # company 形如 "000004.SZ"，csv 中 company_code 形如 "000004.SZ"
    key = (company, year)
    pdfs = pdf_mapping.get(key, set())
    existing = [PDF_BASE_DIR / p for p in pdfs if (PDF_BASE_DIR / p).exists()]
    return existing


TAXONOMY_PROMPT_LIST = "\n".join(
    f"{code} {name}" for code, name in TAXONOMY_NAMES.items()
)


def _classify_inquiry_letter(text: str) -> dict:
    """用 LLM 对问询函原文做 45 类主题分类与证据抽取。"""
    prompt = f"""你是上市公司监管问询函主题分类专家。请阅读以下交易所对上市公司发出的问询函，识别监管机构关注的主要问题类别。

要求：
1. 只依据给出的问询函正文内容，不要补充外部知识。
2. 从以下 45 类二级主题中选择最相关的标签（可多选）：
{TAXONOMY_PROMPT_LIST}
3. 对每个选中的标签，给出问询函中支持该标签的一句原文作为证据。证据必须逐字来自正文。
4. 只输出 JSON，格式如下：
{{"labels":["A03","B03"],"evidences":[{{"label":"A03","text":"原文片段"}}]}}

问询函正文：
{text[:MAX_TEXT_CHARS]}
"""
    result = chat_json("", prompt, max_tokens=4000)
    if not isinstance(result, dict):
        return {"labels": [], "evidences": []}
    labels = [
        str(l).upper().strip()
        for l in result.get("labels", [])
        if re.fullmatch(r"[A-H]\d{2}", str(l).upper().strip())
    ]
    evidences = []
    for ev in result.get("evidences", []):
        if isinstance(ev, dict):
            evidences.append({
                "label": str(ev.get("label", "")).upper().strip(),
                "text": str(ev.get("text", "")).strip(),
            })
    return {"labels": labels, "evidences": evidences}


def _jaccard(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    return len(pred & true) / len(pred | true)


def _evidence_recall(pred_evidences: list[str], focus_points: list[str]) -> float:
    """
    计算证据片段召回率：人工标注 focus_points 中被预测 evidence 覆盖的比例。
    规则：若某条 focus_point 与任意预测 evidence 有 ≥8 个连续中文字符重合，视为命中。
    """
    if not pred_evidences or not focus_points:
        return 0.0
    hit = 0
    for fp in focus_points:
        fp_clean = re.sub(r"\s+", "", fp)
        found = False
        for ev in pred_evidences:
            ev_clean = re.sub(r"\s+", "", ev)
            if _longest_common_substring_len(ev_clean, fp_clean) >= 8:
                found = True
                break
        if found:
            hit += 1
    return hit / len(focus_points)


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    max_len = 0
    # 简单滑动窗口
    for length in range(min(8, len(a)), len(a) + 1):
        if length < max_len:
            continue
        seen = set()
        for i in range(len(a) - length + 1):
            seen.add(a[i : i + length])
        for i in range(len(b) - length + 1):
            if b[i : i + length] in seen:
                max_len = length
                break
    return max_len


def run_evaluation(sample_size: int = SAMPLE_SIZE):
    random.seed(42)
    case_db = _load_case_db()
    pdf_mapping = _load_pdf_mapping()

    # 只保留能映射到 PDF 的案例
    mappable = []
    for case in case_db:
        pdfs = _map_case_to_pdfs(case, pdf_mapping)
        if pdfs:
            mappable.append((case, pdfs))

    _logger.info("case_db 总数: %d, 可映射到 PDF: %d", len(case_db), len(mappable))

    if len(mappable) > sample_size:
        mappable = random.sample(mappable, sample_size)
        _logger.info("随机采样 %d 条进行测试", sample_size)

    predictions = []

    for idx, (case, pdfs) in enumerate(mappable, 1):
        pdf_path = pdfs[0]
        _logger.info("[%d/%d] 处理 %s - %s", idx, len(mappable), case.get("company"), pdf_path.name)
        text = _extract_pdf_text(pdf_path)
        if not text:
            _logger.warning("  文本为空，跳过")
            continue

        try:
            llm_result = _classify_inquiry_letter(text)
        except Exception as exc:
            _logger.error("  LLM 分类异常: %s", exc)
            continue

        pred_labels = sorted(set(llm_result["labels"]))
        pred_evidences = [ev["text"] for ev in llm_result["evidences"] if ev.get("text")]
        true_labels = set(case.get("taxonomy_labels", []))
        focus_points = case.get("focus_points", []) or []

        acc = _jaccard(set(pred_labels), true_labels)
        rec = _evidence_recall(pred_evidences, focus_points)

        predictions.append({
            "company": case.get("company"),
            "publish_date": case.get("publish_date"),
            "pdf_path": str(pdf_path),
            "pred_labels": pred_labels,
            "pred_evidences": pred_evidences,
            "true_labels": sorted(true_labels),
            "focus_points": focus_points,
            "accuracy": acc,
            "evidence_recall": rec,
            "llm_status": "ok",
        })
        _logger.info("  准确率 %.3f, 证据召回 %.3f, 预测标签 %s, 真实标签 %s", acc, rec, pred_labels, sorted(true_labels))

    if not predictions:
        _logger.error("没有成功生成任何预测")
        return None

    mean_acc = sum(p["accuracy"] for p in predictions) / len(predictions)
    mean_rec = sum(p["evidence_recall"] for p in predictions) / len(predictions)

    _logger.info("=" * 60)
    _logger.info("样本数: %d", len(predictions))
    _logger.info("平均分类准确率 (Jaccard): %.4f (%.2f%%)", mean_acc, mean_acc * 100)
    _logger.info("平均证据召回率: %.4f (%.2f%%)", mean_rec, mean_rec * 100)
    _logger.info("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "sample_size": len(predictions),
        "mean_accuracy": mean_acc,
        "mean_evidence_recall": mean_rec,
        "predictions": predictions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _logger.info("预测结果已保存: %s", OUTPUT_PATH)

    return {
        "sample_size": len(predictions),
        "mean_accuracy": mean_acc,
        "mean_evidence_recall": mean_rec,
        "predictions": predictions,
    }


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_SIZE
    run_evaluation(sample_size)
