#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描 PDF 按页 OCR：原生文本优先，RapidOCR 只处理低文本图像页。"""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version


OCR_PIPELINE_VERSION = "rapidocr_pdf_page_v1"


class OCRUnavailable(RuntimeError):
    """OCR 依赖没有安装或无法初始化。"""


def meaningful_char_count(text: str) -> int:
    """统计适合判断文本层是否有效的中英文和数字字符。"""
    return len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]", text or ""))


def page_needs_ocr(page, native_text: str, min_page_chars: int = 40) -> bool:
    """仅把低文本且确有视觉内容的页面判定为 OCR 候选。"""
    if meaningful_char_count(native_text) >= int(min_page_chars):
        return False
    try:
        has_images = bool(page.get_images(full=True))
    except Exception:
        has_images = False
    if has_images:
        return True
    try:
        return len(page.get_drawings()) >= 20
    except Exception:
        return False


class RapidOCRPageEngine:
    """延迟初始化的 RapidOCR/ONNX Runtime 页面识别器。"""

    name = "RapidOCR/ONNX Runtime"

    def __init__(self):
        self._engine = None

    @property
    def version(self) -> str:
        try:
            return version("rapidocr")
        except PackageNotFoundError:
            return "not_installed"

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import RapidOCR
        except Exception as exc:
            raise OCRUnavailable(
                "RapidOCR 未安装；请安装 rapidocr 和 onnxruntime"
            ) from exc
        try:
            self._engine = RapidOCR()
        except Exception as exc:
            raise OCRUnavailable(f"RapidOCR 初始化失败：{exc}") from exc
        return self._engine

    def recognize_page(self, page, dpi: int = 180, min_confidence: float = 0.50) -> dict:
        """渲染 PDF 页并返回通过置信度门槛的识别文本。"""
        import numpy as np
        import pymupdf

        pixmap = page.get_pixmap(
            dpi=int(dpi),
            colorspace=pymupdf.csRGB,
            alpha=False,
            annots=True,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        result = self._load()(image)
        texts = list(getattr(result, "txts", ()) or ())
        scores = [float(value) for value in (getattr(result, "scores", ()) or ())]
        accepted = [
            (str(text).strip(), score)
            for text, score in zip(texts, scores)
            if str(text).strip() and score >= float(min_confidence)
        ]
        return {
            "text": "\n".join(text for text, _ in accepted),
            "line_count": len(accepted),
            "mean_confidence": (
                round(sum(score for _, score in accepted) / len(accepted), 4)
                if accepted else None
            ),
            "min_confidence": float(min_confidence),
            "dpi": int(dpi),
        }


def extract_pdf_text(
    document,
    *,
    engine=None,
    enabled: bool = True,
    dpi: int = 180,
    min_page_chars: int = 40,
    min_confidence: float = 0.50,
    max_pages: int = 80,
) -> tuple[str, dict]:
    """提取整份 PDF 文本并返回可审计 OCR 元数据。"""
    engine = engine or RapidOCRPageEngine()
    page_texts = []
    details = []
    candidate_pages = 0
    attempted_pages = 0
    succeeded_pages = 0
    failed_pages = 0
    skipped_pages = 0
    confidences = []
    unavailable_error = ""

    for page_number, page in enumerate(document, start=1):
        native_text = (page.get_text("text", sort=True) or "").replace("\x00", "").strip()
        native_chars = meaningful_char_count(native_text)
        if not page_needs_ocr(page, native_text, min_page_chars):
            page_texts.append(native_text)
            details.append(
                {"page": page_number, "action": "native", "native_chars": native_chars}
            )
            continue

        candidate_pages += 1
        if not enabled:
            page_texts.append(native_text)
            details.append(
                {"page": page_number, "action": "ocr_disabled", "native_chars": native_chars}
            )
            continue
        if attempted_pages >= int(max_pages) or unavailable_error:
            skipped_pages += 1
            page_texts.append(native_text)
            details.append(
                {
                    "page": page_number,
                    "action": "ocr_skipped_limit" if not unavailable_error else "ocr_unavailable",
                    "native_chars": native_chars,
                }
            )
            continue

        attempted_pages += 1
        try:
            recognized = engine.recognize_page(
                page, dpi=int(dpi), min_confidence=float(min_confidence)
            )
            ocr_text = (recognized.get("text") or "").strip()
            if not ocr_text:
                raise ValueError("OCR 未返回达到置信度门槛的文本")
            succeeded_pages += 1
            if recognized.get("mean_confidence") is not None:
                confidences.append(float(recognized["mean_confidence"]))
            page_texts.append(ocr_text if len(ocr_text) >= len(native_text) else native_text)
            details.append(
                {
                    "page": page_number,
                    "action": "ocr_succeeded",
                    "native_chars": native_chars,
                    "ocr_chars": meaningful_char_count(ocr_text),
                    "line_count": recognized.get("line_count", 0),
                    "mean_confidence": recognized.get("mean_confidence"),
                }
            )
        except OCRUnavailable as exc:
            failed_pages += 1
            unavailable_error = str(exc)
            page_texts.append(native_text)
            details.append(
                {
                    "page": page_number,
                    "action": "ocr_unavailable",
                    "native_chars": native_chars,
                    "error": unavailable_error[:200],
                }
            )
        except Exception as exc:
            failed_pages += 1
            page_texts.append(native_text)
            details.append(
                {
                    "page": page_number,
                    "action": "ocr_failed",
                    "native_chars": native_chars,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )

    if candidate_pages == 0:
        status = "not_needed"
    elif not enabled:
        status = "disabled"
    elif unavailable_error:
        status = "not_available"
    elif skipped_pages:
        status = "partial_truncated" if succeeded_pages else "truncated"
    elif failed_pages and succeeded_pages:
        status = "partial_failed"
    elif failed_pages:
        status = "failed"
    else:
        status = "completed"

    text = "\n".join(value for value in page_texts if value).strip()
    metadata = {
        "ocr_pipeline_version": OCR_PIPELINE_VERSION,
        "ocr_engine": getattr(engine, "name", type(engine).__name__),
        "ocr_engine_version": getattr(engine, "version", "test_or_unknown"),
        "ocr_status": status,
        "ocr_candidate_pages": candidate_pages,
        "ocr_attempted_pages": attempted_pages,
        "ocr_succeeded_pages": succeeded_pages,
        "ocr_failed_pages": failed_pages,
        "ocr_skipped_pages": skipped_pages,
        "ocr_mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "ocr_dpi": int(dpi),
        "ocr_min_confidence": float(min_confidence),
        "ocr_error": unavailable_error[:200],
        "ocr_page_details": details,
    }
    return text, metadata
