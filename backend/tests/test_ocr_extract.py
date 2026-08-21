from io import BytesIO

import pymupdf
from PIL import Image, ImageDraw

from backend.skills.ocr_extract import extract_pdf_text, page_needs_ocr


class FakeOCREngine:
    name = "fake-ocr"
    version = "test"

    def __init__(self, text="公司存在重大诉讼风险"):
        self.text = text
        self.calls = 0

    def recognize_page(self, page, dpi=180, min_confidence=0.5):
        self.calls += 1
        return {
            "text": self.text,
            "line_count": 1,
            "mean_confidence": 0.96,
        }


def _image_bytes():
    image = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 80, 750, 220), outline="black", width=4)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_scanned_image_page_uses_ocr_and_records_audit_metadata():
    document = pymupdf.open()
    page = document.new_page(width=800, height=300)
    page.insert_image(page.rect, stream=_image_bytes())
    engine = FakeOCREngine()

    text, metadata = extract_pdf_text(document, engine=engine)

    assert text == "公司存在重大诉讼风险"
    assert engine.calls == 1
    assert metadata["ocr_status"] == "completed"
    assert metadata["ocr_candidate_pages"] == 1
    assert metadata["ocr_succeeded_pages"] == 1
    assert metadata["ocr_mean_confidence"] == 0.96
    assert metadata["ocr_page_details"][0]["action"] == "ocr_succeeded"


def test_native_text_page_does_not_call_ocr():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Native annual report text " * 5)
    engine = FakeOCREngine()

    text, metadata = extract_pdf_text(document, engine=engine, min_page_chars=20)

    assert "Native annual report text" in text
    assert engine.calls == 0
    assert metadata["ocr_status"] == "not_needed"


def test_blank_page_is_not_misclassified_as_scanned_page():
    document = pymupdf.open()
    page = document.new_page()

    assert page_needs_ocr(page, "", min_page_chars=40) is False
