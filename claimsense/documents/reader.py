"""Get text out of a document.

  Digital PDF        -> read the PDF's text layer (fast and exact; no OCR needed)
  Photo / scan (PNG) -> OpenCV cleanup + Tesseract OCR (pip install opencv-python-headless pytesseract,
                        and install the tesseract program: brew/apt install tesseract)

Real insurers get both kinds, so ClaimSense handles both. Note that a PDF text layer also
contains text a human can't see (white or tiny text), which is exactly where hidden
prompt injections live. The injection scanner runs on this text afterwards.
"""
from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..schemas import DocumentText


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    s = get_settings()
    for base in (s.synthetic_dir, s.upload_dir, s.data_dir):
        if (base / p).exists():
            return base / p
    return p


def detect_type(text: str) -> str:
    t = text.upper()
    if "REPAIR ESTIMATE" in t or "ESTIMATE #" in t:
        return "repair_estimate"
    if "INCIDENT REPORT" in t or "REPORT #" in t:
        return "police_report"
    return "unknown"


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def clean_for_ocr(path: Path):
    """Deskew-free cleanup that helps Tesseract on phone photos: grayscale, denoise, threshold."""
    import cv2  # opencv-python-headless
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def read_image(path: Path, cleanup: bool = True) -> str:
    import pytesseract
    if cleanup:
        img = clean_for_ocr(path)
    else:
        from PIL import Image
        img = Image.open(path)
    return pytesseract.image_to_string(img, config="--psm 6")


def read_document(path: str | Path) -> DocumentText:
    p = _resolve(path)
    if not p.exists():
        return DocumentText(path=str(path), text="", method="none")
    if p.suffix.lower() == ".pdf":
        text, method = read_pdf(p), "pdf_text_layer"
    else:
        try:
            text, method = read_image(p), "ocr"
        except ImportError:
            return DocumentText(path=str(path), text="", method="none")
    return DocumentText(path=str(path), doc_type=detect_type(text), text=text, method=method)
