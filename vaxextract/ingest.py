"""File bytes -> pages of layout tokens. PDF text layers are used when present (no OCR cost)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .layout import Token
from .preprocess import preprocess


@dataclass
class Page:
    tokens: list[Token]
    source: str  # "text_layer" | "ocr"


@dataclass
class Document:
    pages: list[Page]
    kind: str  # "pdf" | "image"


def _ocr_rgb(rgb: np.ndarray, get_ocr: Callable):
    gray = preprocess(rgb)
    return get_ocr().run(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def _pix_to_rgb(pix) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB) if pix.n == 1 else arr[:, :, :3].copy()


def load(data: bytes, get_ocr: Callable, max_pages: int = 10, min_words: int = 15) -> Document:
    if data[:5] == b"%PDF-":
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            toks = [Token(w[4], w[0], w[1], w[2], w[3]) for w in page.get_text("words") if w[4].strip()]
            if len(toks) >= min_words:
                pages.append(Page(toks, "text_layer"))
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
            pages.append(Page(_ocr_rgb(_pix_to_rgb(pix), get_ocr), "ocr"))
        return Document(pages, "pdf")

    from PIL import Image, ImageOps, ImageSequence

    img = Image.open(io.BytesIO(data))
    pages = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i >= max_pages:
            break
        rgb = np.array(ImageOps.exif_transpose(frame).convert("RGB"))
        pages.append(Page(_ocr_rgb(rgb, get_ocr), "ocr"))
    return Document(pages, "image")


def render_page_images(data: bytes, max_pages: int = 4, max_side: int = 1800) -> list[bytes]:
    """PNG bytes per page, for the optional vision-fallback tier."""
    out = []
    if data[:5] == b"%PDF-":
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            out.append(page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False).tobytes("png"))
        return out
    from PIL import Image, ImageOps, ImageSequence

    for i, frame in enumerate(ImageSequence.Iterator(Image.open(io.BytesIO(data)))):
        if i >= max_pages:
            break
        im = ImageOps.exif_transpose(frame).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        out.append(buf.getvalue())
    return out
