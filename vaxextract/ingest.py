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


def _text_layer_tokens(page) -> list[Token]:
    """Words of a PDF text layer, each tagged with its span's (size, colour) so the layout can split a
    differently styled label printed next to a name (a 'Historical' badge). A table cell wrapped inside an
    open parenthesis ('LEPTO (1' / 'YR)') is put back on its first line, where the name starts: only a short
    tail that closes the parenthesis, so paragraphs keep their lines."""
    spans = [(s["bbox"], (round(s["size"], 1), s["color"]))
             for b in page.get_text("dict")["blocks"] for ln in b.get("lines", []) for s in ln["spans"]]

    def style(w):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return next((st for (x0, y0, x1, y1), st in spans if x0 <= cx <= x1 and y0 <= cy <= y1), None)

    lines: dict = {}
    for w in page.get_text("words"):
        if w[4].strip():
            lines.setdefault((w[5], w[6]), []).append(w)
    keys = list(lines)
    toks = []
    for k in keys:
        ws = lines.pop(k, None)
        if not ws:
            continue
        row = [Token(w[4], w[0], w[1], w[2], w[3], style=style(w)) for w in ws]
        text = " ".join(w[4] for w in ws)
        nxt = lines.get((k[0], k[1] + 1))
        if (text.count("(") > text.count(")") and nxt and len(nxt) <= 3 and abs(nxt[0][0] - ws[0][0]) < ws[0][3] - ws[0][1]
                and (text + " ".join(w[4] for w in nxt)).count("(") == (text + " ".join(w[4] for w in nxt)).count(")")):
            row[-1].text += " " + " ".join(w[4] for w in lines.pop((k[0], k[1] + 1)))
        toks += row
    return toks


def load(data: bytes, get_ocr: Callable, max_pages: int = 10, min_words: int = 15,
         max_text_pages: int = 50) -> Document:
    """`max_pages` caps the pages that need OCR; text-layer PDF pages cost no OCR, so long text PDFs
    (patient histories) are read up to `max_text_pages`."""
    if data[:5] == b"%PDF-":
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        pages, n_ocr = [], 0
        for i, page in enumerate(doc):
            if i >= max_text_pages:
                break
            toks = _text_layer_tokens(page)
            if len(toks) >= min_words:
                pages.append(Page(toks, "text_layer"))
                continue
            if n_ocr >= max_pages:
                continue
            n_ocr += 1
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
