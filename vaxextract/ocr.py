"""Pluggable OCR engines returning layout.Token lists. Engines are imported lazily."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .layout import Token, polygon_to_token

TALL_RATIO = 1.6  # pages taller than this x their width (phone screenshots) get the larger detector size


class OcrEngine(Protocol):
    def run(self, img_bgr: np.ndarray) -> list[Token]: ...


class RapidEngine:
    """RapidOCR (PP-OCR models on onnxruntime): no paddle runtime, fastest CPU option."""

    def __init__(self):
        self._ocr = self._make(960)
        self._tall = None

    @staticmethod
    def _make(limit: int):
        from rapidocr import LangRec, RapidOCR  # type: ignore

        # limit by the LONG side: the default ("min") keeps tall phone photos at full size and detection crawls
        return RapidOCR(params={"Rec.lang_type": LangRec.EN, "Global.use_cls": False,
                                "Det.limit_type": "max", "Det.limit_side_len": limit})

    def run(self, img_bgr: np.ndarray) -> list[Token]:
        # on a 1:2 phone screenshot a 960 px long side leaves small print (invoice dates, a due date under a
        # wrapped row) below the detector's size floor; 1440 recovers it for ~0.4 s, a fraction of strip tiling
        h, w = img_bgr.shape[:2]
        if h > TALL_RATIO * w:
            if self._tall is None:
                self._tall = self._make(1440)
            r = self._tall(img_bgr)
        else:
            r = self._ocr(img_bgr)
        if r.boxes is None:
            return []
        return [polygon_to_token(b, t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


_CACHE: dict[str, OcrEngine] = {}


def get_engine(name: str = "rapid") -> OcrEngine:
    if name not in _CACHE:
        if name != "rapid":
            raise ValueError(f"unknown OCR engine '{name}' (only 'rapid')")
        _CACHE[name] = RapidEngine()
    return _CACHE[name]
