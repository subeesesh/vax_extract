"""Pluggable OCR engines returning layout.Token lists. Engines are imported lazily."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .layout import Token, polygon_to_token


class OcrEngine(Protocol):
    def run(self, img_bgr: np.ndarray) -> list[Token]: ...


class PaddleEngine:
    """PaddleOCR 2.x (`.ocr`) and 3.x (`.predict`) are both supported."""

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR  # type: ignore

        try:
            # 3.x: preprocess.py already deskews, so skip the slow doc-orientation/unwarping models;
            # oneDNN is off because paddle 3.x on Windows CPU fails with "ConvertPirAttribute2RuntimeAttribute"
            # mobile det/rec: ~2.5x faster than the default server models on CPU with the same lines on certificates
            self._ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",
                                  text_recognition_model_name="PP-OCRv5_mobile_rec", use_textline_orientation=False,
                                  use_doc_orientation_classify=False, use_doc_unwarping=False, enable_mkldnn=False)
            self._v3 = True
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
            self._v3 = False

    def run(self, img_bgr: np.ndarray) -> list[Token]:
        toks: list[Token] = []
        if self._v3 and hasattr(self._ocr, "predict"):
            for res in self._ocr.predict(img_bgr):
                data = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
                texts = data.get("rec_texts", []) or []
                scores = data.get("rec_scores", []) or [1.0] * len(texts)
                polys = data.get("rec_polys")
                if polys is None:
                    polys = data.get("dt_polys", [])
                for t, s, p in zip(texts, scores, polys):
                    toks.append(polygon_to_token(p, t, float(s)))
            return toks
        result = self._ocr.ocr(img_bgr, cls=True)
        for page in result or []:
            for box, (text, conf) in page or []:
                toks.append(polygon_to_token(box, text, float(conf)))
        return toks


class RapidEngine:
    """RapidOCR (PP-OCR models on onnxruntime): no paddle runtime, fastest CPU option."""

    def __init__(self):
        from rapidocr import LangRec, RapidOCR  # type: ignore

        # limit by the LONG side: the default ("min") keeps tall phone photos at full size and detection crawls
        self._ocr = RapidOCR(params={"Rec.lang_type": LangRec.EN, "Global.use_cls": False,
                                     "Det.limit_type": "max", "Det.limit_side_len": 960})

    def run(self, img_bgr: np.ndarray) -> list[Token]:
        r = self._ocr(img_bgr)
        if r.boxes is None:
            return []
        return [polygon_to_token(b, t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


class TesseractEngine:
    def __init__(self, lang: str = "eng", config: str = ""):
        import pytesseract  # type: ignore

        self._pt, self._lang, self._config = pytesseract, lang, config

    def run(self, img_bgr: np.ndarray) -> list[Token]:
        d = self._pt.image_to_data(img_bgr, lang=self._lang, config=self._config, output_type=self._pt.Output.DICT)
        toks = []
        for i, text in enumerate(d["text"]):
            if text.strip() and float(d["conf"][i]) > 30:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                toks.append(Token(text, x, y, x + w, y + h, float(d["conf"][i]) / 100))
        return toks


_CACHE: dict[str, OcrEngine] = {}


def get_engine(name: str = "rapid") -> OcrEngine:
    if name not in _CACHE:
        if name == "rapid":
            _CACHE[name] = RapidEngine()
        elif name == "paddle":
            _CACHE[name] = PaddleEngine()
        elif name == "tesseract":
            _CACHE[name] = TesseractEngine()
        else:
            raise ValueError(f"unknown OCR engine '{name}' (use 'rapid', 'paddle' or 'tesseract')")
    return _CACHE[name]
