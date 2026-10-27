"""Orchestration: ingest -> layout text -> LLM -> post-process -> (escalate if low confidence)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from . import ingest, llm, ocr
from .layout import serialize
from .llm import LlmConfig, LlmError
from .postprocess import build_data
from .schema import Data, Meta, Result, TokenUsage


def load_dotenv(path: Path) -> None:
    """KEY=value lines into os.environ; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


@dataclass
class Config:
    primary: LlmConfig
    fallback: Optional[LlmConfig] = None
    ocr_engine: str = "rapid"
    max_pages: int = 10  # pages that need OCR; text-layer PDF pages are read up to ingest's max_text_pages
    min_text_layer_words: int = 15

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ
        primary = LlmConfig(e["VAX_LLM_BASE_URL"], e.get("VAX_LLM_API_KEY", ""), e["VAX_LLM_MODEL"],
                            reasoning_effort=e.get("VAX_LLM_REASONING_EFFORT") or None,
                            hedge_after=float(e.get("VAX_LLM_HEDGE_SECONDS", "10") or 0))
        fb = None
        if e.get("VAX_FALLBACK_MODEL"):
            fb = LlmConfig(
                e.get("VAX_FALLBACK_BASE_URL", primary.base_url),
                e.get("VAX_FALLBACK_API_KEY", primary.api_key),
                e["VAX_FALLBACK_MODEL"],
                vision=e.get("VAX_FALLBACK_VISION", "0") == "1",
            )
        return cls(primary, fb, e.get("VAX_OCR", "rapid"))


def _usage(acc: TokenUsage, u: dict) -> None:
    acc.inputTokens += u.get("prompt_tokens", 0)
    acc.outputTokens += u.get("completion_tokens", 0)
    acc.totalTokens = acc.inputTokens + acc.outputTokens


def _score(r: Result):
    return (len(r.meta.reviewReasons), -sum(len(p.events) for p in r.data.pets))


def _run(tier: str, lcfg: LlmConfig, text: str, images, today, targets, usage: TokenUsage, meta_base: dict) -> Result:
    raw, u = llm.extract(lcfg, text, images, targets)
    _usage(usage, u)
    data, reasons = build_data(raw, text, today, targets)
    meta = Meta(tier=tier, needsHumanReview=bool(reasons), reviewReasons=reasons, **meta_base)
    return Result(data=data, meta=meta, tokenUsage=usage.model_copy())


def extract_bytes(data: bytes, cfg: Config, today: Optional[date] = None,
                  targets: Optional[list[str]] = None) -> Result:
    """`targets`: the requested vaccine names (per request) that extracted rows are mapped to."""
    t0 = time.time()
    doc = ingest.load(data, lambda: ocr.get_engine(cfg.ocr_engine), cfg.max_pages, cfg.min_text_layer_words)
    text = serialize([p.tokens for p in doc.pages])
    t_ocr = time.time() - t0
    sources = {p.source for p in doc.pages}
    meta_base = {"pages": len(doc.pages), "source": sources.pop() if len(sources) == 1 else "mixed"}
    usage = TokenUsage()
    result: Optional[Result] = None
    try:
        result = _run("primary", cfg.primary, text, None, today, targets, usage, meta_base)
    except LlmError as e:
        if not cfg.fallback:
            raise
        primary_err = str(e)
    if cfg.fallback and (result is None or result.meta.needsHumanReview):
        images = ingest.render_page_images(data) if cfg.fallback.vision else None
        try:
            fb = _run("fallback", cfg.fallback, text, images, today, targets, usage, meta_base)
            if result is None or _score(fb) < _score(result):
                result = fb
        except LlmError:
            if result is None:
                raise
        result.usedFallback = True
    assert result is not None
    result.duration = round(time.time() - t0, 3)
    result.meta.ocrSeconds = round(t_ocr, 3)
    result.meta.llmSeconds = round(result.duration - t_ocr, 3)
    result.tokenUsage = usage
    return result


def extract_file(path: str | Path, cfg: Config, today: Optional[date] = None,
                 targets: Optional[list[str]] = None) -> Result:
    return extract_bytes(Path(path).read_bytes(), cfg, today, targets)
