"""Split LLM latency into waiting (time to first token) and generating, per model, on one document's real request.

    python tools/latency_split.py dataset/a.pdf [--models gemini-3.5-flash-lite,gemini-3.1-flash-lite] [--runs 3]

A long first-token time with a short generation time means provider-side queueing (e.g. free-tier priority),
not output size or our code. Streams the same messages, schema and reasoning_effort the pipeline sends.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import httpx

from vaxextract import ingest, llm, ocr
from vaxextract.layout import serialize
from vaxextract.pipeline import Config, load_dotenv
from vaxextract.prompt import SYSTEM_PROMPT, user_prompt

DEFAULT_TARGETS = "Rabies;DHLPP / FVRCP;Bordetella;Leptospirosis"


def _stream(url: str, key: str, payload: dict, timeout: float):
    """-> (first_token_s, total_s, chars) or raises."""
    t, first, chars = time.time(), None, 0
    with httpx.stream("POST", url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} {r.read()[:150]!r}")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                first = first or time.time()
                chars += len(delta)
    return (first - t if first else float("nan")), time.time() - t, chars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--models", help="comma-separated; default: VAX_LLM_MODEL")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="requested vaccine names, ';'-separated")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    load_dotenv(Path.cwd() / ".env")
    cfg = Config.from_env()
    p = cfg.primary
    targets = [t.strip() for t in a.targets.split(";") if t.strip()]

    t = time.time()
    doc = ingest.load(Path(a.file).read_bytes(), lambda: ocr.get_engine(cfg.ocr_engine), cfg.max_pages,
                      cfg.min_text_layer_words)
    text = serialize([pg.tokens for pg in doc.pages])
    print(f"ingest {time.time() - t:.1f}s ({doc.pages[0].source if doc.pages else '-'}, {len(doc.pages)} page(s))")
    expansion, _ = llm.expand_targets(p, targets)  # cached after the first run; not part of the timing
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(text, targets=targets, expansion=expansion)}]
    url = p.base_url.rstrip("/") + "/chat/completions"

    for model in (a.models.split(",") if a.models else [p.model]):
        payload = {"model": model.strip(), "messages": messages, "temperature": 0, "max_tokens": p.max_tokens,
                   "stream": True, "response_format": {"type": "json_schema", "json_schema": {
                       "name": "extraction", "strict": True, "schema": llm.output_schema(targets)}}}
        if p.reasoning_effort:
            payload["reasoning_effort"] = p.reasoning_effort
        firsts, totals = [], []
        for _ in range(a.runs):
            try:
                first, total, chars = _stream(url, p.api_key, payload, a.timeout)
            except Exception as e:
                print(f"  {model:28} {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            firsts.append(first)
            totals.append(total)
            print(f"  {model:28} first token {first:5.1f}s   generating {total - first:4.1f}s   total {total:5.1f}s"
                  f"   {chars} chars", flush=True)
        if totals:
            print(f"  {model:28} median: first token {st.median(firsts):.1f}s, total {st.median(totals):.1f}s")


if __name__ == "__main__":
    main()
