"""Run real extractions and time every LLM HTTP call (incl. hedged duplicates), plus the OCR / LLM split.

    python tools/probe.py dataset/a.jpg dataset/b.pdf [--targets "Rabies;DHLPP / FVRCP"]

Reads .env like the CLI. Shows whether a slow document is one slow call, a hedge, a retry or a format fallback.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import httpx

from vaxextract.pipeline import Config, extract_file, load_dotenv

DEFAULT_TARGETS = "Rabies;DHLPP / FVRCP;Bordetella;Leptospirosis"


def _timed(real):
    def post(url, json=None, **kw):
        t, who = time.time(), threading.current_thread().name
        fmt = ((json or {}).get("response_format") or {}).get("type")
        try:
            r = real(url, json=json, **kw)
        except Exception as e:
            print(f"    call [{who}] {time.time() - t:5.1f}s {type(e).__name__}", flush=True)
            raise
        u = r.json().get("usage", {}) if r.status_code == 200 else {}
        err = "" if r.status_code == 200 else " " + r.text[:120]
        print(f"    call [{who}] {time.time() - t:5.1f}s HTTP {r.status_code} format={fmt} "
              f"in={u.get('prompt_tokens')} out={u.get('completion_tokens')}{err}", flush=True)
        return r
    return post


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="requested vaccine names, ';'-separated")
    a = ap.parse_args()
    load_dotenv(Path.cwd() / ".env")
    httpx.post = _timed(httpx.post)
    cfg = Config.from_env()
    targets = [t.strip() for t in a.targets.split(";") if t.strip()]
    for f in a.files:
        print(Path(f).name, flush=True)
        r = extract_file(f, cfg, targets=targets)
        m = r.meta
        print(f"  total {r.duration:.1f}s = ocr {m.ocrSeconds:.1f}s ({m.source}, {m.pages} page(s)) + llm {m.llmSeconds:.1f}s"
              f"   tier={m.tier} fallback={r.usedFallback}", flush=True)


if __name__ == "__main__":
    main()
