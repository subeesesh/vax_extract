"""Web UI: upload a document, pick the requested vaccine names, get the extraction back.

    python -m vaxextract.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .llm import LlmError
from .pipeline import Config, extract_bytes, load_dotenv, read_layout
from .sheet import update_sheet, verified_by

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 25 * 1024 * 1024
log = logging.getLogger(__name__)


load_dotenv(Path.cwd() / ".env")
app = FastAPI(title="vaxextract")
_lock = threading.Lock()  # the OCR engine is a shared singleton


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/targets")
def targets():
    """Suggested names for the picker (VAX_TARGETS, ';'-separated); the user can add any others."""
    raw = os.environ.get("VAX_TARGETS", "Bordetella;Canine Influenza;DHLPP / FVRCP;Parvo;Rabies")
    return [t.strip() for t in raw.split(";") if t.strip()]


@app.post("/api/extract")
def extract(file: UploadFile = File(...), targets: str = Form(""), full: bool = Form(False)):
    """Returns the slim payload; `full=true` returns the whole Result (history, meta) for the UI, plus the slim
    payload under `output`."""
    names = [t.strip() for t in targets.split(";") if t.strip()]  # none -> extraction only, no mapping
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    try:
        with _lock:
            cfg = Config.from_env()
            res = extract_bytes(data, cfg, targets=names)
            if os.environ.get("VAX_SHEET"):
                try:
                    update_sheet(os.environ["VAX_SHEET"], res, file.filename,
                                 verified_by(cfg.primary.model, cfg.ocr_engine))
                except Exception:  # the sheet is a side record; never fail the extraction over it
                    log.exception("sheet update failed")
    except LlmError as e:
        raise HTTPException(502, f"LLM error: {e}")
    except Exception as e:  # unreadable / unsupported file
        raise HTTPException(422, f"could not process {file.filename}: {e}")
    return {**res.model_dump(), "output": res.to_output()} if full else res.to_output()


@app.post("/api/layout")
def layout(file: UploadFile = File(...)):
    """OCR + layout only, no LLM call: the serialized text the LLM would receive."""
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    t0 = time.time()
    try:
        with _lock:
            doc, text = read_layout(data, os.environ.get("VAX_OCR", "rapid"))
    except Exception as e:  # unreadable / unsupported file
        raise HTTPException(422, f"could not process {file.filename}: {e}")
    return {"text": text, "kind": doc.kind, "pages": len(doc.pages),
            "sources": [p.source for p in doc.pages], "duration": round(time.time() - t0, 3)}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("VAX_HOST", "127.0.0.1"), port=int(os.environ.get("VAX_PORT", "8000")))


if __name__ == "__main__":
    main()
