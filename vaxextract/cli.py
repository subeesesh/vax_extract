from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .eval import evaluate
from .pipeline import Config, extract_file
from .sheet import update_sheet, verified_by

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vaxextract")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="extract JSON from files/directories")
    ex.add_argument("paths", nargs="+")
    ex.add_argument("--out", help="output directory (one <name>.json per input); default stdout")
    ex.add_argument("--no-fallback", action="store_true")
    ex.add_argument("--targets", help="requested vaccine names to map to, separated by ';' (e.g. 'Rabies;DHLPP / FVRCP')")
    ex.add_argument("--sheet", default=os.environ.get("VAX_SHEET"),
                    help="workbook to update with one row per pet x vaccine (default: $VAX_SHEET)")
    ex.add_argument("--full", action="store_true", help="write the full result (events, meta); needed for eval")
    ev = sub.add_parser("eval", help="score predictions against gold labels")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--gold", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "eval":
        r = evaluate(a.pred, a.gold)
        print(json.dumps(r, indent=2))
        return 0

    cfg = Config.from_env()
    if a.no_fallback:
        cfg.fallback = None
    targets = [t.strip() for t in (a.targets or "").split(";") if t.strip()]
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    rc = 0
    for f in files:
        try:
            r = extract_file(f, cfg, targets=targets)
            res = r.model_dump_json(indent=2) if a.full else json.dumps(r.to_output(), indent=2)
            if a.sheet and not update_sheet(a.sheet, r, str(f.resolve()), verified_by(cfg.primary.model, cfg.ocr_engine)):
                print(f"{f}: could not write {a.sheet} (is it open in Excel?)", file=sys.stderr)
                rc = 1
        except Exception as e:  # keep batch going
            print(f"{f}: ERROR {e}", file=sys.stderr)
            rc = 1
            continue
        if a.out:
            (Path(a.out) / (f.stem + ".json")).write_text(res, encoding="utf-8")
        else:
            print(res)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
