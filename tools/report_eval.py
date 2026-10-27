"""Score extracted due dates against a kennel report (PetName, POName, Immunization_Type_Name, Expiration_date
[, Document_URL]) exported as .xlsx or .csv.

    python tools/report_eval.py report.csv results/report_preds --docs dataset results/sheet_docs
    python tools/report_eval.py testreport.xlsx results/testreport_preds

With --docs, each document of the report that has no prediction yet is extracted first, with the names the report
requests for it (cached as <preds>/<file stem>.json). Rows are matched to a prediction by the file name at the end
of Document_URL, else (no URL column) to the prediction whose pet name shares the row's first word. Inside a file
the pet is the one sharing the row's first word, or the only pet. A row is correct when the extractor's expiry for that
requested name (the pet's `vaccinations` roll-up) equals the report's date; [review] marks rows it flags for review.

Report expirations are the end of the printed due date (Reggie's chart prints Bordetella 04-26-27; the report says
2027-04-27), so the printed date is the report's date minus one day.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import openpyxl

COLS = ["PetName", "POName", "Immunization_Type_Name", "Expiration_date", "Document_URL"]


def first_word(name: str) -> str:
    return (name or "").strip().lower().replace("-", " ").split(" ")[0]


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        it = openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True)
        head = [str(h).strip() for h in next(it)]
        rows = [dict(zip(head, r)) for r in it]
    out, seen = [], set()
    for r in rows:
        if not r.get("PetName"):
            continue
        exp = r["Expiration_date"]
        if not hasattr(exp, "date"):
            s = str(exp).strip().split(" ")[0]
            exp = datetime.strptime(s, "%m/%d/%Y" if "/" in s else "%Y-%m-%d")
        url = str(r.get("Document_URL") or "").strip()
        row = (str(r["PetName"]).strip(), str(r["Immunization_Type_Name"]).strip(), exp.date(),
               url.rsplit("/", 1)[-1] if url else None)
        if row not in seen:  # the kennel export repeats some rows
            seen.add(row)
            out.append(row)
    return out


def extract_missing(rows, preds: Path, folders: list[Path]) -> None:
    from vaxextract.pipeline import Config, extract_file, load_dotenv

    load_dotenv(Path.cwd() / ".env")
    cfg = Config.from_env()
    files = {p.name: p for d in folders for p in d.rglob("*") if p.is_file()}
    targets = defaultdict(list)
    for _pet, vax, _exp, doc in rows:
        if doc and vax not in targets[doc]:
            targets[doc].append(vax)
    todo = [d for d in targets if not (preds / (Path(d).stem + ".json")).exists()]
    for i, doc in enumerate(todo, 1):
        if doc not in files:
            print(f"[{i}/{len(todo)}] {doc}: not found in {[str(f) for f in folders]}", file=sys.stderr)
            continue
        try:
            res = extract_file(files[doc], cfg, targets=targets[doc]).model_dump_json(indent=2)
        except Exception as e:  # keep the batch going
            print(f"[{i}/{len(todo)}] {doc}: ERROR {e}", file=sys.stderr)
            continue
        (preds / (Path(doc).stem + ".json")).write_text(res, encoding="utf-8")
        print(f"[{i}/{len(todo)}] {doc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("preds")
    ap.add_argument("--docs", nargs="+", type=Path, help="folders holding the report's documents: extract missing ones")
    ap.add_argument("--no-shift", action="store_true", help="compare report dates as-is (no minus one day)")
    a = ap.parse_args()

    rows = load_rows(Path(a.report))
    preds = Path(a.preds)
    preds.mkdir(parents=True, exist_ok=True)
    if a.docs:
        extract_missing(rows, preds, a.docs)
    pets_in = {f.stem: json.loads(f.read_text(encoding="utf-8"))["data"]["pets"] for f in sorted(preds.glob("*.json"))}
    by_name = {}
    for stem, pets in pets_in.items():
        for p in pets:
            by_name.setdefault(first_word(p.get("name")), (stem, p))

    shift = timedelta(0 if a.no_shift else 1)
    by_status, used, fails, reviewed = Counter(), set(), [], []
    for pet, vax, exp, doc in rows:
        want = (exp - shift).isoformat()
        if doc:
            stem = Path(doc).stem
            pets = pets_in.get(stem)
            same = lambda p: (lambda a, b: a and b and (a.startswith(b) or b.startswith(a)))(  # noqa: E731
                first_word(p.get("name")), first_word(pet))  # 'Theo' is 'Theodore' on the document
            hit = None if pets is None else next(((stem, p) for p in pets if same(p)),
                                                 (stem, pets[0]) if len(pets) == 1 else None)
        else:
            hit = by_name.get(first_word(pet))
        if not hit:
            got, status = "", "NO PREDICTION" if doc and Path(doc).stem not in pets_in else "PET NOT FOUND"
        else:
            used.add(hit[0])
            evs = hit[1].get("events", [])
            dues = [e.get("dueDate") for e in evs if e.get("target") == vax]
            roll = next((v for v in hit[1].get("vaccinations", []) if v.get("name") == vax), {})
            got = roll.get("expirationDate") or ""  # the extractor's own answer for this requested name
            status = "OK" if got == want else "NOT EXTRACTED" if not dues else "NO DUE DATE" if not got else "WRONG DATE"
            # the report's date was read, but the extractor chose another of the pet's rows for this name
            if status != "OK" and want in dues:
                status = "READ, NOT CHOSEN"
            elif status != "OK" and want in {e.get("dueDate") for e in evs}:
                status = "READ, OTHER NAME"
            if roll.get("needsHumanReview"):
                status += " [review]"
                reviewed.append((pet, vax, roll.get("reviewReasons")))
        by_status[status] += 1
        if not status.startswith("OK"):
            fails.append((doc or (hit[0] if hit else ""), pet, vax, want, got, status))
    ok = by_status["OK"] + by_status["OK [review]"]
    print(f"{ok}/{len(rows)} report rows correct ({ok / len(rows):.1%}); " +
          ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())))
    print(f"{len(reviewed)} rows sent to review by the extractor")
    docs = {r[3] for r in rows if r[3]}
    bad_docs = {f[0] for f in fails}
    if docs:
        print(f"{len(docs) - len(bad_docs & docs)}/{len(docs)} documents fully correct")
    if fails:
        print(f"\n{'Document':42} {'Pet':12} {'Vaccine':24} {'Report':10} {'Extracted':10} Result")
        for doc, pet, vax, want, got, status in sorted(fails):
            print(f"{doc[:42]:42} {pet[:12]:12} {vax[:24]:24} {want:10} {got or '-':10} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
