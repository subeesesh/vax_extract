# vaxextract

Cost-efficient replacement for the Document AI call: **any vet vaccination document -> vaccination events (JSON)**.
Output: one entry per requested vaccine per pet, with the printed rows behind it, plus `owner`, `clinic`,
`veterinarian`, `duration` and `tokenUsage`; see `final_json_format.txt` for an example.

## Flow
```
bytes -> ingest (PDF text layer = free; else preprocess + OCR) -> layout text (rows, ' | ' cells)
      -> small text LLM (JSON, dates copied as printed) -> postprocess (deterministic)
      -> needsHumanReview? -> fallback LLM (optionally with page images) -> Result
```
No vaccine names are hardcoded. The caller sends the **requested vaccine names** per request (e.g. the kennel's
`Bordetella`, `DHLPP / FVRCP`, `Parvo`, `Rabies`); the LLM extracts every printed row and, in the same call, says which
requested names each row covers (`matches`, with `partial` and any per-component printed interval). Names it returns that
are not in the request are dropped and flagged.

Postprocess (`postprocess.py`) owns everything that must be correct and knows nothing about vaccines: per-document
MDY/DMY inference, ISO dates, one event per (row x matched name), join of reminder/"due" rows to the latest administered
event of the same requested name (a reminder whose date does not fit the dose's own printed interval is left alone),
derived due dates (only from a printed interval, flagged `dueDateSource: "derived"`),
validation, coverage check (a dated text line containing a requested name with no extracted event on that date ->
`possible missed events`), `isLatest`, and the per-requested-name roll-up.

## Rules
Applied deterministically in `postprocess.py` / `dates.py` (no vaccine name tables anywhere):

**Expiry per requested name**
1. **Combination = earliest expiry.** When several printed rows cover one requested name (e.g. `DHLPP / FVRCP` from
   DA2PP 3 yr + Lepto 1 yr), the part that expires first sets the date (client output, roll-up and sheet).
2. **Missing part -> review.** If the rows in force cover only part of a requested name and the rest is not on the
   document, the name gets `partialMatch` and `needsHumanReview`.
3. **Latest dose wins.** An older dose is superseded by a later dose of the same name that expires no earlier.

**Dates**
4. **Month-only dates** (`MM/YYYY`, `YYYY/MM`, `Mon YYYY`, `MM/YY` in reminder lists) mean the **1st of that month**.
5. **MDY vs DMY** is inferred once per document from its unambiguous dates; `VAX_DEFAULT_DATE_ORDER` (default MDY) is
   used only when none exists.
6. Dates are copied **as printed**. A due date is derived only from a printed interval (`dueDateSource: "derived"`); the
   given date is never back-computed from the due date.
7. Lot / product expiry, date of birth and tag dates are never vaccination dates.

**Rows and reminders**
8. All historical events are kept; reminder-only rows are kept with `dateAdministered: null`.
9. A reminder / "due" row joins the latest dose of the same requested name, unless its date doesn't fit that dose's
   printed interval. Different products are never merged.
10. Rows matching no requested name stay in `events` with `target: null` and are not flagged.
11. A dated line that literally contains a requested name but produced no event -> `possible missed events` (review).

## Output
API and CLI return the client payload by default (example in `final_json_format.txt`):
```
pets[{name, species, breed, sex, weight, color, microchipNumber,
      vaccinations[{name, dateAdministered, expirationDate, partialMatch?, needsHumanReview?,
                    matchedComponents[{name, dateAdministered?, expirationDate?}]}]}]
owner / clinic / veterinarian  {name, address, phone, email, website, licenseNumber}
duration, tokenUsage{inputTokens, outputTokens, totalTokens}
```
- `pets[].name` is the pet; a multi-pet document gives several pets, each with its own `vaccinations`.
- `vaccinations[]` has **one entry per requested vaccine name**, in request order; `name` is the requested name and the
  dates are those of its latest dose (the expiry may be worked out from a printed interval). A combination expires with
  its first-expiring part. Printed vaccines nobody asked for are left out.
- `matchedComponents[]`: every printed row matched to that name, newest first, with the name and dates **exactly as on
  the document**; a date that is not printed is left out, never calculated.
- Requested but not on the document: `dateAdministered` / `expirationDate` are `null`, `matchedComponents` is `[]`.
- `partialMatch` / `needsHumanReview` appear only when true. Empty pet/party fields are left out. No confidence score.

`POST /api/extract` with `full=true` (the web UI does this) or `cli extract --full` returns the full result below;
**eval needs `--full` predictions**.

### Full result, per pet
- `vaccinations[]` - one per requested name, in request order (Document AI shape): `name`, `partialMatch`,
  `matchedComponents[{extractedName, dateAdministered, expiryDate, expiryDateSource, evidence}]`, `vaccine_name`,
  `dateAdministered`, `expirationDate`, `expirationDateSource`, `needsHumanReview`, `reviewReasons`. Built from the
  latest administered record (else the latest reminder); a requested name with no match has empty `matchedComponents`.
- `events[]` - full history: `rawName`, `target`, `partial`, `dateAdministered`, `dueDate`, `dueDateSource`, `isLatest`,
  `needsHumanReview`, `reviewReasons`, `evidence`.
- `diagnosticTests[]`.
- `printedRows[]` - the rows the client payload is built from (row x matched name, before merging).
Document-level: `meta.needsHumanReview`, `meta.reviewReasons`, `meta.tier`, `meta.source`, `meta.ocrSeconds`,
`meta.llmSeconds`.

## Verification sheet
With `VAX_SHEET` set (or `cli extract --sheet <file>`), every extraction (web UI, API, CLI) writes one row per
pet x requested vaccine into the workbook's columns (`PetName`, `POName`, `Immunization_Type_Name`, `Expiration_date`,
`status` Verified / Needs review / Not found, `Results - Comment`, `is_verified_by_ai`, `final_json` = the document's full
client payload as in `final_json_format.txt`, ...). Re-running a document replaces its rows (keyed by `Document_URL`: the absolute path from the
CLI, the upload's file name from the web UI). Excel locks an open workbook, so close it first; a failed write is logged
and never fails the extraction.

## Cost vs Document AI
| | Document AI (Custom Extractor) | vaxextract (Gemini 3.5 Flash-Lite + RapidOCR) |
|---|---|---|
| Billing | per page | per token; OCR runs locally (free) |
| Price | $30 / 1,000 pages (first 1M pages/month), $20 after; + $0.05/h per deployed custom processor (~$438/yr) | $0.30 / 1M input tokens, $2.50 / 1M output tokens |
| Per document | ~$0.034 (1.14 pages avg) | ~$0.0033 (1,958 input + 1,078 output tokens avg) |
| **Per 1,000 documents** | **~$34** | **~$3.3** (up to ~$6.6 if every call were hedged) |

About **10x cheaper**. Our averages come from 208 extractions in `results/` (159 with page counts; no fallback calls).
Text-layer PDFs skip OCR entirely; layout-only mode (below) makes no LLM call at all. Hedged requests
(`VAX_LLM_HEDGE_SECONDS`) can bill a slow call twice, and a fallback model (if set) adds its own cost on flagged
documents. Assumes Document AI runs a Custom Extractor (Form Parser is the same price); list prices as of 2026-10-27:
[Document AI pricing](https://cloud.google.com/document-ai/pricing),
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). The Batch tier halves the Gemini price
($0.15 / $1.25) where latency doesn't matter.

## How to run
Use the project's virtual environment (`.venv`); the system Python lacks the dependencies (e.g. PyMuPDF).
```
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e ".[rapid,web,dev]"
```
Put the LLM settings in `.env` (see Setup below), then:

- **Web UI:** `python -m vaxextract.server` -> http://127.0.0.1:8000. Upload a document, tick the vaccine names to
  check and press **Check record**. Tick **Layout only (no LLM)** to see just the OCR/layout text (no API key or quota
  needed).
- **CLI:** `python -m vaxextract.cli extract <file or folder> --targets "Rabies;Parvo" [--out preds/ --full]`
- **API:** `POST /api/extract` (multipart `file`, `targets="Rabies;Parvo"`) -> JSON;
  `POST /api/layout` (multipart `file`) -> `{text, kind, pages, sources, duration}`, no LLM call.
- **Tests:** `pytest`

## Setup
```
pip install -e ".[rapid,dev]"
export VAX_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   # any OpenAI-compatible API (Gemini, vLLM, Ollama...)
export VAX_LLM_API_KEY=...
export VAX_LLM_MODEL=gemini-3.5-flash-lite
export VAX_LLM_REASONING_EFFORT=low     # optional; lower thinking = lower latency
export VAX_LLM_HEDGE_SECONDS=10         # resend once if no answer after N s, first reply wins (0 = off)
export VAX_EXPANSION_CACHE=.vax_cache/expansions.json  # cache of requested-name definitions ('' = off)
# optional escalation tier
export VAX_FALLBACK_MODEL=<stronger model>
export VAX_DEFAULT_DATE_ORDER=MDY      # used only when a document has no unambiguous date
export VAX_SHEET=vax_verify_failures.xlsx  # optional; every extraction updates this workbook
```
Instead of exporting, put the same `KEY=value` lines in `.env` in the working directory; the CLI and the web server
both read it (variables already set in the shell win).
```
python -m vaxextract.cli extract docs/ --out preds/ --full --targets "Bordetella;DHLPP / FVRCP;Parvo;Rabies"
python tools/report_eval.py <kennel report .xlsx/.csv> <preds dir> --docs <document folders>   # accuracy
pytest
```
Library: `extract_file("doc.pdf", Config.from_env(), targets=["Rabies", "Bordetella"])`.

Web UI (`pip install -e ".[rapid,web]"`): `python -m vaxextract.server` -> http://127.0.0.1:8000. Upload a document, tick
the vaccine names to request (suggestions come from `VAX_TARGETS`, ';'-separated; any other name can be typed in), and
see the per-name result, full history and raw JSON next to the document preview. API: `POST /api/extract` (multipart `file`, `targets="Rabies;Parvo"`) -> the same JSON as the CLI.
The **Layout only (no LLM)** toggle calls `POST /api/layout` instead and shows the serialized layout text.

## Status / handoff
- Tested (pytest, 51 tests): dates, layout, post-processing on realistic LLM outputs (incl. `matches`) for the Rubi,
  Sam, Silvie, Jimmy, Odin and Luna layouts, client payload shape, combination expiry, reminder joining,
  text-layer PDF -> pipeline -> fallback routing (fake LLM), hedged requests, deskew/colour-cast preprocessing.
- Live run (Gemini Flash-Lite + RapidOCR, 16 real documents: photos, screenshots, PDFs, multi-pet invoices):
  output spot-checked against the images; vaccine rows and dates correct on all but one photo (a faded reminder
  list where dates and names don't line up). The model sometimes drops a printed interval ("3 Year") from a name,
  leaving no expiry, or lists an exam package as a vaccine.
- **Accuracy (AI-checked, not yet human-verified; the gold-label tooling has since been removed):** 50-document sample (seed 0, targets
  Bordetella; Canine Influenza; DHLPP / FVRCP; Parvo; Rabies), each document checked by an AI pass against its image;
  47 scored (2 unreadable photos left for a human, 1 duplicate file). Event F1 **91.8 %** strict (target + date given +
  due date; P 92.2 %, R 91.4 %), **94.5 %** on target + date given; pet names 93.6 %; 37/47 documents exactly right.
  Errors by layer: OCR/layout - handwriting (3 shelter cards), small light text dropped on tall phone screenshots
  (detector limit 960 px on the long side), one digit misread; LLM/mapping - Lepto rows not mapped to DHLPP,
  reminder lines counted as a second dose, printed reminders not used as due dates, a "DECLINED" vaccine taken as
  given. Treat as an estimate until the sample is spot-checked by a person (`gold_draft/review.xlsx`).
- **Latency is above target:** ~6 s was the goal; these runs took 7-32 s per document (mostly 10-25 s), mainly the provider's
  long-tail response times. Needs a proper benchmark (on AC power) and possibly a different hedge setting or model.
  Diagnosis (2026-10-26, free-tier key): OCR ~2 s; the LLM generates in ~2 s but waits 5-70 s for the first token
  (provider queueing; reasoning_effort makes no difference). Re-measure on the paid key, on AC power, with:
  `python tools/probe.py <files>` (time per LLM call + OCR/LLM split) and
  `python tools/latency_split.py <file> --models a,b` (waiting vs generating, per model).
- **Accuracy is now measured against the kennel's own reports** with `tools/report_eval.py` (see docs/EXPIRY_RULES.md).
- Next steps: (1) tune the prompt against the kennel-report eval; (2) shadow-run vs Document AI and pick
  the escalation policy; (3) optional rule-parser tier for high-volume clean-table templates; (4) optional LoRA
  fine-tune of a small model on verified labels; (5) perspective correction for heavily skewed phone photos.
- Known limits: pet names are kept as printed (e.g. "Coco Terrasi"); mapping quality depends on the LLM (check it in
  the eval); the coverage check only fires when a requested name is printed literally on a dated line.
