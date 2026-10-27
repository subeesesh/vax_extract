# vaxextract

Cost-efficient replacement for the Document AI call: **any vet vaccination document -> vaccination events (JSON)**.
Output mirrors the existing Document AI payload (`data.pets[].vaccinations`, `owner`, `clinic`, `veterinarian`,
`duration`, `tokenUsage`, `usedFallback`); see `final_json_format.txt` for an example.

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

Policy defaults (change in `postprocess.py`): all historical events kept; reminder-only items kept with
`dateAdministered: null`; given date is never back-computed from due; lot/product expiry, DOB, tags are never dates;
rows that match no requested name are kept in `events` with `target: null` and are not flagged; when several printed
rows give one requested name different expiries on the same day (combination products, e.g. Lepto 1 yr + DHPP 3 yr),
the **earliest expiry wins** (client output, roll-up and sheet).

## Output
API and CLI return the client payload by default (example in `final_json_format.txt`):
```
data.pets[{name, species, breed, sex, weight, color, microchipNumber,
           vaccinations[{partialMatch, matchedComponents, name, dateAdministered, expirationDate, needsHumanReview}]}]
data.owner / data.clinic / data.veterinarian  {name, address, phone, email, website, licenseNumber}
duration, tokenUsage{inputTokens, outputTokens, totalTokens}, usedFallback
```
- `pets[].name` is the pet; a multi-pet document gives several pets, each with its own `vaccinations`.
- `vaccinations[]` has **one entry per printed vaccine row**, `name` exactly as printed, history included: given doses
  newest first, then reminder-only rows (no `dateAdministered`) soonest due first. A reminder row joins a dose printed
  under the same name (invoice line + reminder table); different products are never merged.
- `partialMatch` / `matchedComponents`: the requested names the row covers only partly (e.g. `DAPP` for
  `DHLPP / FVRCP`); `[]` otherwise.
- A combination row whose components expire at different dates gets the earliest expiry.
- Anything not found is omitted (no `expirationDate`, no `veterinarian`, ...). There is no confidence score.

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

## Setup
```
pip install -e ".[rapid,dev]"         # or ".[paddle,dev]" / ".[tesseract,dev]" (needs the tesseract binary)
export VAX_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   # any OpenAI-compatible API (Gemini, vLLM, Ollama...)
export VAX_LLM_API_KEY=...
export VAX_LLM_MODEL=gemini-3.5-flash-lite
export VAX_LLM_REASONING_EFFORT=low     # optional; lower thinking = lower latency
export VAX_LLM_HEDGE_SECONDS=10         # resend once if no answer after N s, first reply wins (0 = off)
export VAX_EXPANSION_CACHE=.vax_cache/expansions.json  # cache of requested-name definitions ('' = off)
# optional escalation tier
export VAX_FALLBACK_MODEL=<stronger model>  VAX_FALLBACK_VISION=1   # vision=1 sends page images
export VAX_OCR=rapid                   # rapid | paddle | tesseract
export VAX_DEFAULT_DATE_ORDER=MDY      # used only when a document has no unambiguous date
export VAX_SHEET=vax_verify_failures.xlsx  # optional; every extraction updates this workbook
```
Instead of exporting, put the same `KEY=value` lines in `.env` in the working directory; the CLI and the web server
both read it (variables already set in the shell win).
```
python -m vaxextract.cli extract docs/ --out preds/ --full --targets "Bordetella;DHLPP / FVRCP;Parvo;Rabies"
python -m vaxextract.cli eval --pred preds/ --gold gold/
pytest
```
Library: `extract_file("doc.pdf", Config.from_env(), targets=["Rabies", "Bordetella"])`.

Web UI (`pip install -e ".[rapid,web]"`): `python -m vaxextract.server` -> http://127.0.0.1:8000. Upload a document, tick
the vaccine names to request (suggestions come from `VAX_TARGETS`, ';'-separated; any other name can be typed in), and
see the per-name result, full history and raw JSON next to the document preview. API: `POST /api/extract` (multipart `file`, `targets="Rabies;Parvo"`) -> the same JSON as the CLI.

## Status / handoff
- Tested (pytest, 46 tests): dates, layout, post-processing on realistic LLM outputs (incl. `matches`) for the Rubi,
  Sam, Silvie, Jimmy, Odin and Luna layouts, client payload shape, combination expiry, reminder joining,
  text-layer PDF -> pipeline -> fallback routing (fake LLM), hedged requests, deskew/colour-cast preprocessing.
- Live run (Gemini Flash-Lite + RapidOCR, 16 real documents: photos, screenshots, PDFs, multi-pet invoices):
  output spot-checked against the images; vaccine rows and dates correct on all but one photo (a faded reminder
  list where dates and names don't line up). The model sometimes drops a printed interval ("3 Year") from a name,
  leaving no expiry, or lists an exam package as a vaccine.
- **Latency is above target:** ~6 s was the goal; these runs took 7-32 s per document (mostly 10-25 s), mainly the provider's
  long-tail response times. Needs a proper benchmark (on AC power) and possibly a different hedge setting or model.
  Diagnosis (2026-10-26, free-tier key): OCR ~2 s; the LLM generates in ~2 s but waits 5-70 s for the first token
  (provider queueing; reasoning_effort makes no difference). Re-measure on the paid key, on AC power, with:
  `python tools/probe.py <files>` (time per LLM call + OCR/LLM split) and
  `python tools/latency_split.py <file> --models a,b` (waiting vs generating, per model).
- **Not run here:** `PaddleEngine` / `TesseractEngine` (need the OCR packages).
- Next steps: (1) run Document AI over a sample, human-verify -> gold labels in `gold/` (one JSON per doc with its `targets`, same
  shape as `gold/rubi.json`); (2) tune the prompt against the eval; (3) shadow-run vs Document AI and pick
  the escalation policy; (4) optional rule-parser tier for high-volume clean-table templates; (5) optional LoRA
  fine-tune of a small model on verified labels; (6) perspective correction for heavily skewed phone photos.
- Known limits: pet names are kept as printed (e.g. "Coco Terrasi"); mapping quality depends on the LLM (check it in
  the eval); the coverage check only fires when a requested name is printed literally on a dated line.
