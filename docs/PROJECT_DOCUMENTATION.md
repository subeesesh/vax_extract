# vaxextract — Project Documentation

As of 27 October 2026. Companion pages: [pipeline.html](pipeline.html) (architecture diagram) and
[extraction-trace-cheddar.html](extraction-trace-cheddar.html) (one real document traced through every stage).

## Overview

vaxextract turns any veterinary vaccination document (a PDF, photo or screenshot) into vaccination events as JSON. It is a lower-cost replacement for the current Document AI call, and its output matches the existing Document AI payload (`data.pets[].vaccinations`, `owner`, `clinic`, `veterinarian`, `duration`, `tokenUsage`, `usedFallback`).

- **How it keeps cost down:** PDF text layers are read for free. Images go through local OCR (RapidOCR). One small text LLM (Gemini Flash-Lite) does the reading. Everything that has to be correct is handled by deterministic Python.
- **No hardcoded vaccine names:** each request sends the kennel's own vaccine names (e.g. `Bordetella`, `DHLPP / FVRCP`, `Parvo`, `Rabies`). The LLM matches each printed row to those names.
- **The product is the final JSON** (`to_output`). The web UI and the verification sheet are views of it.
- **Latency target:** a few seconds per certificate, about 6 s on average. Measured runs currently take 7–32 s. Most of that is time waiting for the provider's first token (see Known limitations).
- **Ownership:** this module covers OCR and layout extraction. The LLM and mapping layer, which uses the paid API, is owned by others.

## Architecture & flow

Each document gets one call to the primary LLM. A second call goes to the fallback model only when postprocessing flags the result and a fallback is configured. When both run, the result with fewer review reasons wins.

```
File bytes ─► Text layer usable? ─yes─► Read text layer ─┐
                   │ no                                   ▼
                   └──► Preprocess + OCR ───────────► Layout text
                                                          │
                                                          ▼
Result JSON ◄─no── Needs review? ◄── Postprocess ◄── Primary LLM
     ▲                  │ yes, if a fallback is set
     └─ better result ─ Fallback LLM (+ page images, optional)
```

The full diagram is in [pipeline.html](pipeline.html).

| Module | Role |
| --- | --- |
| `ingest.py` | Bytes → pages of tokens (text layer or OCR); page images for the vision fallback |
| `preprocess.py` | Downscale, illumination normalisation, deskew |
| `ocr.py` | RapidOCR / PaddleOCR / Tesseract engines behind one interface |
| `layout.py` | Tokens → row text with `' \| '` cells |
| `dates.py` | Date parsing and per-document MDY/DMY inference |
| `prompt.py`, `llm.py` | Prompt, OpenAI-compatible client, hedged requests, name-expansion cache (LLM side) |
| `postprocess.py` | Deterministic rules, validation, roll-up |
| `schema.py` | Pydantic models; `to_output` builds the client payload |
| `pipeline.py` | `Config.from_env`, `extract_bytes` / `extract_file`, fallback routing, timings |
| `cli.py`, `server.py`, `sheet.py`, `eval.py` | CLI, web UI/API, verification workbook, scoring |

## Extraction module (OCR & layout)

The extraction module turns file bytes into one plain-text page per document page, laid out in rows with `' | '` between cells. OCR is used only when a page has no usable text layer.

### Ingest (`ingest.py`)

- **PDF pages** use the PDF text layer first. A page needs at least 15 words and must pass `_text_layer_usable`. Up to 50 text-layer pages are read.
- **Text layers are rejected and the page goes to OCR** when (a) more than half of the text is invisible (type 3: a scanner's OCR layer laid over the scan, which gives garbage like `10t12t2026`), or (b) on pages with 150 or more letters, fewer than 20 % of the letters are vowels (a font without a usable character map, e.g. `5DELHV` for `Rabies`).
- **OCR pages** (scanned PDFs, rendered at 200 dpi, and images) are capped at `max_pages` = 10. Multi-frame images (TIFF) are read frame by frame, and the EXIF rotation is applied.
- **Text-layer words keep their style**, meaning font size and colour. This lets the layout split off a differently styled badge (e.g. *Historical*) printed right after a name.
- **Wrapped table cells are rejoined.** When a line opens a parenthesis and the next line (3 words or fewer, same indent) closes it (`LEPTO (1` / `YR)`), the tail moves back onto the first line.
- `render_page_images` produces PNGs (150 dpi, longest side 1800 px or less) for the optional vision fallback tier.

### Preprocessing (`preprocess.py`, OCR path only)

1. Downscale so the longest side is 2400 px or less.
2. Normalise illumination: divide by a blurred, dilated background. This removes colour casts, shadows and glare gradients.
3. Deskew: projection-profile search over ±10° in 0.5° steps. The image is rotated only when the angle is 0.3° or more.

### OCR engines (`ocr.py`)

| Engine | `VAX_OCR` | Install extra | Status |
| --- | --- | --- | --- |
| RapidOCR (ONNX) | `rapid` (default) | `.[rapid]` | Used in all live runs |
| PaddleOCR | `paddle` | `.[paddle]` | Not run here |
| Tesseract | `tesseract` | `.[tesseract]` + binary | Not run here |

Each engine returns `Token`s: text plus a bounding box, a confidence and an optional style.

### Layout serialisation (`layout.py`)

- **Row grouping:** tokens are sorted by vertical centre. A token joins the nearest open row within 0.6 × the median token height, but only if it does not overlap that row's boxes horizontally. This keeps the tightly spaced rows of phone screenshots apart. A refinement pass, repeated up to 4 times, moves any box that sits closer to another row.
- **Cells:** within a row, a gap wider than 1.5 average character widths, or a change in both font size and colour, becomes `' | '`.
- **Blocks:** a vertical gap larger than 1.5 × the median height adds a blank line. Pages are headed `=== PAGE n ===`.

### Dates (`dates.py`)

- **Formats parsed:** numeric (`10/21/26`, `2026-10-21`), compact (`102126`, `10212026`), month-first (`Oct 21, 2026`, `Jun7, 2029`, `Oct-06-2026`), day-first (`21st Oct 2026`) and year-month-day (`2026-Sep-21`).
- Words that are not months, such as `Exp`, are skipped before a month-first date.
- **Day/month order is inferred per document** from dates that are unambiguous (`infer_order`). `VAX_DEFAULT_DATE_ORDER` (MDY by default) is used only when no date settles it.
- **Two-digit years:** 69 and below map to 20xx; 70 and above map to 19xx.

## Postprocess rules & policy defaults

`postprocess.py` owns everything that must be correct. It has no built-in knowledge of vaccine names. It takes the LLM's raw rows (dates copied exactly as printed), the layout text and the requested names, and returns the final data plus the reasons it needs review.

### Rules, in order

1. **Dates:** infer MDY or DMY per document, parse to ISO, and clean names.
2. **Events:** create one event per (printed row × matched requested name). Names the LLM returns that were not requested are dropped and flagged.
3. **Merge duplicates:** when one dose has two different due dates, flag *conflicting due dates*.
4. **Join reminders:** a reminder ("due") row attaches to the latest administered event of the same requested name. If the reminder date does not fit that dose's own printed interval, it is left unattached. Diagnostic tests are joined the same way.
5. **Derive due dates** only from a printed interval (e.g. *3 Year*). These are marked `dueDateSource: "derived"`.
6. **Validate** and set `isLatest`, then build the roll-up per requested name.
7. **Coverage check:** a dated text line that prints a requested name but has no extracted event on that date adds *possible missed events*.

### Policy defaults

- All historical events are kept.
- Reminder-only rows are kept with `dateAdministered: null`.
- A given date is never back-computed from a due date, and a future given date never becomes a due date.
- Lot/product expiry, date of birth and tag dates are never vaccination dates.
- Rows that match no requested name stay in `events` with `target: null`. They are not flagged.
- **Combination products:** when several rows give one requested name different expiries on the same day (e.g. Lepto 1 yr + DHPP 3 yr), the earliest expiry wins.

### Review reasons

| Level | Reason |
| --- | --- |
| Document | no pet found · pet name missing · pet name equals owner/veterinarian name · no vaccination events found · *n* vaccination event(s) flagged · possible missed events |
| Event | no dates · due date before given date · due date more than 6 years after given date · given date in the future · name looks like a date or number · conflicting due dates · matched names not in the request |

Any document-level reason sets `needsHumanReview`. When a fallback model is configured, that also triggers the fallback tier.

## Output format

The default output is the client payload, which uses the Document AI shape. Adding `full=true` to the API call or `--full` to the CLI returns the full result instead, which evaluation needs.

### Client payload (default)

```json
{
  "data": {
    "pets": [{
      "name": "Buddy", "species": "Canine", "breed": "Labrador Retriever",
      "sex": "Male (Neutered)", "weight": "64.2 lbs", "color": "Black",
      "microchipNumber": "985112000000000",
      "vaccinations": [{
        "partialMatch": true,
        "matchedComponents": ["DHLPP / FVRCP"],
        "name": "DAPP Annual Vaccine",
        "dateAdministered": "2026-10-20",
        "expirationDate": "2027-10-20",
        "needsHumanReview": false
      }]
    }],
    "owner": {"name": "Jane Doe", "phone": "555-010-0000"},
    "clinic": {"name": "Example Animal Hospital", "website": "www.example.vet"},
    "veterinarian": {"name": "Dr. A. Smith, DVM", "licenseNumber": "TX 12345"}
  },
  "duration": 6.2,
  "tokenUsage": {"inputTokens": 2100, "outputTokens": 640, "totalTokens": 2740},
  "usedFallback": false
}
```

- **One pet entry per pet.** Multi-pet invoices produce several.
- **`vaccinations[]` has one entry per printed row,** with `name` exactly as printed. Given doses come first, newest first, followed by reminder-only rows, soonest due first. A reminder joins a dose printed under the same name. Different products are never merged.
- **`partialMatch` / `matchedComponents`** list the requested names that a row covers only partly (e.g. `DAPP` for `DHLPP / FVRCP`).
- **Fields that were not found are omitted.** There is no confidence score. The full example is in `final_json_format.txt`.

### Full result (`--full`), per pet

| Field | What it holds |
| --- | --- |
| `vaccinations[]` | One entry per requested name, in request order: `matchedComponents[{extractedName, dateAdministered, expiryDate, expiryDateSource, evidence}]`, `expirationDateSource`, `needsHumanReview`, `reviewReasons` |
| `events[]` | Full history: `rawName`, `target`, `partial`, `dateAdministered`, `dueDate`, `dueDateSource`, `isLatest`, `evidence` |
| `diagnosticTests[]` | Diagnostic test rows, with reminder due dates joined to the latest test of the same name |
| `printedRows[]` | The rows the client payload is built from (row × matched name) |
| `meta` | `needsHumanReview`, `reviewReasons`, `tier`, `source` (text_layer / ocr / mixed), `ocrSeconds`, `llmSeconds` |

## Setup, configuration & usage

### Install

```bash
pip install -e ".[rapid,dev]"        # RapidOCR + pytest
pip install -e ".[rapid,web]"        # + web UI / API (FastAPI, uvicorn)
# alternatives: ".[paddle,dev]" or ".[tesseract,dev]" (needs the tesseract binary)
```

Requires Python 3.10 or later. Settings can be exported in the shell or written as `KEY=value` lines in `.env` in the working directory. The CLI and the server both read `.env`, and values already set in the shell take precedence.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAX_LLM_BASE_URL` | required | Any OpenAI-compatible API (Gemini, vLLM, Ollama…) |
| `VAX_LLM_API_KEY` | empty | API key |
| `VAX_LLM_MODEL` | required | e.g. `gemini-3.5-flash-lite` |
| `VAX_LLM_REASONING_EFFORT` | unset | Lower thinking effort for lower latency |
| `VAX_LLM_HEDGE_SECONDS` | 10 | Resend once if no answer after N s; first reply wins (0 = off) |
| `VAX_FALLBACK_MODEL` | unset | Stronger model, used when the primary result needs review |
| `VAX_FALLBACK_BASE_URL` / `_API_KEY` | primary's | Fallback endpoint |
| `VAX_FALLBACK_VISION` | 0 | 1 = also send page images to the fallback model |
| `VAX_OCR` | rapid | rapid · paddle · tesseract |
| `VAX_DEFAULT_DATE_ORDER` | MDY | Used only when no date in the document is unambiguous |
| `VAX_EXPANSION_CACHE` | `.vax_cache/expansions.json` | Cache of requested-name definitions ('' = off) |
| `VAX_SHEET` | unset | Verification workbook updated on every extraction |
| `VAX_TARGETS` | Bordetella;Canine Influenza;DHLPP / FVRCP;Parvo;Rabies | Name suggestions in the web UI |
| `VAX_HOST` / `VAX_PORT` | 127.0.0.1 / 8000 | Web server address |

### Running it

- **CLI:**
    - `python -m vaxextract.cli extract docs/ --out preds/ --full --targets "Bordetella;DHLPP / FVRCP;Parvo;Rabies"`
    - The `extract` command also takes `--no-fallback` and `--sheet <file>`.
- **Library:** `extract_file("doc.pdf", Config.from_env(), targets=["Rabies", "Bordetella"])`
- **Web UI:** `python -m vaxextract.server`, then open http://127.0.0.1:8000.
    - Upload a document and tick or type the requested names.
    - The page shows the per-name result, the full history and the raw JSON next to a preview of the document.
- **API:** `POST /api/extract` takes multipart `file` and `targets="Rabies;Parvo"`, plus optional `full=true`.

### Verification sheet

When `VAX_SHEET` is set, or `--sheet` is passed, every extraction writes one row per pet × requested vaccine. The columns are:

- `PetName`, `POName`, `Immunization_Type_Name`, `Expiration_date`
- `status`: Verified, Needs review or Not found
- `Results - Comment`, `is_verified_by_ai`
- `final_json`: the document's full client payload

Re-running a document replaces its rows. Rows are keyed by `Document_URL`, which is the absolute path for the CLI and the upload's file name for the web UI. Excel locks an open workbook, so close it first. A failed write is logged and never fails the extraction.

## Testing, evaluation & benchmarks

All 46 unit tests pass (`pytest`, 2.2 s, run on 27 October 2026). None of them call a real LLM; the pipeline tests use a fake one.

| Test file | Tests | Covers |
| --- | --- | --- |
| `test_core.py` | 23 | Date parsing, postprocess on realistic LLM outputs (Rubi, Sam, Silvie, Jimmy, Odin, Luna, Benzo layouts), combination expiry, reminder joining, validation flags, coverage check, client payload shape |
| `test_pipeline.py` | 16 | Layout rows/columns, text-layer handling (badges, wrapped cells, invisible and letter-shifted layers, long PDFs), fake-LLM pipeline and fallback routing, hedged requests, name-expansion cache, deskew |
| `test_server.py` | 4 | `POST /api/extract` |
| `test_sheet.py` | 3 | Verification workbook rows |

### Evaluation

- `python -m vaxextract.cli eval --pred preds/ --gold gold/` scores at the event level. It compares (requested name, date given, due date) between `--full` predictions and gold labels.
- Gold labels are one JSON file per document, with its `targets`, in the same shape as `gold/rubi.json`. Only that one gold file exists so far.

### Latency tools

- `python tools/probe.py <files>` times every LLM HTTP call, including hedged duplicates, and shows the OCR/LLM split per document.
- `python tools/latency_split.py <file> --models a,b` separates time spent waiting for the first token from time spent generating, per model.
- Benchmark on AC power. Battery throttling skews the OCR times.

### Live results so far

- **Setup:** 16 real documents (photos, screenshots, PDFs, multi-pet invoices), Gemini Flash-Lite + RapidOCR, checked by hand against the images.
- **Accuracy:** vaccine rows and dates were correct on all but one document, a faded reminder list where the names and dates don't line up.
- **Latency:** 7–32 s per document. The latency diagnosis on 26 October 2026, with a free-tier key, measured about 2 s for OCR and about 2 s of LLM generation, plus 5–70 s of waiting for the first token. Setting `reasoning_effort` made no difference.
- **Test folders in the repo (not committed):** `dataset/` (101 files), `newtestdata/` (100), `testingdata/` (40) and `Unique Dataset/` (12).

## Known limitations & next steps

The biggest open issue is latency. Runs take 7–32 s against a target of about 6 s, and most of that is the provider's queue before the first token.

### Limitations

- **Latency:** this needs a proper benchmark with the paid key, on AC power. A different hedge setting or model may also help.
- **LLM misses:** the model sometimes drops a printed interval (*3 Year*) from a name, which leaves no expiry. It also sometimes lists an exam package as a vaccine.
- **Faded or misaligned photos:** a reminder list whose names and dates don't line up can be paired wrongly.
- **Coverage check:** it fires only when a requested name is printed literally on a dated line.
- **Pet names** are kept exactly as printed (e.g. *Coco Terrasi*).
- **Mapping quality** depends on the LLM, which belongs to the mapping owners. It should be measured in the eval.
- **Untested engines:** `PaddleEngine` and `TesseractEngine` have not been run here.

### Next steps

- [ ] Run Document AI over a sample and have a person verify the results, to build gold labels in `gold/`
- [ ] Re-measure latency on the paid key (`tools/probe.py`, `tools/latency_split.py`)
- [ ] Tune the prompt against the eval
- [ ] Run alongside Document AI in shadow mode and choose the escalation policy
- [ ] Perspective correction for heavily skewed phone photos
- [ ] Optional: rule-based parser tier for high-volume clean-table templates
- [ ] Optional: LoRA fine-tune of a small model on the verified labels
