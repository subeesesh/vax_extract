# Vaccine Expiry Extraction Rules

As of 2026-10-27.

With these rules the extractor gets **97.6%** of checkable rows right (200 of 205) against the kennel's report for
20–26 Oct 2026. Against the kennel's dates exactly as recorded, it is **89.7%** (191 of 213 rows). Both figures
cover 64 of the 74 documents; the other 10 hit the Gemini API rate limit and still need extracting.

The rules decide which expiry date the extractor reports for each requested vaccine name (the pet's
`vaccinations` roll-up). Rules 1–3 are business rules; rules 4–7 support them.

## Core rules

1. **A combination takes the earliest expiry.** When a requested combination (e.g. `DHLPP / FVRCP`) is covered by
   several printed vaccines (e.g. DA2PP 3 yr + Lepto 1 yr), its expiry is the date that expires first.
   *Example:* Cooper — DA2PP due 2029-09-21, Lepto due 2027-09-21 → 2027-09-21.
2. **A missing part goes to review.** When the only valid row for a combination covers just part of it (e.g. DA2PP
   with no Lepto), the roll-up is flagged `needsHumanReview` with the reason
   `combination not complete: only '<printed name>' is printed for <requested name>`.
   *Example:* Holly — only `DHP/P-3 Year` printed for DHLPP → review.
   Without vaccine/component lists (none are hard-coded), two or more different partial products that are still
   valid are assumed to complete each other.
3. **Month-only dates are the 1st of the month.** `MM/YYYY`, `YYYY/MM`, `MM/YY` and `Feb 2027` all mean day 1 of
   that month: `02/2027`, `2027/02`, `02/27` → 2027-02-01. Applies only when the whole date field is month-only.

## Supporting rules

4. **Expired history drops out.** An expired row is ignored when another row for the same requested name is still
   valid, so an old shot can't become the "earliest" date.
   *Example:* Sadie — `Nobivac 4/Third Vaccination` due 2021-02-12 next to `Nobivac 4 Vaccination` due 2027-10-20
   → 2027-10-20.
5. **A single (non-combination) vaccine uses the latest dose's own expiry,** not the earliest date on the page.
   *Example:* Gracie — certificate expiry 2029-10-12, reminder line 2029-10-11 → 2029-10-12.
6. **A combination line printed twice with two due dates on the same day takes the earlier date.**
   *Example:* Finn — `DHPP 3 YR + LEPTO BOOSTER` due 3/6/2027 and 3/6/2029 → 2027-03-06.
7. **A list of only future dates is read as due dates.** When every single-date row in a pet's list is in the
   future (at least 3 rows, within 6 years), those are next-due dates, not misreads. A single future date among past
   ones is still dropped as a likely OCR misread.
   *Example:* Bentley, Murphy — "Vaccinations done" lists like `02-11-27 | Kennel Cough-oral`.

Implemented in `vaxextract/postprocess.py` (`_current`, `_latest`, `_rollup`, `_merge_dups`, `_due_list`) and
`vaxextract/dates.py` (`MONTH_ONLY`; month-name dates with a 2-digit year such as `APR 14 26` are also parsed).

## How accuracy is measured

Scored with `tools/report_eval.py` against the kennel's "Completed Valid Documents" report (254 rows, 74 documents):

```
.venv/Scripts/python.exe tools/report_eval.py results/valid_docs_2026-10-20_to_26.csv results/sheet_preds --docs dataset newtestdata testingdata results/sheet_docs
```

- **Date shift:** the report stores the end of the printed due date, so its date minus 1 day is the printed date
  (Reggie's chart prints Bordetella 04-26-27; the report says 2027-04-27).
- **Matching:** each row is matched to its file through `Document_URL`, and to the pet by name; a shortened name
  counts ("Theo" = "Theodore").
- **Correct** = the extractor's expiry for that requested name equals the report's date.

| Measured against | Rows correct | % |
| --- | --- | --- |
| The kennel's report as it is | 191 / 213 | 89.7% |
| Our rules (9 combination rows where the kennel chose the later date count as correct) | 200 / 213 | 93.9% |
| Our rules, without 8 rows whose dates aren't printed on the documents | 200 / 205 | 97.6% |

- The 9 rule-driven combination rows: Ruby, Maverick ×2, Josie, Zoe, Cooper, Willow, Cece, Finn — the kennel
  recorded the 3-year date, rule 1 gives the earlier Lepto date.
- The 8 rows not printed on the documents: 4 Flea and Tick rows (Reggie, Dottie, Lucy, Zeus) and 4 rows from
  Maggie's invoice, which prints no due dates (the kennel computed them).

## Remaining failures and open items

| Document | Pet | Row | Cause |
| --- | --- | --- | --- |
| dd46f922-….pdf | Holly | DHLPP / FVRCP | LLM copied the due year wrong (2027 instead of 2029); flagged for review |
| 48a31eeb-….jpg | Mando | DAP | `DHLPP 3YR` not matched to DAP (LLM mapping step) |
| 9d86164f-….pdf | Maximus | Bordetella | Three Bordetella products; we return the newest dose, the kennel used an older one |
| cb767aee-….png | Maverick | Rabies, Bordetella | No dates on that document; both flagged for review |

- 10 documents still need extracting once the Gemini quota resets (re-run the command above; finished documents are
  cached in `results/sheet_preds/`). Tippy's month-only reminders (rule 3) are among them and not yet confirmed on a
  real document.
- The LLM step varies run to run, so a single run can be a row or two off either way.
- A month/day printed without a year (e.g. `10/12`) would be read as October 2012 by rule 3; not seen on these
  documents.
