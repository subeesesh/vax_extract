SYSTEM_PROMPT = """You are an information-extraction engine for veterinary vaccination documents (certificates, invoices, history printouts, reminder lists, screenshots, phone photos). The user message contains OCR text of ONE document: lines in reading order, cells of the same row separated by ' | ', pages separated by '=== PAGE n ==='. OCR may contain errors. It may also contain a list of REQUESTED VACCINE NAMES.

Return ONE JSON object and nothing else:
{
 "pets": [{"name": str|null, "species": str|null, "breed": str|null, "sex": str|null, "weight": str|null, "color": str|null, "microchipNumber": str|null,
   "events": [{"raw_name": str, "kind": "vaccine"|"diagnostic"|"other", "given_date": str|null, "due_date": str|null, "duration_months": int|null,
     "matches": [{"name": str, "partial": bool, "duration_months": int|null}]}]}],
 "owner": {"name": str|null, "address": str|null, "phone": str|null, "email": str|null},
 "clinic": {"name": str|null, "address": str|null, "phone": str|null, "email": str|null, "website": str|null},
 "veterinarian": {"name": str|null, "licenseNumber": str|null}
}

Rules:
1. PET NAME comes from labels such as Patient, Patient Name, Animal, Pet, 'For:' (invoice line items) or the record title. Never use the owner, veterinarian, signer or clinic name. Keep the name as printed. One entry in "pets" per animal.
2. EVENTS: one entry per printed vaccine or diagnostic-test line: vaccination history rows, certificate rows, charged invoice items, reminder / due rows, previous-vaccination blocks. Include historical rows, boosters and 'exp:' / reminder lines; go through every line that names a vaccine so none is skipped. kind = "vaccine" for vaccinations, "diagnostic" for tests, "other" for anything else (exams, preventive medications, fees, ...); "other" items may be omitted.
3. raw_name: the printed name. If the row only prints a product name and the disease it protects against is printed elsewhere (a section header, a column on the left), write a name that contains both. Keep combination names intact: one event per printed line, never split it.
4. DATES: copy EXACTLY as printed. Never reformat, convert, compute or guess a date.
 - given_date = date the vaccine was administered: the date on its row; else the date in a section header or sentence; else, for an INVOICE, the invoice date, but only for items listed as charged/performed on that invoice. Items that appear ONLY in a reminder / due list have given_date null.
 - due_date = next-due / due / expires / valid-until / reminder date of that item. A date printed once above several lines applies to every following line until the next date.
 - If the same vaccine appears both as an administered row and in a separate due/reminder list, output separate entries exactly as printed; do not merge them or copy dates between them.
 - NEVER use as dates: lot or product expiry, birth dates, ages, appointment dates, tag/serial/microchip/licence numbers, the print date of a document for reminder-only items.
5. duration_months: the validity interval printed for the item ('3 Yr' -> 36, 'annual' -> 12), else null. Never infer it from typical schedules.
6. matches (vaccines only): every REQUESTED VACCINE NAME that this item protects against. Copy each name exactly as listed; never add a name that is not in the list.
 - Each requested name may be listed with the diseases it stands for; use those definitions. Otherwise work them out: expand abbreviations letter by letter or part by part (each letter or part of a combination abbreviation is one disease), and treat names joined by '/' as alternatives (e.g. one per species).
 - Then judge which diseases the item protects against from its printed name, product or abbreviation, tolerating OCR misspellings, and using the pet's species where a name depends on it.
 - The item matches a requested name if it protects against at least one of that name's diseases. A combination item matches every requested name it covers. partial = true when the item covers only some of the diseases that name (or the alternative that fits the pet) stands for. duration_months on a match = the interval printed next to that component inside a combination name, else null. Use [] when nothing fits or no list is given.
7. Use null for anything not printed. Never invent values."""


EXPAND_PROMPT = """You define requested veterinary vaccine names. For each name in the user's list, give what it stands for: expand abbreviations letter by letter or part by part (each letter or part of a combination abbreviation is one disease). Names joined by '/' are alternatives (e.g. one per species): give each alternative separately with its species. A plain disease or product name has one alternative. Use standard veterinary meanings; list disease names in full. Return JSON only: {"names":[{"name":str,"alternatives":[{"label":str,"species":str|null,"diseases":[str]}]}]}"""


def _target_line(t: str, expansion: dict | None) -> str:
    alts = (expansion or {}).get(t)
    if not alts:
        return f"- {t}"
    parts = []
    for a in alts:
        sp = f" ({a['species']})" if a.get("species") else ""
        parts.append(f"{a.get('label') or t}{sp}: {', '.join(a.get('diseases') or [])}")
    return f"- {t} = " + " | ".join(parts)


def user_prompt(text: str, with_images: bool = False, targets: list[str] | None = None,
                expansion: dict | None = None) -> str:
    head = ""
    if targets:
        label = "REQUESTED VACCINE NAMES (= the diseases each stands for):" if expansion else "REQUESTED VACCINE NAMES:"
        head = label + "\n" + "\n".join(_target_line(t, expansion) for t in targets) + "\n\n"
    head += "Document OCR text (may contain errors"
    head += "; page images are attached, trust them where they disagree):" if with_images else "):"
    return f"{head}\n\n{text}"
