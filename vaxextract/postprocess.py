"""Deterministic post-processing: dates, due-date join, derivation, validation, per-target roll-up.

Which requested vaccine name a row belongs to is decided by the LLM (`matches`); nothing here knows vaccine names.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Optional

from . import dates as D
from .schema import (Data, DiagnosticTest, MatchedComponent, Pet, RawExtraction, RawPet, Vaccination,
                     VaccinationEvent)

MAX_INTERVAL_DAYS = 366 * 6


@dataclass
class _Ev:
    raw_name: str
    target: Optional[str]
    partial: bool
    given: Optional[date]
    due: Optional[date]
    evidence: Optional[str]
    months: Optional[int] = None
    due_src: Optional[str] = None
    reasons: list = field(default_factory=list)
    also: list = field(default_factory=list)  # other raw names merged into this event


def _pdate(s: Optional[str], order: str):
    if not s or not s.strip():
        return None, []
    d = D.parse_date(s, order)
    return (d, []) if d else (None, [f"unparseable date '{s}'"])


def _clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s or None


def _find_line(name: str, lines: list[str]) -> Optional[str]:
    """Evidence = the first OCR line that prints the row's name (the LLM no longer echoes it, to save tokens)."""
    n = name.lower()
    return next((ln.strip()[:160] for ln in lines if n in ln.lower()), None)


def _months(v: Optional[int]) -> Optional[int]:
    return v if v and 0 < v <= 12 * 6 else None


def _due_list(rp: RawPet, order: str, today: date) -> bool:
    """Every row that prints one date prints a future one: a 'vaccinations done' list of next-due dates
    (Cobblestone certificates: '02-11-27 | Kennel Cough-oral'), not misreads - a misread is one stray future date
    among past ones. Rows with both dates (the certificate's own 'Date of Rabies Vaccination / Next ...') say
    nothing either way."""
    rows = [e for e in rp.events if (e.kind or "vaccine").strip().lower() != "other"
            and not _pdate(e.due_date, order)[0]]
    dated = [g for g in (_pdate(e.given_date, order)[0] for e in rows) if g]
    return len(dated) >= 3 and all(today + timedelta(days=1) < g <= today + timedelta(days=MAX_INTERVAL_DAYS)
                                   for g in dated)


def _build_pet(rp: RawPet, order: str, today: date, targets: list[str], lines: list[str], seen: set):
    canon = {t.strip().lower(): t for t in targets}
    evs: list[_Ev] = []
    diags: dict = {}
    due_list = _due_list(rp, order, today)
    for e in rp.events:
        name = re.sub(r"\s+", " ", e.raw_name or "").strip()  # a name wrapped over two lines is one name
        kind = (e.kind or "vaccine").strip().lower()
        if not name or kind == "other":
            continue
        g, r1 = _pdate(e.given_date, order)
        d, r2 = _pdate(e.due_date, order)
        reasons = r1 + r2
        if not any(c.isalpha() for c in name):
            reasons.append("name is not a vaccine name (looks like a date or number)")
        evidence = _clean(e.evidence) or _find_line(name, lines)
        if g and g > today + timedelta(days=1) and due_list:
            d, g = g, None
        elif g and g > today + timedelta(days=1):
            # often a next-due date printed in a 'vaccinations done' list, but just as often an OCR misread
            # ('10-19-26' read as '10-19-28'): never used as a date, only shown to the reviewer
            reasons.append(f"given date {g.isoformat()} is in the future (a due date or a misread?)")
            g = None
        if kind == "diagnostic":
            key = (name.lower(), g, d)
            if key not in diags:
                rs = list(reasons)
                if g is None and d is None:
                    rs.append("no dates")
                diags[key] = DiagnosticTest(
                    name=name,
                    rawName=name,
                    dateAdministered=g.isoformat() if g else None,
                    dueDate=d.isoformat() if d else None,
                    needsHumanReview=bool(rs),
                    reviewReasons=rs,
                    evidence=evidence,
                )
            continue
        matches, unknown = {}, []
        for m in e.matches:
            t = canon.get((m.name or "").strip().lower())
            if t is None:
                if m.name:
                    unknown.append(m.name)
            elif t not in matches:
                matches[t] = m
        if unknown:
            reasons.append("matched names not in the request: " + ", ".join(unknown))
        if not matches:
            evs.append(_Ev(name, None, False, g, d, evidence, _months(e.duration_months),
                           "document" if d else None, reasons))
            continue
        for t, m in matches.items():
            evs.append(_Ev(name, t, m.partial, g, d, evidence,
                           _months(m.duration_months) or _months(e.duration_months),
                           "document" if d else None, list(reasons)))
            if g:
                seen.add((t, g))
            if d:
                seen.add((t, d))
    return _finalize(rp, evs, _join_due_tests(list(diags.values())), targets, today)


def _join_due_tests(tests: list[DiagnosticTest]) -> list[DiagnosticTest]:
    """A reminder-only test row gives its due date to the latest performed test of the same name that has none."""
    out: list[DiagnosticTest] = []
    for t in tests:
        if t.dateAdministered is None and t.dueDate:
            done = [x for x in out if x.rawName.lower() == t.rawName.lower() and x.dateAdministered]
            last = max(done, key=lambda x: x.dateAdministered, default=None)
            if last and last.dueDate is None and t.dueDate > last.dateAdministered:
                last.dueDate = t.dueDate
                last.evidence = " || ".join(dict.fromkeys(x for x in (last.evidence, t.evidence) if x))[:400] or None
                continue
            if last and last.dueDate == t.dueDate:
                continue
        out.append(t)
    return out


def _merge_dups(evs: list[_Ev]) -> list[_Ev]:
    out: dict = {}
    for e in evs:
        undated = e.given is None and e.due is None  # never merge rows just because both lack dates
        key = (e.target, e.given, None if e.given else e.due,
               e.raw_name.lower() if e.target is None or undated else "")
        k = out.get(key)
        if k is None:
            out[key] = e
            continue
        if k.due is None and e.due:
            k.due, k.due_src = e.due, e.due_src
        elif e.due and k.due and e.due < k.due:
            # several products for this name on one day, or one combination printed once per part
            # ('DHPP 3 YR + LEPTO BOOSTER' due 3/6/2027 and 3/6/2029): the earliest expiry wins
            k.due, k.due_src = e.due, e.due_src
        if e.raw_name != k.raw_name and e.raw_name not in k.also:
            k.also.append(e.raw_name)
        k.partial = k.partial and e.partial
        k.months = k.months or e.months
        k.evidence = k.evidence or e.evidence
    return list(out.values())


def _join_due(evs: list[_Ev], key=lambda e: e.target) -> list[_Ev]:
    """Attach due-only (reminder) rows to the latest administered event with the same key (default: requested name)."""
    by_target: dict = defaultdict(list)
    for e in evs:
        by_target[key(e)].append(e)
    out: list[_Ev] = []
    for target, items in by_target.items():
        if target is None:
            out += items
            continue
        admin = sorted([e for e in items if e.given], key=lambda e: e.given)
        due_only = sorted([e for e in items if not e.given and e.due], key=lambda e: e.due)
        neither = [e for e in items if not e.given and not e.due]
        latest = admin[-1] if admin else None
        remaining = []
        # a reminder far from the dose's own printed interval is another product sharing the name (Lepto vs DHPP)
        implied = D.add_months(latest.given, latest.months) if latest and latest.months else None
        for d in due_only:
            fits = implied is None or abs((d.due - implied).days) <= 31
            if latest and latest.due is None and d.due > latest.given and fits:
                latest.due, latest.due_src = d.due, "document"
                latest.evidence = " || ".join(dict.fromkeys(x for x in (latest.evidence, d.evidence) if x))[:400] or None
            elif latest and latest.due == d.due:
                continue
            else:
                remaining.append(d)
        out += admin + remaining + neither
    return out


def _derive_and_validate(evs: list[_Ev]) -> None:
    for e in evs:
        if e.given and e.due is None and e.months:
            e.due, e.due_src = D.add_months(e.given, e.months), "derived"
        if e.given is None and e.due is None:
            e.reasons.append("no dates")
        if e.given and e.due:
            if e.due < e.given:
                e.reasons.append("due date before given date")
            elif (e.due - e.given).days > MAX_INTERVAL_DAYS:
                e.reasons.append("due date more than 6 years after given date")


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _current(mine: list[_Ev], today: date) -> list[_Ev]:
    """The rows still in force for one requested name: an expired row is history once another row for the name is
    still valid (last year's annual shot next to this year's 3-year one; 'Nobivac 4/Third' due 2021 next to
    'Nobivac 4' due 2027). Valid rows all stay - without knowing the components, a later dose may be another part of
    the combination (Lepto given after DHPP), not a replacement. Rows printing no expiry stay if they are the latest dose."""
    valid = any(s.due and s.due >= today for s in mine)
    last = max((s.given for s in mine if s.given), default=None)
    return [r for r in mine if (r.due and (r.due >= today or not valid))
            or (not r.due and (r.given is None or r.given == last))]


def _latest(rows: list[_Ev]) -> list[_Ev]:
    """The most recent record among rows of the same coverage: the latest dose(s), else the latest reminder."""
    admin = [e for e in rows if e.given]
    if admin:
        return [e for e in admin if e.given == max(x.given for x in admin)]
    due = [e for e in rows if e.due]
    return [e for e in due if e.due == max(x.due for x in due)] if due else rows


def _rollup(target: str, evs: list[_Ev], today: date) -> Vaccination:
    """One requested name: the latest record of the rows that cover it fully, and next to it every part of a
    combination still in force; the one that expires first wins (a combination is only as current as its
    first-expiring part)."""
    mine = [e for e in evs if e.target == target]
    if not mine:
        return Vaccination(name=target)
    cur = _current(mine, today)
    full = _latest([e for e in cur if not e.partial])
    chosen = full + [e for e in cur if e.partial] if full else cur
    dated = [e for e in chosen if e.due]
    head = min(dated, key=lambda e: (e.due, e.due_src != "document")) if dated else _latest(chosen)[0]
    chosen = [head] + [e for e in chosen if e is not head]
    names = []
    for e in chosen:
        names += [n for n in [e.raw_name, *e.also] if n not in names]
    reasons = sorted({r for e in chosen for r in e.reasons})
    # every row in force covers only part of the name and they are all one product: the rest is not on the document
    if all(e.partial for e in cur) and len({n.lower() for e in cur for n in [e.raw_name, *e.also]}) == 1:
        reasons.append(f"combination not complete: only '{cur[0].raw_name}' is printed for {target}")
    return Vaccination(
        name=target,
        partialMatch=all(e.partial for e in cur),
        matchedComponents=[MatchedComponent(extractedName=n, dateAdministered=_iso(e.given), expiryDate=_iso(e.due),
                                            expiryDateSource=e.due_src if e.due else None, evidence=e.evidence)
                           for e in chosen for n in [e.raw_name, *e.also]],
        vaccine_name=" | ".join(names),
        dateAdministered=_iso(head.given),
        expirationDate=_iso(head.due),
        expirationDateSource=head.due_src if head.due else None,
        needsHumanReview=bool(reasons),
        reviewReasons=reasons,
    )


def _event(e: _Ev, is_latest: bool = False) -> VaccinationEvent:
    return VaccinationEvent(rawName=e.raw_name, target=e.target, partial=e.partial, dateAdministered=_iso(e.given),
                            dueDate=_iso(e.due), dueDateSource=e.due_src if e.due else None, isLatest=is_latest,
                            needsHumanReview=bool(e.reasons), reviewReasons=e.reasons, evidence=e.evidence)


def _finalize(rp: RawPet, evs: list[_Ev], diags: list[DiagnosticTest], targets: list[str], today: date) -> Pet:
    # rows as printed: no merging, and a reminder only joins a dose printed under the same name
    printed = _join_due([replace(e, reasons=list(e.reasons)) for e in evs], key=lambda e: e.raw_name.lower())
    _derive_and_validate(printed)
    evs = _join_due(_merge_dups(evs))
    _derive_and_validate(evs)
    latest: dict = {}
    for e in evs:
        if e.target and e.given and (e.target not in latest or e.given > latest[e.target]):
            latest[e.target] = e.given

    def sort_key(e: _Ev):
        ti = targets.index(e.target) if e.target in targets else len(targets)
        return (ti, -(e.given.toordinal() if e.given else 0), e.raw_name)

    events = [_event(e, bool(e.target and e.given and latest.get(e.target) == e.given)) for e in sorted(evs, key=sort_key)]
    return Pet(
        name=_clean(rp.name), species=_clean(rp.species), breed=_clean(rp.breed), sex=_clean(rp.sex),
        weight=_clean(rp.weight), color=_clean(rp.color), microchipNumber=_clean(rp.microchipNumber),
        vaccinations=[_rollup(t, evs, today) for t in targets], events=events, diagnosticTests=diags,
        printedRows=[_event(e) for e in printed],
    )


def _terms(target: str) -> list[re.Pattern]:
    parts = [p.strip() for p in re.split(r"[/,|+&]", target)]
    return [re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)", re.I) for p in parts if len(p) >= 3]


def _coverage_warnings(text: str, seen: set, order: str, targets: list[str]) -> list[str]:
    """Dated lines that print a requested name but have no extracted event with that date -> likely a dropped row."""
    terms = {t: _terms(t) for t in targets}
    warns = []
    for line in text.splitlines():
        hits = [t for t, rxs in terms.items() if any(rx.search(line) for rx in rxs)]
        if not hits:
            continue
        ds = D.dates_in_text(line, order)
        if ds and any(not any((t, d) in seen for d in ds) for t in hits):
            warns.append(line.strip()[:80])
    return warns


def build_data(raw: RawExtraction, text: str, today: Optional[date] = None, targets: Optional[list[str]] = None):
    """RawExtraction + OCR text + requested vaccine names -> (Data, review_reasons)."""
    today = today or date.today()
    targets = list(dict.fromkeys(t.strip() for t in targets or [] if t and t.strip()))
    strings = [s for p in raw.pets for e in p.events for s in (e.given_date, e.due_date)]
    order = D.infer_order(strings)
    seen: set = set()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pets = [_build_pet(p, order, today, targets, lines, seen) for p in raw.pets]
    reasons: list[str] = []
    if not pets:
        reasons.append("no pet found")
    owner = (raw.owner.name or "").strip().lower() if raw.owner else ""
    vet = (raw.veterinarian.name or "").strip().lower() if raw.veterinarian else ""
    for p in pets:
        n = (p.name or "").strip().lower()
        if not n:
            reasons.append("pet name missing")
        elif n in (owner, vet):
            reasons.append("pet name equals owner/veterinarian name")
        if not p.events and not p.diagnosticTests:
            reasons.append("no vaccination events found")
        flagged = sum(v.needsHumanReview for v in p.events)
        if flagged:
            reasons.append(f"{flagged} vaccination event(s) flagged")
    missed = _coverage_warnings(text, seen, order, targets)
    if missed:
        reasons.append(f"possible missed events ({len(missed)}): " + " ; ".join(missed[:3]))
    data = Data(pets=pets, owner=raw.owner, clinic=raw.clinic, veterinarian=raw.veterinarian)
    return data, reasons
