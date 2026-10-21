"""Deterministic post-processing: dates, due-date join, derivation, validation, per-target roll-up.

Which requested vaccine name a row belongs to is decided by the LLM (`matches`); nothing here knows vaccine names.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
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


def _build_pet(rp: RawPet, order: str, today: date, targets: list[str], lines: list[str], seen: set):
    canon = {t.strip().lower(): t for t in targets}
    evs: list[_Ev] = []
    diags: dict = {}
    for e in rp.events:
        name = (e.raw_name or "").strip()
        kind = (e.kind or "vaccine").strip().lower()
        if not name or kind == "other":
            continue
        g, r1 = _pdate(e.given_date, order)
        d, r2 = _pdate(e.due_date, order)
        reasons = r1 + r2
        evidence = _clean(e.evidence) or _find_line(name, lines)
        if g and g > today + timedelta(days=1):
            reasons.append("given date is in the future")
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
    return _finalize(rp, evs, list(diags.values()), targets)


def _merge_dups(evs: list[_Ev]) -> list[_Ev]:
    out: dict = {}
    for e in evs:
        key = (e.target, e.given, None if e.given else e.due, e.raw_name.lower() if e.target is None else "")
        k = out.get(key)
        if k is None:
            out[key] = e
            continue
        if k.due is None and e.due:
            k.due, k.due_src = e.due, e.due_src
        elif e.due and k.due and e.due != k.due:
            k.reasons.append("conflicting due dates for the same event")
        if e.raw_name != k.raw_name and e.raw_name not in k.also:
            k.also.append(e.raw_name)
        k.partial = k.partial and e.partial
        k.months = k.months or e.months
        k.evidence = k.evidence or e.evidence
    return list(out.values())


def _join_due(evs: list[_Ev]) -> list[_Ev]:
    """Attach due-only (reminder) rows to the latest administered event of the same requested name."""
    by_target: dict = defaultdict(list)
    for e in evs:
        by_target[e.target].append(e)
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
        for d in due_only:
            if latest and latest.due is None and d.due > latest.given:
                latest.due, latest.due_src = d.due, "document"
                latest.evidence = " || ".join(x for x in (latest.evidence, d.evidence) if x)[:400] or None
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


def _rollup(target: str, evs: list[_Ev]) -> Vaccination:
    """Latest record for one requested name: the latest administered events, else the latest reminder."""
    mine = [e for e in evs if e.target == target]
    admin = [e for e in mine if e.given]
    if admin:
        g = max(e.given for e in admin)
        chosen = [e for e in admin if e.given == g]
    else:
        due = [e for e in mine if e.due]
        chosen = [e for e in due if e.due == max(x.due for x in due)] if due else mine
    if not chosen:
        return Vaccination(name=target)
    names = []
    for e in chosen:
        names += [n for n in [e.raw_name, *e.also] if n not in names]
    dues = {e.due for e in chosen if e.due}
    head = max(chosen, key=lambda e: (e.due is not None, e.due_src == "document"))
    reasons = sorted({r for e in chosen for r in e.reasons})
    if len(dues) > 1:
        reasons.append("conflicting due dates")
    return Vaccination(
        name=target,
        partialMatch=all(e.partial for e in chosen),
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


def _finalize(rp: RawPet, evs: list[_Ev], diags: list[DiagnosticTest], targets: list[str]) -> Pet:
    evs = _join_due(_merge_dups(evs))
    _derive_and_validate(evs)
    latest: dict = {}
    for e in evs:
        if e.target and e.given and (e.target not in latest or e.given > latest[e.target]):
            latest[e.target] = e.given

    def sort_key(e: _Ev):
        ti = targets.index(e.target) if e.target in targets else len(targets)
        return (ti, -(e.given.toordinal() if e.given else 0), e.raw_name)

    events = [
        VaccinationEvent(
            rawName=e.raw_name,
            target=e.target,
            partial=e.partial,
            dateAdministered=_iso(e.given),
            dueDate=_iso(e.due),
            dueDateSource=e.due_src if e.due else None,
            isLatest=bool(e.target and e.given and latest.get(e.target) == e.given),
            needsHumanReview=bool(e.reasons),
            reviewReasons=e.reasons,
            evidence=e.evidence,
        )
        for e in sorted(evs, key=sort_key)
    ]
    return Pet(
        name=_clean(rp.name), species=_clean(rp.species), breed=_clean(rp.breed), sex=_clean(rp.sex),
        weight=_clean(rp.weight), color=_clean(rp.color), microchipNumber=_clean(rp.microchipNumber),
        vaccinations=[_rollup(t, evs) for t in targets], events=events, diagnosticTests=diags,
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
