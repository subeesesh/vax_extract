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


_INTERVAL = re.compile(r"(?<![\w.])(\d{1,2})\s*-?\s*(?:(yrs?|years?|y)|(mos?|months?))(?![a-z])", re.I)


def _single(months: set[int]) -> Optional[int]:
    """The one interval a name prints. Used for a matched item only when it matches a single requested name: in
    'DA2/PV/Lepto/Rabies(1yr)' the interval is the Rabies part's, not DHPP's or Lepto's."""
    return next(iter(months)) if len(months) == 1 else None


def _printed_months(name: str) -> set[int]:
    """Validity intervals printed in an item's name ('DAPP 3 year' -> {36}, 'DA2P-Pv 3yr 4L 1yr' -> {36, 12})."""
    found = {int(n) * 12 if y else int(n) for n, y, _ in _INTERVAL.findall(name)}
    return {m for m in found if _months(m)}


def _llm_months(e, name: str, value: Optional[int]) -> Optional[int]:
    """The model's interval, kept only when the name prints it: as a number and unit ('3yr', '12 mo') or as the words
    the model quotes in interval_text ('Annual'). A number read out of a code or dose number ('DA2PPvL-1') is dropped,
    so no expiry is derived from it and the empty expiry goes to review instead."""
    v = _months(value)
    if not v:
        return None
    if v in _printed_months(name):
        return v
    quoted = (e.interval_text or "").strip()
    printed = re.sub(r"\s+", "", quoted).lower() in re.sub(r"\s+", "", name).lower()
    if not (quoted and printed and _UNIT.search(quoted)):
        return None
    stated = _printed_months(quoted)  # a quoted '3 year' must mean 36: Kairo's '(3 year)' quoted for a 12
    return v if not stated or v in stated else None


_UNIT = re.compile(r"(annual|yearly|years?|yrs?|months?|mos?|weeks?|wks?)\b", re.I)  # time units, as in _INTERVAL


def _item_months(name: str, llm_months: Optional[int]) -> Optional[int]:
    """The validity interval of a whole printed item (no requested name to split it by): the interval printed in its
    name wins over the model's (which it sometimes leaves out); a combination printing different intervals for its
    parts ('DA2P-Pv 3yr 4L 1yr') is current only until its first part runs out, so the shortest. A matched part keeps
    its own interval instead."""
    printed = _printed_months(name)
    return min(printed) if printed else _months(llm_months)


_NEGATED = re.compile(r"\b(?:no(?!\.)|non|not|without|excl\w*|minus|w/o)\W*$", re.I)


def _shown(name: str, quote: Optional[str]) -> bool:
    """A match counts only if the words it quotes as showing the disease are printed in the row's name, and not
    after 'no' / 'without' ('DHPPV (NO LEPTO)'). A one- or two-letter quote is an abbreviation letter: it must be
    printed in that case ('L' of 'DHLPP', not the l of 'Annual'). Spaces and punctuation are ignored. A list of
    quotes ('D, A, P' of 'DHLPP') counts if every one of them is shown."""
    pieces = [p for p in re.split(r"\s*(?:[,;&+]|\band\b)\s*", quote or "") if p.strip()]
    if len(pieces) == 1 and all(len(w) <= 2 for w in pieces[0].split()):  # 'D P': letters apart
        pieces = pieces[0].split()
    if len(pieces) > 1:
        return all(_shown(name, p) for p in pieces)
    q = re.sub(r"[\W_]+", "", quote or "")
    if not q:
        return False
    flat, at = "", []  # name without spaces/punctuation, and where each kept character came from
    for i, c in enumerate(name):
        if c.isalnum():
            flat += c
            at.append(i)
    short = len(q) <= 2
    hay, needle = (flat, q) if short else (flat.lower(), q.lower())
    for k in [k for k in range(len(hay)) if hay.startswith(needle, k)]:
        if _NEGATED.search(name[:at[k]]):
            continue
        if short and not _abbreviation(name, at[k]):
            continue
        return True
    return False


def _abbreviation(name: str, i: int) -> bool:
    """The word around name[i] reads as an abbreviation: two or more capitals ('DHLPP', 'DAPPv', 'DA2PP'), or all
    lower case with at most one vowel ('dhlpp', 'da2pp') - not an ordinary word ('Annual')."""
    word = re.findall(r"\w+", name[:i][::-1][:40])  # characters of the word before i, reversed
    word = (word[0][::-1] if word and name[:i] and name[i - 1].isalnum() else "") + re.match(r"\w*", name[i:]).group()
    if _INTERVAL.fullmatch(word):  # '3YR' is an interval, not an abbreviation
        return False
    letters = [c for c in word if c.isalpha()]
    if sum(c.isupper() for c in letters) >= 2 or letters == [name[i]] and name[i].isupper():  # or 'L' of 'DA2PP+L'
        return True
    return all(c.islower() for c in letters) and sum(c in "aeiou" for c in letters) <= 1


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


# a row whose name says the dose was not given ('DECLINED: 1st Canine influenza'): status words, not vaccine names
_NOT_GIVEN = re.compile(r"\b(declined|refused|not given)\b", re.I)


def _words(disease: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", disease.lower()))


def _same_disease(a: str, b: str) -> bool:
    """'Canine Distemper' = 'Distemper' (one name's words inside the other's); 'Canine Influenza' != 'Parainfluenza'."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and (wa <= wb or wb <= wa)


_SPECIES = {"dog": "dog", "dogs": "dog", "canine": "dog", "k9": "dog", "cat": "cat", "cats": "cat", "feline": "cat"}


def _species(s: Optional[str]) -> set:
    return {_SPECIES[w] for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w in _SPECIES}


def _diseases(expansion: dict, target: str, species: set) -> list[list[str]]:
    """The disease lists a requested name stands for, for the pet's species (all of them if the species is unknown)."""
    out = []
    for alt in expansion.get(target) or []:
        sp = _species(alt.get("species"))
        if not sp or not species or sp & species:
            out.append([d for d in alt.get("diseases") or [] if d])
    return [d for d in out if d]


def _parts_of(expansion: dict, targets: list[str], species: set) -> dict:
    """For each requested name B, the requested combination names A that contain every disease of B (Leptospirosis
    -> DHLPP / FVRCP for a dog): a row of B is also a part of A. From the name definitions the LLM wrote for this
    request; nothing here knows vaccine names. {} without definitions."""
    out: dict = defaultdict(list)
    for b in targets:
        for a in targets:
            if a == b:
                continue
            for db in _diseases(expansion, b, species):
                if any(len(da) > len(db) and all(any(_same_disease(x, y) for y in da) for x in db)
                       for da in _diseases(expansion, a, species)):
                    out[b].append(a)
                    break
    return out


def _build_pet(rp: RawPet, order: str, today: date, targets: list[str], lines: list[str], seen: set,
               expansion: Optional[dict] = None):
    canon = {t.strip().lower(): t for t in targets}
    parts_of = _parts_of(expansion or {}, targets, _species(rp.species))
    evs: list[_Ev] = []
    diags: dict = {}
    due_list = _due_list(rp, order, today)
    for e in rp.events:
        name = re.sub(r"\s+", " ", e.raw_name or "").strip()  # a name wrapped over two lines is one name
        kind = (e.kind or "vaccine").strip().lower()
        if not name or kind == "other" or _NOT_GIVEN.search(name):
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
            if not _shown(name, m.shown_by):  # nothing printed in the name shows that disease: not a match
                continue
            t = canon.get((m.name or "").strip().lower())
            if t is None:
                if m.name:
                    unknown.append(m.name)
            elif t not in matches:
                matches[t] = m
        for b, m in list(matches.items()):  # a Lepto row is also a part of the requested DHLPP
            for a in parts_of.get(b, []):
                if a not in matches:
                    matches[a] = m.model_copy(update={"partial": True})
        if unknown:
            reasons.append("matched names not in the request: " + ", ".join(unknown))
        if not matches:
            evs.append(_Ev(name, None, False, g, d, evidence, _item_months(name, _llm_months(e, name, e.duration_months)),
                           "document" if d else None, reasons))
            continue
        for t, m in matches.items():
            evs.append(_Ev(name, t, m.partial, g, d, evidence,
                           _llm_months(e, name, m.duration_months) or _llm_months(e, name, e.duration_months)
                           or (_single(_printed_months(name)) if len(matches) == 1 else None),
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
    """The same dose printed twice (invoice line + certificate row) is one row. Rows of one day that match different
    requested names and print different expiries are different parts ('Bordetella / Parainfluenza' due 2024 and
    'DHPP 3yr' due 2026, both given 05-23-23): they stay apart, so the combination takes its first-expiring part in
    force and an expired part is seen."""
    out: dict = {}
    for e in evs:
        undated = e.given is None and e.due is None  # never merge rows just because both lack dates
        key = (e.target, e.given, None if e.given else e.due,
               e.raw_name.lower() if e.target is None or undated else "")
        k = out.get(key)
        if k is not None and e.due and k.due and e.due != k.due and _cover(e, evs) != _cover(k, evs):
            key = key + (len(out),)  # another part given the same day
            k = None
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


def _current(mine: list[_Ev], today: date, evs: list[_Ev]) -> list[_Ev]:
    """The rows still in force for one requested name: an expired row is history once another row for the name is
    still valid (last year's annual shot next to this year's 3-year one; 'Nobivac 4/Third' due 2021 next to
    'Nobivac 4' due 2027). Valid rows all stay - without knowing the components, a later dose may be another part of
    the combination (Lepto given after DHPP), not a replacement. Rows printing no expiry stay if they are the latest dose."""
    valid = any(s.due and s.due >= today for s in mine)
    if not valid:
        # all expired: the latest dose of each part is the record, not the oldest due date (Sasuke's 2021 shot);
        # rows matching the same requested names are one part ('DHPP 3yr' and 'DHPP Annual'; 'Lepto Annual' and 'Lepto 1st')
        # A part counts only if it was still in force when the newest dose was given (Zelda: DHPP 3yr due 2026-05-22
        # next to the 2025 Lepto); one that had lapsed by then was renewed by it (Sasuke: 2023 DA2PP, then DA2PP+L)
        newest = _latest(mine)
        last = max((e.given for e in newest if e.given), default=None)
        if last is None:
            return newest
        parts: dict = defaultdict(list)
        for e in mine:
            parts[_cover(e, evs)].append(e)
        out = list(newest)
        for rows in parts.values():
            out += [e for e in _latest(rows) if e not in out and e.due and e.due > last]
        return out
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


def _cover(e: _Ev, evs: list[_Ev]) -> frozenset:
    """Every requested name the row's printed line matches."""
    return frozenset(x.target for x in evs if x.target and x.raw_name.lower() == e.raw_name.lower())


def _dose_or_part(cur: list[_Ev], evs: list[_Ev]) -> list[str]:
    """Two valid partial rows matching the same requested names, the older one due first (Coco: 'Canine Da2PP First'
    due 2029-03-25, 'Canine DaPP Third' due 2029-06-16). A later dose of the same product means the later date; another
    part of the combination (Lepto given after DHPP) means the earlier one. Without the components the two can't be told
    apart, so the earliest stays (rule 1) and a person decides. When the older row is due later, both readings agree."""
    rows = [e for e in cur if e.partial and e.due and e.given]
    out = []
    for a in rows:
        for b in rows:
            if a.given < b.given and a.due < b.due and _cover(a, evs) == _cover(b, evs):
                out.append(f"older dose or another part? '{a.raw_name}' given {a.given.isoformat()} due "
                           f"{a.due.isoformat()}, '{b.raw_name}' given {b.given.isoformat()} due {b.due.isoformat()}")
    return out


def _expired_parts(mine: list[_Ev], cur: list[_Ev], evs: list[_Ev], today: date) -> list[str]:
    """Expired parts of a combination that `_current` dropped while another part is still valid (Lepto due
    2026-10-09 next to DA2PP due 2027): the roll-up then shows the valid part's date, which is right by the rules,
    but the expired part needs a person. A row's coverage = every requested name its printed line matches; an
    expired row with the same coverage as a row still in force is just an older dose and stays history."""
    cover = lambda e: _cover(e, evs)  # noqa: E731

    def redosed(e: _Ev) -> bool:  # a later dose of the same printed name exists anywhere on the document (Onyx)
        return any(x.raw_name.lower() == e.raw_name.lower() and x.given and e.given and x.given > e.given for x in evs)

    in_force = {cover(c) for c in cur}
    out = []
    for e in mine:
        if (e.partial and e.due and e.due < today and not any(e is c for c in cur)
                and not redosed(e) and cover(e) not in in_force):
            r = f"part expired: '{e.raw_name}' due {e.due.isoformat()}"
            if r not in out:
                out.append(r)
    return out


def _rollup(target: str, evs: list[_Ev], today: date) -> Vaccination:
    """One requested name: the latest record of the rows that cover it fully, and next to it every part of a
    combination still in force; the one that expires first wins (a combination is only as current as its
    first-expiring part)."""
    mine = [e for e in evs if e.target == target]
    if not mine:
        return Vaccination(name=target)
    cur = _current(mine, today, evs)
    full = _latest([e for e in cur if not e.partial])
    chosen = full + [e for e in cur if e.partial] if full else cur
    dated = [e for e in chosen if e.due]
    head = min(dated, key=lambda e: (e.due, e.due_src != "document")) if dated else _latest(chosen)[0]
    extra = []
    if head.due is None and head.given:
        # the latest row prints no due date (Kiya: a 2024 rabies tag row after the 2023 3-year vaccine): show the
        # latest row that has one, and let a person check
        older = [e for e in mine if e.due and e.given and e.given < head.given]
        if older:
            prev = max(older, key=lambda e: (e.given, e.due))
            extra.append(f"latest dose prints no due date: '{head.raw_name}' given {head.given.isoformat()}")
            chosen = [prev] + chosen
            head = prev
    chosen = [head] + [e for e in chosen if e is not head]
    names = []
    for e in chosen:
        names += [n for n in [e.raw_name, *e.also] if n not in names]
    reasons = sorted({r for e in chosen for r in e.reasons})
    # every row in force covers only part of the name and they are all one product: the rest is not on the document.
    # When every dose has expired only the latest is in force; judge the parts on all rows, as the history shows them
    valid = any(e.due and e.due >= today for e in mine)
    basis = cur if valid else mine
    if all(e.partial for e in basis) and len({n.lower() for e in basis for n in [e.raw_name, *e.also]}) == 1:
        reasons.append(f"combination not complete: only '{basis[0].raw_name}' is printed for {target}")
    if head.due is None:  # printed rows match the name but none gives or implies an expiry: a person has to look
        reasons.append(f"no expiry printed for {target}")
    reasons += extra + _dose_or_part(cur, evs) + (_expired_parts(mine, cur, evs, today) if valid else [])
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


def _same(field: str, a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    if field == "phone":
        da, db = re.sub(r"\D", "", a)[-10:], re.sub(r"\D", "", b)[-10:]
        return len(da) >= 7 and da == db
    return re.sub(r"\W", "", a).lower() == re.sub(r"\W", "", b).lower()


def _not_the_clinic(owner, clinic):
    """The model sometimes fills the owner's empty phone/email/address with the clinic's (Leo: owner phone and email
    = Elgin Animal Hospital's). A contact detail identical to the clinic's is the clinic's: drop it from the owner."""
    if not owner or not clinic:
        return owner
    copied = {f for f in ("address", "phone", "email") if _same(f, getattr(owner, f), getattr(clinic, f))}
    return owner.model_copy(update={f: None for f in copied}) if copied else owner


def _merge_surname_pets(pets: list[RawPet]) -> list[RawPet]:
    """One animal listed twice, once with the owner's surname attached ('Ollie' on the invoice lines, 'Ollie
    McAuliffe's Reminders' below): merge them under the bare name, and drop that surname from the other pets that
    carry it ('Mila McAuliffe' -> 'Mila')."""
    names = [(p.name or "").strip() for p in pets]
    surnames = set()
    for a in names:
        for b in names:
            extra = b[len(a):].strip() if a and b.lower().startswith(a.lower() + " ") else ""
            if re.fullmatch(r"[A-Z][A-Za-z'\-]{2,}", extra):
                surnames.add(extra.lower())
    if not surnames:
        return pets
    out: dict = {}
    for p in pets:
        name = (p.name or "").strip()
        words = name.split()
        if len(words) > 1 and words[-1].lower() in surnames:
            name = " ".join(words[:-1])
        key = name.lower()
        if key not in out:
            out[key] = p.model_copy(update={"name": name, "events": list(p.events)})
            continue
        k = out[key]  # same animal: keep every event, fill fields the first entry lacks
        k.events += p.events
        for f in ("species", "breed", "sex", "weight", "color", "microchipNumber"):
            if not getattr(k, f) and getattr(p, f):
                setattr(k, f, getattr(p, f))
    return list(out.values())


_NAMED_PET = [re.compile(p) for p in (
    r"Reminders\s+for:?\s+([A-Z][A-Za-z\-]+)",                         # 'Reminders for: Lillie (Weight...'
    r"\b([A-Z][A-Za-z\-]+)(?:\s+[A-Z][A-Za-z\-]+)?'s\s+(?:Expired\s+)?Reminders",  # "Mila McAuliffe's Reminders"
    r"(?i:\bpatient)\s*:\s*([A-Z][A-Za-z\-]+)",                          # 'PATIENT: Zoey', 'Patient: Archie'
    r"(?i:\bpatient chart for)\s+([A-Z][A-Za-z\-]+)",                     # 'Patient Chart for Lillie'
)]
MISSING_PET = "pet named in the document but not extracted"


def _missing_pets(text: str, pets: list[Pet]) -> list[str]:
    """Pet names the document prints in a patient / reminder heading that no extracted pet carries: the model
    sometimes leaves a whole animal out of a multi-pet document (Mila on the McAuliffe invoice, about half the runs)."""
    have = {w.lower() for p in pets for w in re.findall(r"[A-Za-z\-]+", p.name or "")}
    found = []
    for rx in _NAMED_PET:
        for m in rx.finditer(text):
            n = m.group(1)
            if len(n) > 1 and n.lower() not in have and n not in found:
                found.append(n)
    return found


def build_data(raw: RawExtraction, text: str, today: Optional[date] = None, targets: Optional[list[str]] = None,
               expansion: Optional[dict] = None):
    """RawExtraction + OCR text + requested vaccine names (+ the LLM's definitions of them) -> (Data, review_reasons)."""
    today = today or date.today()
    targets = list(dict.fromkeys(t.strip() for t in targets or [] if t and t.strip()))
    strings = [s for p in raw.pets for e in p.events for s in (e.given_date, e.due_date)]
    order = D.infer_order(strings)
    seen: set = set()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pets = [_build_pet(p, order, today, targets, lines, seen, expansion) for p in _merge_surname_pets(raw.pets)]
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
    gone = _missing_pets(text, pets) if pets else []
    if gone:
        reasons.append(f"{MISSING_PET}: " + ", ".join(gone))
    missed = _coverage_warnings(text, seen, order, targets)
    if missed:
        reasons.append(f"possible missed events ({len(missed)}): " + " ; ".join(missed[:3]))
    data = Data(pets=pets, owner=_not_the_clinic(raw.owner, raw.clinic), clinic=raw.clinic,
                veterinarian=raw.veterinarian)
    return data, reasons
