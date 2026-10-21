"""Date parsing with per-document day/month order inference."""
from __future__ import annotations

import calendar
import os
import re
from datetime import date
from typing import Iterable, Optional

_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9

NUM_DATE = re.compile(r"(?<![\d/.\-])(\d{1,4})[-/.](\d{1,2})[-/.](\d{2,4})(?!\d|[/.\-]\d)")
MON_FIRST = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
DAY_FIRST = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")


def default_order() -> str:
    v = os.environ.get("VAX_DEFAULT_DATE_ORDER", "MDY").upper()
    return v if v in ("MDY", "DMY") else "MDY"


def _year(c: str) -> Optional[int]:
    if len(c) == 4:
        return int(c)
    if len(c) == 2:
        v = int(c)
        return 2000 + v if v <= 69 else 1900 + v
    return None


def parse_date(s: Optional[str], order: str = "MDY") -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    m = NUM_DATE.search(s)
    if m:
        a, b, c = m.groups()
        if len(a) == 4:
            cands = [(int(a), int(b), int(c))]
        else:
            yr = _year(c)
            if yr is None:
                return None
            x, y = int(a), int(b)
            cands = [(yr, y, x), (yr, x, y)] if order == "DMY" else [(yr, x, y), (yr, y, x)]
        for yy, mm, dd in cands:
            try:
                return date(yy, mm, dd)
            except ValueError:
                continue
        return None
    m = MON_FIRST.search(s)
    if m and m.group(1).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    m = DAY_FIRST.search(s)
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None
    return None


def infer_order(date_strings: Iterable[Optional[str]]) -> str:
    """Decide MDY vs DMY for one document from unambiguous dates (e.g. 28/10 or 10/28)."""
    dmy = mdy = 0
    for s in date_strings:
        if not s:
            continue
        for m in NUM_DATE.finditer(s):
            a, b, _ = m.groups()
            if len(a) == 4:
                continue
            x, y = int(a), int(b)
            if x > 12 >= y:
                dmy += 1
            elif y > 12 >= x:
                mdy += 1
    if dmy > mdy:
        return "DMY"
    if mdy > dmy:
        return "MDY"
    return default_order()


def dates_in_text(line: str, order: str) -> list[date]:
    out: list[date] = []
    for rx in (NUM_DATE, MON_FIRST, DAY_FIRST):
        for m in rx.finditer(line):
            d = parse_date(m.group(0), order)
            if d:
                out.append(d)
    return out


def add_months(d: date, n: int) -> date:
    idx = d.month - 1 + n
    y, m = d.year + idx // 12, idx % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))
