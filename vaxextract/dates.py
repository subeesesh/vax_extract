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
# the day may carry an ordinal or OCR noise ('21st', '21e'); phone fonts drop spaces ('Jun7, 2029', 'Mar 17,2027');
# some clinics hyphenate ('Oct-06-2026'), some print a two-digit year ('APR 14 26')
MON_FIRST = re.compile(r"\b([A-Za-z]{3,9})\.?[\s\-/.]*(\d{1,2})[A-Za-z]{0,2}(?:,\s*|[\s\-/.]+)(\d{4}|\d{2})\b")
DAY_FIRST = re.compile(r"\b(\d{1,2})[A-Za-z]{0,2}[\s\-/.]+([A-Za-z]{3,9})\.?,?[\s\-/.]+(\d{4}|\d{2})\b")
COMPACT = re.compile(r"(\d{2})(\d{2})(\d{2}|\d{4})")  # whole value only: '102126', '10212026'
YEAR_MON = re.compile(r"\b(\d{4})[\s\-/.]+([A-Za-z]{3,9})\.?[\s\-/.]+(\d{1,2})\b")  # 2026-Sep-21
# whole value only, read as the 1st of the month: '02/2027', '2027/02', '02/27' (reminder lists), 'Feb 2027'
MONTH_ONLY = re.compile(r"(\d{1,2})[-/.](\d{4}|\d{2})|(\d{4})[-/.](\d{1,2})|([A-Za-z]{3,9})\.?,?\s*(\d{4})")


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
    m = NUM_DATE.search(s) or COMPACT.fullmatch(s)
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
    m = next((m for m in MON_FIRST.finditer(s) if m.group(1).lower() in _MONTHS), None)  # skip 'Exp-12-2026'-like words
    if m:
        try:
            return date(_year(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    m = DAY_FIRST.search(s)
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(_year(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None
    m = YEAR_MON.search(s)
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3)))
        except ValueError:
            return None
    m = MONTH_ONLY.fullmatch(s)
    if m:
        mm, yy, yy4, mm4, name, yyn = m.groups()
        month = int(mm) if mm else int(mm4) if mm4 else _MONTHS.get(name.lower())
        year = _year(yy) if yy else int(yy4 or yyn)
        if month and 1 <= month <= 12 and year:
            return date(year, month, 1)
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
    for rx in (NUM_DATE, MON_FIRST, DAY_FIRST, YEAR_MON):
        for m in rx.finditer(line):
            d = parse_date(m.group(0), order)
            if d:
                out.append(d)
    return out


def add_months(d: date, n: int) -> date:
    idx = d.month - 1 + n
    y, m = d.year + idx // 12, idx % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))
