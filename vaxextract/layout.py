"""Token (word/line + bbox) -> row-structured text that preserves table/section structure."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Token:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0


def polygon_to_token(points, text: str, conf: float = 1.0) -> Token:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Token(text, min(xs), min(ys), max(xs), max(ys), conf)


def serialize_page(tokens: list[Token], col_gap_chars: float = 3.0) -> str:
    toks = [t for t in tokens if t.text and t.text.strip()]
    if not toks:
        return ""
    heights = sorted(max(t.y1 - t.y0, 1e-6) for t in toks)
    med_h = heights[len(heights) // 2]
    yc = lambda t: (t.y0 + t.y1) / 2  # noqa: E731
    toks.sort(key=yc)
    rows, cur, cur_y = [], [toks[0]], yc(toks[0])
    for t in toks[1:]:
        if abs(yc(t) - cur_y) <= 0.6 * med_h:
            cur.append(t)
            cur_y = sum(yc(a) for a in cur) / len(cur)
        else:
            rows.append(cur)
            cur, cur_y = [t], yc(t)
    rows.append(cur)

    lines, prev_bottom = [], None
    for row in rows:
        row.sort(key=lambda t: t.x0)
        char_w = sum(t.x1 - t.x0 for t in row) / max(1, sum(len(t.text) for t in row))
        parts = [row[0].text]
        for a, b in zip(row, row[1:]):
            parts.append(" | " if (b.x0 - a.x1) > col_gap_chars * char_w else " ")
            parts.append(b.text)
        top, bottom = min(t.y0 for t in row), max(t.y1 for t in row)
        if prev_bottom is not None and (top - prev_bottom) > 1.5 * med_h:
            lines.append("")
        lines.append("".join(parts))
        prev_bottom = bottom
    return "\n".join(lines)


def serialize(pages: list[list[Token]]) -> str:
    out = []
    for i, toks in enumerate(pages, 1):
        out.append(f"=== PAGE {i} ===")
        out.append(serialize_page(toks))
    return "\n".join(out)
