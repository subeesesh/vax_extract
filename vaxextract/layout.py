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
    style: tuple | None = None  # (font size, colour) of a PDF text-layer word


def _style_break(a: Token, b: Token) -> bool:
    """Neighbouring words that differ in both font size and colour are separate fields (a status badge
    printed right after a name), whatever the gap."""
    return a.style is not None and b.style is not None and all(x != y for x, y in zip(a.style, b.style))


def polygon_to_token(points, text: str, conf: float = 1.0) -> Token:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Token(text, min(xs), min(ys), max(xs), max(ys), conf)


def serialize_page(tokens: list[Token], col_gap_chars: float = 1.5) -> str:
    toks = [t for t in tokens if t.text and t.text.strip()]
    if not toks:
        return ""
    heights = sorted(max(t.y1 - t.y0, 1e-6) for t in toks)
    med_h = heights[len(heights) // 2]
    yc = lambda t: (t.y0 + t.y1) / 2  # noqa: E731
    toks.sort(key=yc)
    # a box joins the nearest open line within the height tolerance that it does not stack under/over:
    # boxes of one line never overlap horizontally (tight line spacing on phone screenshots puts
    # neighbouring rows within the tolerance, and this keeps them apart)
    stacked = lambda a, b: min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 * min(a.x1 - a.x0, b.x1 - b.x0)  # noqa: E731
    tol = 0.6 * med_h
    rows: list[list[Token]] = []
    centers: list[float] = []
    open_ = []  # indexes of rows still within reach
    for t in toks:
        y = yc(t)
        open_ = [i for i in open_ if y - centers[i] <= tol]
        fits = [i for i in open_ if not any(stacked(t, a) for a in rows[i])]
        if fits:
            i = min(fits, key=lambda i: abs(y - centers[i]))
            rows[i].append(t)
            centers[i] = sum(yc(a) for a in rows[i]) / len(rows[i])
        else:
            rows.append([t])
            centers.append(y)
            open_.append(len(rows) - 1)
    # refinement: a box grouped before its own line existed moves to the line it sits closest to; repeat while
    # boxes move, since one leaving a line can unblock another (it stacked with the box that left)
    for _ in range(4):
        moved = False
        for i in range(len(rows)):
            for t in list(rows[i]):
                if len(rows[i]) == 1:
                    break
                y = yc(t)
                own = sum(yc(a) for a in rows[i] if a is not t) / (len(rows[i]) - 1)
                best = min((j for j in range(len(rows)) if j != i and rows[j] and abs(y - centers[j]) <= tol
                            and not any(stacked(t, a) for a in rows[j])),
                           key=lambda j: abs(y - centers[j]), default=None)
                if best is not None and abs(y - centers[best]) < abs(y - own):
                    rows[i].remove(t)
                    rows[best].append(t)
                    centers[i], centers[best] = own, sum(yc(a) for a in rows[best]) / len(rows[best])
                    moved = True
        if not moved:
            break
    rows = [r for r in rows if r]

    lines, prev_bottom = [], None
    for row in rows:
        row.sort(key=lambda t: t.x0)
        char_w = sum(t.x1 - t.x0 for t in row) / max(1, sum(len(t.text) for t in row))
        parts = [row[0].text]
        for a, b in zip(row, row[1:]):
            parts.append(" | " if (b.x0 - a.x1) > col_gap_chars * char_w or _style_break(a, b) else " ")
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
