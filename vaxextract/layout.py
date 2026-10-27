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
    slope: float = 0.0  # dy/dx of the OCR box's top edge: text tilt on a bent or angled photo


def _style_break(a: Token, b: Token) -> bool:
    """Neighbouring words that differ in both font size and colour are separate fields (a status badge
    printed right after a name), whatever the gap."""
    return a.style is not None and b.style is not None and all(x != y for x, y in zip(a.style, b.style))


def polygon_to_token(points, text: str, conf: float = 1.0) -> Token:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]  # top-left -> top-right corner
    slope = dy / dx if len(xs) == 4 and dx > 0 and abs(dy / dx) < 0.25 else 0.0
    return Token(text, min(xs), min(ys), max(xs), max(ys), conf, slope=slope)


def _untilted_centers(toks: list[Token]) -> dict[int, float]:
    """Vertical centre of each box with the local text tilt removed. A photo of a bent page can tilt a table
    by 5+ degrees while the page as a whole is straight (so deskew leaves it), and then a due date at the right
    edge sits a full row below its own given date. Each box takes the median slope of the wide boxes around it
    (short ones like 'MP' report no usable angle) and is moved along it to the page's middle x - but only where
    those boxes agree: on a curled photo the left column can tilt while the price columns stay flat, and there
    the flat rows are the better guess."""
    straight_h = lambda t: max((t.y1 - t.y0) - abs(t.slope) * (t.x1 - t.x0), 1e-6)  # noqa: E731
    yc = lambda t: (t.y0 + t.y1) / 2  # noqa: E731
    wide = [t for t in toks if (t.x1 - t.x0) > 4 * straight_h(t)]
    if not any(abs(t.slope) > 0.01 for t in wide):
        return {id(t): yc(t) for t in toks}
    hs = sorted(map(straight_h, toks))
    reach = 4 * hs[len(hs) // 2]
    xs = sorted((t.x0 + t.x1) / 2 for t in toks)
    x_mid = xs[len(xs) // 2]
    out = {}
    for t in toks:
        near = sorted(w.slope for w in wide if abs(yc(w) - yc(t)) <= reach)
        s = near[len(near) // 2] if near else 0.0
        if len(near) < 3 or sum(abs(v - s) <= 0.03 for v in near) < 0.8 * len(near):
            s = 0.0
        out[id(t)] = yc(t) - s * ((t.x0 + t.x1) / 2 - x_mid)
    return out


def serialize_page(tokens: list[Token], col_gap_chars: float = 1.5) -> str:
    toks = [t for t in tokens if t.text and t.text.strip()]
    if not toks:
        return ""
    # a tilted box is taller than its text line: measure the line, or the row tolerance spans several rows
    heights = sorted(max((t.y1 - t.y0) - abs(t.slope) * (t.x1 - t.x0), 1e-6) for t in toks)
    med_h = heights[len(heights) // 2]
    ys = _untilted_centers(toks)
    yc = lambda t: ys[id(t)]  # noqa: E731
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
