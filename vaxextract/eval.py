"""Event-level evaluation. Gold/pred files: <name>.json with pets[].events[{target,dateAdministered,dueDate}]."""
from __future__ import annotations

import json
from pathlib import Path


def _pets(obj: dict) -> list:
    return (obj.get("data") or obj).get("pets", [])


def _events(obj: dict) -> set:
    return {
        (v.get("target"), v.get("dateAdministered"), v.get("dueDate"))
        for p in _pets(obj)
        for v in p.get("events", [])
        if v.get("target")  # rows that match no requested name are not scored
    }


def _names(obj: dict) -> list:
    return [(p.get("name") or "").strip().lower() for p in _pets(obj)]


def evaluate(pred_dir: str, gold_dir: str) -> dict:
    tp = fp = fn = 0
    tp_g = fp_g = fn_g = 0  # target + given date only
    name_ok = n = 0
    rows = []
    for gp in sorted(Path(gold_dir).glob("*.json")):
        pp = Path(pred_dir) / gp.name
        if not pp.exists():
            rows.append((gp.stem, "MISSING PREDICTION"))
            continue
        g, p = json.loads(gp.read_text()), json.loads(pp.read_text())
        ge, pe = _events(g), _events(p)
        tp += len(ge & pe); fp += len(pe - ge); fn += len(ge - pe)
        gg = {(c, a) for c, a, _ in ge}; pg = {(c, a) for c, a, _ in pe}
        tp_g += len(gg & pg); fp_g += len(pg - gg); fn_g += len(gg - pg)
        n += 1
        name_ok += int(_names(g) == _names(p))
        rows.append((gp.stem, f"missing={sorted(map(str, ge - pe))} extra={sorted(map(str, pe - ge))}"))

    def prf(a, b, c):
        pr = a / (a + b) if a + b else 0.0
        rc = a / (a + c) if a + c else 0.0
        return round(pr, 4), round(rc, 4), round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0

    return {
        "docs": n,
        "event_strict_prf": prf(tp, fp, fn),
        "event_given_only_prf": prf(tp_g, fp_g, fn_g),
        "pet_name_accuracy": round(name_ok / n, 4) if n else 0.0,
        "details": rows,
    }
