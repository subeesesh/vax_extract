from datetime import date

import fitz
import numpy as np

from vaxextract import ingest, llm as llm_mod
from vaxextract.layout import Token, polygon_to_token, serialize_page
from vaxextract.pipeline import Config, extract_bytes
from vaxextract.llm import LlmConfig
from vaxextract.preprocess import estimate_skew, preprocess
from vaxextract.schema import RawExtraction


def make_pdf(lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for ln in lines:
        page.insert_text((60, y), ln, fontsize=11)
        y += 18
    return doc.tobytes()


def test_layout_rows_and_columns():
    toks = [Token("Rabies", 10, 10, 60, 20), Token("08-13-2026", 200, 10, 290, 20), Token("08-13-2027", 400, 11, 490, 21),
            Token("DA2PP", 10, 25, 55, 35), Token("08-13-2026", 200, 25, 290, 35)]
    out = serialize_page(toks).splitlines()
    assert out[0] == "Rabies | 08-13-2026 | 08-13-2027"
    assert out[1] == "DA2PP | 08-13-2026"


def test_pipeline_text_layer_with_fake_llm(monkeypatch):
    pdf = make_pdf(["Patient: Luna", "Vaccination History", "Rabies 10/28/2025 Expires 10/28/2026"]
                   + [f"filler line {i}" for i in range(20)])
    seen = {}

    def fake(cfg, text, images=None, targets=None):
        seen["text"] = text
        raw = RawExtraction.model_validate({"pets": [{"name": "Luna", "events": [
            {"raw_name": "Rabies", "kind": "vaccine", "given_date": "10/28/2025", "due_date": "10/28/2026",
             "matches": targets}]}]})
        return raw, {"prompt_tokens": 100, "completion_tokens": 20}

    monkeypatch.setattr(llm_mod, "extract", fake)
    cfg = Config(primary=LlmConfig("http://x", "k", "m"))
    res = extract_bytes(pdf, cfg, today=date(2026, 10, 2), targets=["Rabies"])
    assert "Rabies 10/28/2025 Expires 10/28/2026" in seen["text"]
    v = res.data.pets[0].events[0]
    assert (v.target, v.dateAdministered, v.dueDate) == ("Rabies", "2025-10-28", "2026-10-28")
    assert res.data.pets[0].vaccinations[0].expirationDate == "2026-10-28"
    assert res.meta.source == "text_layer" and not res.meta.needsHumanReview
    assert res.tokenUsage.totalTokens == 120 and not res.usedFallback


def test_fallback_used_when_primary_flagged(monkeypatch):
    pdf = make_pdf(["Patient: Luna", "Rabies 10/28/2025"] + [f"filler {i}" for i in range(20)])
    calls = []

    def fake(cfg, text, images=None, targets=None):
        calls.append(cfg.model)
        events = [] if cfg.model == "small" else [
            {"raw_name": "Rabies", "given_date": "10/28/2025", "due_date": "10/28/2026"}]
        return RawExtraction.model_validate({"pets": [{"name": "Luna", "events": events}]}), {}

    monkeypatch.setattr(llm_mod, "extract", fake)
    cfg = Config(primary=LlmConfig("u", "k", "small"), fallback=LlmConfig("u", "k", "big"))
    res = extract_bytes(pdf, cfg, today=date(2026, 10, 2))
    assert calls == ["small", "big"] and res.usedFallback and res.meta.tier == "fallback"
    assert len(res.data.pets[0].events) == 1


def test_deskew_estimate():
    import cv2
    img = np.full((600, 800), 255, np.uint8)
    for i in range(12):
        cv2.putText(img, f"Rabies Vaccination 08-13-2026 line {i}", (40, 60 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    m = cv2.getRotationMatrix2D((400, 300), 3.0, 1.0)
    tilted = cv2.warpAffine(img, m, (800, 600), borderValue=255)
    assert abs(abs(estimate_skew(tilted)) - 3.0) <= 1.0
    assert abs(estimate_skew(img)) <= 0.5
    pink = np.stack([np.clip(img.astype(int) * [1.0, 0.7, 0.8][i], 0, 255).astype(np.uint8) for i in range(3)], -1)
    out = preprocess(pink)
    assert out.ndim == 2 and out.shape == img.shape


def test_blank_line_between_sections():
    toks = [Token("A", 10, 10, 20, 20), Token("B", 10, 80, 20, 90)]
    assert serialize_page(toks).splitlines() == ["A", "", "B"]


def test_target_expansion_cached_and_in_prompt(monkeypatch):
    calls = []

    def fake_post(cfg, messages, schema=None, fmt="json_schema"):
        calls.append(messages[0]["content"][:20])
        if len(calls) == 1:  # expansion call
            return ('{"names":[{"name":"DHLPP / FVRCP","alternatives":['
                    '{"label":"DHLPP","species":"canine","diseases":["Distemper","Leptospirosis"]},'
                    '{"label":"FVRCP","species":"feline","diseases":["Panleukopenia"]}]}]}'), {"prompt_tokens": 5}
        assert "- DHLPP / FVRCP = DHLPP (canine): Distemper, Leptospirosis | FVRCP (feline): Panleukopenia" in messages[1]["content"]
        assert "\n- Rabies\n" in messages[1]["content"]  # no definition -> plain name
        return '{"pets":[{"name":"Luna","events":[]}]}', {"prompt_tokens": 7}

    monkeypatch.setattr(llm_mod, "_post", fake_post)
    llm_mod._EXPANSIONS.clear()
    cfg = LlmConfig("u", "k", "m")
    _, u = llm_mod.extract(cfg, "text", targets=["DHLPP / FVRCP", "Rabies"])
    _, u2 = llm_mod.extract(cfg, "text", targets=["DHLPP / FVRCP", "Rabies"])
    assert len(calls) == 3 and u["prompt_tokens"] == 12 and u2["prompt_tokens"] == 7  # expansion ran once


def test_extraction_survives_failed_expansion(monkeypatch):
    def fake_post(cfg, messages, schema=None, fmt="json_schema"):
        if "define requested" in messages[0]["content"]:
            raise llm_mod.LlmError("down")
        assert "REQUESTED VACCINE NAMES:\n- Rabies" in messages[1]["content"]
        return '{"pets":[{"name":"Luna","events":[]}]}', {}

    monkeypatch.setattr(llm_mod, "_post", fake_post)
    llm_mod._EXPANSIONS.clear()
    raw, _ = llm_mod.extract(LlmConfig("u", "k", "m2"), "text", targets=["Rabies"])
    assert raw.pets[0].name == "Luna"


def test_hedged_request_uses_first_answer(monkeypatch):
    import threading
    import time

    import httpx

    calls, release = [], threading.Event()

    def fake_post(url, json=None, headers=None, timeout=None):
        n = len(calls)
        calls.append(n)
        if n == 0:  # the first request hangs until the test ends
            release.wait(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"answer {n}"}}], "usage": {}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)
    t = time.time()
    out, _ = llm_mod._post(LlmConfig("http://x", "k", "m", hedge_after=0.05), [{"role": "user", "content": "hi"}])
    release.set()
    assert out == "answer 1" and len(calls) == 2 and time.time() - t < 2


def test_expansion_cached_on_disk(monkeypatch, tmp_path):
    calls = []

    def fake_post(cfg, messages, schema=None, fmt="json_schema"):
        calls.append(1)
        return '{"names":[{"name":"Rabies","alternatives":[{"label":"Rabies","species":null,"diseases":["Rabies"]}]}]}', {}

    monkeypatch.setattr(llm_mod, "_post", fake_post)
    monkeypatch.setenv("VAX_EXPANSION_CACHE", str(tmp_path / "exp.json"))
    llm_mod._EXPANSIONS.clear()
    a, _ = llm_mod.expand_targets(LlmConfig("u", "k", "m3"), ["Rabies"])
    llm_mod._EXPANSIONS.clear()  # simulate a restart
    b, _ = llm_mod.expand_targets(LlmConfig("u", "k", "m3"), ["Rabies"])
    assert a == b and "Rabies" in a and len(calls) == 1


def test_layout_keeps_tightly_spaced_rows_apart():
    # phone screenshot: rows 16 px apart, boxes 27 px tall -> within the height tolerance, but the boxes stack
    toks = [Token("10/3/2024", 122, 1204, 192, 1227), Token("Lyme Annual (on own)", 270, 1200, 433, 1231),
            Token("10/15/2026", 484, 1204, 565, 1227), Token("10/3/2024", 119, 1217, 194, 1246),
            Token("Lepto Annual (on own) inc PE", 272, 1219, 478, 1246), Token("10/15/2026", 484, 1217, 567, 1246),
            Token("x", 0, 1000, 10, 1027), Token("y", 0, 1100, 10, 1127)]  # sets the median height
    assert serialize_page(toks).splitlines()[-2:] == [
        "10/3/2024 | Lyme Annual (on own) | 10/15/2026", "10/3/2024 | Lepto Annual (on own) inc PE 10/15/2026"]


def test_layout_follows_tilted_table_rows():
    # bent photo: the table tilts ~5.6 deg (straight page elsewhere), so each name sits ~44 px and each due date
    # ~80 px below its own given date, while rows are only 30 px apart (polygons from RapidOCR)
    def tok(text, x0, y0, x1, h=27):
        return polygon_to_token([(x0, y0), (x1, y0 + 0.098 * (x1 - x0)), (x1, y0 + 0.098 * (x1 - x0) + h),
                                 (x0, y0 + h)], text)
    toks = []
    for i, (given, name, due) in enumerate([("06-03-26", "Booster Galaxy VI-DA2PPv", "06-03-27"),
                                            ("03-27-25", "2 year Canine Rabies", "03-27-27"),
                                            ("07-10-20", "Reissue Rabies Tag", None)]):
        y = 1750 + 30 * i
        toks += [tok(given, 174, y, 293), tok(name, 520, y + 34, 852)]
        if due:
            toks.append(tok(due, 992, y + 80, 1118))
    assert serialize_page(toks).splitlines() == [
        "06-03-26 | Booster Galaxy VI-DA2PPv | 06-03-27", "03-27-25 | 2 year Canine Rabies | 03-27-27",
        "07-10-20 | Reissue Rabies Tag"]


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_text_layer_badge_split_from_name():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 72), _words(20), fontsize=9)
    page.insert_text((60, 100), "DA2PP vaccine", fontsize=9, color=(0.2, 0.2, 0.2))
    page.insert_text((128, 100), "Historical", fontsize=10, color=(0.3, 0.3, 0.9))
    page.insert_text((300, 100), "Aug 30, 2024", fontsize=9, color=(0.2, 0.2, 0.2))
    d = ingest.load(doc.tobytes(), lambda: None)
    assert "DA2PP vaccine | Historical | Aug 30, 2024" in serialize_page(d.pages[0].tokens)


def test_text_layer_wrapped_cell_rejoined():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 72), _words(20), fontsize=9)
    page.insert_textbox(fitz.Rect(60, 90, 260, 130), "DIST/PARVO/C (3 YR) / LEPTO (1\nYR)", fontsize=10)
    page.insert_text((300, 100), "6/17/2025", fontsize=10)
    d = ingest.load(doc.tobytes(), lambda: None)
    assert "DIST/PARVO/C (3 YR) / LEPTO (1 YR) | 6/17/2025" in serialize_page(d.pages[0].tokens)


def test_long_text_pdf_read_past_ocr_page_limit():
    doc = fitz.open()
    for i in range(12):
        doc.new_page().insert_text((60, 72), f"page{i} " + _words(20), fontsize=9)
    assert len(ingest.load(doc.tobytes(), lambda: None, max_pages=10).pages) == 12


class _FakeOcr:
    def run(self, img):
        return [Token("OCR", 10, 10, 40, 20)]


def _text_pdf(text, render_mode=0):
    doc = fitz.open()
    page = doc.new_page()
    for i, ln in enumerate(text.splitlines()):
        page.insert_text((60, 72 + i * 18), ln, fontsize=11, render_mode=render_mode)
    return doc.tobytes()


_CERT = "\n".join(["Vaccination Certificate for Gracie", "Rabies Vaccination 3yr given 10/12/2026"]
                  + [f"Canine Bordetella intranasal due 10/12/2027 line {i}" for i in range(6)])


def test_good_text_layer_is_used():
    assert ingest.load(_text_pdf(_CERT), lambda: _FakeOcr()).pages[0].source == "text_layer"


def test_invisible_scanner_text_layer_is_ocred():
    assert ingest.load(_text_pdf(_CERT, render_mode=3), lambda: _FakeOcr()).pages[0].source == "ocr"


def test_letter_shifted_text_layer_is_ocred():
    shifted = "".join(chr(ord(c) - 29) if c.isalpha() else c for c in _CERT)  # 'Rabies' -> '5DELHV'-style
    assert ingest.load(_text_pdf(shifted), lambda: _FakeOcr()).pages[0].source == "ocr"
