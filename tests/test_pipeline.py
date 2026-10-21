from datetime import date

import fitz
import numpy as np

from vaxextract import llm as llm_mod
from vaxextract.layout import Token, serialize_page
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
