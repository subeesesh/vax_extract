import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from vaxextract import server
from vaxextract.schema import Data, Party, Pet, Result, VaccinationEvent


def test_index_and_targets(monkeypatch):
    monkeypatch.setenv("VAX_TARGETS", "Rabies; Bordetella ;")
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    assert c.get("/api/targets").json() == ["Rabies", "Bordetella"]


def test_extract_passes_selected_names(monkeypatch):
    seen = {}

    def fake(data, cfg, targets=None):
        seen["data"], seen["targets"] = data, targets
        return Result(data=Data(pets=[Pet(name="Luna", printedRows=[VaccinationEvent(rawName="RABIES 3YR", target="Rabies")])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies;DHLPP / FVRCP; "})
    assert r.status_code == 200
    assert seen == {"data": b"%PDF", "targets": ["Rabies", "DHLPP / FVRCP"]}
    assert [v["name"] for v in r.json()["data"]["pets"][0]["vaccinations"]] == ["RABIES 3YR"]


def test_extract_requires_a_vaccine():
    r = TestClient(server.app).post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": " ; "})
    assert r.status_code == 400


def test_extract_slim_and_full(monkeypatch):
    def fake(data, cfg, targets=None):
        dhpp = dict(rawName="DHPP-L", dateAdministered="2026-02-28", dueDate="2027-02-28")
        return Result(duration=4.2, data=Data(clinic=Party(name="Comal Pet Hospital", email="staff@comalvet.com",
                                                           website="comalvet.com"), pets=[Pet(name="Luna", species="dog", printedRows=[
            VaccinationEvent(**dhpp, target="DHLPP / FVRCP", partial=True),
            VaccinationEvent(**{**dhpp, "dueDate": "2029-02-28"}, target="Lepto", partial=True),  # same row, longer interval
            VaccinationEvent(rawName="Rabies 1 Year", dateAdministered="2022-05-03", target="Rabies", needsHumanReview=True),
            VaccinationEvent(rawName="Giardia", dateAdministered="2025-01-01")])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "DHLPP / FVRCP;Rabies;Lepto"})
    assert r.json() == {"data": {"pets": [{"name": "Luna", "species": "dog", "vaccinations": [
        {"partialMatch": True, "matchedComponents": ["DHLPP / FVRCP", "Lepto"], "name": "DHPP-L",
         "dateAdministered": "2026-02-28", "expirationDate": "2027-02-28", "needsHumanReview": False},
        {"partialMatch": False, "matchedComponents": [], "name": "Giardia", "dateAdministered": "2025-01-01",
         "needsHumanReview": False},
        {"partialMatch": False, "matchedComponents": [], "name": "Rabies 1 Year", "dateAdministered": "2022-05-03",
         "needsHumanReview": True}]}],
        "clinic": {"name": "Comal Pet Hospital", "email": "staff@comalvet.com", "website": "comalvet.com"}},
        "duration": 4.2, "tokenUsage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "usedFallback": False}
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies", "full": "true"})
    assert {"meta", "tokenUsage"} <= r.json().keys() and r.json()["data"]["pets"][0]["species"] == "dog"
