import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from vaxextract import server
from vaxextract.schema import Data, MatchedComponent, Party, Pet, Result, Vaccination, VaccinationEvent


def test_index_and_targets(monkeypatch):
    monkeypatch.setenv("VAX_TARGETS", "Rabies; Bordetella ;")
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    assert c.get("/api/targets").json() == ["Rabies", "Bordetella"]


def test_extract_passes_selected_names(monkeypatch):
    seen = {}

    def fake(data, cfg, targets=None):
        seen["data"], seen["targets"] = data, targets
        return Result(data=Data(pets=[Pet(name="Luna", printedRows=[VaccinationEvent(rawName="RABIES 3YR", target="Rabies")],
                                          vaccinations=[Vaccination(name="Rabies", matchedComponents=[
                                              MatchedComponent(extractedName="RABIES 3YR")])])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies;DHLPP / FVRCP; "})
    assert r.status_code == 200
    assert seen == {"data": b"%PDF", "targets": ["Rabies", "DHLPP / FVRCP"]}
    assert r.json()["pets"][0]["vaccinations"] == [{"name": "Rabies", "dateAdministered": None, "expirationDate": None,
                                                    "matchedComponents": [{"name": "RABIES 3YR"}]}]


def test_extract_without_vaccines_is_extraction_only(monkeypatch):
    seen = {}

    def fake(data, cfg, targets=None):
        seen["targets"] = targets
        return Result(data=Data(pets=[Pet(name="Luna", printedRows=[VaccinationEvent(rawName="RABIES 3YR")])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    monkeypatch.delenv("VAX_SHEET", raising=False)
    r = TestClient(server.app).post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": " ; "})
    assert r.status_code == 200 and seen["targets"] == []
    assert r.json()["pets"][0]["vaccinations"] == [  # nothing requested: the printed vaccines, as printed
        {"name": "RABIES 3YR", "dateAdministered": None, "expirationDate": None, "matchedComponents": [{"name": "RABIES 3YR"}]}]


def test_extract_slim_and_full(monkeypatch):
    def fake(data, cfg, targets=None):
        dhpp = dict(rawName="DHPP-L", dateAdministered="2026-02-28")
        comp = [MatchedComponent(extractedName="DHPP-L")]
        return Result(duration=4.2, data=Data(clinic=Party(name="Comal Pet Hospital", email="staff@comalvet.com",
                                                           website="comalvet.com"), pets=[Pet(
            name="Luna", species="dog",
            vaccinations=[Vaccination(name="DHLPP / FVRCP", partialMatch=True, matchedComponents=comp,
                                      dateAdministered="2026-02-28", expirationDate="2027-02-28"),
                          Vaccination(name="Rabies", needsHumanReview=True, reviewReasons=["no dates"]),
                          Vaccination(name="Lepto", matchedComponents=comp, dateAdministered="2026-02-28",
                                      expirationDate="2029-02-28")],
            printedRows=[
                VaccinationEvent(**dhpp, dueDate="2027-02-28", dueDateSource="document", target="DHLPP / FVRCP", partial=True),
                VaccinationEvent(**dhpp, dueDate="2029-02-28", dueDateSource="derived", target="Lepto"),
                VaccinationEvent(rawName="Giardia", dateAdministered="2025-01-01")])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = slim = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "DHLPP / FVRCP;Rabies;Lepto"})
    assert r.json() == {"pets": [{"name": "Luna", "species": "dog", "vaccinations": [
        {"name": "DHLPP / FVRCP", "dateAdministered": "2026-02-28", "expirationDate": "2027-03-01", "partialMatch": True,
         "matchedComponents": [{"name": "DHPP-L", "dateAdministered": "2026-02-28", "expirationDate": "2027-03-01"}]},
        {"name": "Rabies", "dateAdministered": None, "expirationDate": None, "matchedComponents": []},  # not found
        {"name": "Lepto", "dateAdministered": "2026-02-28", "expirationDate": "2029-03-01",
         "matchedComponents": [{"name": "DHPP-L", "dateAdministered": "2026-02-28"}]}]}],  # derived expiry not in component
        "clinic": {"name": "Comal Pet Hospital", "email": "staff@comalvet.com", "website": "comalvet.com"},
        "duration": 4.2, "tokenUsage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}}
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies", "full": "true"})
    assert {"meta", "tokenUsage"} <= r.json().keys() and r.json()["data"]["pets"][0]["species"] == "dog"
    assert r.json()["output"] == slim.json()  # the final JSON rides along for the UI


def test_layout_only_needs_no_llm(monkeypatch):
    from vaxextract.ingest import Document, Page

    def fake(data, ocr_engine="rapid"):
        assert data == b"%PDF"
        return Document(pages=[Page(tokens=[], source="text_layer")], kind="pdf"), "=== PAGE 1 ===\nRABIES | 8/10/2029"

    monkeypatch.setattr(server, "read_layout", fake)
    monkeypatch.setattr(server, "extract_bytes", lambda *a, **k: pytest.fail("layout must not call the LLM pipeline"))
    r = TestClient(server.app).post("/api/layout", files={"file": ("a.pdf", b"%PDF")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"].endswith("RABIES | 8/10/2029")
    assert (body["kind"], body["pages"], body["sources"]) == ("pdf", 1, ["text_layer"])
