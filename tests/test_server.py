import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from vaxextract import server
from vaxextract.schema import Data, Pet, Result, Vaccination


def test_index_and_targets(monkeypatch):
    monkeypatch.setenv("VAX_TARGETS", "Rabies; Bordetella ;")
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    assert c.get("/api/targets").json() == ["Rabies", "Bordetella"]


def test_extract_passes_selected_names(monkeypatch):
    seen = {}

    def fake(data, cfg, targets=None):
        seen["data"], seen["targets"] = data, targets
        return Result(data=Data(pets=[Pet(name="Luna", vaccinations=[Vaccination(name=t) for t in targets])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies;DHLPP / FVRCP; "})
    assert r.status_code == 200
    assert seen == {"data": b"%PDF", "targets": ["Rabies", "DHLPP / FVRCP"]}
    assert [v["name"] for v in r.json()["data"]["pets"][0]["vaccinations"]] == ["Rabies", "DHLPP / FVRCP"]


def test_extract_requires_a_vaccine():
    r = TestClient(server.app).post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": " ; "})
    assert r.status_code == 400


def test_extract_slim_and_full(monkeypatch):
    from vaxextract.schema import MatchedComponent

    def fake(data, cfg, targets=None):
        return Result(data=Data(pets=[Pet(name="Luna", species="dog", vaccinations=[
            Vaccination(name="DHLPP / FVRCP", partialMatch=True, expirationDate="2027-02-28",
                        matchedComponents=[MatchedComponent(extractedName="DHPP")]),
            Vaccination(name="Rabies", needsHumanReview=True)])]))

    monkeypatch.setattr(server, "extract_bytes", fake)
    monkeypatch.setattr(server.Config, "from_env", classmethod(lambda cls: None))
    c = TestClient(server.app)
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "DHLPP / FVRCP;Rabies"})
    assert r.json() == {"data": {"pets": [{"name": "Luna", "vaccinations": [
        {"partialMatch": True, "matchedComponents": ["DHPP"], "name": "DHLPP / FVRCP",
         "expirationDate": "2027-02-28", "needsHumanReview": False},
        {"partialMatch": False, "matchedComponents": [], "name": "Rabies", "needsHumanReview": True}]}]}}
    r = c.post("/api/extract", files={"file": ("a.pdf", b"%PDF")}, data={"targets": "Rabies", "full": "true"})
    assert {"meta", "tokenUsage"} <= r.json().keys() and r.json()["data"]["pets"][0]["species"] == "dog"
