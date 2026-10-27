import json

import pytest

openpyxl = pytest.importorskip("openpyxl")

from vaxextract.schema import Data, MatchedComponent, Party, Pet, Result, Vaccination, VaccinationEvent
from vaxextract.sheet import COLUMNS, update_sheet


def _result(exp="2027-02-28"):
    return Result(data=Data(owner=Party(name="Gary"), pets=[Pet(name="Roo", vaccinations=[
        Vaccination(name="Rabies", vaccine_name="RABIES 3YR", dateAdministered="2024-02-28", expirationDate=exp,
                    expirationDateSource="document",
                    matchedComponents=[MatchedComponent(extractedName="RABIES 3YR")]),
        Vaccination(name="Bordetella")])]))


def _rows(path):
    ws = openpyxl.load_workbook(path).active
    return [dict(zip(COLUMNS, r)) for r in ws.iter_rows(min_row=2, values_only=True)]


def test_creates_appends_and_replaces(tmp_path):
    p = tmp_path / "s.xlsx"
    assert update_sheet(p, _result(), "a.jpg", "vx")
    assert update_sheet(p, _result(), "b.jpg", "vx")
    assert update_sheet(p, _result("2028-01-01"), "a.jpg", "vx")  # re-run replaces a.jpg's rows
    rows = _rows(p)
    assert [(r["Document_URL"], r["Immunization_Type_Name"]) for r in rows] == [
        ("b.jpg", "Rabies"), ("b.jpg", "Bordetella"), ("a.jpg", "Rabies"), ("a.jpg", "Bordetella")]
    rab, bord = rows[2], rows[3]
    assert (rab["PetName"], rab["POName"], rab["status"], rab["is_verified_by_ai"]) == ("Roo", "Gary", "Verified", True)
    assert rab["Expiration_date"].date().isoformat() == "2028-01-01"
    assert json.loads(rab["final_json"]) == _result("2028-01-01").to_output()  # full client payload
    assert set(json.loads(rab["final_json"])) == {"data", "duration", "tokenUsage", "usedFallback"}
    assert (bord["status"], bord["Results - Comment"]) == ("Not found", "not on document")
    assert bord["final_json"] == rab["final_json"]


def test_no_pets_gives_one_not_found_row(tmp_path):
    p = tmp_path / "s.xlsx"
    update_sheet(p, Result(data=Data()), "c.jpg", "vx")
    [r] = _rows(p)
    assert (r["status"], r["Document_URL"]) == ("Not found", "c.jpg")
    assert json.loads(r["final_json"])["data"] == {"pets": []}


def test_no_requested_names_gives_one_row_per_printed_vaccination(tmp_path):
    p = tmp_path / "s.xlsx"
    rows = [VaccinationEvent(rawName="RABIES 3YR", dateAdministered="2024-02-28", dueDate="2027-02-28"),
            VaccinationEvent(rawName="DA2PP", dateAdministered="2024-02-28", dueDate="2025-02-28", needsHumanReview=True)]
    update_sheet(p, Result(data=Data(pets=[Pet(name="Roo", printedRows=rows)])), "d.jpg", "vx")
    got = _rows(p)
    assert [(r["Immunization_Type_Name"], r["status"]) for r in got] == [("RABIES 3YR", "Extracted"), ("DA2PP", "Needs review")]
    assert got[0]["Expiration_date"].date().isoformat() == "2027-02-28"
    assert json.loads(got[0]["final_json"])["data"]["pets"][0]["vaccinations"][0]["matchedComponents"] == []
