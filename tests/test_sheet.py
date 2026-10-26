import json

import pytest

openpyxl = pytest.importorskip("openpyxl")

from vaxextract.schema import Data, MatchedComponent, Party, Pet, Result, Vaccination
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
    assert json.loads(rab["final_json"]) == {"partialMatch": False, "matchedComponents": [], "name": "Rabies",
                                             "dateAdministered": "2024-02-28", "expirationDate": "2028-01-01", "needsHumanReview": False}
    assert (bord["status"], bord["Results - Comment"]) == ("Not found", "not on document")
    assert "expirationDate" not in json.loads(bord["final_json"])


def test_no_pets_gives_one_not_found_row(tmp_path):
    p = tmp_path / "s.xlsx"
    update_sheet(p, Result(data=Data()), "c.jpg", "vx")
    [r] = _rows(p)
    assert (r["status"], r["Document_URL"]) == ("Not found", "c.jpg")
