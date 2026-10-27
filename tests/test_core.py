import json
from datetime import date
from pathlib import Path

from vaxextract import dates as D
from vaxextract.postprocess import build_data
from vaxextract.schema import RawExtraction, Result

TODAY = date(2026, 10, 2)
ROOT = Path(__file__).resolve().parents[1]


def ev(name, g=None, d=None, kind="vaccine", m=(), dur=None):
    """m: requested names the (fake) LLM matched; a tuple (name, months) carries a per-component interval."""
    matches = [{"name": x[0], "duration_months": x[1]} if isinstance(x, tuple) else x if isinstance(x, dict)
               else {"name": x} for x in m]
    return {"raw_name": name, "kind": kind, "given_date": g, "due_date": d, "duration_months": dur,
            "evidence": name, "matches": matches}


def run(pet_events, targets, text=""):
    raw = RawExtraction.model_validate({"pets": [{"name": "X", "events": pet_events}]})
    data, reasons = build_data(raw, text, TODAY, targets)
    return data.pets[0], reasons


def triples(pet, with_reminders=False):
    return {(v.target, v.dateAdministered, v.dueDate) for v in pet.events if with_reminders or v.dateAdministered}


# ---------------- dates ----------------
def test_dates():
    assert D.parse_date("02-19-26") == date(2026, 2, 19)
    assert D.parse_date("9/3/2026") == date(2026, 9, 3)
    assert D.parse_date("Jun 26, 2027") == date(2027, 6, 26)
    assert D.parse_date("September 29, 2025") == date(2025, 9, 29)
    assert D.parse_date("28/10/2025", "MDY") == date(2025, 10, 28)  # invalid month -> falls back to DMY
    assert D.parse_date("03/04/2026", "DMY") == date(2026, 4, 3)
    assert D.infer_order(["28/10/2025", "03/04/2026"]) == "DMY"
    assert D.infer_order(["10/28/2025", "03/04/2026"]) == "MDY"
    assert D.add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)
    assert D.parse_date("22.1-44lb") is not None  # NUM_DATE is permissive; callers only feed event dates
    assert D.parse_date("102126") == date(2026, 10, 21)  # printed without separators
    assert D.parse_date("10212026") == date(2026, 10, 21)
    assert D.parse_date("Tag 250458 issued") is None  # only a whole value is read as a compact date
    assert D.parse_date("April 21e, 2027") == date(2027, 4, 21)  # OCR noise after the day
    assert D.parse_date("Expires: Jun7, 2029") == date(2029, 6, 7)  # phone font without spaces
    assert D.parse_date("Expires: Mar 17,2027") == date(2027, 3, 17)


def test_future_given_date_dropped_and_shown_to_reviewer():
    # Ralph: '10-19-26' misread as '10-19-28'; Bentley: next-due dates in a 'vaccinations done' list
    pet, _ = run([ev("Rabios 1year, #00637", "10-19-28", m=["Rabies"]),
                  ev("4DX-Heartworm + Tick Diseases", "04-13-27", kind="diagnostic")], ["Rabies"])
    [rab] = pet.events
    assert (rab.dateAdministered, rab.dueDate) == (None, None)
    assert rab.needsHumanReview and "2028-10-19 is in the future" in rab.reviewReasons[0]
    assert pet.vaccinations[0].expirationDate is None
    t = pet.diagnosticTests[0]
    assert (t.dateAdministered, t.dueDate, t.needsHumanReview) == (None, None, True)


def test_wrapped_name_joined():
    pet, _ = run([ev("DA2P+ Parvo\nVaccine Annual", None, "Mar 30, 2027")], [])
    assert pet.events[0].rawName == "DA2P+ Parvo Vaccine Annual"


# ---------------- post-processing on realistic LLM outputs ----------------
def test_rubi_gold():
    gold = json.loads((ROOT / "gold" / "rubi.json").read_text())
    pet, reasons = run([
        ev("Rabies Vaccination", "02-19-26", m=["Rabies"]),
        ev("Bordetella - with other vaccines", "02-19-26", m=["Bordetella"]),
        ev("DA2/PV/Rabies(3yr)Lepto(1yr), #64162961", "02-19-26", m=["DHPP", ("Rabies", 36), ("Lepto", 12)]),
        ev("Bordetella/Lepto - annual vaccine", "02-11-25", m=["Bordetella", "Lepto"], dur=12),
        ev("DA2/PV/Lepto/Rabies(1yr) - 16 wk", "03-24-22", m=["DHPP", "Lepto", ("Rabies", 12)]),
        ev("DA2/PV/Lepto/Bordetella - 12 wk", "02-10-22", m=["DHPP", "Lepto", "Bordetella"]),
        ev("DA2/PV - 8 wk", "01-10-22", m=["DHPP"]),
        ev("Heartworm Test", None, "04-02-27", kind="diagnostic"),
        ev("Bordetella", None, "02-19-27", m=["Bordetella"]),
        ev("Reminder DAP 3yr", None, "02-18-29", m=["DHPP"]),
        ev("Lepto - annual vaccine", None, "02-19-27", m=["Lepto"]),
        ev("Canine Rabies 3yr", None, "02-18-29", m=["Rabies"]),
        ev("Heartworm/Flea Prevention", None, "04-24-26", kind="other"),
    ], gold["targets"])
    assert triples(pet) == {(g["target"], g["dateAdministered"], g["dueDate"]) for g in gold["pets"][0]["events"]}
    assert triples(pet, True) == triples(pet)  # nothing left as reminder-only
    assert pet.diagnosticTests[0].dueDate == "2027-04-02"
    assert {v.target for v in pet.events if v.isLatest} == set(gold["targets"])
    rab = [v for v in pet.events if v.target == "Rabies" and v.dateAdministered == "2022-03-24"][0]
    assert rab.dueDateSource == "derived"
    # one roll-up per requested name, in request order; both 2026-02-19 rabies rows are listed
    assert [v.name for v in pet.vaccinations] == gold["targets"]
    rabies = pet.vaccinations[0]
    assert (rabies.dateAdministered, rabies.expirationDate) == ("2026-02-19", "2029-02-18")
    assert {c.extractedName for c in rabies.matchedComponents} == {
        "Rabies Vaccination", "DA2/PV/Rabies(3yr)Lepto(1yr), #64162961"}


def test_invoice_reminders_sam():
    pet, _ = run([
        ev("DAPP vac", "2/5/2026", "02/05/2027", m=["DHPP"]),
        ev("Bordetella oral annual vacc.", "2/5/2026", "02/05/2027", m=["Bordetella"]),
        ev("Lyme Disease annual vacc.", "2/5/2026", "02/05/2027"),
        ev("Heartworm test Flex 4", None, "03/11/2027", kind="diagnostic"),
        ev("Rabies canine 3 yr. vacc.", None, "03/11/2028", m=["Rabies"]),
    ], ["Rabies", "DHPP", "Bordetella"])
    assert triples(pet, True) == {
        ("DHPP", "2026-02-05", "2027-02-05"), ("Bordetella", "2026-02-05", "2027-02-05"),
        (None, "2026-02-05", "2027-02-05"), ("Rabies", None, "2028-03-11"),
    }
    lyme = [v for v in pet.events if v.target is None][0]
    assert lyme.rawName == "Lyme Disease annual vacc." and not lyme.needsHumanReview  # not requested, still kept
    assert pet.diagnosticTests[0].dueDate == "2027-03-11"
    rabies = pet.vaccinations[0]
    assert rabies.dateAdministered is None and rabies.expirationDate == "2028-03-11"


def test_booster_series_silvie():
    pet, _ = run([
        ev("H3N8/H3N2 Vaccine 1st of 2", "09-29-26", m=["Canine Influenza"]),
        ev("Bordetella Vaccine", None, "09-04-27", m=["Bordetella"]),
        ev("DA2PP 3 Year Vaccine", None, "04-29-27", m=["DHPP"]),
        ev("Rabies Vaccine Canine 3yr", None, "04-29-27", m=["Rabies"]),
        ev("H3N8/H3N2 Vaccine 2nd of 2", None, "10-29-26", m=["Canine Influenza"]),
    ], ["Rabies", "DHPP", "Bordetella", "Canine Influenza"])
    assert triples(pet, True) == {
        ("Canine Influenza", "2026-09-29", "2026-10-29"), ("Bordetella", None, "2027-09-04"),
        ("DHPP", None, "2027-04-29"), ("Rabies", None, "2027-04-29"),
    }


def test_cross_section_join_jimmy():
    pet, _ = run([
        ev("Annual Bordetella", "Jun 26, 2026", "Jun 26, 2027", m=["Bordetella"]),
        ev("Annual Lyme", "Jun 26, 2026", "Jun 26, 2027", m=["Lyme"]),
        ev("Triannual DHPP Vacc", None, "Sep 18, 2027", m=["DHPP"]),
        ev("Triannual Rabies Vacc", None, "Sep 28, 2028", m=["Rabies"]),
        ev("Rabies (Zoetis Vanguard 3)", "September 29, 2025", m=["Rabies"]),
        ev("Rabies (Zoetis Defensor 3)", "September 20, 2022", m=["Rabies"]),
    ], ["Rabies", "DHPP", "Bordetella", "Lyme"])
    assert triples(pet, True) == {
        ("Bordetella", "2026-06-26", "2027-06-26"), ("Lyme", "2026-06-26", "2027-06-26"),
        ("DHPP", None, "2027-09-18"), ("Rabies", "2025-09-29", "2028-09-28"), ("Rabies", "2022-09-20", None),
    }


def test_combo_and_partial_match_luna():
    # prod example: target "DHLPP / FVRCP" matched by a DA2PP row (partial) and a Lepto row (partial)
    targets = ["Bordetella", "DHLPP / FVRCP", "Parvo", "Rabies"]
    pet, _ = run([
        ev("DA2PP 1st Year 0.5ml Vaccine +", "10/28/2025", "10/28/2026",
           m=[{"name": "DHLPP / FVRCP", "partial": True}, "Parvo"]),
        ev("Lepto 0.5ml Vaccine Inclusive*", "10/28/2025", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
        ev("Canine Rabies 1st Year - Imrab*", None, None, m=["Rabies"]),
    ], targets)
    v = {x.name: x for x in pet.vaccinations}
    assert list(v) == targets
    assert v["Bordetella"].matchedComponents == [] and not v["Bordetella"].needsHumanReview
    dhlpp = v["DHLPP / FVRCP"]
    assert dhlpp.partialMatch and dhlpp.dateAdministered == "2025-10-28" and dhlpp.expirationDate == "2026-10-28"
    assert dhlpp.vaccine_name == "DA2PP 1st Year 0.5ml Vaccine + | Lepto 0.5ml Vaccine Inclusive*"
    assert not v["Parvo"].partialMatch
    assert v["Rabies"].needsHumanReview and "no dates" in v["Rabies"].reviewReasons


def test_derived_due_only_from_printed_interval_odin():
    pet, _ = run([
        ev("Vanguard DAPP+CV Injectable 1ml", "06/22/2026", m=["DHPP", "Coronavirus"]),
        ev("Rabies - Vanguard Rabies 3 Year 1ml", "06/22/2026", m=["Rabies"], dur=36),
    ], ["Rabies", "DHPP", "Coronavirus"])
    assert triples(pet) == {
        ("DHPP", "2026-06-22", None), ("Coronavirus", "2026-06-22", None), ("Rabies", "2026-06-22", "2029-06-22"),
    }


def test_unknown_match_names_dropped_and_flagged():
    pet, _ = run([ev("Rabies", "01-01-26", m=["Rabies", "Rabbies 3yr"])], ["Rabies"])
    (e,) = pet.events
    assert e.target == "Rabies" and any("not in the request" in r for r in e.reviewReasons)


def test_match_names_normalised_to_request_spelling():
    pet, _ = run([ev("rabies", "01-01-26", m=["  RABIES "])], ["Rabies"])
    assert pet.events[0].target == "Rabies"


def test_no_targets_extracts_rows_only():
    pet, _ = run([ev("DA2PP", "01-01-26"), ev("Annual Wellness exam", "01-01-26", kind="other")], [])
    assert [(e.rawName, e.target) for e in pet.events] == [("DA2PP", None)] and pet.vaccinations == []


def test_validation_flags():
    pet, reasons = run([ev("Rabies", "01-01-27", m=["Rabies"]), ev("Lyme", "03-01-26", "01-01-26", m=["Lyme"])],
                       ["Rabies", "Lyme"])
    rabies = [v for v in pet.events if v.target == "Rabies"][0]
    assert rabies.dateAdministered is None and rabies.needsHumanReview  # future given date dropped
    lyme = [v for v in pet.events if v.target == "Lyme"][0]
    assert "due date before given date" in lyme.reviewReasons


def test_coverage_detects_dropped_row():
    text = "02-19-26 JJ Bordetella - with other vaccines\n02-11-25 WC Bordetella/Lepto - annual vaccine"
    targets = ["Bordetella", "Lepto"]
    _, reasons = run([ev("Bordetella - with other vaccines", "02-19-26", m=["Bordetella"])], targets, text)
    assert any("possible missed events" in r for r in reasons)
    _, ok = run([ev("Bordetella - with other vaccines", "02-19-26", m=["Bordetella"]),
                 ev("Bordetella/Lepto - annual vaccine", "02-11-25", m=["Bordetella", "Lepto"])], targets, text)
    assert not any("possible missed events" in r for r in ok)


def test_year_month_name_dates_benzo():
    """Oakridge certificate prints '2026-Sep-21'; Lepto and DAPP must stay separate rows."""
    assert D.parse_date("2026-Sep-21") == date(2026, 9, 21)
    assert D.parse_date("21-Sep-2026") == date(2026, 9, 21)
    targets = ["Bordetella", "DHLPP / FVRCP", "Rabies"]
    pet, _ = run([
        ev("Bordetella Vaccine", "2026-Sep-21", "2027-Sep-20", m=["Bordetella"]),
        ev("Lepto Vaccine", "2026-Sep-21", "2027-Sep-20", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
        ev("DAPP Vaccine", "2025-Sep-26", "2028-Sep-24", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
        ev("Rabies Vaccine", "2025-Sep-26", "2028-Sep-25", m=["Rabies"]),
    ], targets)
    assert triples(pet) == {("Bordetella", "2026-09-21", "2027-09-20"), ("DHLPP / FVRCP", "2026-09-21", "2027-09-20"),
                            ("DHLPP / FVRCP", "2025-09-26", "2028-09-24"), ("Rabies", "2025-09-26", "2028-09-25")}
    v = {x.name: x for x in pet.vaccinations}
    assert (v["Bordetella"].expirationDate, v["Rabies"].expirationDate) == ("2027-09-20", "2028-09-25")
    assert not any(x.needsHumanReview for x in pet.vaccinations)


def test_undated_rows_not_merged():
    pet, _ = run([ev("Lepto Vaccine", m=["DHLPP / FVRCP"]), ev("DAPP Vaccine", m=["DHLPP / FVRCP"])],
                 ["DHLPP / FVRCP"])
    assert sorted(e.rawName for e in pet.events) == ["DAPP Vaccine", "Lepto Vaccine"]


def _slim(raw_events, targets):
    raw = RawExtraction.model_validate({"pets": [{"name": "X", "events": raw_events}]})
    data, _ = build_data(raw, "", TODAY, targets)
    return Result(data=data).to_output()["data"]["pets"][0]["vaccinations"]


def test_slim_output_lists_every_printed_row():
    targets = ["DHLPP / FVRCP", "Rabies"]
    rows = _slim([ev("DA2P", "9/22/2026", "8/10/2027", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("PARVO", "9/22/2026", "8/10/2027", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("RABIES", "9/22/2026", "8/10/2029", m=["Rabies"])], targets)
    assert [(r["name"], r["expirationDate"]) for r in rows] == [
        ("DA2P", "2027-08-10"), ("PARVO", "2027-08-10"), ("RABIES", "2029-08-10")]


def test_combination_row_gets_earliest_component_expiry():
    rows = _slim([ev("DHPP-L", "2026-02-01", m=[("DHLPP / FVRCP", 36), ("Leptospirosis", 12)])],
                 ["DHLPP / FVRCP", "Leptospirosis"])
    assert [(r["name"], r["expirationDate"]) for r in rows] == [("DHPP-L", "2027-02-01")]


def test_reminder_of_other_product_not_joined():
    # Greta: DHPPV 3 Year given; the 'Lepto 1 Year' reminder also matches DHLPP but is not DHPPV's due date
    targets = ["DHLPP / FVRCP", "Leptospirosis"]
    raw = [ev("DHPPV K9 Distemper Vaccine: 3 Year", "9/18/2026", dur=36, m=[{"name": "DHLPP / FVRCP", "partial": True}]),
           ev("Lepto Vaccine, 1 Year", None, "12/10/2026",
              m=[{"name": "DHLPP / FVRCP", "partial": True}, "Leptospirosis"])]
    pet, _ = run(raw, targets)
    assert pet.vaccinations[0].expirationDate == "2029-09-18"
    assert [(r["name"], r.get("expirationDate")) for r in _slim(raw, targets)] == [
        ("DHPPV K9 Distemper Vaccine: 3 Year", "2029-09-18"), ("Lepto Vaccine, 1 Year", "2026-12-10")]


def test_reminder_row_joins_dose_with_same_printed_name():
    # Harley: invoice line + reminder-table row for the same product
    rows = _slim([ev("Bordetella Oral Vaccine", "08/26/26", m=["Bordetella"]),
                  ev("Bordetella Oral Vaccine", None, "2/24/2027", m=["Bordetella"])], ["Bordetella"])
    assert [(r["name"], r["dateAdministered"], r["expirationDate"]) for r in rows] == [
        ("Bordetella Oral Vaccine", "2026-08-26", "2027-02-24")]


def test_rollup_takes_earliest_expiry_of_same_day_rows():
    # Cheddar: Lepto 1 yr and DHPP 3 yr given the same day both count for DHLPP
    pet, _ = run([ev("Canine Leptospirosis 1-Year", "2026-08-27", "2027-08-27", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("Canine Distemper Parvo DHPP", "2026-08-27", "2029-08-26", m=[{"name": "DHLPP / FVRCP", "partial": True}])],
                 ["DHLPP / FVRCP"])
    v = pet.vaccinations[0]
    assert (v.dateAdministered, v.expirationDate, v.needsHumanReview) == ("2026-08-27", "2027-08-27", False)


def test_name_without_letters_is_flagged():
    from vaxextract.postprocess import build_data
    from vaxextract.schema import RawExtraction
    raw = RawExtraction.model_validate({"pets": [{"name": "Onyx", "events": [
        {"raw_name": "9/8/2025", "kind": "vaccine", "given_date": "9/8/2025", "due_date": "3/8/2026", "matches": []}]}]})
    data, _ = build_data(raw, "", date(2026, 10, 27))
    [e] = data.pets[0].printedRows
    assert e.needsHumanReview and "looks like a date" in e.reviewReasons[0]


def test_reminder_due_date_joins_performed_test():
    from vaxextract.postprocess import build_data
    from vaxextract.schema import RawExtraction
    raw = RawExtraction.model_validate({"pets": [{"name": "Bailey", "events": [
        {"raw_name": "Heartworm / Antigen Test", "kind": "diagnostic", "given_date": "9/30/2026", "due_date": None, "matches": []},
        {"raw_name": "Heartworm / Antigen Test", "kind": "diagnostic", "given_date": None, "due_date": "09/30/2027", "matches": []}]}]})
    data, _ = build_data(raw, "", date(2026, 10, 27))
    [t] = data.pets[0].diagnosticTests
    assert (t.dateAdministered, t.dueDate) == ("2026-09-30", "2027-09-30")
