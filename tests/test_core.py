import json
from datetime import date
from pathlib import Path

from vaxextract import dates as D
from vaxextract.postprocess import build_data
from vaxextract.schema import RawExtraction, Result

TODAY = date(2026, 10, 2)
ROOT = Path(__file__).resolve().parents[1]


def ev(name, g=None, d=None, kind="vaccine", m=(), dur=None, iv=None):
    """m: requested names the (fake) LLM matched; a tuple (name, months) carries a per-component interval.
    iv: the interval words the LLM quotes from the name ('annual')."""
    matches = [{"name": x[0], "duration_months": x[1]} if isinstance(x, tuple) else dict(x) if isinstance(x, dict)
               else {"name": x} for x in m]
    for x in matches:  # by default the fake LLM quotes the whole printed name as what shows the disease
        x.setdefault("shown_by", name)
    return {"raw_name": name, "kind": kind, "given_date": g, "due_date": d, "interval_text": iv,
            "duration_months": dur, "evidence": name, "matches": matches}


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
    assert D.parse_date("Oct-06-2026") == date(2026, 10, 6)  # hyphenated month-first
    assert D.parse_date("Exp-12-2026 Oct-05-2029") == date(2029, 10, 5)  # a non-month word before it is skipped
    assert D.parse_date("APR 14 26") == date(2026, 4, 14)  # month name with a two-digit year
    assert D.parse_date("12 MAY 27") == date(2027, 5, 12)
    assert D.parse_date("Lot 12 34") is None
    assert D.parse_date("02/2027") == D.parse_date("2027/02") == D.parse_date("02/27") == date(2027, 2, 1)
    assert D.parse_date("Feb 2027") == date(2027, 2, 1) and D.parse_date("13/2027") is None


def test_future_given_date_dropped_and_shown_to_reviewer():
    # Ralph: '10-19-26' misread as '10-19-28'; Bentley: next-due dates in a 'vaccinations done' list
    pet, _ = run([ev("Rabios 1year, #00637", "10-19-28", m=["Rabies"]),
                  ev("4DX-Heartworm + Tick Diseases", "04-13-27", kind="diagnostic")], ["Rabies"])
    [rab] = pet.events
    assert (rab.dateAdministered, rab.dueDate) == (None, None)
    assert rab.needsHumanReview and "2028-10-19 is in the future" in rab.reviewReasons[0]
    assert pet.vaccinations[0].expirationDate is None
    t = pet.diagnosticTests[0]  # less than a year ahead: read as its due date, still flagged
    assert (t.dateAdministered, t.dueDate, t.needsHumanReview) == (None, "2027-04-13", True)


def test_list_of_only_future_dates_read_as_due_dates():
    # Bentley: every row of 'vaccinations done' prints a next-due date and nothing else -> a due list, not misreads
    pet, _ = run([ev("Kennel Cough-oral", "02-11-27", m=["Bordetella"]),
                  ev("DA2PPV 3 year", "02-10-29", m=["DHLPP / FVRCP"]),
                  ev("Rabies, 2 year", "03-20-27", m=["Rabies"]),
                  ev("4DX-Heartworm + Tick Diseases", "04-13-27", kind="diagnostic"),
                  ev("Date of Rabies Vaccination", "03-20-25", "03-20-27", m=["Rabies"])],  # certificate header
                 ["Bordetella", "DHLPP / FVRCP", "Rabies"])
    assert triples(pet, True) >= {("Bordetella", None, "2027-02-11"), ("DHLPP / FVRCP", None, "2029-02-10"),
                                  ("Rabies", "2025-03-20", "2027-03-20")}
    assert not any(v.needsHumanReview for v in pet.events)
    # one past date in the list -> the future ones may be misreads again: within a year they are read as due dates
    # but flagged, further ahead they are not used
    pet, _ = run([ev("Kennel Cough-oral", "02-11-27", m=["Bordetella"]), ev("DA2PPV 3 year", "02-10-29", m=["Rabies"]),
                  ev("Rabies, 2 year", "03-20-27", m=["Rabies"]), ev("Lepto", "02-11-26", m=["Bordetella"])],
                 ["Bordetella", "Rabies"])
    assert {v.dueDate for v in pet.events} == {"2027-02-11", "2027-03-20", None}  # 02-11-27 joins the 02-11-26 dose
    assert all(v.needsHumanReview for v in pet.events if v.dueDate != None or v.dateAdministered is None)


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
        ev("Bordetella/Lepto - annual vaccine", "02-11-25", m=["Bordetella", "Lepto"], dur=12, iv="annual"),
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
    return Result(data=data).to_output()["pets"][0]["vaccinations"]


def test_output_one_entry_per_requested_name():
    targets = ["DHLPP / FVRCP", "Rabies", "Bordetella"]
    rows = _slim([ev("DA2P", "9/22/2026", "8/10/2027", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("PARVO", "9/22/2026", "8/10/2027", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("RABIES 3 YR", "9/22/2026", dur=36, m=["Rabies"]),
                  ev("RABIES 1 YR", "9/01/2025", "9/01/2026", m=["Rabies"]),
                  ev("Giardia", "9/22/2026", "9/22/2027")], targets)  # not requested -> not in the output
    assert rows == [
        {"name": "DHLPP / FVRCP", "dateAdministered": "2026-09-22", "expirationDate": "2027-08-11", "partialMatch": True,
         "matchedComponents": [{"name": "DA2P", "dateAdministered": "2026-09-22", "expirationDate": "2027-08-11"},
                               {"name": "PARVO", "dateAdministered": "2026-09-22", "expirationDate": "2027-08-11"}]},
        # latest dose on top (expiry worked out from '3 YR'); components keep only printed dates, older dose too
        {"name": "Rabies", "dateAdministered": "2026-09-22", "expirationDate": "2029-09-23",
         "matchedComponents": [{"name": "RABIES 3 YR", "dateAdministered": "2026-09-22"},
                               {"name": "RABIES 1 YR", "dateAdministered": "2025-09-01", "expirationDate": "2026-09-02"}]},
        {"name": "Bordetella", "dateAdministered": None, "expirationDate": None, "matchedComponents": []}]


def test_combination_row_gets_earliest_component_expiry():
    rows = _slim([ev("DHPP 3yr-L 1yr", "2026-02-01", m=[("DHLPP / FVRCP", 36), ("Leptospirosis", 12)])],
                 ["DHLPP / FVRCP", "Leptospirosis"])
    assert [(r["name"], r["expirationDate"]) for r in rows] == [("DHLPP / FVRCP", "2029-02-02"),
                                                               ("Leptospirosis", "2027-02-02")]
    assert rows[0]["matchedComponents"] == [{"name": "DHPP 3yr-L 1yr", "dateAdministered": "2026-02-01"}]


def test_empty_expiry_goes_to_review():
    # Maximus (62959cb7): an invoice prints 'TruCan Bordetella (Intranasal)' with no due date and no interval
    pet, _ = run([ev("TruCan Bordetella (Intranasal)", "02/06/2026", m=["Bordetella"]),
                  ev("Elanco 3yr Rabies Vaccine", "02/06/2026", m=["Rabies"], dur=36, iv="3yr")],
                 ["Bordetella", "Rabies", "Leptospirosis"])
    bord, rab, lepto = pet.vaccinations
    assert (bord.expirationDate, bord.needsHumanReview) == (None, True)
    assert "no expiry printed for Bordetella" in bord.reviewReasons
    assert (rab.expirationDate, rab.needsHumanReview) == ("2029-02-06", False)  # derived from a printed '3yr'
    assert (lepto.expirationDate, lepto.needsHumanReview) == (None, False)  # not on the document: nothing to review


def test_interval_read_from_a_code_is_not_used():
    # Apollo (58a66256): 'Canine DA2PPvL-1' prints no interval; the model read '-1' as one year
    pet, _ = run([ev("Canine DA2PPvL-1", "07-27-26", m=["DHLPP / FVRCP"], dur=12, iv="-1"),
                  ev("Canine Bordetella Oral - Annual", "07-27-26", m=["Bordetella"], dur=12, iv="Annual"),
                  ev("Canine Rabies 1 Year", "07-27-26", m=["Rabies"], dur=12)],
                 ["DHLPP / FVRCP", "Bordetella", "Rabies"])
    dhlpp, bord, rab = pet.vaccinations
    assert (dhlpp.expirationDate, dhlpp.needsHumanReview) == (None, True)
    assert "no expiry printed for DHLPP / FVRCP" in dhlpp.reviewReasons
    assert (bord.expirationDate, bord.expirationDateSource) == ("2027-07-27", "derived")  # 'Annual' is printed
    assert (rab.expirationDate, rab.expirationDateSource) == ("2027-07-27", "derived")  # '1 Year' is printed


def test_expired_part_goes_to_review():
    # Calla: Lepto (part of DHLPP) expired before today, DA2PP still valid -> the earlier date (Lepto's), sent to review.
    # Last cycle's DA2PP dose has expired too, but it is the same product: plain history, no flag.
    pet, _ = run([ev("Canine DA2PP 3 year", "11-28-25", "11-01-28", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("Canine DA2PP 3 year", "11-01-22", "11-01-25", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("Canine Leptospirosis 1 year", "09-01-25", "09-01-26",
                     m=[{"name": "DHLPP / FVRCP", "partial": True}, "Leptospirosis"])],
                 ["DHLPP / FVRCP", "Leptospirosis"])
    dhlpp, lepto = pet.vaccinations
    assert dhlpp.expirationDate == "2026-09-01" and dhlpp.needsHumanReview
    assert [r for r in dhlpp.reviewReasons if r.startswith("part expired")] == [
        "part expired: 'Canine Leptospirosis 1 year' due 2026-09-01"]
    assert lepto.expirationDate == "2026-09-01"  # its own (expired) date; a single vaccine, not a part


def test_older_doses_of_a_part_are_history():
    # Onyx (2301c002): yearly 'Leptospirosis' doses; the model linked the old ones to DHLPP but not this year's
    lepto = lambda g, d, both: ev("Leptospirosis", g, d, m=[{"name": "DHLPP / FVRCP", "partial": True}, "Leptospirosis"]
                                  if both else ["Leptospirosis"])
    pet, _ = run([ev("DAPP", "03-05-26", "03-05-27", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  lepto("03-05-26", "03-05-27", False), lepto("02-15-24", "02-15-25", True),
                  lepto("03-21-22", "03-21-23", True)], ["DHLPP / FVRCP", "Leptospirosis"])
    assert not [r for r in pet.vaccinations[0].reviewReasons if r.startswith("part expired")]


def test_interval_needs_a_printed_unit():
    # Kairo: 'Lepto 1' and a product code quoted as the interval: no unit printed -> not used
    for name, quoted in (("Recombitek Lepto 1", "Lepto 1"), ("Canine DA2PPvL-1", "DA2PPvL-1")):
        pet, _ = run([ev(name, "05-29-25", m=["Leptospirosis"], dur=12, iv=quoted)], ["Leptospirosis"])
        assert (pet.vaccinations[0].expirationDate, pet.vaccinations[0].needsHumanReview) == (None, True), name
    # a quoted '(3 year)' given as 12 months: the model's 12 is dropped. Matched to two names, nothing is left
    # (review); matched to one, the printed 3 years is used
    two, _ = run([ev("Recombitek Da2pp (3 year) + Lepto 1", "05-29-25", m=["DHLPP / FVRCP", "Leptospirosis"],
                     dur=12, iv="(3 year)")], ["DHLPP / FVRCP", "Leptospirosis"])
    assert [(v.expirationDate, v.needsHumanReview) for v in two.vaccinations] == [(None, True), (None, True)]
    one, _ = run([ev("Rabies (3 year)", "05-29-25", m=["Rabies"], dur=12, iv="(3 year)")], ["Rabies"])
    assert one.vaccinations[0].expirationDate == "2028-05-29"


def test_reminder_of_other_product_not_joined():
    # Greta: DHPPV 3 Year given; the 'Lepto 1 Year' reminder also matches DHLPP but is not DHPPV's due date.
    # DHLPP as a whole expires with its first-expiring part, the Lepto
    targets = ["DHLPP / FVRCP", "Leptospirosis"]
    raw = [ev("DHPPV K9 Distemper Vaccine: 3 Year", "9/18/2026", dur=36, m=[{"name": "DHLPP / FVRCP", "partial": True}]),
           ev("Lepto Vaccine, 1 Year", None, "12/10/2026",
              m=[{"name": "DHLPP / FVRCP", "partial": True}, "Leptospirosis"])]
    pet, _ = run(raw, targets)
    assert (pet.vaccinations[0].expirationDate, pet.vaccinations[0].needsHumanReview) == ("2026-12-10", False)
    dhlpp = _slim(raw, targets)[0]
    assert dhlpp["expirationDate"] == "2026-12-11" and dhlpp["matchedComponents"] == [
        {"name": "DHPPV K9 Distemper Vaccine: 3 Year", "dateAdministered": "2026-09-18"},
        {"name": "Lepto Vaccine, 1 Year", "expirationDate": "2026-12-11"}]


def test_reminder_row_joins_dose_with_same_printed_name():
    # Harley: invoice line + reminder-table row for the same product
    rows = _slim([ev("Bordetella Oral Vaccine", "08/26/26", m=["Bordetella"]),
                  ev("Bordetella Oral Vaccine", None, "2/24/2027", m=["Bordetella"])], ["Bordetella"])
    assert rows[0]["matchedComponents"] == [
        {"name": "Bordetella Oral Vaccine", "dateAdministered": "2026-08-26", "expirationDate": "2027-02-25"}]


def test_rollup_takes_earliest_expiry_of_same_day_rows():
    # Cheddar: Lepto 1 yr and DHPP 3 yr given the same day both count for DHLPP
    pet, _ = run([ev("Canine Leptospirosis 1-Year", "2026-08-27", "2027-08-27", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
                  ev("Canine Distemper Parvo DHPP", "2026-08-27", "2029-08-26", m=[{"name": "DHLPP / FVRCP", "partial": True}])],
                 ["DHLPP / FVRCP"])
    v = pet.vaccinations[0]
    assert (v.dateAdministered, v.expirationDate, v.needsHumanReview) == ("2026-08-27", "2027-08-27", False)


def test_combination_expires_with_first_part_and_old_doses_are_superseded():
    t = ["DHLPP / FVRCP"]
    dhlpp = lambda: [{"name": "DHLPP / FVRCP", "partial": True}]  # noqa: E731
    # Bentley: due-only list, DA2PPV 3 year + Lepto annual -> the Lepto date
    pet, _ = run([ev("DA2PPV 3 year", None, "02-10-29", m=dhlpp()), ev("Annual Leptospirosis Vaccine", None, "02-11-27",
                                                                       m=dhlpp())], t)
    assert (pet.vaccinations[0].expirationDate, pet.vaccinations[0].needsHumanReview) == ("2027-02-11", False)
    # Dottie: this 3-year Distemper/Parvo replaced last year's annual one; no Lepto printed -> review
    pet, _ = run([ev("3-year Distemper/Parvo Vacc.", "10-16-25", "10-15-28", m=dhlpp()),
                  ev("Annual Distemper/Parvo Vacc.", "09-05-24", "09-05-25", m=dhlpp())], t)
    v = pet.vaccinations[0]
    assert v.expirationDate == "2028-10-15" and v.needsHumanReview
    assert any("combination not complete" in r for r in v.reviewReasons)
    # Sadie: an old reminder row long expired next to the current one -> the current one (single product -> review)
    pet, _ = run([ev("Nobivac 4 Vaccination", None, "10-20-27", m=dhlpp()),
                  ev("Nobivac 4/Third Vaccination", None, "02-12-21", m=dhlpp())], t)
    assert (pet.vaccinations[0].expirationDate, pet.vaccinations[0].needsHumanReview) == ("2027-10-20", True)
    # Gracie: not a combination - the dose's own expiry, not a reminder line printed a day earlier
    pet, _ = run([ev("Rabies Vaccination. 3yr", "10-12-26", "10-12-29", m=["Rabies"]),
                  ev("Rabies Vaccination, 3yr", None, "10-11-29", m=["Rabies"])], ["Rabies"])
    assert pet.vaccinations[0].expirationDate == "2029-10-12"
    # one combination line printed per part, same day: the earliest expiry
    pet, _ = run([ev("DHPP 3 YR + LEPTO BOOSTER", "3/6/2026", "3/6/2027", m=dhlpp()),
                  ev("DHPP 3 YR + LEPTO BOOSTER", "3/6/2026", "3/6/2029", m=dhlpp())], t)
    assert pet.vaccinations[0].expirationDate == "2027-03-06"


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


def test_eval_ignores_rows_matching_no_requested_name(tmp_path):
    from vaxextract.eval import evaluate

    ev = {"target": "Rabies", "dateAdministered": "2026-01-02", "dueDate": "2029-01-01"}
    (tmp_path / "g").mkdir(); (tmp_path / "p").mkdir()
    (tmp_path / "g" / "d.json").write_text(json.dumps({"pets": [{"name": "Rex", "events": [ev]}]}))
    pred = {"data": {"pets": [{"name": "Rex", "events": [ev, {"target": None, "dateAdministered": "2026-01-02"}]}]}}
    (tmp_path / "p" / "d.json").write_text(json.dumps(pred))
    assert evaluate(str(tmp_path / "p"), str(tmp_path / "g"))["event_strict_prf"] == (1.0, 1.0, 1.0)


def test_nothing_requested_lists_printed_vaccines():
    rows = _slim([ev("Rabies 3 Yr", "05/01/2026", dur=36), ev("Rabies 1 Yr", "04/20/2025", "04/20/2026"),
                  ev("DA2PP", "05/01/2026", "05/01/2027"), ev("DA2PP", "04/20/2025", "04/20/2026")], [])
    assert rows == [
        {"name": "Rabies 3 Yr", "dateAdministered": "2026-05-01", "expirationDate": "2029-05-02",
         "matchedComponents": [{"name": "Rabies 3 Yr", "dateAdministered": "2026-05-01"}]},
        {"name": "DA2PP", "dateAdministered": "2026-05-01", "expirationDate": "2027-05-02",
         "matchedComponents": [{"name": "DA2PP", "dateAdministered": "2026-05-01", "expirationDate": "2027-05-02"},
                               {"name": "DA2PP", "dateAdministered": "2025-04-20", "expirationDate": "2026-04-21"}]},
        {"name": "Rabies 1 Yr", "dateAdministered": "2025-04-20", "expirationDate": "2026-04-21",
         "matchedComponents": [{"name": "Rabies 1 Yr", "dateAdministered": "2025-04-20", "expirationDate": "2026-04-21"}]}]


def test_combination_printing_two_intervals_expires_with_the_shorter():
    # Rosie: 'DA2P-Pv 3yr 4L 1yr' - the model says 36 months, but the Lepto part runs out after one year
    rows = _slim([ev("DA2P-Pv 3yr 4L 1yr Vaccine", "1/21/2026", dur=36),
                  ev("Rabies Vaccine 3yr Canine", "1/21/2026", dur=36)], [])
    assert {r["name"]: r["expirationDate"] for r in rows} == {
        "DA2P-Pv 3yr 4L 1yr Vaccine": "2027-01-22", "Rabies Vaccine 3yr Canine": "2029-01-22"}


def test_owner_contact_copied_from_clinic_is_dropped():
    # Leo: the model gave the owner the clinic's phone and email
    raw = RawExtraction.model_validate({"pets": [{"name": "Leo", "events": []}],
        "owner": {"name": "Travis Ryan", "address": "94 Wayside Lane St Thomas", "phone": "(519) 631-0430",
                  "email": "info@elginanimalhospital.com"},
        "clinic": {"name": "Elgin Animal Hospital", "address": "9789 Sunset Road St. Thomas", "phone": "519-631-0430",
                   "email": "Info@ElginAnimalHospital.com"}})
    data, _ = build_data(raw, "", TODAY, [])
    assert data.owner.model_dump(exclude_none=True) == {"name": "Travis Ryan", "address": "94 Wayside Lane St Thomas"}
    assert data.clinic.phone == "519-631-0430"


def test_pet_listed_again_with_owner_surname_is_merged():
    # Mila/Ollie: invoice lines say 'Ollie', reminder headings say 'Ollie McAuliffe' and 'Mila McAuliffe'
    raw = RawExtraction.model_validate({"pets": [
        {"name": "Ollie", "events": [ev("Lyme Vaccine", None, "1/6/2027")]},
        {"name": "Mila McAuliffe", "species": "Canine", "events": [ev("Bordetella Vaccine", None, "11/14/2026")]},
        {"name": "Ollie McAuliffe", "species": "Canine", "events": [ev("DA2PP 3 Yr", None, "1/6/2029")]}]})
    data, _ = build_data(raw, "", TODAY, [])
    assert [(p.name, p.species, sorted(e.rawName for e in p.printedRows)) for p in data.pets] == [
        ("Ollie", "Canine", ["DA2PP 3 Yr", "Lyme Vaccine"]), ("Mila", "Canine", ["Bordetella Vaccine"])]


def test_different_pets_sharing_a_first_word_are_kept_apart():
    raw = RawExtraction.model_validate({"pets": [{"name": "Max", "events": []}, {"name": "Max 2", "events": []},
                                                 {"name": "Bella", "events": []}]})
    data, _ = build_data(raw, "", TODAY, [])
    assert [p.name for p in data.pets] == ["Max", "Max 2", "Bella"]


def test_interval_printed_in_name_used_when_model_leaves_it_out():
    # Indiana: no expiry printed anywhere; the model sometimes returns no duration for 'DAPP 3 year'
    rows = _slim([ev("DAPP 3 year", "06/18/2025"), ev("Bordetella", "06/18/2025")], [])
    assert {r["name"]: r["expirationDate"] for r in rows} == {"DAPP 3 year": "2028-06-19", "Bordetella": None}
    pet, _ = run([ev("Canine Rabies 3 Year", "05/05/2022", m=["Rabies"])], ["Rabies"])
    assert pet.vaccinations[0].expirationDate == "2025-05-05"


def test_declined_row_is_not_a_vaccination():
    # Sasuke: 'DECLINED: 1st Canine influenza H3N2/H3N8' on 6/19/2025 was not given
    rows = _slim([ev("DECLINED: 1st Canine influenza H3N2/H3N8", "6/19/2025"), ev("Oral Bordetella, Annual", "6/19/2025")], [])
    assert [r["name"] for r in rows] == ["Oral Bordetella, Annual"]


def test_same_dose_on_invoice_and_reminder_row_keeps_the_printed_expiry():
    # Cosmo: invoice line (no expiry) + reminder row 'Last done 03-06-26, due 03/27' share one name
    rows = _slim([ev("Oral Bordetella Vaccine", "03-06-26"), ev("Oral Bordetella Vaccine", "03-06-26", "03/27")], [])
    assert [(r["name"], r["dateAdministered"], r["expirationDate"]) for r in rows] == [
        ("Oral Bordetella Vaccine", "2026-03-06", "2027-03-07")]  # 03/27 takes the dose's day, +1 in the output


def test_pet_named_in_the_document_but_missing_is_flagged():
    # Mila/Ollie: the model sometimes returns only Ollie although the text prints "Mila McAuliffe's Reminders"
    text = "Ollie | Simplera Otic Solution | 5/14/2026\nMila McAuliffe's Reminders\nLyme Booster | 6/4/2026\n" \
           "Ollie McAuliffe's Reminders\nLyme Vaccine | 1/6/2027"
    raw = RawExtraction.model_validate({"pets": [{"name": "Ollie", "events": [ev("Lyme Vaccine", None, "1/6/2027")]}]})
    _, reasons = build_data(raw, text, TODAY, [])
    assert any(r.startswith("pet named in the document but not extracted") and "Mila" in r for r in reasons)
    raw = RawExtraction.model_validate({"pets": [{"name": "Ollie", "events": []}, {"name": "Mila McAuliffe", "events": []}]})
    _, reasons = build_data(raw, text, TODAY, [])
    assert not any("not extracted" in r for r in reasons)


# ---------------- P2: latest dose ----------------
def test_all_expired_uses_latest_dose():
    # Sasuke (saske.webp): five yearly DA2PP doses, all expired by today -> the 2025 dose, not the 2021 one
    T = ["DHLPP / FVRCP", "Leptospirosis"]
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("DA2PP+L Annual DA2PP+Lepto", "06-19-25", "06-19-26", m=[part, "Leptospirosis"]),
                  ev("Recombitek: C8 Lepto 4 + DAPPv 25x1", "06-19-24", "06-19-25", m=[part, "Leptospirosis"]),
                  ev("Annual DA2PP", "06-27-23", "06-27-24", m=[part]),
                  ev("Annual DA2PP", "07-20-21", "07-20-22", m=[part])], T)
    assert [v.expirationDate for v in pet.vaccinations] == ["2026-06-19", "2026-06-19"]


def test_latest_row_without_due_date_falls_back_to_dated_row():
    # Kiya: a 2024 'Rabies' tag row with no due date after the 2023 3-year vaccine due 2026-03-02
    pet, _ = run([ev("Rabies", "04-24-24", m=["Rabies"]),
                  ev("GS Rabies Vaccine: Canine 3yr, #38139", "03-03-23", "03-02-26", m=["Rabies"])], ["Rabies"])
    rab = pet.vaccinations[0]
    assert (rab.expirationDate, rab.needsHumanReview) == ("2026-03-02", True)
    assert rab.reviewReasons == ["latest dose prints no due date: 'Rabies' given 2024-04-24"]


def test_older_dose_due_first_goes_to_review():
    # Coco: two DaPP doses still valid, the older one due first: a later dose or another part? -> review
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("Canine DaPP Third", "06-16-26", "06-16-29", m=[part]),
                  ev("Canine Da2PP First", "03-25-26", "03-25-29", m=[part])], ["DHLPP / FVRCP"])
    v = pet.vaccinations[0]
    assert v.expirationDate == "2029-03-25" and v.needsHumanReview
    assert any(r.startswith("older dose or another part? 'Canine Da2PP First'") for r in v.reviewReasons)


def test_combination_parts_given_apart_stay_unflagged():
    # DHPP 3 yr then a Lepto booster a year later: the older row is due LATER, so dose and part readings agree
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("DHPP 3 Year", "09-18-25", "09-18-28", m=[part]),
                  ev("Lepto 1 Year", "09-01-26", "09-01-27", m=[part])], ["DHLPP / FVRCP"])
    v = pet.vaccinations[0]
    assert (v.expirationDate, v.needsHumanReview) == ("2027-09-01", False)


def test_all_expired_history_raises_no_part_flags():
    # Sasuke again, as the model returned it on the re-run: every row partial for DHLPP and all expired.
    # The latest dose is shown; the older 'Annual DA2PP' doses are history, not an expired part or a missing one
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("DA2PP+L Annual DA2PP+Lepto", "06-19-25", "06-19-26", m=[part]),
                  ev("Annual DA2PP", "06-27-23", "06-27-24", m=[part]),
                  ev("Annual DA2PP", "07-20-21", "07-20-22", m=[part])], ["DHLPP / FVRCP"])
    v = pet.vaccinations[0]
    assert (v.expirationDate, v.reviewReasons) == ("2026-06-19", [])


# ---------------- P3: a match must be shown by the printed name ----------------
def test_negated_disease_is_not_matched():
    # COCO: 'DHPPV (NO LEPTO) 1Y' - the LLM matched Leptospirosis quoting 'LEPTO'
    pet, _ = run([ev("DHPPV (NO LEPTO) 1Y", "09/26/2026", "09/26/2027",
                     m=[{"name": "DHLPP / FVRCP", "partial": True, "shown_by": "DHPPV"},
                        {"name": "Leptospirosis", "partial": False, "shown_by": "LEPTO"}])],
                 ["DHLPP / FVRCP", "Leptospirosis"])
    by = {v.name: v for v in pet.vaccinations}
    assert by["DHLPP / FVRCP"].expirationDate == "2027-09-26"
    assert by["Leptospirosis"].expirationDate is None and not by["Leptospirosis"].matchedComponents


def test_match_quoting_letters_not_printed_is_dropped():
    # Bear: 'DHPP vaccine - 1 year' matched to Leptospirosis via an 'L' that is not printed; a lower-case l
    # inside a word ('Annual') is not an abbreviation letter either; no quote at all is no match
    pet, _ = run([ev("DHPP vaccine - 1 year", "10/16/2026", "10/16/2027",
                     m=[{"name": "Leptospirosis", "shown_by": "L"}]),
                  ev("Distemper Combo Annual", "03-06-26", "03-06-27", m=[{"name": "Leptospirosis", "shown_by": "l"}]),
                  ev("Distemper Annual Vacc", "03-06-26", "03-06-27", m=[{"name": "Leptospirosis", "shown_by": None}])],
                 ["Leptospirosis"])
    assert pet.vaccinations[0].expirationDate is None


def test_quoted_abbreviation_letter_and_word_are_kept():
    pet, _ = run([ev("DHLPP 3YR", "10/13/2024", "10/13/2027", m=[{"name": "Leptospirosis", "shown_by": "L"},
                                                                {"name": "DAP", "shown_by": "DHLPP"}]),
                  ev("Dose No. 2 lepto", "10/13/2026", "10/13/2027", m=[{"name": "Lepto 2", "shown_by": "Lepto"}]),
                  ev("Nobivac Canine DAPPv 1", "09/14/2025", "09/28/2026", m=[{"name": "DA2PP", "shown_by": "DAPPv"}])],
                 ["Leptospirosis", "DAP", "Lepto 2", "DA2PP"])
    assert [v.expirationDate for v in pet.vaccinations] == ["2027-10-13", "2027-10-13", "2027-10-13", "2026-09-28"]


def test_quoted_list_of_letters_is_kept():
    # Mando: DAP requested, 'DHLPP 3YR' printed; the LLM quotes 'D, P, P'
    pet, _ = run([ev("DHLPP 3YR", None, "Oct 13, 2027", m=[{"name": "DAP", "shown_by": "D, P, P"},
                                                         {"name": "Rabies", "shown_by": "D, R"}])], ["DAP", "Rabies"])
    assert [v.expirationDate for v in pet.vaccinations] == ["2027-10-13", None]


def test_all_expired_combination_takes_parts_in_force_at_the_last_dose():
    # Zelda (today 2026-11-06): DHPP 3yr due 2026-05-22 was still in force when Lepto was given 2025-06-05: the
    # combination ran out on 2026-05-22. A Bordetella/Parainfluenza dose that lapsed in 2024 is no part of it any more
    T = ["DHLPP / FVRCP", "Leptospirosis", "Bordetella"]
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("Lepto Annual", "06-05-25", "06-05-26", m=[part, "Leptospirosis"]),
                  ev("Bordetella / Parainfluenza IN", "05-23-23", "05-22-24", m=["Bordetella", part]),
                  ev("DHPP 3yr", "05-23-23", "05-22-26", m=[part]),
                  ev("DHPP Annual", "05-10-22", None, m=[part])], T)
    assert [v.expirationDate for v in pet.vaccinations] == ["2026-05-22", "2026-06-05", "2024-05-22"]


def test_quoted_letters_with_spaces_and_lone_letter_are_kept():
    pet, _ = run([ev("DHLPP 3YR", None, "Oct 13, 2027", m=[{"name": "DAP", "shown_by": "D P"}]),
                  ev("DA2PP+L Annual", "06-19-25", "06-19-26", m=[{"name": "Leptospirosis", "shown_by": "L"}])],
                 ["DAP", "Leptospirosis"])
    assert [v.expirationDate for v in pet.vaccinations] == ["2027-10-13", "2026-06-19"]



# ---------------- P4: a single-disease row is a part of the requested combination ----------------
EXP = {"DHLPP / FVRCP": [{"label": "DHLPP", "species": "dog",
                          "diseases": ["Distemper", "Hepatitis", "Leptospirosis", "Parainfluenza", "Parvovirus"]},
                         {"label": "FVRCP", "species": "cat",
                          "diseases": ["Feline Viral Rhinotracheitis", "Calicivirus", "Panleukopenia"]}],
       "Leptospirosis": [{"label": "Leptospirosis", "species": None, "diseases": ["Leptospirosis"]}],
       "k9 Influenza": [{"label": "k9 Influenza", "species": "canine", "diseases": ["Canine Influenza"]}]}
T4 = ["DHLPP / FVRCP", "Leptospirosis", "k9 Influenza"]


def run4(pet_events, species="Canine", expansion=EXP):
    raw = RawExtraction.model_validate({"pets": [{"name": "X", "species": species, "events": pet_events}]})
    data, _ = build_data(raw, "", TODAY, T4, expansion)
    return {v.name: v for v in data.pets[0].vaccinations}


def test_lepto_row_joins_the_combination():
    # Calla-like: DA2PP 3 year due 2028, Lepto 1 year due 2027 matched only to Leptospirosis by the LLM
    v = run4([ev("Canine DA2PP 3 year", "11-28-25", "11-28-28", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
              ev("Canine Leptospirosis 1 year", "11-28-25", "11-26-27", m=["Leptospirosis"])])
    assert (v["DHLPP / FVRCP"].expirationDate, v["DHLPP / FVRCP"].needsHumanReview) == ("2027-11-26", False)
    assert v["Leptospirosis"].expirationDate == "2027-11-26"


def test_no_link_without_definitions_or_for_a_cat_or_an_unrelated_disease():
    rows = [ev("FVRCP", "11-28-25", "11-28-28", m=[{"name": "DHLPP / FVRCP", "partial": False}]),
            ev("Lepto", "11-28-25", "11-26-27", m=["Leptospirosis"]),
            ev("Canine Influenza H3N2", "11-28-25", "11-26-26", m=["k9 Influenza"])]
    for species, exp in [("Feline", EXP), ("Canine", {})]:
        assert run4(rows, species, exp)["DHLPP / FVRCP"].expirationDate == "2028-11-28"
    v = run4([ev("DA2PP", "11-28-25", "11-28-28", m=[{"name": "DHLPP / FVRCP", "partial": True}]),
              ev("Canine Influenza H3N2", "11-28-25", "11-26-26", m=["k9 Influenza"])])
    assert v["DHLPP / FVRCP"].expirationDate == "2028-11-28"  # influenza is not parainfluenza



# ---------------- P5: reminder dates ----------------
def test_month_only_given_date_is_a_reminder():
    # Moose / Ace: the LLM put a reminder column's 'MM-YY' into given_date
    pet, _ = run([ev("Ultra DAP+4L (FINAL BOOSTER)", "01-27", m=["DHLPP / FVRCP"]),
                  ev("Rabies 1 year", "12-13-24", "12-13-25", m=["Rabies"])], ["DHLPP / FVRCP", "Rabies"])
    v = pet.vaccinations[0]
    assert (v.dateAdministered, v.expirationDate) == (None, "2027-01-01")


def test_mixed_case_abbreviation_with_digit_shows_its_letters():
    pet, _ = run([ev("Da2pl+cpv Booster", "09-25-26", "09-27", m=[{"name": "Leptospirosis", "shown_by": "l"}])],
                 ["Leptospirosis"])
    assert pet.vaccinations[0].expirationDate == "2027-09-25"  # month-only '09-27' takes the dose's day



def test_single_future_date_is_read_as_due_and_flagged():
    # Xena: DATE GIVEN | DATE DUE, 'Rabies Vaccination | 11/26/2026' (given cell empty). Two years ahead stays a misread
    pet, _ = run([ev("Rabies Vaccination", "11/26/2026", m=["Rabies"]),
                  ev("DHPP", "10-19-28", m=["DHLPP / FVRCP"])], ["Rabies", "DHLPP / FVRCP"])
    rab, dh = pet.vaccinations
    assert (rab.expirationDate, rab.needsHumanReview) == ("2026-11-26", True)
    assert dh.expirationDate is None and dh.needsHumanReview



def test_earlier_reminder_beats_a_later_dose_expiry_and_is_flagged():
    # Leo: 'DA2PP+L' given 12/04/2025 prints 12/04/2028; the Lepto 1yr reminder is due 12/04/2026
    pet, _ = run([ev("Distemper+Leptospirosis (DA2PP+L) Vaccine", "12/04/2025", "12/04/2028", m=["Leptospirosis"]),
                  ev("Canine Lepto 1yr Vaccine", None, "12/04/2026", m=["Leptospirosis"])], ["Leptospirosis"])
    v = pet.vaccinations[0]
    assert (v.expirationDate, v.needsHumanReview) == ("2026-12-04", True)


def test_ocr_spelling_of_the_quoted_word_is_tolerated():
    pet, _ = run([ev("Oral Bordetella Annual", "07-31-26", "07-31-27", m=[{"name": "Bordetella",
                                                                         "shown_by": "Bordatella Vaccine"}])],
                 ["Bordetella"])
    assert pet.vaccinations[0].expirationDate == "2027-07-31"



# ---------------- P6: pet names ----------------
def test_pet_name_drops_record_numbers_and_owner_or_vet_names():
    raw = RawExtraction.model_validate({"owner": {"name": "Alexandra Szabo"}, "veterinarian": {"name": "Katherine Ames"},
                                        "pets": [{"name": "26-187 KANE SZABO"}, {"name": "Bear (26-7430)"},
                                                 {"name": "Jumanji Katherine"}, {"name": "Szabo"}]})
    data, _ = build_data(raw, "", TODAY, [])
    assert [p.name for p in data.pets] == ["KANE", "Bear", "Jumanji", "Szabo"]



# ---------------- P7 ----------------
def test_only_one_part_of_the_combination_printed_under_two_names_goes_to_review():
    # Kismet: 'Lepto Annual' and 'Lepto 1st' (both Leptospirosis), the DHPP row cut off the photo
    T = ["DHLPP / FVRCP", "Leptospirosis"]
    part = {"name": "DHLPP / FVRCP", "partial": True}
    pet, _ = run([ev("Lepto Annual", "07-01-25", "07-01-26", m=[part, "Leptospirosis"]),
                  ev("Lepto 1st", "06-05-25", None, m=[part, "Leptospirosis"])], T)
    v = pet.vaccinations[0]
    assert v.expirationDate == "2026-07-01" and any("combination not complete" in r for r in v.reviewReasons)
    # DA2PP + Lepto when Leptospirosis is not requested: two parts, nothing missing
    pet, _ = run([ev("DA2PP 3yr", "07-01-25", "07-01-28", m=[part]), ev("Lepto Annual", "07-01-25", "07-01-26", m=[part])],
                 ["DHLPP / FVRCP"])
    assert not pet.vaccinations[0].needsHumanReview



# ---------------- P8 ----------------
def test_document_that_is_not_a_vaccine_record_gives_no_dates():
    # MIKO: a screenshot of the upload/scan screen; 'DHLPP / FVRCP | Wed 09/09/2026 05:58 AM' is a timestamp
    raw = RawExtraction.model_validate({"document_type": "other", "pets": [{"name": "MIKO", "events": [
        ev("DHLPP / FVRCP", None, "09/09/2026", m=["DHLPP / FVRCP"])]}]})
    data, reasons = build_data(raw, "", TODAY, ["DHLPP / FVRCP"])
    v = data.pets[0].vaccinations[0]
    assert v.expirationDate is None and not v.matchedComponents and "not a vaccine record" in reasons


# ---------------- held-out fixes: interval vs lot expiry; a whole combination row covers its parts ----------------
def test_a_printed_due_date_is_used_as_is():
    # the user's rule: a printed date wins; the interval is only for rows that print no due date
    pet, _ = run([ev("Canine Rabies Vac 1yr, #246335", "05-20-26", "08/13/27", m=[("Rabies", 12)], dur=12, iv="1yr")],
                 ["Rabies"])
    assert pet.vaccinations[0].expirationDate == "2027-08-13"
    pet, _ = run([ev("Canine Rabies Vac 1yr", "05-20-26", None, m=[("Rabies", 12)], dur=12, iv="1yr")], ["Rabies"])
    assert pet.vaccinations[0].expirationDate == "2027-05-20"


def test_row_printing_the_combination_name_covers_the_requested_part():
    # Homer: 'DHLPP Vaccination' matched only to DHLPP / FVRCP; Leptospirosis is requested too
    v = run4([ev("DHLPP Vaccination", "05-27-26", "05-27-27", m=[{"name": "DHLPP / FVRCP", "partial": False}])])
    assert v["Leptospirosis"].expirationDate == "2027-05-27"
    # the model's partial=False on a DHPP / Distemper row is not enough (Cosmo, Benito)
    for name in ("DHPP 1 YR BOOSTER", "Distemper Annual Vacc", "Nobivac Canine DAPPv 1"):
        v = run4([ev(name, "05-27-26", "05-27-27", m=[{"name": "DHLPP / FVRCP", "partial": False}])])
        assert not v["Leptospirosis"].matchedComponents, name
    # a cat's FVRCP never covers Leptospirosis
    v = run4([ev("FVRCP", "05-27-26", "05-27-27", m=[{"name": "DHLPP / FVRCP", "partial": False}])], "Feline")
    assert not v["Leptospirosis"].matchedComponents


def test_month_only_due_takes_the_dose_day_and_output_adds_one_day():
    # senior's rules (2026-11-10): 'MM/YYYY' due -> the administered date's day (1st if no dose date);
    # the client payload gives every expiry as the last valid day + 1 (the kennel's convention)
    raw = [ev("Rabies 1 year", "01/31/2026", "02/2027", m=["Rabies"])]
    pet, _ = run(raw, ["Rabies"])
    assert pet.vaccinations[0].expirationDate == "2027-02-28"  # day 31 clamped to February
    assert _slim(raw, ["Rabies"])[0]["expirationDate"] == "2027-03-01"


def test_combination_row_printing_two_intervals_takes_the_shorter():
    # Rosie: 'DA2P-Pv 3yr 4L 1yr' given 1/21/2026, no due printed -> DHLPP is due when the Lepto part is
    v = run4([ev("DA2P-Pv 3yr 4L 1yr Vaccine", "01-21-26", None,
                 m=[{"name": "DHLPP / FVRCP", "partial": False, "duration_months": 36}, "Leptospirosis"], dur=36)])
    assert v["DHLPP / FVRCP"].expirationDate == "2027-01-21"


def test_same_day_parts_with_different_intervals_stay_apart():
    # 2_expiry invoice: 'DAPP, Triannual, 3 years' + 'Lepto, Annual, 1 year' given the same day, no due printed
    v = run4([ev("**DAPP, Triannual, 3 years", "2026-Jun-23", None,
                 m=[{"name": "DHLPP / FVRCP", "partial": True, "duration_months": 36}], dur=36, iv="3 years"),
              ev("**Lepto, Annual, 1 year", "2026-Jun-23", None,
                 m=[{"name": "DHLPP / FVRCP", "partial": True, "duration_months": 12},
                    {"name": "Leptospirosis", "duration_months": 12}], dur=12, iv="1 year")])
    assert v["DHLPP / FVRCP"].expirationDate == "2027-06-23"
