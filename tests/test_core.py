import json
from datetime import date
from pathlib import Path

from vaxextract import dates as D
from vaxextract.postprocess import build_data
from vaxextract.schema import RawExtraction

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
