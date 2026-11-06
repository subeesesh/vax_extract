"""Raw (LLM-facing) and final (Document-AI-compatible) data models."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def _num_to_str(cls, v, info):
        if info.field_name == "duration_months":
            try:
                return int(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
        return v


# ---------- raw model returned by the LLM (dates are strings exactly as printed) ----------
class Party(_Lenient):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    licenseNumber: Optional[str] = None


class RawMatch(_Lenient):
    name: Optional[str] = None
    partial: bool = False
    duration_months: Optional[int] = None


class RawEvent(_Lenient):
    raw_name: Optional[str] = None
    kind: Optional[str] = None
    given_date: Optional[str] = None
    due_date: Optional[str] = None
    duration_months: Optional[int] = None
    evidence: Optional[str] = None
    matches: list[RawMatch] = Field(default_factory=list)

    @field_validator("matches", mode="before")
    @classmethod
    def _str_matches(cls, v):
        if isinstance(v, list):
            return [{"name": m} if isinstance(m, str) else m for m in v]
        return v or []


class RawPet(_Lenient):
    name: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    sex: Optional[str] = None
    weight: Optional[str] = None
    color: Optional[str] = None
    microchipNumber: Optional[str] = None
    events: list[RawEvent] = Field(default_factory=list)


class RawExtraction(_Lenient):
    pets: list[RawPet] = Field(default_factory=list)
    owner: Optional[Party] = None
    clinic: Optional[Party] = None
    veterinarian: Optional[Party] = None


# ---------- final output (shape mirrors the existing Document AI payload) ----------
class VaccinationEvent(BaseModel):
    """One printed row x one requested name it matched (target=None if it matched none)."""
    rawName: str
    target: Optional[str] = None
    partial: bool = False
    dateAdministered: Optional[str] = None  # ISO yyyy-mm-dd
    dueDate: Optional[str] = None  # ISO yyyy-mm-dd
    dueDateSource: Optional[Literal["document", "derived"]] = None
    isLatest: bool = False
    needsHumanReview: bool = False
    reviewReasons: list[str] = Field(default_factory=list)
    evidence: Optional[str] = None


class MatchedComponent(BaseModel):
    extractedName: str
    dateAdministered: Optional[str] = None
    expiryDate: Optional[str] = None
    expiryDateSource: Optional[Literal["document", "derived"]] = None
    evidence: Optional[str] = None


class Vaccination(BaseModel):
    """One per requested vaccine name: its latest record (Document AI payload shape)."""
    name: str
    partialMatch: bool = False
    matchedComponents: list[MatchedComponent] = Field(default_factory=list)
    vaccine_name: Optional[str] = None
    dateAdministered: Optional[str] = None
    expirationDate: Optional[str] = None
    expirationDateSource: Optional[Literal["document", "derived"]] = None
    needsHumanReview: bool = False
    reviewReasons: list[str] = Field(default_factory=list)


class DiagnosticTest(BaseModel):
    name: str
    rawName: str
    dateAdministered: Optional[str] = None
    dueDate: Optional[str] = None
    needsHumanReview: bool = False
    reviewReasons: list[str] = Field(default_factory=list)
    evidence: Optional[str] = None


class Pet(BaseModel):
    name: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    sex: Optional[str] = None
    weight: Optional[str] = None
    color: Optional[str] = None
    microchipNumber: Optional[str] = None
    vaccinations: list[Vaccination] = Field(default_factory=list)
    events: list[VaccinationEvent] = Field(default_factory=list)
    diagnosticTests: list[DiagnosticTest] = Field(default_factory=list)
    printedRows: list[VaccinationEvent] = Field(default_factory=list)  # (row x name) before merge/join; slim output


class Data(BaseModel):
    pets: list[Pet] = Field(default_factory=list)
    owner: Optional[Party] = None
    clinic: Optional[Party] = None
    veterinarian: Optional[Party] = None


class TokenUsage(BaseModel):
    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0


class Meta(BaseModel):
    tier: str = "primary"  # primary | fallback
    source: str = "ocr"  # text_layer | ocr | mixed
    pages: int = 0
    ocrSeconds: float = 0.0  # ingest: text layer / preprocess + OCR
    llmSeconds: float = 0.0  # LLM call(s) + post-processing, all tiers
    needsHumanReview: bool = False
    reviewReasons: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # per page: text_layer | ocr
    layoutText: str = ""  # the serialized OCR/layout text the LLM read (shown in the web UI)


class Result(BaseModel):
    data: Data
    duration: float = 0.0
    tokenUsage: TokenUsage = Field(default_factory=TokenUsage)
    usedFallback: bool = False
    meta: Meta = Field(default_factory=Meta)

    def to_output(self) -> dict:
        """The client payload (final_json_format.txt): one vaccination per requested name with its latest dose,
        matchedComponents = the printed rows behind it (names and dates only as printed on the document),
        owner/clinic/veterinarian, duration and tokenUsage. Flags appear only when true, empty party fields are left out."""
        out = {"pets": [{**{k: getattr(p, k) for k in _PET_FIELDS if getattr(p, k)},
                         "vaccinations": [_vaccination(v, p.printedRows) for v in p.vaccinations]
                                         or _unrequested(p.printedRows)}
                        for p in self.data.pets]}
        for k in ("owner", "clinic", "veterinarian"):
            party = getattr(self.data, k)
            fields = {f: v for f, v in party.model_dump().items() if v and f in _PARTY_FIELDS} if party else {}
            if fields:
                out[k] = fields
        return {**out, "duration": self.duration, "tokenUsage": self.tokenUsage.model_dump()}


_PET_FIELDS = ("name", "species", "breed", "sex", "weight", "color", "microchipNumber")
_PARTY_FIELDS = ("name", "address", "phone", "email", "website", "licenseNumber")


def _component(e: VaccinationEvent) -> dict:
    due = e.dueDate if e.dueDateSource == "document" else None  # never a date worked out from '3 Yr'
    return {"name": e.rawName, **({"dateAdministered": e.dateAdministered} if e.dateAdministered else {}),
            **({"expirationDate": due} if due else {})}


def _newest_first(comps) -> list[dict]:
    return sorted(comps, key=lambda c: (c.get("dateAdministered") or "", c.get("expirationDate") or ""), reverse=True)


def _unrequested(printed: list[VaccinationEvent]) -> list[dict]:
    """No vaccine names requested: one entry per printed vaccine name (as printed), in the same shape - the latest
    dose's dates on top, every printed row of that name in matchedComponents."""
    groups: dict = {}
    for e in printed:
        groups.setdefault(e.rawName.lower(), []).append(e)
    out = []
    for rows in groups.values():
        dosed = [e for e in rows if e.dateAdministered]
        if dosed:
            # the same dose printed twice (invoice line + reminder row 'Last done'): the row with an expiry heads it,
            # the earliest if they differ (a combination is current until its first part runs out)
            last = max(e.dateAdministered for e in dosed)
            same = [e for e in dosed if e.dateAdministered == last]
            head = min(same, key=lambda e: (e.dueDate is None, e.dueDate or ""))
        else:
            head = max(rows, key=lambda e: e.dueDate or "")
        entry = {"name": head.rawName, "dateAdministered": head.dateAdministered, "expirationDate": head.dueDate}
        if head.needsHumanReview:
            entry["needsHumanReview"] = True
        comps = {(e.rawName, e.dateAdministered, e.dueDate if e.dueDateSource == "document" else None): _component(e)
                 for e in rows}
        out.append({**entry, "matchedComponents": _newest_first(comps.values())})
    return sorted(out, key=lambda v: v["dateAdministered"] or "", reverse=True)


def _vaccination(v: Vaccination, printed: list[VaccinationEvent]) -> dict:
    """A requested name: the dates of its latest dose (expiry may be calculated), then every printed row matched to
    it, newest first, each with only the dates the document prints. Not found -> null dates, no components."""
    out = {"name": v.name, "dateAdministered": v.dateAdministered if v.matchedComponents else None,
           "expirationDate": v.expirationDate if v.matchedComponents else None}
    if v.partialMatch:
        out["partialMatch"] = True
    if v.needsHumanReview and v.matchedComponents:
        out["needsHumanReview"] = True
    comps = {(e.rawName, e.dateAdministered, e.dueDate if e.dueDateSource == "document" else None): _component(e)
             for e in printed if e.target == v.name}
    out["matchedComponents"] = _newest_first(comps.values())
    return out
