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


class Result(BaseModel):
    data: Data
    duration: float = 0.0
    tokenUsage: TokenUsage = Field(default_factory=TokenUsage)
    usedFallback: bool = False
    meta: Meta = Field(default_factory=Meta)

    def to_output(self) -> dict:
        """The slim client payload (final_json_format.txt); empty expirationDate is omitted."""
        return OutputResult(data=OutputData(pets=[
            OutputPet(name=p.name, vaccinations=[
                OutputVaccination(
                    partialMatch=v.partialMatch,
                    matchedComponents=[c.extractedName for c in v.matchedComponents] if v.partialMatch else [],
                    name=v.name, expirationDate=v.expirationDate, needsHumanReview=v.needsHumanReview)
                for v in p.vaccinations])
            for p in self.data.pets])).model_dump(exclude_none=True)


# ---------- slim client payload (final_json_format.txt) ----------
class OutputVaccination(BaseModel):
    partialMatch: bool = False
    matchedComponents: list[str] = Field(default_factory=list)  # printed names, only for a partial match
    name: str
    expirationDate: Optional[str] = None
    needsHumanReview: bool = False


class OutputPet(BaseModel):
    name: Optional[str] = None
    vaccinations: list[OutputVaccination] = Field(default_factory=list)


class OutputData(BaseModel):
    pets: list[OutputPet] = Field(default_factory=list)


class OutputResult(BaseModel):
    data: OutputData
