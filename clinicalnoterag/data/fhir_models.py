"""Pydantic models for the subset of HL7 FHIR R4 used by this project.

The models are intentionally small: they validate the fields the pipeline
depends on (subject references, code systems, encounter linkage, claim
adjudication) while allowing additional FHIR fields through. Every generated
bundle is validated before it is written to disk.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
CPT_SYSTEM = "http://www.ama-assn.org/go/cpt"
SNOMED_SYSTEM = "http://snomed.info/sct"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
CARC_SYSTEM = "https://x12.org/codes/claim-adjustment-reason-codes"
SYNTHETIC_TAG_SYSTEM = "https://clinicalnoterag.dev/synthetic"


class Coding(BaseModel):
    """A FHIR ``Coding``."""

    model_config = ConfigDict(extra="allow")

    system: str
    code: str
    display: str


class CodeableConcept(BaseModel):
    """A FHIR ``CodeableConcept`` with at least one coding."""

    model_config = ConfigDict(extra="allow")

    coding: list[Coding]
    text: str | None = None

    @field_validator("coding")
    @classmethod
    def _non_empty(cls, value: list[Coding]) -> list[Coding]:
        if not value:
            raise ValueError("CodeableConcept.coding must not be empty")
        return value


class Reference(BaseModel):
    """A FHIR resource reference."""

    model_config = ConfigDict(extra="allow")

    reference: str
    display: str | None = None


class HumanName(BaseModel):
    """A FHIR ``HumanName``."""

    model_config = ConfigDict(extra="allow")

    use: str = "official"
    family: str
    given: list[str]


class Identifier(BaseModel):
    """A FHIR ``Identifier``."""

    model_config = ConfigDict(extra="allow")

    system: str
    value: str


class Address(BaseModel):
    """A FHIR ``Address``."""

    model_config = ConfigDict(extra="allow")

    line: list[str]
    city: str
    state: str
    postalCode: str
    country: str = "US"


class ContactPoint(BaseModel):
    """A FHIR ``ContactPoint``."""

    model_config = ConfigDict(extra="allow")

    system: Literal["phone", "email", "fax", "url"]
    value: str
    use: str | None = None


class Period(BaseModel):
    """A FHIR ``Period``."""

    model_config = ConfigDict(extra="allow")

    start: str
    end: str | None = None


class Meta(BaseModel):
    """A FHIR ``Meta`` block."""

    model_config = ConfigDict(extra="allow")

    tag: list[Coding] = Field(default_factory=list)


class FHIRResource(BaseModel):
    """Base class for a FHIR resource."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    resourceType: str
    id: str
    meta: Meta | None = None


class PatientResource(FHIRResource):
    """Minimal validated FHIR Patient."""

    resourceType: Literal["Patient"]
    identifier: list[Identifier]
    name: list[HumanName]
    gender: Literal["male", "female", "other", "unknown"]
    birthDate: str
    address: list[Address]
    telecom: list[ContactPoint]


class EncounterClass(BaseModel):
    """FHIR Encounter.class coding."""

    model_config = ConfigDict(extra="allow")

    system: str
    code: str
    display: str


class EncounterResource(FHIRResource):
    """Minimal validated FHIR Encounter."""

    resourceType: Literal["Encounter"]
    status: Literal["planned", "arrived", "triaged", "in-progress", "onleave", "finished"]
    class_: EncounterClass = Field(alias="class")
    subject: Reference
    period: Period
    reasonCode: list[CodeableConcept] | None = None


class ConditionResource(FHIRResource):
    """Minimal validated FHIR Condition using ICD-10-CM."""

    resourceType: Literal["Condition"]
    clinicalStatus: CodeableConcept
    verificationStatus: CodeableConcept
    code: CodeableConcept
    subject: Reference
    encounter: Reference
    onsetDateTime: str

    @field_validator("code")
    @classmethod
    def _icd10(cls, value: CodeableConcept) -> CodeableConcept:
        if not any(c.system == ICD10_SYSTEM for c in value.coding):
            raise ValueError("Condition.code must contain an ICD-10-CM coding")
        return value


class ProcedureResource(FHIRResource):
    """Minimal validated FHIR Procedure using CPT."""

    resourceType: Literal["Procedure"]
    status: Literal["preparation", "in-progress", "not-done", "on-hold", "stopped", "completed"]
    code: CodeableConcept
    subject: Reference
    encounter: Reference
    performedPeriod: Period

    @field_validator("code")
    @classmethod
    def _cpt(cls, value: CodeableConcept) -> CodeableConcept:
        if not any(c.system == CPT_SYSTEM for c in value.coding):
            raise ValueError("Procedure.code must contain a CPT coding")
        return value


class Quantity(BaseModel):
    """FHIR Quantity value."""

    model_config = ConfigDict(extra="allow")

    value: float
    unit: str
    system: str = "http://unitsofmeasure.org"
    code: str


class ObservationResource(FHIRResource):
    """Minimal validated FHIR Observation using SNOMED CT."""

    resourceType: Literal["Observation"]
    status: Literal["registered", "preliminary", "final", "amended"]
    code: CodeableConcept
    subject: Reference
    encounter: Reference
    effectiveDateTime: str
    valueQuantity: Quantity | None = None
    valueString: str | None = None

    @field_validator("code")
    @classmethod
    def _snomed(cls, value: CodeableConcept) -> CodeableConcept:
        if not any(c.system == SNOMED_SYSTEM for c in value.coding):
            raise ValueError("Observation.code must contain a SNOMED CT coding")
        return value


class MedicationRequestResource(FHIRResource):
    """Minimal validated FHIR MedicationRequest using RxNorm."""

    resourceType: Literal["MedicationRequest"]
    status: Literal["active", "completed", "stopped"]
    intent: Literal["order", "plan", "proposal"]
    medicationCodeableConcept: CodeableConcept
    subject: Reference
    encounter: Reference
    authoredOn: str

    @field_validator("medicationCodeableConcept")
    @classmethod
    def _rxnorm(cls, value: CodeableConcept) -> CodeableConcept:
        if not any(c.system == RXNORM_SYSTEM for c in value.coding):
            raise ValueError("MedicationRequest must contain an RxNorm coding")
        return value


class ClaimItem(BaseModel):
    """A FHIR Claim line item."""

    model_config = ConfigDict(extra="allow")

    sequence: int
    productOrService: CodeableConcept
    net: dict[str, Any]
    servicedDate: str | None = None


class ClaimResource(FHIRResource):
    """Minimal validated FHIR Claim using CPT line items."""

    resourceType: Literal["Claim"]
    status: Literal["active", "cancelled", "draft", "entered-in-error"]
    type: CodeableConcept
    patient: Reference
    created: str
    provider: Reference
    priority: CodeableConcept
    insurance: list[dict[str, Any]]
    item: list[ClaimItem]
    total: dict[str, Any]
    billablePeriod: Period

    @field_validator("item")
    @classmethod
    def _cpt_items(cls, value: list[ClaimItem]) -> list[ClaimItem]:
        if not value:
            raise ValueError("Claim.item must not be empty")
        for item in value:
            if not any(c.system == CPT_SYSTEM for c in item.productOrService.coding):
                raise ValueError("Claim line productOrService must contain a CPT coding")
        return value


class Adjudication(BaseModel):
    """A FHIR ClaimResponse adjudication entry."""

    model_config = ConfigDict(extra="allow")

    category: CodeableConcept
    amount: dict[str, Any] | None = None
    reason: CodeableConcept | None = None


class ClaimResponseResource(FHIRResource):
    """Minimal validated FHIR ClaimResponse with a CARC denial reason."""

    resourceType: Literal["ClaimResponse"]
    status: Literal["active", "cancelled", "draft", "entered-in-error"]
    type: CodeableConcept
    patient: Reference
    created: str
    request: Reference
    outcome: Literal["queued", "complete", "error", "partial"]
    adjudication: list[Adjudication]
    total: list[Adjudication]
    item: list[dict[str, Any]] = Field(default_factory=list)


FHIRResourceUnion = Annotated[
    Union[
        PatientResource,
        EncounterResource,
        ConditionResource,
        ProcedureResource,
        ObservationResource,
        MedicationRequestResource,
        ClaimResource,
        ClaimResponseResource,
    ],
    Field(discriminator="resourceType"),
]


class BundleEntry(BaseModel):
    """A FHIR Bundle entry."""

    model_config = ConfigDict(extra="allow")

    fullUrl: str
    resource: FHIRResourceUnion


class Bundle(BaseModel):
    """A validated FHIR R4 collection bundle."""

    model_config = ConfigDict(extra="allow")

    resourceType: Literal["Bundle"]
    type: Literal["collection"]
    id: str
    timestamp: str
    entry: list[BundleEntry]

    @field_validator("entry")
    @classmethod
    def _has_patient(cls, value: list[BundleEntry]) -> list[BundleEntry]:
        if not any(entry.resource.resourceType == "Patient" for entry in value):
            raise ValueError("Bundle must contain at least one Patient resource")
        return value


def validate_bundle(data: dict[str, Any]) -> Bundle:
    """Validate a raw FHIR bundle dict and return the parsed model.

    Raises:
        pydantic.ValidationError: If required fields or code systems are invalid.
    """
    return Bundle.model_validate(data)


__all__ = [
    "CARC_SYSTEM",
    "CPT_SYSTEM",
    "ICD10_SYSTEM",
    "RXNORM_SYSTEM",
    "SNOMED_SYSTEM",
    "Bundle",
    "validate_bundle",
]
