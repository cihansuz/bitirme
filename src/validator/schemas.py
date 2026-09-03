from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RawLabItem(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: str = ""


class PatientInput(BaseModel):
    patient_id: str = "P-1001"
    age: Optional[int] = 45
    gender: Optional[str] = "unspecified"
    biomarkers: List[RawLabItem]
    medications: List[str] = []
    allergies: List[str] = []
    target_calories: Optional[float] = 1800.0


class BiomarkerResult(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: str
    status: str  # LOW, NORMAL, HIGH, CRITICAL


class RetrievedEvidence(BaseModel):
    chunk_id: str
    title: str
    text: str


class DecisionPacket(BaseModel):
    patient_id: str
    abnormal_biomarkers: List[BiomarkerResult]
    active_constraints: Dict[str, Any]
    retrieved_evidence: List[RetrievedEvidence]
    macro_plan: Optional[Dict[str, Any]] = None
    contraindications: Optional[List[str]] = None


class Finding(BaseModel):
    biomarker: str
    interpretation: str
    citation_ids: List[str] = Field(default_factory=list)


class DietaryGuideline(BaseModel):
    parameter: str
    target: str
    rationale: str
    citation_ids: List[str] = Field(default_factory=list)


class DietitianDecisionSupportOutput(BaseModel):
    patient_id: Optional[str] = "P-UNKNOWN"
    clinical_summary: str
    findings: List[Finding]
    dietary_guidelines_suggested: List[DietaryGuideline]
    macro_plan: Optional[Dict[str, Any]] = None
    contraindications_flagged: List[str]
    citations_verified: bool = True
    status: str = "PENDING_DIETITIAN_REVIEW"  # PENDING_DIETITIAN_REVIEW, APPROVED, EDITED_AND_APPROVED


class DietitianReviewRequest(BaseModel):
    patient_id: str
    approved: bool
    edited_output: Optional[DietitianDecisionSupportOutput] = None
    dietitian_notes: Optional[str] = ""
