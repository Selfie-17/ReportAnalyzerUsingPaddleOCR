"""
schemas.py
Canonical Pydantic models for structured observation report extraction,
section-wise aggregation, validation, and batch manifest generation.
"""

from typing import List, Dict, Optional, Any, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class VariableItem(BaseModel):
    """Represents a single variable and its described purpose."""
    model_config = ConfigDict(extra="ignore")
    variable: str = Field(..., description="Variable name mentioned in the report")
    purpose: str = Field(..., description="Stated purpose of the variable")


# ============================================================
# DYNAMIC EVALUATION SCHEMAS (RUNTIME QUESTIONS & MANUALS)
# ============================================================

class AssignedQuestion(BaseModel):
    """Runtime representation of an assigned question/problem statement."""
    model_config = ConfigDict(extra="ignore")
    question_number: int = Field(..., description="1-indexed question sequence number")
    question_text: str = Field(..., description="Full text or statement of the question")


class RequirementEvaluation(BaseModel):
    """Evaluation of a specific instruction manual requirement for a given question."""
    model_config = ConfigDict(extra="ignore")
    requirement: str = Field(..., description="Name of the requirement (e.g. Logic, Variables)")
    status: Literal["PRESENT", "PARTIAL", "MISSING"] = Field(
        default="MISSING",
        description="PRESENT if supported by student evidence, PARTIAL if incomplete, MISSING if absent"
    )
    score: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Marks awarded: 2 for PRESENT, 1 for PARTIAL, 0 for MISSING"
    )
    max_score: int = Field(
        default=2,
        description="Maximum marks possible for this criterion"
    )
    evidence: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Direct quote or list of verbatim excerpts from student's OCR text. None if MISSING."
    )
    evaluation: Optional[str] = Field(
        default=None,
        description="Objective assessment of how student evidence satisfies or fails the requirement"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Evaluator's explanation for why status was classified as PRESENT, PARTIAL, or MISSING"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Confidence level reflecting OCR clarity and evidence certainty"
    )
    grounded: bool = Field(
        default=True,
        description="Whether Python post-validation verified this evidence exists in the OCR text"
    )


class QuestionEvaluation(BaseModel):
    """Dynamic evaluation for a single assigned question against the student report."""
    model_config = ConfigDict(extra="ignore")
    question_number: int = Field(..., description="1-indexed question sequence number")
    question_text: str = Field(..., description="Full statement of the assigned question")
    match_status: Literal["FOUND_AND_COVERED", "FOUND_BUT_INCOMPLETE", "NOT_FOUND"] = Field(
        default="NOT_FOUND",
        description="FOUND_AND_COVERED: valid explanation present; FOUND_BUT_INCOMPLETE: title/list only; NOT_FOUND: no evidence"
    )
    match_confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Confidence level in matching student content to this assigned question"
    )
    matched_heading: Optional[str] = Field(
        default=None,
        description="The heading or label under which this question appeared in the student report"
    )
    match_evidence: Optional[List[str]] = Field(
        default=None,
        description="Grounded OCR fragments/quotes supporting the question association"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Overall confidence in matching and evidence identification"
    )
    score: int = Field(
        default=0,
        description="Total obtained marks for this question (sum of criteria)"
    )
    max_score: int = Field(
        default=8,
        description="Maximum marks possible for this question (sum of criteria max scores, e.g. 8)"
    )
    requirements: List[RequirementEvaluation] = Field(
        default_factory=list,
        description="List of requirement evaluations dynamically parsed from the Instruction Manual"
    )


class ObjectiveEvaluation(BaseModel):
    """Evaluation of overall lab objective / session-level requirements."""
    model_config = ConfigDict(extra="ignore")
    requirement: str = Field(default="Objective of the Lab", description="Requirement name")
    status: Literal["PRESENT", "PARTIAL", "MISSING"] = Field(
        default="MISSING",
        description="PRESENT if stated in student's own words; PARTIAL if copied titles; MISSING if absent"
    )
    score: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Marks awarded: 2 for PRESENT, 1 for PARTIAL, 0 for MISSING"
    )
    max_score: int = Field(
        default=2,
        description="Maximum marks possible for objective"
    )
    evidence: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Direct quote or list of verbatim excerpts from student's OCR text"
    )
    evaluation: Optional[str] = Field(
        default=None,
        description="Objective assessment on the objective statement"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Evaluator's explanation for why status was classified as PRESENT, PARTIAL, or MISSING"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Confidence level in extraction"
    )
    grounded: bool = Field(
        default=True,
        description="Whether evidence was verified against OCR text"
    )


class DynamicEvaluationResult(BaseModel):
    """
    Complete dynamic, hallucination-resistant evaluation result.
    Grounds all findings strictly in student OCR text, without invented scores or grades.
    """
    model_config = ConfigDict(extra="ignore")
    objective: Optional[ObjectiveEvaluation] = Field(
        default=None,
        description="Session-wide objective evaluation"
    )
    questions: List[QuestionEvaluation] = Field(
        default_factory=list,
        description="List of question evaluations, exactly matching runtime assigned questions"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic metrics (coverage, counts, percentages) computed in Python"
    )


class ProgramDetails(BaseModel):
    """
    Structured extraction details for a single program (e.g. P1..P10).
    All fields default to None/empty to represent absent or unreadable content without hallucinations.
    """
    model_config = ConfigDict(extra="ignore")
    status: Literal["detected", "not_detected"] = "not_detected"
    source_pages: List[int] = Field(
        default_factory=list,
        description="List of 1-indexed page numbers in original document where this program was found."
    )
    problem_understanding: Optional[str] = Field(
        default=None,
        description="Student's problem understanding in their own words. None if missing."
    )
    logic_approach: Optional[str] = Field(
        default=None,
        description="Step-by-step logic or approach. None if missing."
    )
    important_variables: List[VariableItem] = Field(
        default_factory=list,
        description="List of variables and purposes from table or text. Empty if missing."
    )
    what_i_observed: Optional[str] = Field(
        default=None,
        description="Actual observed program behavior. None if missing."
    )


class ExtractionResult(BaseModel):
    """
    Complete structured extraction for a student's observation report.
    """
    model_config = ConfigDict(extra="ignore")
    status: Literal["success", "partial", "failed"] = "success"
    objective_of_lab: Optional[str] = Field(
        default=None,
        description="Overall objective of the lab session. None if missing."
    )
    programs: Dict[str, ProgramDetails] = Field(
        default_factory=dict,
        description="Mapping from program identifier ('P1'..'P10') to ProgramDetails"
    )
    detected_programs: List[str] = Field(
        default_factory=list,
        description="List of detected program keys, e.g. ['P1', 'P2']"
    )
    missing_programs: List[str] = Field(
        default_factory=list,
        description="List of programs among P1..P10 that were not detected"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Diagnostic warnings or parsing issues encountered during extraction"
    )


class SourceMeta(BaseModel):
    """Metadata about the input document source."""
    model_config = ConfigDict(extra="ignore")
    filename: str
    num_pages: int = 1


class OcrResult(BaseModel):
    """Preserves raw OCR output and page breakdown alongside structured extraction."""
    model_config = ConfigDict(extra="ignore")
    status: str = "success"
    text: str = ""
    num_pages: int = 1
    page_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    device: str = "gpu:0"
    paddle_version: Optional[str] = None
    total_time: float = 0.0
    error: Optional[str] = None


class StudentObservationReport(BaseModel):
    """
    Canonical top-level JSON schema for a single student's successfully extracted report.
    Saved as <student_id>.json.
    """
    model_config = ConfigDict(extra="ignore")
    student_id: str
    section_id: str = "SEC1"
    week_id: str = "week-01"
    status: Literal["completed", "failed"] = "completed"
    source: SourceMeta
    ocr: OcrResult
    extraction: ExtractionResult
    evaluation: Optional[str] = None
    dynamic_evaluation: Optional[DynamicEvaluationResult] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class FailedStudentError(BaseModel):
    """Details of failure at a specific stage of processing."""
    model_config = ConfigDict(extra="ignore")
    stage: str = Field(..., description="Stage where failure occurred: discovery, ocr, extraction, validation, persistence")
    message: str = Field(..., description="Error message description")


class FailedStudentReport(BaseModel):
    """
    Persisted representation for a student whose extraction failed.
    Saved as <student_id>.json without fabricating fake observations.
    """
    model_config = ConfigDict(extra="ignore")
    student_id: str
    section_id: str
    week_id: str
    status: Literal["failed"] = "failed"
    error: FailedStudentError
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CompleteSectionReport(BaseModel):
    """
    Complete aggregated section JSON containing all student reports (both completed and failed).
    Saved as <section_id>_<week_id>_observation_reports.json.
    """
    model_config = ConfigDict(extra="ignore")
    section_id: str
    week_id: str
    total_students: int = 0
    successful: int = 0
    failed: int = 0
    pending: int = 0
    students: List[Union[StudentObservationReport, FailedStudentReport, Dict[str, Any]]] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SectionSummary(BaseModel):
    """
    Section-level summary tallying detection stats for P1..P10 calculated directly from student JSONs.
    Saved as <section_id>_<week_id>_summary.json.
    """
    model_config = ConfigDict(extra="ignore")
    section_id: str
    week_id: str
    total_students: int = 0
    successful: int = 0
    failed: int = 0
    pending: int = 0
    program_detection: Dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchStudentStatus(BaseModel):
    """Tracking entry for a student within a batch run."""
    model_config = ConfigDict(extra="allow")
    student_id: str
    section_id: Optional[str] = None
    status: Literal[
        "pending",
        "processing_ocr",
        "ocr_complete",
        "extracting",
        "evaluating",
        "completed",
        "failed"
    ] = "pending"
    stage: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    time_taken: float = 0.0
    ocr_status: Optional[str] = None
    extraction_status: Optional[str] = None
    evaluation_status: Optional[str] = None


class BatchManifest(BaseModel):
    """Batch manifest recording cohort processing statistics and individual student outputs."""
    model_config = ConfigDict(extra="ignore")
    week_id: str
    section_id: Optional[str] = None
    status: Literal["pending", "running", "completed", "partial", "failed"] = "pending"
    total_students: int = 0
    successful: int = 0
    failed: int = 0
    pending: int = 0
    students: List[BatchStudentStatus] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
