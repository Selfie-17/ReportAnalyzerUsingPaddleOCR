from __future__ import annotations
"""
schemas.py
Canonical Pydantic models for structured observation report extraction,
section-wise aggregation, validation, and batch manifest generation.
"""


from typing import List, Dict, Optional, Any, Literal, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, model_validator


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


class ReportProgramEntry(BaseModel):
    """
    Structured observation report entry documented by student in OCR text.
    Uses R1, R2, R3... to designate report order without assuming official P-number.
    """
    model_config = ConfigDict(extra="ignore")
    report_program_id: str = Field(..., description="Canonical report entry id: R1, R2, ...")
    student_written_program_number: Optional[str] = Field(default=None, description="Number written by student e.g. '7' or None")
    program_title: str = Field(default="Untitled Program", description="Student-written program title")
    status: Literal["detected", "not_detected"] = "detected"
    problem_understanding: Optional[str] = Field(default=None, description="Extracted problem understanding or None")
    logic_approach: Optional[str] = Field(default=None, description="Extracted logic/approach or None")
    important_variables: List[VariableItem] = Field(default_factory=list, description="Extracted variables table")
    what_i_observed: Optional[str] = Field(default=None, description="Extracted observation or None")
    source_pages: List[int] = Field(default_factory=list, description="Source pages where program was detected")
    matched_problem_id: Optional[str] = Field(default=None, description="Matched code problem ID e.g. 'P3'")
    matched_source_file: Optional[str] = Field(default=None, description="Matched source code file name e.g. 'p3.c'")
    match_confidence: Optional[float] = Field(default=None, description="Match confidence score between 0.0 and 1.0")
    match_status: Optional[str] = Field(default=None, description="Match status: 'matched', 'needs_review', or 'unmatched'")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            if other == self.report_program_id:
                return True
            if self.student_written_program_number and (
                other == self.student_written_program_number or
                other == f"P{self.student_written_program_number}"
            ):
                return True
            if self.report_program_id.replace("R", "P") == other:
                return True
            if other == self.program_title:
                return True
            return False
        return super().__eq__(other)


class ExtractionResult(BaseModel):
    """
    Complete structured extraction for a student's observation report.
    Answers strictly 'What did the student actually write?'
    """
    model_config = ConfigDict(extra="ignore")
    status: Literal["success", "partial", "failed"] = "success"
    objective_of_lab: Optional[str] = Field(
        default=None,
        description="Overall objective of the lab session. None if missing."
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="Overall conclusion of the lab session. None if missing."
    )
    detected_programs: List[Union[ReportProgramEntry, str]] = Field(
        default_factory=list,
        description="List of detected report entries (R1, R2, ...) or legacy key strings"
    )
    programs: Dict[str, ProgramDetails] = Field(
        default_factory=dict,
        description="Legacy mapping from program identifier to ProgramDetails for backward compatibility"
    )
    missing_programs: List[str] = Field(
        default_factory=list,
        description="Legacy missing programs list"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Diagnostic warnings or parsing issues encountered during extraction"
    )

    def get_report_entries(self) -> List[ReportProgramEntry]:
        """Returns normalized list of ReportProgramEntry items."""
        import re
        entries = []
        for p in self.detected_programs:
            if isinstance(p, ReportProgramEntry):
                entries.append(p)
            elif isinstance(p, dict):
                entries.append(ReportProgramEntry(**p))
        if not entries and self.programs:
            # Fallback adapter for legacy dictionary format
            for k, pd in self.programs.items():
                if pd.status == "detected":
                    r_id = k.replace("P", "R") if k.startswith("P") else f"R{len(entries)+1}"
                    entries.append(ReportProgramEntry(
                        report_program_id=r_id,
                        student_written_program_number=re.sub(r"[^\d]", "", k) or None,
                        program_title=pd.problem_understanding[:50] if pd.problem_understanding else f"Program {k}",
                        problem_understanding=pd.problem_understanding,
                        logic_approach=pd.logic_approach,
                        important_variables=pd.important_variables,
                        what_i_observed=pd.what_i_observed,
                        source_pages=pd.source_pages
                    ))
        return entries


# Alias for explicit naming
ReportExtractionResult = ExtractionResult


# ============================================================
# STAGE 1: OBSERVATION REPORT SCHEMA (SCHEMA VERSION 1.0)
# ============================================================

class ObservationReportVariable(BaseModel):
    """Represents a single variable and its described purpose in the report."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    name: str = Field(..., alias="variable", description="Variable name mentioned in the report")
    purpose: str = Field(default="", description="Stated purpose of the variable")

    @property
    def variable(self) -> str:
        return self.name


class ObservationReportDocument(BaseModel):
    """Metadata about the observation PDF document."""
    model_config = ConfigDict(extra="ignore")
    filename: str = Field(default="observation_report.pdf", description="Name of the PDF file")
    page_count: int = Field(default=0, description="Total number of pages")


class ObservationReportObjective(BaseModel):
    """Overall laboratory objective extracted from the document."""
    model_config = ConfigDict(extra="ignore")
    text: str = Field(default="", description="Extracted objective text")
    confidence: float = Field(default=0.0, description="Confidence score 0.0 to 1.0")


class ObservationReportEntry(BaseModel):
    """Single program entry documented in the observation report."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    report_id: str = Field(..., alias="report_program_id", description="Canonical report entry id: R1, R2, ...")
    program_title: str = Field(default="Untitled Program", description="Student-written program title")
    problem_understanding: Optional[str] = Field(default=None, description="Extracted problem understanding or None")
    logic_used: Optional[str] = Field(default=None, alias="logic_approach", description="Extracted logic/approach or None")
    important_variables: List[ObservationReportVariable] = Field(default_factory=list, description="Extracted variables table")
    observations: Optional[str] = Field(default=None, alias="what_i_observed", description="Extracted observation or None")
    conclusion: Optional[str] = Field(default=None, description="Student-written conclusion for this entry or None")
    page_numbers: List[int] = Field(default_factory=list, alias="source_pages", description="Pages where program was detected")
    section_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence per section")
    section_status: Dict[str, str] = Field(default_factory=dict, description="Status per section: detected, not_provided, ocr_detection_failed, uncertain")
    extraction_status: str = Field(default="detected", description="Status: detected, uncertain, partial")

    @property
    def report_program_id(self) -> str:
        return self.report_id

    @property
    def logic_approach(self) -> Optional[str]:
        return self.logic_used

    @property
    def what_i_observed(self) -> Optional[str]:
        return self.observations

    @property
    def source_pages(self) -> List[int]:
        return self.page_numbers


class ObservationReport(BaseModel):
    """
    Stage 1 Canonical JSON: Structured Observation Report (Schema 1.0).
    Captures strictly what OCR and document extraction discovered in the handwritten PDF.
    """
    model_config = ConfigDict(extra="ignore")
    schema_version: str = Field(default="1.0", description="Schema version identifier")
    student_id: str = Field(default="", description="Student ID e.g. N241003")
    week: str = Field(default="week-04", description="Week identifier e.g. week-04")
    document: ObservationReportDocument = Field(default_factory=ObservationReportDocument)
    objective: ObservationReportObjective = Field(default_factory=ObservationReportObjective)
    entries: List[ObservationReportEntry] = Field(default_factory=list, description="List of extracted report entries")
    document_conclusion: Optional[str] = Field(default=None, description="Overall laboratory conclusion if present")
    extraction_status: Literal["success", "partial", "failed", "empty_report"] = Field(
        default="success",
        description="Extraction outcome: success (entries detected), partial (some info/objective found), failed (OCR garbled/unparseable), empty_report (blank document)"
    )
    extraction_warnings: List[str] = Field(default_factory=list, description="Warnings or issues encountered during extraction")

    @property
    def conclusion(self) -> Optional[str]:
        return self.document_conclusion

    def to_extraction_result(self) -> ExtractionResult:
        """Converts to backward-compatible ExtractionResult."""
        prog_entries: List[ReportProgramEntry] = []
        for e in self.entries:
            vars_list = [VariableItem(variable=v.name, purpose=v.purpose) for v in e.important_variables]
            prog_entries.append(ReportProgramEntry(
                report_program_id=e.report_id,
                program_title=e.program_title,
                status="detected" if e.extraction_status == "detected" else "not_detected",
                problem_understanding=e.problem_understanding,
                logic_approach=e.logic_used,
                important_variables=vars_list,
                what_i_observed=e.observations,
                source_pages=e.page_numbers
            ))
        status_val: Literal["success", "partial", "failed"] = "success" if prog_entries else ("partial" if self.objective.text else "failed")
        return ExtractionResult(
            status=status_val,
            objective_of_lab=self.objective.text or None,
            conclusion=self.document_conclusion,
            detected_programs=prog_entries,
            errors=self.extraction_warnings
        )

    @classmethod
    def from_extraction_result(
        cls,
        ext: ExtractionResult,
        student_id: str = "",
        week: str = "",
        filename: str = "observation_report.pdf",
        page_count: int = 0
    ) -> "ObservationReport":
        """Builds ObservationReport from an ExtractionResult object or dict."""
        if isinstance(ext, dict):
            ext = ExtractionResult(**ext)
        entries_list: List[ObservationReportEntry] = []
        raw_entries = ext.get_report_entries() if hasattr(ext, "get_report_entries") else []
        for r in raw_entries:
            v_items = [
                ObservationReportVariable(name=v.variable if hasattr(v, "variable") else v.get("variable", ""), purpose=v.purpose if hasattr(v, "purpose") else v.get("purpose", ""))
                for v in (r.important_variables or [])
            ]
            sec_conf = {
                "program_title": 0.95 if r.program_title else 0.0,
                "problem_understanding": 0.90 if r.problem_understanding else 0.0,
                "logic_used": 0.90 if r.logic_approach else 0.0,
                "important_variables": 0.85 if v_items else 0.0,
                "observations": 0.88 if r.what_i_observed else 0.0,
                "conclusion": 0.0
            }
            sec_status = {
                "program_title": "detected" if (r.program_title and r.program_title != "Untitled Program") else "not_provided",
                "problem_understanding": "detected" if r.problem_understanding else "not_provided",
                "logic_used": "detected" if r.logic_approach else "not_provided",
                "important_variables": "detected" if v_items else "not_provided",
                "observations": "detected" if r.what_i_observed else "not_provided",
                "conclusion": "not_provided"
            }
            entries_list.append(ObservationReportEntry(
                report_id=r.report_program_id,
                program_title=r.program_title or f"Program {r.report_program_id}",
                problem_understanding=r.problem_understanding,
                logic_used=r.logic_approach,
                important_variables=v_items,
                observations=r.what_i_observed,
                conclusion=None,
                page_numbers=r.source_pages or [],
                section_confidence=sec_conf,
                section_status=sec_status,
                extraction_status="detected"
            ))

        obj_text = ext.objective_of_lab or ""
        ext_status: Literal["success", "partial", "failed", "empty_report"] = (
            "success" if entries_list else ("partial" if obj_text else ("failed" if ext.status == "failed" else "empty_report"))
        )
        return cls(
            schema_version="1.0",
            student_id=student_id,
            week=week,
            document=ObservationReportDocument(filename=filename, page_count=page_count),
            objective=ObservationReportObjective(text=obj_text, confidence=0.95 if obj_text else 0.0),
            entries=entries_list,
            document_conclusion=ext.conclusion,
            extraction_status=ext_status,
            extraction_warnings=ext.errors or []
        )



# ============================================================
# STUDENT CODE & MULTI-FILE CODE INGESTION SCHEMAS
# ============================================================

class StudentCodeSnippet(BaseModel):
    """Represents a single student source code file or problem solution."""
    model_config = ConfigDict(extra="ignore")
    problem_number: Optional[int] = Field(default=None, description="1-indexed problem sequence number")
    problem_id: Optional[str] = Field(default=None, description="Identifier e.g. 'P1', 'p1.c', 'week-04-p1'")
    problem_title: Optional[str] = Field(default=None, description="Title of the problem")
    problem_statement: Optional[str] = Field(default=None, description="Full problem statement if available")
    source_file: Optional[str] = Field(default=None, description="Filename e.g. 'p1.c'")
    source_code: str = Field(default="", description="Complete raw source code")

    @property
    def source_filename(self) -> Optional[str]:
        return self.source_file


StudentCodeProblem = StudentCodeSnippet


class StudentCodeCollection(BaseModel):
    """Complete collection of ALL code files submitted by the student (e.g. all 17 files)."""
    model_config = ConfigDict(extra="ignore")
    student_id: Optional[str] = Field(default=None, description="Student ID e.g. 'N241003'")
    week: Optional[str] = Field(default=None, description="Week identifier e.g. 'week-04'")
    problems: List[StudentCodeSnippet] = Field(default_factory=list, description="All submitted code problems/files")


# ============================================================
# QUESTION MATCHING SCHEMAS
# ============================================================

class QuestionMatch(BaseModel):
    """Semantic pairing between a detected report entry (R1..Rn) and an official question (P1..Pn)."""
    model_config = ConfigDict(extra="ignore")
    report_program_id: str = Field(..., description="Report entry ID: R1, R2, ...")
    report_program_title: Optional[str] = Field(default=None, description="Title written in the student report")
    matched_problem_id: Optional[str] = Field(default=None, description="Matched official problem ID e.g. 'P3'")
    matched_problem_title: Optional[str] = Field(default=None, description="Title of matched official problem")
    source_file: Optional[str] = Field(default=None, description="Source code file name, e.g. p3.c")
    match_status: Literal["matched", "needs_review", "unmatched"] = Field(
        default="unmatched",
        description="'matched' (confidence >= 0.70), 'needs_review' (0.40 <= conf < 0.70), or 'unmatched'"
    )
    match_confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
    evidence: List[str] = Field(default_factory=list, description="Reasoning/evidence for the match")


# ============================================================
# UNIFIED 5-DIMENSION HOLISTIC EVALUATION SCHEMAS
# ============================================================

class EvaluationEvidence(BaseModel):
    """Evidence citation grounding an evaluation score."""
    model_config = ConfigDict(extra="ignore")
    source: Literal["report", "code", "question", "cross_source"] = Field(..., description="Evidence origin")
    problem_id: Optional[str] = Field(default=None, description="Official problem ID if applicable")
    report_program_id: Optional[str] = Field(default=None, description="Report entry ID if applicable")
    source_file: Optional[str] = Field(default=None, description="Source code file name if applicable")
    content: str = Field(..., description="Verbatim quote or objective static code observation")


class DimensionScore(BaseModel):
    """Individual score for one of the 5 evaluation dimensions (D1 to D5)."""
    model_config = ConfigDict(extra="ignore")
    dimension: str = Field(..., description="Dimension key e.g. 'D1', 'D2', 'D3', 'D4', 'D5'")
    name: str = Field(..., description="Human-readable dimension name")
    score: float = Field(default=0.0, ge=0.0, le=2.0, description="Awarded marks out of 2.0")
    max_score: float = Field(default=2.0, description="Maximum possible marks (2.0)")
    evidence: List[EvaluationEvidence] = Field(default_factory=list, description="Grounded citations supporting the score")
    justification: str = Field(default="", description="Objective evaluator justification")
    confidence: Optional[float] = Field(default=1.0, description="Evaluator confidence")
    consistency_matrix: Dict[str, str] = Field(default_factory=dict)
    components: Optional[List[Any]] = None


class DimensionList(list):
    """List of DimensionScores supporting both list iteration and attribute access (.d1..d5)."""
    @property
    def d1(self): return self._get("D1", 0)
    @property
    def d2(self): return self._get("D2", 1)
    @property
    def d3(self): return self._get("D3", 2)
    @property
    def d4(self): return self._get("D4", 3)
    @property
    def d5(self): return self._get("D5", 4)

    def _get(self, name, idx):
        for item in self:
            if getattr(item, "dimension", "").upper() == name:
                return item
        if idx < len(self):
            return self[idx]
        return None


class DimensionAccessor:
    """Provides dictionary-like and attribute-like (.d1..d5) access to EvaluationReport.evaluations."""
    def __init__(self, evaluations: Optional[Dict[str, Any]] = None):
        self._evaluations = evaluations or {}

    @property
    def d1(self): return self._evaluations.get("D1")
    @property
    def d2(self): return self._evaluations.get("D2")
    @property
    def d3(self): return self._evaluations.get("D3")
    @property
    def d4(self): return self._evaluations.get("D4")
    @property
    def d5(self): return self._evaluations.get("D5")

    def __getitem__(self, key: str):
        return self._evaluations[key]

    def __getattr__(self, name: str):
        u = name.upper()
        if u in self._evaluations:
            return self._evaluations[u]
        raise AttributeError(f"'DimensionAccessor' object has no attribute '{name}'")

    def __iter__(self):
        return iter(self._evaluations.values())

    def __len__(self):
        return len(self._evaluations)

    def get(self, key: str, default: Any = None):
        return self._evaluations.get(key, default)


class InterestingLogicItem(BaseModel):

    """Specific novelty or innovative implementation identified in student code or report for D5."""
    model_config = ConfigDict(extra="ignore")
    problem: str = Field(default="Code", description="Problem identifier e.g. 'P10'")
    problem_id: Optional[str] = Field(default=None, description="Problem identifier e.g. 'P10'")
    source_file: Optional[str] = Field(default=None, description="Source file name e.g. 'p10.c'")
    title: str = Field(..., description="Problem title")
    interesting_logic: str = Field(..., description="Specific algorithmic or structural choice")
    why_interesting: str = Field(..., description="Why this approach is technically interesting or elegant")
    evidence: str = Field(default="", description="Code or report snippet demonstrating the logic")
    presentation_potential: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Readiness and suitability for the 300-second voice presentation"
    )
    suggested_explanation: str = Field(default="", description="Recommended points for student to explain")


class CodeReportConsistencyItem(BaseModel):
    """Consistency audit between what student reported and what they actually implemented in code."""
    model_config = ConfigDict(extra="ignore")
    problem: str = Field(default="Code", description="Problem identifier e.g. 'P5'")
    problem_id: Optional[str] = Field(default=None, description="Problem identifier e.g. 'P5'")
    report_program_id: str = Field(..., description="Report entry ID e.g. 'R2'")
    claim_type: Literal["supported", "inconsistent", "unsupported"] = Field(..., description="Consistency status")
    report_claim: str = Field(..., description="Technique or behavior claimed in the report")
    code_reality: str = Field(..., description="What the static code inspection reveals")
    assessment: str = Field(..., description="Evaluator assessment of the consistency")

    @property
    def status(self) -> str:
        return self.claim_type


class PresentationReadiness(BaseModel):
    """Presentation readiness notes prepared for the student's 300-second (5-minute) presentation."""
    model_config = ConfigDict(extra="ignore")
    interesting_topics: List[str] = Field(default_factory=list, description="Key topics recommended for presentation")
    recommended_explanation_points: List[str] = Field(default_factory=list, description="Structured talking points")
    possible_faculty_questions: List[str] = Field(default_factory=list, description="Anticipated faculty Q&A questions")
    strong_areas: List[str] = Field(default_factory=list, description="Student's demonstrated strengths")
    weak_areas: List[str] = Field(default_factory=list, description="Areas needing clarification or improvement")


class HolisticEvaluationResult(BaseModel):
    """
    Complete Unified 5-Dimension Student Evaluation Result.
    Deterministic arithmetic computed in Python.
    """
    model_config = ConfigDict(extra="allow")
    student_id: Optional[str] = Field(default=None, description="Student ID")
    week_id: Optional[str] = Field(default=None, description="Week ID e.g. 'week-04'")
    week: Optional[str] = Field(default=None, description="Week ID alias e.g. 'week-04'")
    submission_summary: Optional[Any] = Field(default=None, description="Summary counts breakdown")
    dimensions: List[DimensionScore] = Field(default_factory=list, description="D1 to D5 dimension scores")
    total_score_10: float = Field(default=0.0, description="Sum of D1..D5 (out of 10.0)")
    total_score_100: float = Field(default=0.0, description="Total score scaled out of 100.0 (total_10 * 10)")
    programming_score_40: float = Field(default=0.0, description="Programming score out of 40.0 ((D1 + D2) * 10)")
    report_score_20: float = Field(default=0.0, description="Observation report score out of 20.0 (D3 * 10)")
    conceptual_score_20: float = Field(default=0.0, description="Conceptual understanding score out of 20.0 (D4 * 10)")
    novelty_score_20: float = Field(default=0.0, description="Novelty & Innovation score out of 20.0 (D5 * 10)")
    interesting_logic: List[InterestingLogicItem] = Field(default_factory=list, description="D5 interesting logic items")
    code_report_consistency: List[CodeReportConsistencyItem] = Field(default_factory=list, description="D4 consistency items")
    presentation_readiness: Optional[PresentationReadiness] = Field(default=None, description="D5 presentation readiness highlights")
    total_code_problems: int = Field(default=0, description="Total student code problems submitted")
    detected_report_entries_count: int = Field(default=0, description="Total documented report entries detected")
    matched_report_entries_count: int = Field(default=0, description="Total report entries successfully matched to code")
    review_report_entries_count: int = Field(default=0, description="Report entries requiring review")
    unmatched_report_entries_count: int = Field(default=0, description="Unmatched report entries")
    report_entries: List[ReportProgramEntry] = Field(default_factory=list, description="Documented report program entries")
    matches: List[QuestionMatch] = Field(default_factory=list, description="Question matches between report and code")
    evidence_grounding_rate: float = Field(default=100.0, description="Report evidence grounding percentage")
    strengths: List[str] = Field(default_factory=list, description="Key student strengths")
    improvement_areas: List[str] = Field(default_factory=list, description="Areas for student improvement")
    overall_summary: str = Field(default="", description="Executive summary of the evaluation")
    evaluation_notes: Dict[str, bool] = Field(default_factory=lambda: {
        "execution_performed": False,
        "testcases_executed": False,
        "online_judge_used": False,
        "execution_claims_allowed": False
    })
    evaluation_report: Optional[Union["EvaluationReport", Dict[str, Any]]] = Field(default=None, description="Canonical Stage 2 EvaluationReport")

    @property
    def total_report_entries(self) -> int:
        return self.detected_report_entries_count

    @property
    def matched_report_entries(self) -> int:
        return self.matched_report_entries_count

    @property
    def undocumented_code_problems(self) -> int:
        return max(0, self.total_code_problems - self.matched_report_entries_count)

    @model_validator(mode="after")
    def wrap_dimensions(self) -> "HolisticEvaluationResult":
        if self.dimensions and not isinstance(self.dimensions, DimensionList):
            self.dimensions = DimensionList(self.dimensions)
        if self.submission_summary is not None and not isinstance(self.submission_summary, SubmissionSummary):
            if isinstance(self.submission_summary, dict):
                raw = self.submission_summary
                self.submission_summary = SubmissionSummary(
                    code_programs=raw.get("code_programs") or raw.get("total_code_problems", self.total_code_problems),
                    report_entries=raw.get("report_entries") or raw.get("detected_report_entries", self.detected_report_entries_count),
                    matched_programs=raw.get("matched_programs") or raw.get("matched_report_entries", self.matched_report_entries_count),
                    unmatched_report_entries=raw.get("unmatched_report_entries", self.unmatched_report_entries_count),
                    undocumented_code_programs=raw.get("undocumented_code_programs") or raw.get("undocumented_code_problems", self.undocumented_code_problems)
                )
        elif self.submission_summary is None:
            self.submission_summary = SubmissionSummary(
                code_programs=self.total_code_problems,
                report_entries=self.detected_report_entries_count,
                matched_programs=self.matched_report_entries_count,
                unmatched_report_entries=self.unmatched_report_entries_count,
                undocumented_code_problems=self.undocumented_code_problems
            )
        return self


    @property
    def conflicts(self) -> List[Any]:
        if self.evaluation_report and hasattr(self.evaluation_report, "conflicts"):
            return self.evaluation_report.conflicts
        return getattr(self, "_conflicts", [])

    @conflicts.setter
    def conflicts(self, val: List[Any]):
        self._conflicts = val
        if self.evaluation_report and hasattr(self.evaluation_report, "conflicts"):
            self.evaluation_report.conflicts = val

    @property
    def grade(self) -> Optional[str]:
        if self.evaluation_report and hasattr(self.evaluation_report, "grade") and self.evaluation_report.grade:
            return self.evaluation_report.grade
        return getattr(self, "_grade", "A")

    @grade.setter
    def grade(self, val: Optional[str]):
        self._grade = val
        if self.evaluation_report and hasattr(self.evaluation_report, "grade"):
            self.evaluation_report.grade = val

    @property
    def status(self) -> Optional[str]:
        if self.evaluation_report and hasattr(self.evaluation_report, "status") and self.evaluation_report.status:
            return self.evaluation_report.status
        return getattr(self, "_status", "Approved")

    @status.setter
    def status(self, val: Optional[str]):
        self._status = val
        if self.evaluation_report and hasattr(self.evaluation_report, "status"):
            self.evaluation_report.status = val

    @property
    def final_score(self) -> Any:
        if self.evaluation_report and hasattr(self.evaluation_report, "final_score"):
            return self.evaluation_report.final_score
        d_scores = {d.dimension: d.score for d in self.dimensions}
        return EvaluationFinalScore(
            D1=d_scores.get("D1", 0.0),
            D2=d_scores.get("D2", 0.0),
            D3=d_scores.get("D3", 0.0),
            D4=d_scores.get("D4", 0.0),
            D5=d_scores.get("D5", 0.0),
            total=self.total_score_10,
            percentage=self.total_score_100
        )

    def to_markdown(self, layout: str = "unified", format_variant: Optional[str] = None) -> str:
        if self.evaluation_report and hasattr(self.evaluation_report, "to_markdown"):
            return self.evaluation_report.to_markdown(layout=format_variant or layout)
        return f"# Evaluation for {self.student_id}\nScore: {self.total_score_10:.2f} / 10 ({self.total_score_100:.1f}%)"

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Serializes the exact Unified 5-Dimension JSON structure."""
        sub_summary = self.submission_summary or {
            "total_code_problems": self.total_code_problems,
            "detected_report_entries": self.detected_report_entries_count,
            "matched_report_entries": self.matched_report_entries_count,
            "review_report_entries": self.review_report_entries_count,
            "unmatched_report_entries": self.unmatched_report_entries_count,
            "evidence_grounding_rate": round(self.evidence_grounding_rate / 100.0, 2)
        }

        rep_entries = []
        match_dict = {m.report_program_id: m for m in self.matches}
        for r in self.report_entries:
            m = match_dict.get(r.report_program_id)
            p_id = r.matched_problem_id or (m.matched_problem_id if m else None)
            s_file = r.matched_source_file or (m.source_file if m else None)
            conf = r.match_confidence if r.match_confidence is not None else (m.match_confidence if m else 0.0)
            status = r.match_status or (m.match_status if m else "unmatched")
            rep_entries.append({
                "report_program_id": r.report_program_id,
                "program_title": r.program_title,
                "source_pages": r.source_pages,
                "matched_problem_id": p_id,
                "matched_source_file": s_file,
                "match_confidence": conf,
                "match_status": status
            })

        matches_out = []
        for m in self.matches:
            matches_out.append({
                "report_program_id": m.report_program_id,
                "matched_problem_id": m.matched_problem_id,
                "matched_problem_title": m.matched_problem_title,
                "source_file": m.source_file,
                "match_status": m.match_status,
                "match_confidence": m.match_confidence,
                "evidence": m.evidence
            })

        dims_out = []
        for d in self.dimensions:
            ev_out = []
            for ev in d.evidence:
                ev_dict = {
                    "source": ev.source,
                    "content": ev.content
                }
                if ev.problem_id:
                    ev_dict["problem_id"] = ev.problem_id
                if ev.report_program_id:
                    ev_dict["report_program_id"] = ev.report_program_id
                if ev.source_file:
                    ev_dict["source_file"] = ev.source_file
                ev_out.append(ev_dict)

            dims_out.append({
                "dimension": d.dimension,
                "name": d.name,
                "score": d.score,
                "max_score": d.max_score,
                "confidence": getattr(d, "confidence", 0.90) or 0.90,
                "evidence": ev_out,
                "justification": d.justification
            })

        interesting_out = []
        for item in self.interesting_logic:
            p_id = getattr(item, "problem_id", None) or item.problem
            s_file = getattr(item, "source_file", None) or (f"{p_id.lower()}.c" if p_id.startswith("P") else None)
            interesting_out.append({
                "problem_id": p_id,
                "source_file": s_file,
                "title": item.title,
                "interesting_logic": item.interesting_logic,
                "why_interesting": item.why_interesting,
                "evidence": item.evidence,
                "presentation_potential": item.presentation_potential,
                "suggested_explanation": item.suggested_explanation
            })

        cons_out = []
        for item in self.code_report_consistency:
            p_id = getattr(item, "problem_id", None) or item.problem
            cons_out.append({
                "problem_id": p_id,
                "report_program_id": item.report_program_id,
                "claim_type": item.claim_type,
                "report_claim": item.report_claim,
                "code_reality": item.code_reality,
                "assessment": item.assessment
            })

        pres_out = {}
        if self.presentation_readiness:
            pres_out = {
                "interesting_topics": self.presentation_readiness.interesting_topics,
                "recommended_explanation_points": self.presentation_readiness.recommended_explanation_points,
                "possible_faculty_questions": self.presentation_readiness.possible_faculty_questions,
                "strong_areas": self.presentation_readiness.strong_areas,
                "weak_areas": self.presentation_readiness.weak_areas
            }

        return {
            "student_id": self.student_id or "Unknown",
            "week": self.week_id or self.week or "week-01",
            "submission_summary": sub_summary,
            "report_entries": rep_entries,
            "matches": matches_out,
            "dimensions": dims_out,
            "interesting_logic": interesting_out,
            "code_report_consistency": cons_out,
            "presentation_readiness": pres_out,
            "strengths": self.strengths,
            "improvement_areas": self.improvement_areas,
            "overall_summary": self.overall_summary,
            "evaluation_notes": self.evaluation_notes or {
                "execution_performed": False,
                "testcases_executed": False,
                "online_judge_used": False,
                "execution_claims_allowed": False
            }
        }

    def to_evaluation_report(
        self,
        static_analysis: Optional[Dict[str, Any]] = None,
        objective: Optional[str] = None,
        conclusion: Optional[str] = None,
        evidence_package: Optional[Any] = None,
        conflicts: Optional[List[Any]] = None,
        grading_policy: Optional[str] = None,
        grade: Optional[str] = None,
        status: Optional[str] = None,
        model_scores: Optional[Dict[str, Optional[float]]] = None,
        validation_metadata: Optional[Any] = None
    ) -> "EvaluationReport":
        """Converts HolisticEvaluationResult to canonical EvaluationReport (schema 2.0)."""
        from schemas import (
            EvaluationReport,
            SubmissionSummary,
            EvaluationObservationReportSummary,
            EvaluationReportEntryDetail,
            EvaluationCodeAnalysis,
            CodeAnalysisFile,
            CodeSyntaxCheck,
            CodeAlgorithmAnalysis,
            EvaluationMatch,
            DimensionEvaluationDetail,
            EvaluationConsistencyItem,
            EvaluationNotableImplementation,
            EvaluationPresentationPrep,
            EvaluationFinalScore,
            EvaluationFeedback,
            ObservationReportVariable,
            EvaluationConflict
        )

        sub_sum = SubmissionSummary(
            code_programs=self.total_code_problems,
            report_entries=self.detected_report_entries_count,
            matched_programs=self.matched_report_entries_count,
            unmatched_report_entries=self.unmatched_report_entries_count,
            undocumented_code_programs=self.undocumented_code_problems
        )

        rep_entries = []
        for r in self.report_entries:
            vars_list = []
            for v in getattr(r, "important_variables", []):
                vars_list.append(ObservationReportVariable(
                    name=getattr(v, "variable", "") or getattr(v, "name", ""),
                    purpose=getattr(v, "purpose", "")
                ))
            rep_entries.append(EvaluationReportEntryDetail(
                report_id=r.report_program_id,
                program_title=r.program_title or f"Program {r.report_program_id}",
                problem_understanding=getattr(r, "problem_understanding", None),
                logic_used=getattr(r, "logic_approach", None) or getattr(r, "logic_used", None),
                important_variables=vars_list,
                observations=getattr(r, "what_i_observed", None) or getattr(r, "observations", None),
                conclusion=getattr(r, "conclusion", None)
            ))

        obs_summary = EvaluationObservationReportSummary(
            objective=objective,
            conclusion=conclusion,
            entries=rep_entries
        )

        code_files = []
        if static_analysis and "files" in static_analysis:
            for cf in static_analysis["files"]:
                code_files.append(CodeAnalysisFile(
                    program_id=cf.get("program_id", ""),
                    file=cf.get("file", ""),
                    syntax=CodeSyntaxCheck(
                        status=cf.get("syntax", {}).get("status", "valid"),
                        issues=cf.get("syntax", {}).get("issues", [])
                    ),
                    algorithm=CodeAlgorithmAnalysis(
                        description=cf.get("algorithm", {}).get("description", ""),
                        concepts=cf.get("algorithm", {}).get("concepts", [])
                    )
                ))

        code_an = EvaluationCodeAnalysis(
            total_files=self.total_code_problems,
            files=code_files
        )

        matches_list = []
        for m in self.matches:
            matches_list.append(EvaluationMatch(
                report_id=m.report_program_id,
                program_id=m.matched_problem_id or "—",
                file=m.source_file or "—",
                status=m.match_status,
                confidence=m.match_confidence
            ))

        dim_names_map = {
            "D1": "Syntax & Code Validity",
            "D2": "Algorithmic Logic & Functional Correctness",
            "D3": "Observation Report Quality & Completeness",
            "D4": "Conceptual Understanding & Code-Report Consistency",
            "D5": "Novelty, Innovation & Presentation Readiness"
        }
        dim_dict = {}
        for d in self.dimensions:
            ev_items = []
            for e in d.evidence:
                if hasattr(e, "content"):
                    ev_items.append(e.content)
                elif hasattr(e, "finding"):
                    ev_items.append(e.finding)
                elif isinstance(e, dict):
                    ev_items.append(e.get("content") or e.get("finding") or str(e))
                else:
                    ev_items.append(str(e))

            d_components = getattr(d, "components", None)
            if not d_components and static_analysis and d.dimension.lower() == "d3" and "d3" in static_analysis and "components" in static_analysis["d3"]:
                d_components = static_analysis["d3"]["components"]

            dim_dict[d.dimension] = DimensionEvaluationDetail(
                name=d.name or dim_names_map.get(d.dimension, d.dimension),
                score=round(d.score, 2),
                max_score=round(d.max_score, 1),
                assessment=d.justification or "",
                evidence=ev_items,
                components=d_components
            )

        cons_list = []
        for c in self.code_report_consistency:
            cons_list.append(EvaluationConsistencyItem(
                report_id=c.report_program_id or "—",
                program_id=c.problem or "—",
                claim=c.report_claim,
                code_reality=c.code_reality,
                status=c.claim_type,
                assessment=c.assessment
            ))

        c_matrix = {c.report_id: c.status for c in cons_list}
        if "D4" in dim_dict:
            dim_dict["D4"].consistency_matrix = c_matrix
        if hasattr(self.dimensions, "d4") and self.dimensions.d4:
            self.dimensions.d4.consistency_matrix = c_matrix

        notable_list = []
        for it in self.interesting_logic:
            f_name = getattr(it, "source_file", None)
            if not f_name:
                p_id = it.problem.lower() if it.problem else "p"
                f_name = f"{p_id}.c"
            notable_list.append(EvaluationNotableImplementation(
                program_id=it.problem,
                file=f_name,
                interesting_logic=it.interesting_logic,
                why_notable=it.why_interesting,
                presentation_potential=(it.presentation_potential or "medium").lower() if (it.presentation_potential or "").lower() in ("low", "medium", "high") else "medium"
            ))

        pr = self.presentation_readiness
        prep = EvaluationPresentationPrep(
            recommended_topics=pr.interesting_topics if pr else [],
            likely_faculty_questions=pr.possible_faculty_questions if pr else []
        )

        d_scores = {d.dimension: d.score for d in self.dimensions}
        fin_score = EvaluationFinalScore(
            D1=round(d_scores.get("D1", 0.0), 2),
            D2=round(d_scores.get("D2", 0.0), 2),
            D3=round(d_scores.get("D3", 0.0), 2),
            D4=round(d_scores.get("D4", 0.0), 2),
            D5=round(d_scores.get("D5", 0.0), 2),
            total=round(self.total_score_10, 2),
            percentage=round(self.total_score_100, 1),
            model_scores=model_scores,
            validation_metadata=validation_metadata
        )
        fin_score.recompute()

        fb = EvaluationFeedback(
            strengths=self.strengths,
            improvements=self.improvement_areas,
            overall_assessment=self.overall_summary
        )

        conflicts_list = []
        if conflicts:
            for conf in conflicts:
                if isinstance(conf, EvaluationConflict):
                    conflicts_list.append(conf)
                elif isinstance(conf, dict):
                    conflicts_list.append(EvaluationConflict(**conf))

        return EvaluationReport(
            schema_version="2.0",
            student_id=self.student_id or "Unknown",
            week=self.week_id or self.week or "week-01",
            submission_summary=sub_sum,
            observation_report=obs_summary,
            code_analysis=code_an,
            matches=matches_list,
            evaluations=dim_dict,
            code_report_consistency=cons_list,
            notable_implementations=notable_list,
            presentation_preparation=prep,
            final_score=fin_score,
            feedback=fb,
            evidence_package=evidence_package,
            conflicts=conflicts_list,
            grading_policy=grading_policy or getattr(self, "grading_policy", None),
            grade=grade or getattr(self, "grade", None),
            status=status or getattr(self, "status", None)
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


# ============================================================
# 4-STEP PIPELINE: STEP 3 (CALCULATED SCORE) & STEP 4 (LLM JUDGE)
# ============================================================

class CalculatedScore(BaseModel):
    """
    Step 3: Algorithmic / deterministic rubric score calculation.
    Computed via rule-based heuristics, question matching, and static code check.
    Zero LLM ambiguity.
    """
    model_config = ConfigDict(extra="ignore")
    total_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Deterministic calculated score out of 10.0")
    max_score: float = Field(default=10.0, description="Max score achievable")
    grade: str = Field(default="F", description="Letter grade (O, A+, A, B, C, F)")
    status: str = Field(default="Pending", description="Status: Approved / Needs Revision")
    questions_attempted_count: int = Field(default=0, description="Number of programs/questions detected")
    questions_total_assigned: int = Field(default=0, description="Total assigned syllabus questions")
    score_display: str = Field(default="0.0 / 10.0", description="Formatted score string")
    objective: float = Field(default=0.0, description="Objective / Problem statement score (0-2)")
    problem_understanding: float = Field(default=0.0, description="Problem understanding / theory score (0-2)")
    logic_approach: float = Field(default=0.0, description="Logic, flowchart, or algorithm score (0-2)")
    variables_table: float = Field(default=0.0, description="Variables table completeness score (0-2)")
    what_i_observed: float = Field(default=0.0, description="Observations / results score (0-2)")
    code_match_bonus: float = Field(default=0.0, description="Bonus/penalty from code static verification")
    detected_programs: List[str] = Field(default_factory=list, description="Labels of detected programs")
    criteria_breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed criteria score breakdown")
    summary_feedback: str = Field(default="", description="Algorithmic rationale for the calculated score")
    calculated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class LLMJudgeEvaluation(BaseModel):
    """
    Step 4: LLM as a Judge Recommended Score and Detailed Evaluation Report.
    Generated independently by either Ollama or Gemini.
    """
    model_config = ConfigDict(extra="ignore")
    provider: Literal["ollama", "gemini"] = Field(..., description="LLM provider name: 'ollama' or 'gemini'")
    model_name: str = Field(..., description="Specific model used, e.g. qwen2.5-coder:3b or gemini-2.5-flash")
    recommended_score: float = Field(default=0.0, ge=0.0, le=10.0, description="LLM recommended score out of 10.0")
    max_score: float = Field(default=10.0, description="Max score achievable")
    score_display: str = Field(default="0.0 / 10.0", description="Formatted score display string")
    grade: str = Field(default="F", description="Recommended grade")
    status: str = Field(default="Needs Revision", description="Evaluation verdict: Approved / Needs Revision")
    criteria_scores: Dict[str, Any] = Field(default_factory=dict, description="Criteria breakdown (Objective, Logic, Variables, etc.)")
    strengths: List[str] = Field(default_factory=list, description="Key strengths identified by judge")
    recommendations: List[str] = Field(default_factory=list, description="Areas of improvement / revision guidance")
    full_report_markdown: str = Field(default="", description="Complete markdown evaluation report")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response if needed for auditing")
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    evaluation_report: Optional[Union["EvaluationReport", Dict[str, Any]]] = None


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
    holistic_evaluation: Optional[Union[HolisticEvaluationResult, Dict[str, Any]]] = None
    # 4-Step Pipeline outputs
    calculated_score: Optional[Union[CalculatedScore, Dict[str, Any]]] = None
    ollama_evaluation: Optional[Union[LLMJudgeEvaluation, Dict[str, Any]]] = None
    gemini_evaluation: Optional[Union[LLMJudgeEvaluation, Dict[str, Any]]] = None
    # Stage 1 & Stage 2 JSON-first artifacts
    observation_report: Optional[Union["ObservationReport", Dict[str, Any]]] = None
    evaluation_report: Optional[Union["EvaluationReport", Dict[str, Any]]] = None
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
        "calculating_score",
        "evaluating_ollama",
        "evaluating_gemini",
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
    # 4-Step explicit tracking
    calculated_score_status: Optional[str] = None
    ollama_evaluation_status: Optional[str] = None
    gemini_evaluation_status: Optional[str] = None
    calculated_score: Optional[float] = None
    ollama_score: Optional[float] = None
    gemini_score: Optional[float] = None


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


# ============================================================
# STAGE 2: CANONICAL EVALUATION REPORT SCHEMA (SCHEMA VERSION 2.0)
# ============================================================

class SubmissionSummary(BaseModel):
    """Summary metrics of submitted programs versus documented report entries."""
    model_config = ConfigDict(extra="ignore")
    code_programs: int = Field(default=0, description="Total submitted C source files")
    report_entries: int = Field(default=0, description="Total documented report programs")
    matched_programs: int = Field(default=0, description="Matched report-code pairs")
    unmatched_report_entries: int = Field(default=0, description="Report entries with no matching code")
    undocumented_code_programs: int = Field(default=0, description="Submitted code files without report entry")

    @property
    def matched(self) -> int:
        return self.matched_programs

    @property
    def undocumented(self) -> int:
        return self.undocumented_code_programs


class EvaluationSourceQuality(BaseModel):
    """Extraction quality and detected sections for a documented program."""
    model_config = ConfigDict(extra="ignore")
    ocr_confidence: float = Field(default=0.90)
    sections_detected: List[str] = Field(default_factory=list)
    section_status: Dict[str, str] = Field(default_factory=dict)


class EvaluationReportEntryDetail(BaseModel):
    """Documented program details preserved in the canonical evaluation."""
    model_config = ConfigDict(extra="ignore")
    report_id: str = Field(..., description="Report identifier: R1, R2, ...")
    program_title: str = Field(default="Untitled Program")
    problem_understanding: Optional[str] = None
    logic_used: Optional[str] = None
    important_variables: List[ObservationReportVariable] = Field(default_factory=list)
    observations: Optional[str] = None
    conclusion: Optional[str] = None
    source_quality: Optional[EvaluationSourceQuality] = None


class EvaluationObservationReportSummary(BaseModel):
    """Preserved observation report content in the canonical evaluation."""
    model_config = ConfigDict(extra="ignore")
    objective: Optional[str] = None
    entries: List[EvaluationReportEntryDetail] = Field(default_factory=list)


class CodeSyntaxCheck(BaseModel):
    """Static syntax verification result for a source file."""
    model_config = ConfigDict(extra="ignore")
    status: str = "valid"
    issues: List[str] = Field(default_factory=list)


class CodeAlgorithmAnalysis(BaseModel):
    """Static algorithm and conceptual analysis for a source file."""
    model_config = ConfigDict(extra="ignore")
    description: str = ""
    concepts: List[str] = Field(default_factory=list)


class CodeAnalysisFile(BaseModel):
    """Analysis for an individual submitted C source file."""
    model_config = ConfigDict(extra="ignore")
    program_id: str
    file: str
    syntax: CodeSyntaxCheck = Field(default_factory=CodeSyntaxCheck)
    algorithm: CodeAlgorithmAnalysis = Field(default_factory=CodeAlgorithmAnalysis)


class EvaluationCodeAnalysis(BaseModel):
    """Static analysis across all submitted C source files (the code universe)."""
    model_config = ConfigDict(extra="ignore")
    files: List[CodeAnalysisFile] = Field(default_factory=list)


class EvaluationMatch(BaseModel):
    """Grounding match between a report entry and a submitted source file."""
    model_config = ConfigDict(extra="ignore")
    report_id: str
    program_id: str
    file: str
    status: Literal["matched", "uncertain", "unmatched"] = "matched"
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class EvaluationEvidenceFinding(BaseModel):
    """
    Structured evidence finding with strict source provenance.
    Sources: code, observation_report, matching, consistency, static_analysis
    """
    model_config = ConfigDict(extra="ignore")
    type: Literal["code", "observation_report", "matching", "consistency", "static_analysis"] = Field(
        ..., description="Provenance source of this evidence"
    )
    finding: str = Field(..., description="Fact-backed finding or statement")
    program_id: Optional[str] = Field(default=None, description="Related code program identifier e.g. P12 or p6_1")
    file: Optional[str] = Field(default=None, description="Related source file e.g. p12.c")
    report_id: Optional[str] = Field(default=None, description="Related report entry e.g. R4")
    section: Optional[str] = Field(default=None, description="Related report section e.g. logic_used, variables")


class EvidencePackage(BaseModel):
    """
    Unified intermediate evidence package assembled prior to evaluation.
    Contains strictly grounded factual findings across all available dimensions and artifacts.
    """
    model_config = ConfigDict(extra="ignore")
    student_id: str
    week: str
    code_findings: List[EvaluationEvidenceFinding] = Field(default_factory=list)
    observation_findings: List[EvaluationEvidenceFinding] = Field(default_factory=list)
    matching_findings: List[EvaluationEvidenceFinding] = Field(default_factory=list)
    consistency_findings: List[EvaluationEvidenceFinding] = Field(default_factory=list)
    static_analysis_findings: List[EvaluationEvidenceFinding] = Field(default_factory=list)
    deterministic_warnings: List[str] = Field(default_factory=list)
    static_analysis: Optional[Dict[str, Any]] = None


class ScoreValidationItem(BaseModel):
    """Validation record for a single dimension score from an LLM."""
    model_config = ConfigDict(extra="ignore")
    dimension: str = Field(..., description="D1, D2, D3, D4, or D5")
    model_score: Optional[float] = Field(default=None, description="Original raw score proposed by LLM")
    validated_score: float = Field(..., description="Final mathematically valid clamped score [0.0, 2.0]")
    status: Literal["valid", "corrected", "invalid"] = "valid"
    validation_warning: Optional[str] = None

    @property
    def field(self) -> str:
        return self.dimension.lower()

    @property
    def original_value(self) -> Optional[float]:
        return self.model_score

    @property
    def corrected_value(self) -> float:
        return self.validated_score

    @property
    def warning(self) -> Optional[str]:
        return self.validation_warning


class ScoreValidationMetadata(BaseModel):
    """Audit trail of score validations and corrections to prevent silent masking of LLM errors."""
    model_config = ConfigDict(extra="ignore")
    status: Literal["valid", "corrected", "invalid"] = "valid"
    validations: Dict[str, ScoreValidationItem] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @property
    def overall_status(self) -> str:
        return self.status

    @property
    def items(self) -> List[ScoreValidationItem]:
        return list(self.validations.values())


class GradeBoundary(BaseModel):
    """Configurable grade boundary."""
    model_config = ConfigDict(extra="ignore")
    grade: str
    min_percentage: float
    max_percentage: Optional[float] = None
    gpa_point: Optional[float] = None
    description: Optional[str] = None


class ApprovalRule(BaseModel):
    """Configurable approval rule criteria."""
    model_config = ConfigDict(extra="ignore")
    min_total: float = 5.0
    min_percentage: float = 50.0
    min_dimension_score: Optional[float] = None
    required_dimensions: Optional[Dict[str, float]] = None


class GradingPolicy(BaseModel):
    """
    Configurable academic grading policy and approval rule engine.
    Eliminates hard-coded grade thresholds.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    name: str = Field(default="Standard Academic Scale", alias="policy_name")
    boundaries: List[GradeBoundary] = Field(default_factory=lambda: [
        GradeBoundary(grade="A+", min_percentage=90.0, max_percentage=100.0, gpa_point=10.0),
        GradeBoundary(grade="A", min_percentage=80.0, max_percentage=89.99, gpa_point=9.0),
        GradeBoundary(grade="B+", min_percentage=75.0, max_percentage=79.99, gpa_point=8.0),
        GradeBoundary(grade="B", min_percentage=70.0, max_percentage=74.99, gpa_point=7.0),
        GradeBoundary(grade="C", min_percentage=60.0, max_percentage=69.99, gpa_point=6.0),
        GradeBoundary(grade="D", min_percentage=50.0, max_percentage=59.99, gpa_point=5.0),
        GradeBoundary(grade="F", min_percentage=0.0, max_percentage=49.99, gpa_point=0.0),
    ])
    approval_rule: ApprovalRule = Field(default_factory=ApprovalRule)

    def determine_grade(self, percentage: float) -> str:
        sorted_b = sorted(self.boundaries, key=lambda b: b.min_percentage, reverse=True)
        for b in sorted_b:
            if percentage >= b.min_percentage:
                if b.max_percentage is None or percentage <= b.max_percentage:
                    return b.grade
        return "F"

    def get_grade(self, percentage: float) -> str:
        return self.determine_grade(percentage)

    def determine_approval(self, final_score: Any) -> Tuple[str, List[str]]:
        pct = getattr(final_score, "percentage", 0.0)
        tot = getattr(final_score, "total", 0.0)
        reasons: List[str] = []
        if pct < self.approval_rule.min_percentage:
            reasons.append(f"Score {pct:.1f}% is below required {self.approval_rule.min_percentage}% approval threshold.")
        if tot < self.approval_rule.min_total:
            reasons.append(f"Total score {tot:.2f} is below required minimum {self.approval_rule.min_total:.2f}.")

        if self.approval_rule.min_dimension_score is not None:
            for d_key in ("D1", "D2", "D3", "D4", "D5"):
                d_val = getattr(final_score, d_key, getattr(final_score, d_key.lower(), 0.0))
                if d_val < self.approval_rule.min_dimension_score:
                    reasons.append(f"{d_key} score ({d_val:.2f}) below required minimum {self.approval_rule.min_dimension_score:.2f}.")

        if self.approval_rule.required_dimensions:
            for d_key, req_val in self.approval_rule.required_dimensions.items():
                d_val = getattr(final_score, d_key.upper(), getattr(final_score, d_key.lower(), 0.0))
                if d_val < req_val:
                    reasons.append(f"{d_key.lower()} ({d_val:.2f}) below required minimum {req_val:.2f}.")

        status = "Approved" if not reasons else "Needs Revision"
        return status, reasons

    def check_approval(self, final_score: Any) -> Tuple[str, List[str]]:
        return self.determine_approval(final_score)


DEFAULT_GRADING_POLICY = GradingPolicy()


class D3ComponentDetail(BaseModel):
    """Field-level breakdown of D3 evaluation."""
    model_config = ConfigDict(extra="ignore")
    component: Optional[str] = "problem_understanding"
    label: str = "Problem Understanding"
    score: float = Field(default=0.0, ge=0.0, le=2.0)
    max_score: float = 2.0
    status: str = "adequate"
    comments: str = ""
    criterion: Optional[str] = None
    finding: Optional[str] = None

    @model_validator(mode="after")
    def populate_aliases(self) -> "D3ComponentDetail":
        if not self.criterion:
            self.criterion = self.label
        if not self.finding:
            self.finding = self.comments
        return self


class EvaluationConflict(BaseModel):
    """Tracking of conflicts between LLM interpretation and deterministic analysis or cross-providers."""
    model_config = ConfigDict(extra="ignore")
    type: str = "llm_vs_deterministic"
    provider: Optional[str] = None
    finding: str = Field(default="", description="LLM finding or claim")
    deterministic_finding: str = Field(default="", description="Authoritative deterministic finding")
    resolution: str = Field(default="deterministic_evidence_preferred", description="How conflict was resolved")
    details: Optional[Dict[str, Any]] = None
    item: Optional[str] = None
    resolved_by: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def populate_conflict_fields(self) -> "EvaluationConflict":
        if not self.item:
            self.item = self.finding or (self.details.get("report_id") if self.details else "")
        if not self.resolved_by:
            self.resolved_by = self.resolution
        if not self.description:
            self.description = f"{self.finding}. {self.deterministic_finding}"
        return self


class EvidenceTableLines(list):
    """List of table markdown lines that also provides .strip() and .split() for string-like handling."""
    def strip(self) -> str:
        return "\n".join(self).strip()

    def split(self, sep: str = "\n") -> List[str]:
        return "\n".join(self).split(sep)


def format_evidence_table_markdown(evidence_list: List[Any]) -> EvidenceTableLines:
    """Renders structured evidence table: | Source | Program/Report | Finding |"""
    import re
    table_lines = [
        "| Source | Program/Report | Finding |",
        "|---|---|---|"
    ]
    if not evidence_list:
        table_lines.append("| Static analysis | — | Static inspection and reasoning. |")
        return EvidenceTableLines(table_lines)

    for ev in evidence_list:
        src = "Static analysis"
        target = "—"
        finding = ""
        if isinstance(ev, EvaluationEvidenceFinding):
            if ev.type == "static_analysis":
                src = "Static analysis"
            elif ev.type == "observation_report":
                src = "Observation report"
            elif ev.type == "code":
                src = "Code"
            elif ev.type == "matching":
                src = "Matching"
            elif ev.type == "consistency":
                src = "Consistency"
            else:
                src = ev.type.replace("_", " ").title()
            target = ev.program_id or ev.report_id or (ev.file or "—")
            finding = ev.finding
        elif isinstance(ev, dict):
            s_raw = ev.get("source", "static_analysis")
            if "static" in s_raw.lower():
                src = "Static analysis"
            elif "report" in s_raw.lower():
                src = "Observation report"
            elif "code" in s_raw.lower():
                src = "Code"
            elif "cross" in s_raw.lower() or "cons" in s_raw.lower():
                src = "Consistency"
            else:
                src = s_raw.replace("_", " ").title()
            target = ev.get("problem_id") or ev.get("report_program_id") or ev.get("source_file") or ev.get("file") or "—"
            finding = ev.get("finding") or ev.get("content", "")
        else:
            ev_str = str(ev).strip()
            finding = ev_str
            if ev_str.startswith("["):
                m_br = re.match(r"^\[(.*?)\]\s*(.*)", ev_str)
                if m_br:
                    src = m_br.group(1).strip()
                    rest = m_br.group(2).strip()
                    if ":" in rest:
                        target_raw, finding = rest.split(":", 1)
                        target = f"`{target_raw.strip()}`"
                        finding = finding.strip()
                    else:
                        target = "—"
                        finding = rest
            else:
                m_file = re.search(r"\b(p\d+(?:_\d+)?\.c)\b", ev_str, re.IGNORECASE)
                m_p = re.search(r"\b(P\d+(?:_\d+)?)\b", ev_str)
                m_r = re.search(r"\b(R\d+)\b", ev_str)
                if m_file:
                    target = f"`{m_file.group(1)}`"
                    src = "Code Inspection"
                elif m_p:
                    target = m_p.group(1)
                    src = "Code Inspection"
                elif m_r:
                    target = m_r.group(1)
                    src = "Observation report"
                else:
                    target = "General"
                    src = "Evaluator Evidence"

        clean_f = str(finding).replace("|", "\\|").replace("\n", " ").strip()
        table_lines.append(f"| {src} | {target} | {clean_f} |")

    return EvidenceTableLines(table_lines)


class DimensionEvaluationDetail(BaseModel):
    """Evaluation score, assessment, and grounded evidence for one dimension."""
    model_config = ConfigDict(extra="ignore")
    name: str
    score: float = Field(default=0.0, ge=0.0, le=2.0)
    max_score: float = Field(default=2.0)
    assessment: str = ""
    evidence: List[Union[EvaluationEvidenceFinding, str]] = Field(default_factory=list)
    components: Optional[List[D3ComponentDetail]] = None
    consistency_matrix: Dict[str, str] = Field(default_factory=dict)

    @property
    def evidence_strings(self) -> List[str]:
        result = []
        for e in self.evidence:
            if isinstance(e, EvaluationEvidenceFinding):
                result.append(e.finding)
            elif isinstance(e, dict):
                result.append(e.get("finding", str(e)))
            else:
                result.append(str(e))
        return result


class EvaluationConsistencyItem(BaseModel):
    """Comparison between what the report claims and what the code actually does."""
    model_config = ConfigDict(extra="ignore")
    report_id: str
    program_id: str
    claim: str
    code_reality: str
    status: Literal["supported", "inconsistent", "partially_supported", "uncertain"] = "supported"


class EvaluationNotableImplementation(BaseModel):
    """Technically interesting implementation identified from actual code."""
    model_config = ConfigDict(extra="ignore")
    program_id: str
    file: str
    interesting_logic: str
    why_notable: str
    presentation_potential: str = "medium"


class EvaluationPresentationPrep(BaseModel):
    """Presentation preparation based on actual student code and report."""
    model_config = ConfigDict(extra="ignore")
    recommended_topics: List[str] = Field(default_factory=list)
    likely_faculty_questions: List[str] = Field(default_factory=list)


class EvaluationFinalScore(BaseModel):
    """Deterministic final marks breakdown."""
    model_config = ConfigDict(extra="ignore")
    D1: float = Field(default=0.0, ge=0.0, le=2.0)
    D2: float = Field(default=0.0, ge=0.0, le=2.0)
    D3: float = Field(default=0.0, ge=0.0, le=2.0)
    D4: float = Field(default=0.0, ge=0.0, le=2.0)
    D5: float = Field(default=0.0, ge=0.0, le=2.0)
    total: float = Field(default=0.0, ge=0.0, le=10.0)
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    model_scores: Optional[Dict[str, Optional[float]]] = None
    validation_metadata: Optional[ScoreValidationMetadata] = None

    @property
    def d1(self) -> float:
        return self.D1
    @d1.setter
    def d1(self, v: float):
        self.D1 = v

    @property
    def d2(self) -> float:
        return self.D2
    @d2.setter
    def d2(self, v: float):
        self.D2 = v

    @property
    def d3(self) -> float:
        return self.D3
    @d3.setter
    def d3(self, v: float):
        self.D3 = v

    @property
    def d4(self) -> float:
        return self.D4
    @d4.setter
    def d4(self, v: float):
        self.D4 = v

    @property
    def d5(self) -> float:
        return self.D5
    @d5.setter
    def d5(self, v: float):
        self.D5 = v

    @model_validator(mode="before")
    @classmethod
    def clamp_and_compute(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize lowercase keys to uppercase
            for k in ("d1", "d2", "d3", "d4", "d5"):
                if k in data and k.upper() not in data:
                    data[k.upper()] = data[k]

            raw_model = {}
            validations = {}
            val_warnings = []
            has_correction = False

            for k in ("D1", "D2", "D3", "D4", "D5"):
                if k in data and data[k] is not None:
                    try:
                        raw_v = float(data[k])
                        raw_model[k] = raw_v
                        clamped = round(min(2.0, max(0.0, raw_v)), 2)
                        if clamped != raw_v:
                            has_correction = True
                            if raw_v > 2.0:
                                w = f"{k} score {raw_v} exceeded maximum 2.0; corrected to {clamped}"
                            else:
                                w = f"{k} score {raw_v} was below minimum 0.0; corrected to {clamped}"
                            val_warnings.append(w)
                            validations[k] = {
                                "dimension": k,
                                "model_score": raw_v,
                                "validated_score": clamped,
                                "status": "corrected",
                                "validation_warning": w
                            }
                        else:
                            validations[k] = {
                                "dimension": k,
                                "model_score": raw_v,
                                "validated_score": clamped,
                                "status": "valid"
                            }
                        data[k] = clamped
                    except (ValueError, TypeError):
                        data[k] = 0.0
                        raw_model[k] = None
                        has_correction = True
                        validations[k] = {
                            "dimension": k,
                            "model_score": None,
                            "validated_score": 0.0,
                            "status": "invalid",
                            "validation_warning": f"{k} non-numeric value converted to 0.0"
                        }
            d_sum = sum(data.get(k, 0.0) for k in ("D1", "D2", "D3", "D4", "D5"))
            data["total"] = round(d_sum, 2)
            data["percentage"] = round(data["total"] * 10.0, 1)

            if "model_scores" not in data or data["model_scores"] is None:
                data["model_scores"] = raw_model
            if "validation_metadata" not in data or data["validation_metadata"] is None:
                data["validation_metadata"] = {
                    "status": "corrected" if has_correction else "valid",
                    "validations": validations,
                    "warnings": val_warnings
                }
        return data

    def recompute(self) -> None:
        """Deterministically ensures total = sum(D1..D5) and percentage = total * 10."""
        self.D1 = round(min(2.0, max(0.0, float(self.D1))), 2)
        self.D2 = round(min(2.0, max(0.0, float(self.D2))), 2)
        self.D3 = round(min(2.0, max(0.0, float(self.D3))), 2)
        self.D4 = round(min(2.0, max(0.0, float(self.D4))), 2)
        self.D5 = round(min(2.0, max(0.0, float(self.D5))), 2)
        self.total = round(self.D1 + self.D2 + self.D3 + self.D4 + self.D5, 2)
        self.percentage = round(self.total * 10.0, 1)


class EvaluationFeedback(BaseModel):
    """Actionable strengths and improvements."""
    model_config = ConfigDict(extra="ignore")
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    overall_assessment: Optional[str] = Field(default=None, description="Executive overall assessment summary")


class EvaluationReport(BaseModel):
    """
    Stage 2 Canonical JSON: Unified Evaluation Report (Schema Version 2.0).
    Source of truth for all grading, code analysis, matching, and UI rendering.
    """
    model_config = ConfigDict(extra="ignore")
    schema_version: str = Field(default="2.0", description="Schema version identifier")
    student_id: str
    week: str = "week-04"
    submission_summary: SubmissionSummary
    observation_report: EvaluationObservationReportSummary
    code_analysis: EvaluationCodeAnalysis
    matches: List[EvaluationMatch] = Field(default_factory=list)
    evaluations: Dict[str, DimensionEvaluationDetail] = Field(default_factory=dict)
    code_report_consistency: List[EvaluationConsistencyItem] = Field(default_factory=list)
    notable_implementations: List[EvaluationNotableImplementation] = Field(default_factory=list)
    presentation_preparation: EvaluationPresentationPrep = Field(default_factory=EvaluationPresentationPrep)
    final_score: EvaluationFinalScore
    feedback: EvaluationFeedback
    evidence_package: Optional[EvidencePackage] = None
    conflicts: List[EvaluationConflict] = Field(default_factory=list)
    grading_policy: Optional[str] = None
    grade: Optional[str] = None
    status: Optional[str] = None

    @property
    def dimensions(self) -> DimensionAccessor:
        return DimensionAccessor(self.evaluations)

    def to_markdown(self, layout: str = "unified", format_variant: Optional[str] = None) -> str:
        """
        Renders the authoritative Markdown report generated strictly from this canonical JSON.
        - layout='unified': The standard 8-section format matching unified academic evaluation.
        - layout='detailed': The 11-section format including deep report and code breakdown.
        """
        if format_variant:
            layout = format_variant
        if layout == "unified":
            lines = [
                "# Unified Student Evaluation\n",
                f"**Student ID:** `{self.student_id}` | **Week:** `{self.week}`\n",
                "## 1. Submission Summary\n",
                "| Metric | Value |",
                "|---|---:|",
                f"| Code Programs | {self.submission_summary.code_programs} |",
                f"| Report Entries | {self.submission_summary.report_entries} |",
                f"| Matched | {self.submission_summary.matched_programs} |",
                f"| Undocumented | {self.submission_summary.undocumented_code_programs} |\n",
                "## 2. Reported Programs\n"
            ]
            if not self.matches and not self.observation_report.entries:
                lines.append("*No observation report entries were documented in this submission (0 report entries detected).*\n")
            else:
                lines.append("| Report | Code | File | Program | Status |")
                lines.append("|---|---|---|---|---|")
                entry_map = {e.report_id: e for e in self.observation_report.entries}
                match_map = {m.report_id: m for m in self.matches}
                all_rids = list(dict.fromkeys(list(entry_map.keys()) + list(match_map.keys())))
                for rid in all_rids:
                    e = entry_map.get(rid)
                    m = match_map.get(rid)
                    p_id = m.program_id if m and m.program_id != "—" else "—"
                    f_name = m.file if m and m.file != "—" else "—"
                    p_title = e.program_title if e else (m.file if m else "Untitled")
                    status_str = m.status.capitalize() if m and m.status else "Matched"
                    lines.append(f"| {rid} | {p_id} | `{f_name}` | {p_title} | {status_str} |")
                lines.append("")

            # 3. Evaluation
            lines.append("## 3. Evaluation\n")
            dim_order = ["D1", "D2", "D3", "D4", "D5"]
            for d_key in dim_order:
                d_item = self.evaluations.get(d_key)
                if not d_item:
                    continue
                lines.append(f"### {d_key} — {d_item.name}\n")
                lines.append(f"**Score:** {d_item.score:.2f} / {int(d_item.max_score)}\n")
                lines.append("**Assessment**\n")
                lines.append(f"{d_item.assessment}\n")
                lines.append("**Evidence**\n")
                lines.extend(format_evidence_table_markdown(d_item.evidence))
                lines.append("")

                if d_item.components:
                    lines.append("**Component Breakdown**\n")
                    lines.append("| Component | Status | Score | Remarks |")
                    lines.append("|---|---|---:|---|")
                    for comp in d_item.components:
                        lines.append(f"| {comp.label} | {comp.status.capitalize()} | {comp.score:.2f} / {comp.max_score:.1f} | {comp.comments} |")
                    lines.append("")

            # 4. Code–Report Consistency
            lines.append("## 4. Code–Report Consistency\n")
            if not self.code_report_consistency:
                lines.append("*No cross-source claims requiring verification were identified.*\n")
            else:
                lines.append("| Report | Code | Claim | Code Reality | Status |")
                lines.append("|---|---|---|---|---|")
                for c in self.code_report_consistency:
                    st_icon = "✓ Supported" if c.status == "supported" else ("⚠️ Inconsistent" if c.status == "inconsistent" else c.status.capitalize())
                    lines.append(f"| {c.report_id} | {c.program_id} | {c.claim} | {c.code_reality} | {st_icon} |")
                lines.append("")

            # 5. Notable Implementations
            lines.append("## 5. Notable Implementations\n")
            if not self.notable_implementations:
                lines.append("*No non-standard implementations identified during static inspection.*\n")
            else:
                total_cand = len(self.notable_implementations)
                top_items = self.notable_implementations[:4]
                if total_cand > len(top_items):
                    lines.append(f"*{total_cand} candidates identified; top {len(top_items)} selected for presentation relevance.*\n")
                for ni in top_items:
                    lines.append(f"### {ni.program_id}\n")
                    lines.append(f"**File:** `{ni.file}`\n")
                    lines.append(f"**Interesting Logic:** {ni.interesting_logic}\n")
                    lines.append(f"**Why it is interesting:** {ni.why_notable}\n")
                    lines.append(f"**Presentation Potential:** {ni.presentation_potential.upper()}\n")

            # 6. Presentation Preparation
            lines.append("## 6. Presentation Preparation\n")
            if self.presentation_preparation.recommended_topics:
                lines.append("### Recommended Topics\n")
                for t in self.presentation_preparation.recommended_topics[:4]:
                    lines.append(f"- {t}")
                lines.append("")
            if self.presentation_preparation.likely_faculty_questions:
                lines.append("### Likely Faculty Questions\n")
                for idx, q in enumerate(self.presentation_preparation.likely_faculty_questions[:5], 1):
                    lines.append(f"{idx}. {q}")
                lines.append("")

            # 7. Final Score
            lines.append("## 7. Final Score\n")
            lines.append("| Dimension | Score | Weight |")
            lines.append("|---|---:|---:|")
            lines.append(f"| D1 | {self.final_score.D1:.2f} / 2 | 20 |")
            lines.append(f"| D2 | {self.final_score.D2:.2f} / 2 | 20 |")
            lines.append(f"| D3 | {self.final_score.D3:.2f} / 2 | 20 |")
            lines.append(f"| D4 | {self.final_score.D4:.2f} / 2 | 20 |")
            lines.append(f"| D5 | {self.final_score.D5:.2f} / 2 | 20 |")
            lines.append(f"| **Total** | **{self.final_score.total:.2f} / 10** | **100** |\n")
            lines.append(f"**Score:** {self.final_score.total:.2f} / 10 ({self.final_score.percentage:.1f}%)\n")
            lines.append(f"**Final Score:** {int(round(self.final_score.percentage))} / 100\n")
            if self.grade:
                policy_str = f" ({self.grading_policy})" if self.grading_policy else ""
                lines.append(f"**Grade:** `{self.grade}`{policy_str}\n")
            if self.status:
                lines.append(f"**Approval Status:** `{self.status}`\n")
            if self.final_score.validation_metadata and self.final_score.validation_metadata.status == "corrected":
                lines.append(f"*Validation Note:* Scores were audited and normalized ({len(self.final_score.validation_metadata.warnings)} correction(s) applied).\n")

            # 8. Feedback
            lines.append("## 8. Feedback\n")
            lines.append("### Strengths\n")
            for s in self.feedback.strengths:
                lines.append(f"- {s}")
            lines.append("")
            lines.append("### Improvements\n")
            for imp in self.feedback.improvements:
                lines.append(f"- {imp}")
            lines.append("")
            if self.feedback.overall_assessment:
                lines.append("### Overall Assessment\n")
                lines.append(f"{self.feedback.overall_assessment}\n")

            # 9. Optional Conflicts Section
            if self.conflicts:
                lines.append("## 9. Cross-Source & Model Conflicts\n")
                lines.append("| Type | Provider | LLM Finding | Deterministic Finding | Resolution |")
                lines.append("|---|---|---|---|---|")
                for conf in self.conflicts:
                    prov = conf.provider or "—"
                    cf = conf.finding.replace("|", "\\|")
                    cdf = conf.deterministic_finding.replace("|", "\\|")
                    cres = conf.resolution.replace("|", "\\|")
                    lines.append(f"| {conf.type} | {prov} | {cf} | {cdf} | {cres} |")
                lines.append("")

            return "\n".join(lines)

        # Detailed 11-section layout
        lines = [
            "# Unified Student Evaluation\n",
            f"**Student ID:** `{self.student_id}` | **Week:** `{self.week}`\n",
            "## 1. Submission Summary\n",
            "| Metric | Value |",
            "|---|---:|",
            f"| Code Programs | {self.submission_summary.code_programs} |",
            f"| Report Entries | {self.submission_summary.report_entries} |",
            f"| Matched | {self.submission_summary.matched_programs} |",
            f"| Unmatched Report Entries | {self.submission_summary.unmatched_report_entries} |",
            f"| Undocumented Code Programs | {self.submission_summary.undocumented_code_programs} |\n",
            "## 2. Observation Report Summary\n"
        ]

        if self.observation_report.objective:
            lines.extend([
                "### Objective\n",
                f"{self.observation_report.objective}\n"
            ])
        else:
            lines.append("*No laboratory objective was documented in the report.*\n")

        lines.append("## 3. Documented Programs\n")
        if not self.observation_report.entries:
            lines.append("*No observation report entries were documented in this submission (0 report entries detected).*\n")
        else:
            for idx, entry in enumerate(self.observation_report.entries):
                lines.append(f"### {entry.report_id} — {entry.program_title}\n")
                lines.append("#### Problem Understanding\n")
                lines.append(f"{entry.problem_understanding or 'Not documented.'}\n")
                lines.append("#### Logic Used\n")
                lines.append(f"{entry.logic_used or 'Not documented.'}\n")
                lines.append("#### Important Variables\n")
                if entry.important_variables:
                    lines.append("| Variable | Purpose |")
                    lines.append("|---|---|")
                    for v in entry.important_variables:
                        lines.append(f"| `{v.name}` | {v.purpose or 'Not specified'} |")
                    lines.append("")
                else:
                    lines.append("*No variables documented for this program.*\n")
                lines.append("#### What I Observed\n")
                lines.append(f"{entry.observations or 'Not documented.'}\n")
                if entry.conclusion:
                    lines.append("#### Conclusion\n")
                    lines.append(f"{entry.conclusion}\n")
                if idx < len(self.observation_report.entries) - 1:
                    lines.append("---\n")

        # 4. Code Submission Analysis
        lines.append("## 4. Code Submission Analysis\n")
        total_code = self.submission_summary.code_programs
        lines.append(f"{total_code} C source file(s) were submitted and statically inspected.\n")
        if self.code_analysis.files:
            lines.append("| Program | File | Syntax | Concepts / Algorithm |")
            lines.append("|---|---|---|---|")
            for cf in self.code_analysis.files:
                syn_str = cf.syntax.status.capitalize() if cf.syntax.status else "Valid"
                if cf.syntax.issues:
                    syn_str += f" ({len(cf.syntax.issues)} finding{'s' if len(cf.syntax.issues)>1 else ''})"
                concepts_str = ", ".join(cf.algorithm.concepts) if cf.algorithm.concepts else (cf.algorithm.description[:40] if cf.algorithm.description else "Standard logic")
                lines.append(f"| {cf.program_id} | `{cf.file}` | {syn_str} | {concepts_str} |")
            lines.append("")

        # 5. Code–Report Matching
        lines.append("## 5. Code–Report Matching\n")
        if not self.matches:
            lines.append("*No cross-source matches identified between report and code files.*\n")
        else:
            lines.append("| Report | Code | File | Status | Confidence |")
            lines.append("|---|---|---|---|---:|")
            for m in self.matches:
                conf_pct = f"{int(round(m.confidence * 100))}%" if m.confidence else "—"
                lines.append(f"| {m.report_id} | {m.program_id} | `{m.file}` | {m.status.capitalize()} | {conf_pct} |")
            lines.append("")

        # 6. Evaluation (D1..D5)
        lines.append("## 6. Evaluation\n")
        dim_order = ["D1", "D2", "D3", "D4", "D5"]
        for d_key in dim_order:
            d_item = self.evaluations.get(d_key)
            if not d_item:
                continue
            lines.append(f"### {d_key} — {d_item.name}\n")
            lines.append(f"**Score:** {d_item.score:.2f} / {d_item.max_score:.1f}\n")
            lines.append("**Assessment**\n")
            lines.append(f"{d_item.assessment}\n")
            lines.append("**Evidence**\n")
            lines.extend(format_evidence_table_markdown(d_item.evidence))
            lines.append("")

            if d_item.components:
                lines.append("**Component Breakdown**\n")
                lines.append("| Component | Status | Score | Remarks |")
                lines.append("|---|---|---:|---|")
                for comp in d_item.components:
                    lines.append(f"| {comp.label} | {comp.status.capitalize()} | {comp.score:.2f} / {comp.max_score:.1f} | {comp.comments} |")
                lines.append("")

        # 7. Code–Report Consistency
        lines.append("## 7. Code–Report Consistency\n")
        if not self.code_report_consistency:
            lines.append("*No cross-source claims requiring verification were identified.*\n")
        else:
            lines.append("| Report | Code | Report Claim | Code Reality | Status |")
            lines.append("|---|---|---|---|---|")
            for c in self.code_report_consistency:
                st_icon = "✓ Supported" if c.status == "supported" else ("⚠️ Inconsistent" if c.status == "inconsistent" else c.status.capitalize())
                lines.append(f"| {c.report_id} | {c.program_id} | {c.claim} | {c.code_reality} | {st_icon} |")
            lines.append("")

        # 8. Notable Implementations
        lines.append("## 8. Notable Implementations\n")
        if not self.notable_implementations:
            lines.append("*No non-standard implementations identified during static inspection.*\n")
        else:
            for ni in self.notable_implementations:
                lines.append(f"### {ni.program_id}\n")
                lines.append(f"**File:** `{ni.file}`\n")
                lines.append(f"**Interesting Logic:** {ni.interesting_logic}\n")
                lines.append(f"**Why Notable:** {ni.why_notable}\n")
                lines.append(f"**Presentation Potential:** {ni.presentation_potential.upper()}\n")

        # 9. Presentation Preparation
        lines.append("## 9. Presentation Preparation\n")
        if self.presentation_preparation.recommended_topics:
            lines.append("### Recommended Topics\n")
            for t in self.presentation_preparation.recommended_topics:
                lines.append(f"- {t}")
            lines.append("")
        if self.presentation_preparation.likely_faculty_questions:
            lines.append("### Likely Faculty Questions\n")
            for idx, q in enumerate(self.presentation_preparation.likely_faculty_questions, 1):
                lines.append(f"{idx}. {q}")
            lines.append("")

        # 10. Final Score
        lines.append("## 10. Final Score\n")
        lines.append("| Dimension | Score | Weight |")
        lines.append("|---|---:|---:|")
        lines.append(f"| D1 | {self.final_score.D1:.2f} / 2 | 20 |")
        lines.append(f"| D2 | {self.final_score.D2:.2f} / 2 | 20 |")
        lines.append(f"| D3 | {self.final_score.D3:.2f} / 2 | 20 |")
        lines.append(f"| D4 | {self.final_score.D4:.2f} / 2 | 20 |")
        lines.append(f"| D5 | {self.final_score.D5:.2f} / 2 | 20 |")
        lines.append(f"| **Total** | **{self.final_score.total:.2f} / 10** | **100** |\n")
        lines.append(f"**Final Score:** {int(round(self.final_score.percentage))} / 100\n")
        if self.grade:
            policy_str = f" ({self.grading_policy})" if self.grading_policy else ""
            lines.append(f"**Grade:** {self.grade}{policy_str}\n")
        if self.status:
            lines.append(f"**Approval Status:** {self.status}\n")
        if self.final_score.validation_metadata and self.final_score.validation_metadata.status == "corrected":
            lines.append(f"*Validation Note:* Scores were audited and normalized ({len(self.final_score.validation_metadata.warnings)} correction(s) applied).\n")

        # 11. Feedback
        lines.append("## 11. Feedback\n")
        lines.append("### Strengths\n")
        for s in self.feedback.strengths:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("### Improvements\n")
        for imp in self.feedback.improvements:
            lines.append(f"- {imp}")
        lines.append("")

        # 12. Optional Conflicts Section
        if self.conflicts:
            lines.append("## 12. Cross-Source & Model Conflicts\n")
            lines.append("| Type | Provider | LLM Finding | Deterministic Finding | Resolution |")
            lines.append("|---|---|---|---|---|")
            for conf in self.conflicts:
                prov = conf.provider or "—"
                cf = conf.finding.replace("|", "\\|")
                cdf = conf.deterministic_finding.replace("|", "\\|")
                cres = conf.resolution.replace("|", "\\|")
                lines.append(f"| {conf.type} | {prov} | {cf} | {cdf} | {cres} |")
            lines.append("")

        return "\n".join(lines)

