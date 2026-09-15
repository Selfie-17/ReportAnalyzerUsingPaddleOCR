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
    model_config = ConfigDict(extra="ignore")
    student_id: Optional[str] = Field(default=None, description="Student ID")
    week_id: Optional[str] = Field(default=None, description="Week ID e.g. 'week-04'")
    week: Optional[str] = Field(default=None, description="Week ID alias e.g. 'week-04'")
    submission_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary counts breakdown")
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

    @property
    def total_report_entries(self) -> int:
        return self.detected_report_entries_count

    @property
    def matched_report_entries(self) -> int:
        return self.matched_report_entries_count

    @property
    def undocumented_code_problems(self) -> int:
        return max(0, self.total_code_problems - self.matched_report_entries_count)

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
    holistic_evaluation: Optional[Union[HolisticEvaluationResult, Dict[str, Any]]] = None
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
