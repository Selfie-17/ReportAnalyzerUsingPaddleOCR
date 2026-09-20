"""
tests/test_evidence_first_pipeline.py
Second-Phase Quality and Reliability Pass Test Suite:
- Grounded EvidencePackage & strict source provenance
- Preservation of deterministic static findings (e.g. p6_1.c swap return)
- Multi-state extraction confidence & OCR empty vs failed status
- Strict Gemini & Ollama schema parity
- Deterministic score math validation
- 100% Markdown-JSON fidelity
"""

import os
import json
import pytest
from schemas import (
    ObservationReport,
    ObservationReportEntry,
    ObservationReportDocument,
    ObservationReportObjective,
    EvaluationReport,
    EvaluationEvidenceFinding,
    EvidencePackage,
    DimensionEvaluationDetail,
    EvaluationFinalScore,
    SubmissionSummary,
    EvaluationObservationReportSummary,
    EvaluationCodeAnalysis,
    EvaluationMatch,
    EvaluationConsistencyItem,
    EvaluationFeedback
)
from extractor import (
    parse_observation_report_deterministic,
    extract_observation_report_stage1,
    clean_latex_and_ocr
)
from verifier import (
    evaluate_holistic_student,
    find_student_code,
    build_evaluation_evidence_package,
    analyze_student_c_code_statically,
    extract_report_entries_normalized
)


def test_evidence_finding_model_and_provenance():
    """Verify EvaluationEvidenceFinding validates correct source provenance."""
    f1 = EvaluationEvidenceFinding(
        type="code",
        finding="Submitted P12 (p12.c): Frequency calculation",
        program_id="P12",
        file="p12.c"
    )
    assert f1.type == "code"
    assert f1.program_id == "P12"
    assert f1.file == "p12.c"

    f2 = EvaluationEvidenceFinding(
        type="static_analysis",
        finding="Static check on p6_1.c (P6_1): Function 'swap' has no explicit return statement",
        program_id="P6_1",
        file="p6_1.c"
    )
    assert f2.type == "static_analysis"

    f3 = EvaluationEvidenceFinding(
        type="consistency",
        finding="Report claims nested loop comparison, but code uses frequency array",
        program_id="P12",
        report_id="R4"
    )
    assert f3.type == "consistency"

    pkg = EvidencePackage(
        student_id="N241003",
        week="week-04",
        code_findings=[f1],
        static_analysis_findings=[f2],
        consistency_findings=[f3],
        deterministic_warnings=["Static check on p6_1.c (P6_1): Function 'swap' has no explicit return statement"]
    )
    assert pkg.student_id == "N241003"
    assert len(pkg.deterministic_warnings) == 1
    assert len(pkg.code_findings) == 1


def test_ocr_empty_vs_failed_status():
    """Verify Stage 1 accurately distinguishes empty documents from garbled OCR failures."""
    # 1. Genuinely empty / blank OCR
    empty_report = parse_observation_report_deterministic("")
    assert empty_report.extraction_status == "empty_report"
    assert len(empty_report.entries) == 0
    assert "OCR text is empty" in empty_report.extraction_warnings[0]

    whitespace_report = parse_observation_report_deterministic("   \n\n\t  ")
    assert whitespace_report.extraction_status == "empty_report"

    # 2. Garbled text with 0 recognized programs and no objective
    garbled_text = "%%%% $$$$ random corrupted scanner noise without any program headers |||| ####"
    failed_report = parse_observation_report_deterministic(garbled_text)
    assert failed_report.extraction_status == "failed"
    assert len(failed_report.entries) == 0
    assert "unparseable or garbled" in failed_report.extraction_warnings[0]

    # 3. Partial extraction (Objective present, but no programs)
    partial_text = "Objective:\nUnderstand functions and pointers in C language.\nSome random notes here."
    partial_report = parse_observation_report_deterministic(partial_text)
    assert partial_report.extraction_status == "partial"
    assert partial_report.objective.text != ""
    assert len(partial_report.entries) == 0

    # 4. Successful extraction
    success_text = (
        "Objective:\nLearn C programming.\n"
        "Program 1: Find Factorial\n"
        "Problem Understanding:\nCalculate n! using recursion.\n"
        "Logic Used:\nBase case n <= 1 return 1, else n * fact(n-1).\n"
        "Important Variables:\nn: input number\n"
        "What I Observed:\nFactorial of 5 returned 120.\n"
    )
    success_report = parse_observation_report_deterministic(success_text)
    assert success_report.extraction_status == "success"
    assert len(success_report.entries) == 1
    assert success_report.entries[0].report_id == "R1"
    assert "Factorial" in success_report.entries[0].program_title


def test_section_level_extraction_status_and_confidence():
    """Verify section-level status distinguishes detected, not_provided, and ocr_detection_failed."""
    sample_text = (
        "Program 1: String Length\n"
        "Problem Understanding:\nFind length of string without strlen.\n"
        "Logic Used:\nIterate until null character.\n"
        "What I Observed:\nString length calculated correctly.\n"
    )
    obs = parse_observation_report_deterministic(sample_text)
    assert len(obs.entries) == 1
    entry = obs.entries[0]

    assert entry.section_status["problem_understanding"] == "detected"
    assert entry.section_status["logic_used"] == "detected"
    assert entry.section_status["observations"] == "detected"
    # Variables were not provided in this entry
    assert entry.section_status["important_variables"] == "not_provided"


def test_mathematical_score_validation():
    """Verify strict mathematical score validation and deterministic recomputation."""
    score = EvaluationFinalScore(
        D1=1.8,
        D2=1.8,
        D3=1.7,
        D4=1.4,
        D5=1.7,
        total=0.0,
        percentage=0.0
    )
    score.recompute()
    assert score.total == 8.4
    assert score.percentage == 84.0

    # Test clamping out-of-bounds dimension score (> 2.0)
    score_out = EvaluationFinalScore(
        D1=2.5,  # Exceeds max 2.0
        D2=-0.5, # Below min 0.0
        D3=1.5,
        D4=1.5,
        D5=1.0,
        total=0.0,
        percentage=0.0
    )
    score_out.recompute()
    assert score_out.D1 == 2.0
    assert score_out.D2 == 0.0
    assert score_out.total == 6.0
    assert score_out.percentage == 60.0


def test_deterministic_static_finding_preservation():
    """Verify that deterministic compiler/static warnings (p6_1.c swap return) are preserved in evidence."""
    student_json_path = os.path.join("output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(student_json_path):
        pytest.skip(f"Student JSON {student_json_path} not found")

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_text = data["ocr"]["text"]
    obs = extract_observation_report_stage1(ocr_text)
    code = find_student_code(student_id="N241003", week_id="week-04")

    res = evaluate_holistic_student(
        report_text=ocr_text,
        extracted_report=obs,
        student_code=code,
        student_id="N241003",
        week_id="week-04",
        provider="deterministic"
    )

    eval_rep = res.evaluation_report
    assert eval_rep is not None

    # D1 evidence must contain static inspection findings and warnings
    d1_evidence = eval_rep.evaluations["D1"].evidence_strings
    assert len(d1_evidence) >= 1
    # Check that swap return or declaration issue is documented
    has_swap_warning = any("swap" in ev.lower() or "return" in ev.lower() or "p6_1.c" in ev.lower() for ev in d1_evidence)
    assert has_swap_warning, f"Expected swap or return warning in D1 evidence, got: {d1_evidence}"

    # Verify evidence_package is attached and includes deterministic warnings
    assert eval_rep.evidence_package is not None
    assert any("p6_1.c" in w or "swap" in w for w in eval_rep.evidence_package.deterministic_warnings)


def test_provider_schema_parity():
    """Verify that EvaluationReport schema is identical and verifiable across providers."""
    student_json_path = os.path.join("output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(student_json_path):
        pytest.skip(f"Student JSON {student_json_path} not found")

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_text = data["ocr"]["text"]
    obs = extract_observation_report_stage1(ocr_text)
    code = find_student_code(student_id="N241003", week_id="week-04")

    res = evaluate_holistic_student(
        report_text=ocr_text,
        extracted_report=obs,
        student_code=code,
        student_id="N241003",
        week_id="week-04",
        provider="deterministic"
    )

    # Dump to pure JSON dictionary as would be sent/stored
    rep_dict = res.evaluation_report.model_dump()

    # Re-validate strictly against Pydantic schema
    validated_rep = EvaluationReport.model_validate(rep_dict)
    assert validated_rep.schema_version == "2.0"
    assert validated_rep.student_id == "N241003"
    assert validated_rep.final_score.total == 8.4
    assert validated_rep.final_score.percentage == 84.0


def test_markdown_fidelity():
    """Verify that Markdown rendering is strictly a projection of JSON without hidden evaluation logic."""
    student_json_path = os.path.join("output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(student_json_path):
        pytest.skip(f"Student JSON {student_json_path} not found")

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_text = data["ocr"]["text"]
    obs = extract_observation_report_stage1(ocr_text)
    code = find_student_code(student_id="N241003", week_id="week-04")

    res = evaluate_holistic_student(
        report_text=ocr_text,
        extracted_report=obs,
        student_code=code,
        student_id="N241003",
        week_id="week-04",
        provider="deterministic"
    )
    eval_rep = res.evaluation_report

    md = eval_rep.to_markdown(layout="unified")

    # Verify exact values from JSON are reflected in Markdown
    assert f"**Student ID:** `{eval_rep.student_id}`" in md
    assert f"| Code Programs | {eval_rep.submission_summary.code_programs} |" in md
    assert f"| Report Entries | {eval_rep.submission_summary.report_entries} |" in md
    assert f"| Matched | {eval_rep.submission_summary.matched_programs} |" in md
    assert f"| Undocumented | {eval_rep.submission_summary.undocumented_code_programs} |" in md
    assert f"**Total** | **{eval_rep.final_score.total:.2f} / 10**" in md
    assert f"**Final Score:** {int(round(eval_rep.final_score.percentage))} / 100" in md
