import json
import pytest
from pathlib import Path
from schemas import (
    EvaluationReport,
    EvaluationFinalScore,
    ScoreValidationItem,
    ScoreValidationMetadata,
    GradingPolicy,
    GradeBoundary,
    ApprovalRule,
    DEFAULT_GRADING_POLICY,
    D3ComponentDetail,
    EvaluationConflict,
    format_evidence_table_markdown,
    ObservationReport,
    HolisticEvaluationResult,
)
from verifier import (
    evaluate_holistic_student,
    compare_evaluation_reports,
    analyze_student_c_code_statically,
    build_evaluation_evidence_package,
)


def test_score_validation_audit_trail_out_of_bounds():
    """Verify that out-of-bounds scores are corrected with an audit trail, not silently clamped."""
    raw_data = {
        "d1": 2.8,  # > 2.0 max
        "d2": -0.5, # < 0.0 min
        "d3": 1.7,
        "d4": 1.4,
        "d5": 1.7,
    }
    score_obj = EvaluationFinalScore(**raw_data)
    assert score_obj.d1 == 2.0
    assert score_obj.d2 == 0.0
    assert score_obj.d3 == 1.7
    assert score_obj.total == 6.8
    assert score_obj.percentage == 68.0
    assert score_obj.validation_metadata is not None
    assert score_obj.validation_metadata.overall_status == "corrected"
    assert len(score_obj.validation_metadata.items) >= 2

    # Check d1 record
    d1_item = next(item for item in score_obj.validation_metadata.items if item.field == "d1")
    assert d1_item.status == "corrected"
    assert d1_item.original_value == 2.8
    assert d1_item.corrected_value == 2.0
    assert "exceeded maximum" in d1_item.warning

    # Check d2 record
    d2_item = next(item for item in score_obj.validation_metadata.items if item.field == "d2")
    assert d2_item.status == "corrected"
    assert d2_item.original_value == -0.5
    assert d2_item.corrected_value == 0.0
    assert "below minimum" in d2_item.warning


def test_configurable_grading_policy():
    """Verify customizable grading policies and approval rules."""
    policy = GradingPolicy(
        policy_name="Strict Honors",
        boundaries=[
            GradeBoundary(grade="A+", min_percentage=90.0),
            GradeBoundary(grade="A", min_percentage=85.0),
            GradeBoundary(grade="B", min_percentage=75.0),
            GradeBoundary(grade="F", min_percentage=0.0),
        ],
        approval_rule=ApprovalRule(
            min_percentage=75.0,
            required_dimensions={"d1": 1.5, "d4": 1.0}
        )
    )

    # 84% gets B under Strict Honors
    grade_b = policy.get_grade(84.0)
    assert grade_b == "B"

    # Check approval with passing score
    scores_pass = EvaluationFinalScore(d1=1.8, d2=1.8, d3=1.7, d4=1.4, d5=1.7)
    app_status, reasons = policy.check_approval(scores_pass)
    assert app_status == "Approved"
    assert len(reasons) == 0

    # Check approval failing d1 requirement
    scores_fail = EvaluationFinalScore(d1=1.2, d2=1.8, d3=1.7, d4=1.4, d5=1.7)
    app_status2, reasons2 = policy.check_approval(scores_fail)
    assert app_status2 == "Needs Revision"
    assert any("d1 (1.20) below required minimum 1.50" in r for r in reasons2)


def test_field_level_d3_components_and_undocumented_independence():
    """Verify D3 evaluates 6 criteria and does not penalize undocumented programs."""
    evidence = build_evaluation_evidence_package(
        student_id="N241003",
        week="week-04",
        c_files_dir="extracted/N241003",
        obs_json_path="output/sections/week-04/SEC2/students/N241003.json"
    )
    analysis = analyze_student_c_code_statically(evidence)
    d3_data = analysis["d3"]
    assert "components" in d3_data
    components = d3_data["components"]
    assert len(components) == 6

    criterion_names = [c["criterion"] for c in components]
    expected_criteria = [
        "Problem Understanding",
        "Logic Used",
        "Important Variables",
        "Observations",
        "Technical Accuracy",
        "Completeness"
    ]
    for exp in expected_criteria:
        assert exp in criterion_names

    # Ensure all components have non-empty findings and appropriate status
    for c in components:
        assert c["status"] in ("Satisfactory", "Good", "Needs Improvement")
        assert len(c["finding"]) > 0

    # Check evidence note indicates evaluation is grounded in reported programs
    assert "Evaluation grounded strictly across reported programs" in d3_data["assessment"]


def test_strict_d4_consistency_and_conflict_logging():
    """Verify deterministic static analysis overrides LLM claims and logs conflict."""
    evidence = build_evaluation_evidence_package(
        student_id="N241003",
        week="week-04",
        c_files_dir="extracted/N241003",
        obs_json_path="output/sections/week-04/SEC2/students/N241003.json"
    )
    
    # Mock LLM result claiming R4/P12 is "supported" and d1 score is out-of-bounds (2.5)
    mock_llm_result = HolisticEvaluationResult(
        d1=2.5,
        d1_assessment="Looks fine",
        d1_evidence=["Checked static code"],
        d2=1.8,
        d2_assessment="Good",
        d2_evidence=["Checked formatting"],
        d3=1.7,
        d3_assessment="Comprehensive report",
        d3_evidence=["Checked writeup"],
        d4=1.9,
        d4_assessment="Reported logic claims frequency array and code matches",
        d4_evidence=["All programs supported"],
        d4_consistency={"R1": "supported", "R2": "supported", "R3": "supported", "R4": "supported"},
        d5=1.7,
        d5_assessment="Good progress",
        d5_evidence=["17 programs"],
        overall_summary="Solid work",
        recommendations=["Keep going"],
        strengths=["Good code"],
        areas_for_improvement=["None"]
    )

    report = evaluate_holistic_student(
        student_id="N241003",
        week="week-04",
        evidence_package=evidence,
        llm_result=mock_llm_result,
        evaluator_model="test-llm",
        prompt_version="3.0"
    )

    # 1. Strict D4 consistency check: R4 must be inconsistent despite LLM claim
    assert report.dimensions.d4.consistency_matrix["R4"] == "inconsistent"
    assert report.dimensions.d4.score == 1.4

    # 2. Conflict must be logged
    assert report.conflicts is not None
    assert len(report.conflicts) >= 1
    d4_conflict = next(c for c in report.conflicts if "R4" in c.item)
    assert d4_conflict.type == "llm_vs_deterministic"
    assert d4_conflict.resolved_by == "deterministic_precedence"
    assert "frequency array" in d4_conflict.description.lower()

    # 3. Score validation audit trail check: d1=2.5 clamped to 2.0 with metadata
    assert report.final_score.d1 == 2.0
    assert report.final_score.validation_metadata is not None
    d1_val = next(item for item in report.final_score.validation_metadata.items if item.field == "d1")
    assert d1_val.status == "corrected"
    assert d1_val.original_value == 2.5
    assert d1_val.corrected_value == 2.0

    # 4. Grading policy: 8.6/10 -> 86.0% -> Grade A under default policy
    assert report.final_score.total == 8.6
    assert report.final_score.percentage == 86.0
    assert report.grade == "A"
    assert report.status == "Approved"


def test_markdown_evidence_table_formatting():
    """Verify that D1-D5 markdown sections render 3-column evidence tables."""
    evidence_items = [
        "[Static Analyzer] p3.c: Clean function declaration with standard headers",
        "[Code Inspection] p12.c: Uses nested loops to count occurrences",
        "Plain evidence text without brackets"
    ]
    table = format_evidence_table_markdown(evidence_items)
    lines = table.strip().split("\n")
    assert lines[0] == "| Source | Program/Report | Finding |"
    assert lines[1] == "|---|---|---|"
    assert "| Static Analyzer | `p3.c` | Clean function declaration with standard headers |" in lines[2]
    assert "| Code Inspection | `p12.c` | Uses nested loops to count occurrences |" in lines[3]
    assert "| Evaluator Evidence | General | Plain evidence text without brackets |" in lines[4]


def test_provider_comparison_utility():
    """Verify compare_evaluation_reports properly checks parity, score differences, and conflicts."""
    evidence = build_evaluation_evidence_package(
        student_id="N241003",
        week="week-04",
        c_files_dir="extracted/N241003",
        obs_json_path="output/sections/week-04/SEC2/students/N241003.json"
    )
    
    gem_file = Path("output/sections/week-04/SEC2/students/N241003_gemini_evaluation.json")
    oll_file = Path("output/sections/week-04/SEC2/students/N241003_ollama_evaluation.json")
    if gem_file.exists() and oll_file.exists():
        with open(gem_file, "r", encoding="utf-8") as f:
            rep_a = json.load(f)
        with open(oll_file, "r", encoding="utf-8") as f:
            rep_b = json.load(f)
    else:
        rep_a = evaluate_holistic_student(
            student_id="N241003",
            week="week-04",
            evidence_package=evidence,
            evaluator_model="deterministic-gemini",
            prompt_version="3.0"
        )
        rep_b = evaluate_holistic_student(
            student_id="N241003",
            week="week-04",
            evidence_package=evidence,
            evaluator_model="deterministic-ollama",
            prompt_version="3.0"
        )


    comparison = compare_evaluation_reports(rep_a, rep_b, "Gemini", "Ollama")
    assert comparison["schema_parity"] is True
    assert comparison["submission_counts_match"] is True
    assert comparison["code_programs_match"] is True
    assert comparison["observation_entries_match"] is True
    assert comparison["matched_programs_match"] is True
    assert comparison["undocumented_programs_match"] is True
    assert comparison["consistency_matrix_parity"] is True
    assert comparison["scores"]["score_delta"] == 0.0
    assert comparison["total_conflicts"]["Gemini"] == 0
    assert comparison["total_conflicts"]["Ollama"] == 0


def test_golden_regression_n241003():
    """
    Semantic golden regression test for N241003 Week-04:
    - 17 code programs
    - 4 observation report entries
    - 4 matched programs
    - 13 undocumented programs
    - R4/P12 inconsistency detected
    - D1=1.8, D2=1.8, D3=1.7, D4=1.4, D5=1.7, Total=8.4/10, Percentage=84.0
    - Grade A
    """
    evidence = build_evaluation_evidence_package(
        student_id="N241003",
        week="week-04",
        c_files_dir="extracted/N241003",
        obs_json_path="output/sections/week-04/SEC2/students/N241003.json"
    )
    report = evaluate_holistic_student(
        student_id="N241003",
        week="week-04",
        evidence_package=evidence,
        evaluator_model="deterministic-golden",
        prompt_version="3.0"
    )

    # 1. Submission summary
    assert report.submission_summary.code_programs == 17
    assert report.submission_summary.report_entries == 4
    assert report.submission_summary.matched == 4
    assert report.submission_summary.undocumented == 13

    # 2. Dimensions & exact scores
    assert report.dimensions.d1.score == 1.8
    assert report.dimensions.d2.score == 1.8
    assert report.dimensions.d3.score == 1.7
    assert report.dimensions.d4.score == 1.4
    assert report.dimensions.d5.score == 1.7

    # 3. Final score & percentage
    assert report.final_score.total == 8.4
    assert report.final_score.percentage == 84.0
    assert report.grade == "A"
    assert report.status == "Approved"

    # 4. Consistency matrix
    assert report.dimensions.d4.consistency_matrix["R1"] == "supported"
    assert report.dimensions.d4.consistency_matrix["R2"] == "supported"
    assert report.dimensions.d4.consistency_matrix["R3"] == "supported"
    assert report.dimensions.d4.consistency_matrix["R4"] == "inconsistent"

    # 5. D3 components present
    assert report.dimensions.d3.components is not None
    assert len(report.dimensions.d3.components) == 6

    # 6. Markdown output verification
    md = report.to_markdown(format_variant="unified")
    assert "| Code Programs | 17 |" in md
    assert "| Report Entries | 4 |" in md
    assert "| Matched | 4 |" in md
    assert "| Undocumented | 13 |" in md
    assert "| Source | Program/Report | Finding |" in md
    assert "**Grade:** `A`" in md
    assert "**Approval Status:** `Approved`" in md
    assert "**Score:** 8.40 / 10 (84.0%)" in md
