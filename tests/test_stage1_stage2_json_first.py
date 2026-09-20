"""
Unit and integration tests for the JSON-First Architecture:
- Stage 1: ObservationReport (Schema 1.0)
- Stage 2: EvaluationReport (Schema 2.0)
- Deterministic 8-Section and 11-Section Markdown generation
"""

import os
import json
import pytest
from schemas import (
    ObservationReport,
    EvaluationReport,
    ReportProgramEntry,
    HolisticEvaluationResult
)
from extractor import (
    extract_observation_report_stage1,
    parse_observation_report_deterministic,
    clean_program_title
)
from verifier import (
    extract_report_entries_normalized,
    evaluate_holistic_student,
    find_student_code
)


def test_clean_program_title():
    raw1 = "Pind Frequency of each character In a string Problem Understanding:"
    assert clean_program_title(raw1) == "Find Frequency of each character In a string"

    raw2 = "Print Armstrong number  $ \\underline{\\text{within}} $ a range using a  $ \\underline{\\text{Function}} $:"
    assert clean_program_title(raw2) == "Print Armstrong number within a range using a Function"

    raw3 = "Find Length of a String Without using strlen"
    assert clean_program_title(raw3) == "Find Length of a String Without using strlen"


def test_stage1_extraction_n241003():
    student_json_path = os.path.join("output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(student_json_path):
        pytest.skip(f"Student JSON {student_json_path} not found")

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_text = data["ocr"]["text"]
    obs = extract_observation_report_stage1(ocr_text)

    # 1. Validate schema version & entries count
    assert obs.schema_version == "1.0"
    assert len(obs.entries) == 4

    # 2. Validate programs
    r1, r2, r3, r4 = obs.entries
    assert r1.report_id == "R1"
    assert "length" in r1.program_title.lower()
    assert r1.problem_understanding is not None
    assert len(r1.important_variables) >= 2

    assert r2.report_id == "R2"
    assert "factorial" in r2.program_title.lower()

    assert r3.report_id == "R3"
    assert "armstrong" in r3.program_title.lower()

    assert r4.report_id == "R4"
    assert "frequency" in r4.program_title.lower()
    assert "nested loops" in (r4.observations or "").lower()

    # 3. Validate Objective & Conclusion
    assert obs.objective is not None
    assert "functions" in obs.objective.text.lower()
    assert obs.conclusion is not None
    assert "recursion" in obs.conclusion.lower()


def test_stage2_evaluation_n241003():
    student_json_path = os.path.join("output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(student_json_path):
        pytest.skip(f"Student JSON {student_json_path} not found")

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_text = data["ocr"]["text"]
    obs = extract_observation_report_stage1(ocr_text)
    code = find_student_code(student_id="N241003", week_id="week-04")
    assert code is not None
    assert len(code.problems) == 17

    res = evaluate_holistic_student(
        report_text=ocr_text,
        extracted_report=obs,
        student_code=code,
        student_id="N241003",
        week_id="week-04",
        provider="deterministic"
    )

    # Validate Holistic Results
    assert res.total_code_problems == 17
    assert res.detected_report_entries_count == 4
    assert res.matched_report_entries_count == 4
    assert res.undocumented_code_problems == 13
    assert res.total_score_100 == 84.0

    # Validate Canonical EvaluationReport (Schema 2.0)
    assert res.evaluation_report is not None
    eval_rep = res.evaluation_report
    assert isinstance(eval_rep, EvaluationReport)
    assert eval_rep.schema_version == "2.0"
    assert eval_rep.student_id == "N241003"
    assert eval_rep.week == "week-04"

    # Submission summary
    assert eval_rep.submission_summary.code_programs == 17
    assert eval_rep.submission_summary.report_entries == 4
    assert eval_rep.submission_summary.matched_programs == 4
    assert eval_rep.submission_summary.undocumented_code_programs == 13

    # D1..D5 dimensions
    assert set(eval_rep.evaluations.keys()) == {"D1", "D2", "D3", "D4", "D5"}
    assert eval_rep.final_score.total == 8.4
    assert eval_rep.final_score.percentage == 84.0

    # Code-Report Consistency (R4 nested loops vs frequency array)
    inconsistencies = [c for c in eval_rep.code_report_consistency if c.status == "inconsistent"]
    assert len(inconsistencies) >= 1
    assert any("p12" in c.program_id.lower() or "r4" in c.report_id.lower() for c in inconsistencies)

    # Notable implementations
    assert len(eval_rep.notable_implementations) >= 4
    p_ids = [n.program_id for n in eval_rep.notable_implementations]
    assert "P4" in p_ids
    assert "P5" in p_ids
    assert "P6_2" in p_ids
    assert "P9" in p_ids

    # Presentation prep
    assert len(eval_rep.presentation_preparation.likely_faculty_questions) >= 3

    # Render Unified 8-Section Markdown
    md_unified = eval_rep.to_markdown(layout="unified")
    assert "## 1. Submission Summary" in md_unified
    assert "## 2. Reported Programs" in md_unified
    assert "## 3. Evaluation" in md_unified
    assert "## 4. Code–Report Consistency" in md_unified
    assert "## 5. Notable Implementations" in md_unified
    assert "## 6. Presentation Preparation" in md_unified
    assert "## 7. Final Score" in md_unified
    assert "## 8. Feedback" in md_unified
    assert "| Code Programs | 17 |" in md_unified
    assert "| Report Entries | 4 |" in md_unified
    assert "| Matched | 4 |" in md_unified
    assert "| Undocumented | 13 |" in md_unified

    # Render Detailed 11-Section Markdown
    md_detailed = eval_rep.to_markdown(layout="detailed")
    assert "## 2. Observation Report Summary" in md_detailed
    assert "## 3. Documented Programs" in md_detailed
    assert "## 4. Code Submission Analysis" in md_detailed
    assert "## 5. Code–Report Matching" in md_detailed
    assert "## 10. Final Score" in md_detailed
    assert "## 11. Feedback" in md_detailed
