"""
test_four_step_pipeline.py
Unit and integration tests for the 4-step evaluation pipeline:
Step 1: Upload Zip
Step 2: Do OCR
Step 3: Calculate Score (Deterministic Rubric)
Step 4: LLM as Judge (Independent Dual Reports: Ollama & Gemini)
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from schemas import CalculatedScore, LLMJudgeEvaluation, BatchStudentStatus
from verifier import calculate_deterministic_score, evaluate_with_ollama, evaluate_with_gemini
from batch_processor import BatchPipeline


@pytest.fixture
def sample_pipeline(tmp_path):
    out_dir = str(tmp_path / "output")
    p = BatchPipeline(
        week_id="week-04",
        section_id="SEC2",
        output_dir=out_dir
    )
    return p


def test_step3_calculate_deterministic_score():
    """Verifies that Step 3 computes 100% reproducible rubric scores (/10) with no LLM ambiguity."""
    sample_ocr = """
    Program 1: Prime Number
    Aim: To check whether a number is prime.
    Logic: Loop from 2 to n/2 and check divisibility.
    Important Variables:
    - int n: number to test
    - int is_prime: flag indicating prime status
    What I Observed:
    Input 17 -> Prime. Input 20 -> Not Prime. Correct output observed.
    """

    score_res = calculate_deterministic_score(
        student_id="N241003",
        ocr_text=sample_ocr,
        week_id="week-04"
    )

    assert isinstance(score_res, CalculatedScore)
    assert 0.0 <= score_res.total_score <= 10.0
    assert score_res.status in ("Approved", "Needs Revision")
    assert "Objective" in score_res.criteria_breakdown
    assert "Logic / Approach" in score_res.criteria_breakdown
    assert "Variables Table" in score_res.criteria_breakdown
    assert score_res.variables_table > 0.0
    assert len(score_res.detected_programs) >= 1


def test_step4_evaluate_with_ollama_mock():
    """Verifies Step 4 Ollama judge produces an independent LLMJudgeEvaluation."""
    mock_holistic = MagicMock()
    mock_holistic.evaluation_report = None
    mock_holistic.total_score_10 = 8.5
    mock_holistic.dimensions = []
    mock_holistic.strengths = ["Strong logic documentation"]
    mock_holistic.improvement_areas = ["Include more sample cases"]
    mock_holistic.matches = []

    with patch("verifier.evaluate_holistic_student", return_value=mock_holistic), \
         patch("verifier.render_holistic_evaluation_markdown", return_value="# Ollama Report\nScore: 8.5/10"):

        res = evaluate_with_ollama(
            student_id="N241003",
            report_text="Sample OCR",
            week_id="week-04"
        )

        assert isinstance(res, LLMJudgeEvaluation)
        assert res.provider == "ollama"
        assert res.recommended_score == 8.5
        assert res.score_display == "8.5 / 10.0"
        assert "# Ollama Report" in res.full_report_markdown


def test_step4_evaluate_with_gemini_mock():
    """Verifies Step 4 Gemini judge produces an independent LLMJudgeEvaluation with distinct output."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "recommended_score": 9.0,
        "grade": "O",
        "status": "Approved",
        "criteria_scores": {
            "D1_Syntax": 1.9,
            "D2_Logic": 1.8,
            "D3_ReportQuality": 1.8,
            "D4_Understanding": 1.8,
            "D5_Novelty": 1.7
        },
        "strengths": ["Excellent structured report"],
        "recommendations": ["Add complexity analysis"],
        "faculty_questions": ["Explain recursion base case"],
        "overall_summary": "Exemplary work."
    })
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        res = evaluate_with_gemini(
            student_id="N241003",
            report_text="Sample OCR",
            api_key="test_key",
            week_id="week-04"
        )

        assert isinstance(res, LLMJudgeEvaluation)
        assert res.provider == "gemini"
        assert res.recommended_score == 9.0
        assert res.grade == "O"
        assert "Unified Student Evaluation" in res.full_report_markdown or "Google Gemini Laboratory Evaluation Report" in res.full_report_markdown
        assert len(res.strengths) >= 1


def test_pipeline_step3_and_comparison_csv(sample_pipeline):
    """Verifies pipeline step3 score calculation and CSV generation."""
    sid = "N241003"
    student_payload = {
        "student_id": sid,
        "section_id": "SEC2",
        "week_id": "week-04",
        "status": "completed",
        "source": {"filename": "report.pdf", "file_size_bytes": 100, "mime_type": "application/pdf"},
        "ocr": {"text": "Program 1: Armstrong Number\nAim: Test armstrong\nLogic: pow()\nOutput: Passed", "page_breakdown": [], "total_pages": 1, "total_time": 1.0},
        "extraction": {"detected_programs": [], "status": "partial", "programs": {}}
    }
    j_path = os.path.join(sample_pipeline.students_dir, f"{sid}.json")
    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(student_payload, f)

    scores = sample_pipeline.step3_calculate_scores()
    assert sid in scores
    assert scores[sid].total_score > 0.0

    csv_path = sample_pipeline.build_scores_summary_csv()
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert sid in content
    assert "Step 3 Calculated Score" in content
