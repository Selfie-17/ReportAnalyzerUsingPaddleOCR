"""
tests/test_verifier.py
Unit tests for the Observation Report Verification Engine,
Instruction Manual submission guide integration, and assigned questions comparison.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from verifier import (
    build_verification_messages,
    stream_observation_verification,
    verify_observation_report_sync,
    evaluate_observation_report,
    render_verification_markdown,
    check_ollama_status,
    OFFICIAL_INSTRUCTION_MANUAL,
    DEFAULT_MANUAL_QUESTIONS_PRESET,
    VERIFICATION_SYSTEM_PROMPT,
)


def test_build_verification_messages_with_assigned_questions():
    report_text = "Objective: Learn loops in C.\nProgram 1: Factorial\nUnderstanding: Multiply 1 to n."
    assigned_questions = "Program 1: Factorial of a Number\nProgram 2: Prime Number"

    messages = build_verification_messages(
        report_text=report_text,
        assigned_questions=assigned_questions
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Instruction Manual for Writing the Observation Report" in messages[0]["content"]
    assert "OBJECTIVE OF THE LAB (Max 2.0 points)" in messages[0]["content"]
    assert "WHAT I OBSERVED (Max 2.0 points)" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert messages[1]["role"] == "user"
    assert "ASSIGNED LAB QUESTIONS / PROBLEMS" in user_content
    assert "Program 1: Factorial of a Number" in user_content
    assert "Program 2: Prime Number" in user_content
    assert report_text in user_content


def test_build_verification_messages_without_assigned_questions():
    report_text = "Objective: Learn basic C arithmetic."
    messages = build_verification_messages(
        report_text=report_text,
        assigned_questions=None
    )

    assert len(messages) == 2
    user_content = messages[1]["content"]
    assert "ASSIGNED LAB QUESTIONS / PROBLEMS" not in user_content
    assert report_text in user_content


def test_official_instruction_manual_contains_all_core_sections():
    # Verify the 5 required sections and 5 core questions are present in the manual
    assert "## 1. Objective of the Lab" in OFFICIAL_INSTRUCTION_MANUAL
    assert "## 2. Problem Understanding" in OFFICIAL_INSTRUCTION_MANUAL
    assert "## 3. Logic / Approach Used" in OFFICIAL_INSTRUCTION_MANUAL
    assert "## 4. Important Variables and Their Purpose" in OFFICIAL_INSTRUCTION_MANUAL
    assert "## 5. What I Observed" in OFFICIAL_INSTRUCTION_MANUAL
    assert "What was I trying to learn?" in OFFICIAL_INSTRUCTION_MANUAL
    assert "What did I understand or notice while executing the program?" in OFFICIAL_INSTRUCTION_MANUAL


def test_default_manual_questions_preset():
    assert "Program 1: Factors of a Number" in DEFAULT_MANUAL_QUESTIONS_PRESET
    assert "Program 2: Factorial of a Number" in DEFAULT_MANUAL_QUESTIONS_PRESET
    assert "Program 3: Palindrome Number" in DEFAULT_MANUAL_QUESTIONS_PRESET
    assert "Program 4: Prime Number" in DEFAULT_MANUAL_QUESTIONS_PRESET
    assert "Program 5: Fibonacci Series" in DEFAULT_MANUAL_QUESTIONS_PRESET



@patch("verifier.requests.post")
def test_stream_observation_verification(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "objective": None,
                "questions": [
                    {
                        "question_number": 1,
                        "question_text": "Factors",
                        "match_status": "FOUND_AND_COVERED",
                        "requirements": [
                            {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Find factors of n", "confidence": "HIGH"}
                        ]
                    }
                ]
            })
        }
    }
    mock_post.return_value = mock_resp

    chunks = list(stream_observation_verification(
        report_text="Program 1: Factors\nFind factors of n.",
        assigned_questions="Program 1: Factors"
    ))

    assert len(chunks) > 0
    full_output = "".join(chunks)
    assert "Laboratory Observation Report" in full_output
    assert "Assigned Questions Coverage Analysis" in full_output

    # Check that payload sent to Ollama had format='json'
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["format"] == "json"


@patch("verifier.requests.post")
def test_evaluate_observation_report_legacy(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "objective": None,
                "questions": [
                    {
                        "question_number": 1,
                        "question_text": "Factors",
                        "match_status": "FOUND_AND_COVERED",
                        "requirements": [
                            {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Find factors of n", "confidence": "HIGH"}
                        ]
                    }
                ]
            })
        }
    }
    mock_post.return_value = mock_resp

    eval_result = evaluate_observation_report(
        report_text="Program 1: Factors\nFind factors of n.",
        assigned_questions="Program 1: Factors"
    )
    result = render_verification_markdown(eval_result)
    assert "# 📊 Laboratory Observation Report" in result
    assert "## 📊 Final Score" in result


@patch("verifier.requests.post")
def test_verify_observation_report_sync_always_unified(mock_post):
    """
    verify_observation_report_sync MUST have exactly one normal evaluation path:
    evaluate_holistic_student() -> render_holistic_evaluation_markdown().
    Passing assigned_questions must NEVER trigger the legacy 42-mark evaluator.
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "dimensions": [
                    {"dimension": "D1", "name": "Syntax & Code Validity", "score": 2.0, "assessment": "Valid syntax", "evidence": ["Clean compilation"]},
                    {"dimension": "D2", "name": "Algorithmic Logic", "score": 2.0, "assessment": "Correct logic", "evidence": ["Proper loops"]},
                    {"dimension": "D3", "name": "Observation Report Quality", "score": 2.0, "assessment": "Good report", "evidence": ["Clear explanations"]},
                    {"dimension": "D4", "name": "Conceptual Understanding", "score": 2.0, "assessment": "Consistent claims", "evidence": ["Matches code"]},
                    {"dimension": "D5", "name": "Novelty & Presentation Readiness", "score": 2.0, "assessment": "Clean presentation", "evidence": ["Well documented"]}
                ],
                "interesting_logic": [],
                "presentation_readiness": [],
                "strengths": ["Clear logic"],
                "improvement_areas": ["Add comments"],
                "overall_summary": "Solid submission"
            })
        }
    }
    mock_post.return_value = mock_resp

    result = verify_observation_report_sync(
        report_text="Program 1: Factors\nFind factors of n.",
        assigned_questions="Program 1: Factors"
    )
    assert "# Unified Student Evaluation" in result
    assert "D1 — Syntax & Code Validity" in result
    assert "D5 — Novelty" in result
    assert "Assigned Questions" not in result
    assert "42 marks" not in result
    assert "0 / 42" not in result
    assert "Requirements Compliance Matrix" not in result


@patch("verifier.requests.get")
def test_check_ollama_status_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen2.5-coder:3b"}, {"name": "llama3:latest"}]
    }
    mock_get.return_value = mock_resp

    status = check_ollama_status(model="qwen2.5-coder:3b")
    assert status["ok"] is True
    assert status["model_found"] is True
    assert "qwen2.5-coder:3b" in status["models"]


@patch("verifier.requests.get")
def test_check_ollama_status_connection_error(mock_get):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    status = check_ollama_status()
    assert status["ok"] is False
    assert "Cannot connect to Ollama" in status["error"]


def test_format_extraction_for_evaluation():
    from verifier import format_extraction_for_evaluation
    from schemas import ExtractionResult, ProgramDetails, VariableItem

    ext = ExtractionResult(
        objective_of_lab="Understand conditions and loops in C.",
        programs={
            "P1": ProgramDetails(
                status="detected",
                problem_understanding="Check even or odd using modulo.",
                logic_approach="If n % 2 == 0 then even else odd.",
                important_variables=[VariableItem(variable="n", purpose="User integer input")],
                what_i_observed="Program printed even for 4 and odd for 7."
            )
        },
        detected_programs=["P1"]
    )

    formatted = format_extraction_for_evaluation(ext)
    assert "# Laboratory Observation Report" in formatted
    assert "## 1. Objective of the Lab" in formatted
    assert "Understand conditions and loops in C." in formatted
    assert "## Program P1" in formatted
    assert "### Problem Understanding" in formatted
    assert "Check even or odd using modulo." in formatted
    assert "| `n` | User integer input |" in formatted
    assert "Program printed even for 4 and odd for 7." in formatted

