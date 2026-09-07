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
    # Simulate streaming response lines
    mock_resp.iter_lines.return_value = [
        json.dumps({"message": {"content": "# Observation Report Verification Report\n"}}).encode("utf-8"),
        json.dumps({"message": {"content": "## Overall Evaluation\n- Total Score: 9.0 / 10.0"}}).encode("utf-8"),
    ]
    mock_post.return_value = mock_resp

    chunks = list(stream_observation_verification(
        report_text="Sample report text",
        assigned_questions="Program 1: Factors"
    ))

    assert len(chunks) == 2
    full_output = "".join(chunks)
    assert "# Observation Report Verification Report" in full_output
    assert "Total Score: 9.0 / 10.0" in full_output

    # Check that payload sent to Ollama had stream=True and included assigned questions
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["stream"] is True
    user_msg = payload["messages"][1]["content"]
    assert "Program 1: Factors" in user_msg


@patch("verifier.requests.post")
def test_verify_observation_report_sync(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = [
        json.dumps({"message": {"content": "Evaluation result chunk 1 "}}).encode("utf-8"),
        json.dumps({"message": {"content": "chunk 2"}}).encode("utf-8"),
    ]
    mock_post.return_value = mock_resp

    result = verify_observation_report_sync(
        report_text="Sample report",
        assigned_questions="P1: Test"
    )
    assert result == "Evaluation result chunk 1 chunk 2"


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

