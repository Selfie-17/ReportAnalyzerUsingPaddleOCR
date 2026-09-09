"""
tests/test_extractor.py
Unit tests for structured extraction parsing, sanitization, and source page mapping.
"""

import pytest
from extractor import (
    _clean_json_response,
    _parse_and_validate_json,
    _build_extraction_result,
    assign_source_pages,
)


def test_clean_json_response_fences():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert _clean_json_response(raw) == '{"key": "value"}'

    raw_no_tag = "```\n{\"key\": \"value\"}\n```"
    assert _clean_json_response(raw_no_tag) == '{"key": "value"}'


def test_clean_json_response_with_surrounding_text():
    raw = "Here is the extracted JSON:\n{\"objective_of_lab\": \"Learn C\"}\nHope this helps!"
    assert _clean_json_response(raw) == '{"objective_of_lab": "Learn C"}'


def test_parse_and_validate_json_valid():
    raw = '{"objective_of_lab": "Test Lab", "programs": {}}'
    data, err = _parse_and_validate_json(raw)
    assert err is None
    assert data["objective_of_lab"] == "Test Lab"


def test_parse_and_validate_json_invalid():
    raw = 'Not valid json at all'
    data, err = _parse_and_validate_json(raw)
    assert data is None
    assert err is not None


def test_build_extraction_result_full():
    data = {
        "objective_of_lab": "Practice basic control structures in C",
        "programs": {
            "P1": {
                "status": "detected",
                "problem_understanding": "Determine whether a number is even or odd.",
                "logic_approach": "Use the modulus operator % 2 == 0.",
                "important_variables": [
                    {"variable": "num", "purpose": "Input integer value"}
                ],
                "what_i_observed": "Number 4 gave Even, 7 gave Odd."
            },
            "P2": {
                "status": "not_detected",
                "problem_understanding": None,
                "logic_approach": None,
                "important_variables": [],
                "what_i_observed": None
            }
        }
    }
    result = _build_extraction_result(data)
    assert result.status == "success"
    assert result.objective_of_lab == "Practice basic control structures in C"
    assert "P1" in result.detected_programs
    assert "P2" in result.missing_programs
    assert len(result.programs) == 10  # P1 to P10 all present in map
    p1 = result.programs["P1"]
    assert p1.status == "detected"
    assert p1.problem_understanding == "Determine whether a number is even or odd."
    assert len(p1.important_variables) == 1
    assert p1.important_variables[0].variable == "num"
    assert p1.important_variables[0].purpose == "Input integer value"


def test_assign_source_pages():
    data = {
        "objective_of_lab": "Learn C",
        "programs": {
            "P1": {
                "status": "detected",
                "problem_understanding": "Determine whether a number is even or odd.",
                "logic_approach": "Use the modulus operator.",
                "important_variables": [],
                "what_i_observed": "Program ran well."
            },
            "P2": {
                "status": "detected",
                "problem_understanding": "Check leap year",
                "logic_approach": "Divisible by 4 and not 100 or 400.",
                "important_variables": [],
                "what_i_observed": None
            },
            "P3": {
                "status": "not_detected",
                "problem_understanding": None,
                "logic_approach": None,
                "important_variables": [],
                "what_i_observed": None
            }
        }
    }
    result = _build_extraction_result(data)
    page_breakdown = [
        {
            "page": 1,
            "text": "Objective: Learn C\n\n1. Check whether a given number is even or odd\nDetermine whether a number is even or odd."
        },
        {
            "page": 2,
            "text": "Program 2: Check leap year\nDivisible by 4 and not 100 or 400."
        }
    ]
    updated = assign_source_pages(result, page_breakdown)
    assert updated.programs["P1"].source_pages == [1]
    assert updated.programs["P2"].source_pages == [2]
    assert updated.programs["P3"].source_pages == []


from unittest.mock import patch
from extractor import extract_observation_report


@patch("extractor._call_ollama")
def test_extract_observation_report_with_assigned_questions(mock_call):
    mock_call.return_value = (
        True,
        '{"objective_of_lab": "Learn loops", "programs": {"P1": {"status": "detected", "problem_understanding": "Factorial"}}}',
        None
    )

    res = extract_observation_report(
        report_text="Program 1: Factorial",
        assigned_questions="Program 1: Factorial of a Number"
    )

    assert res.status == "success"
    assert res.objective_of_lab == "Learn loops"
    assert "P1" in res.detected_programs

    # Verify that assigned_questions was included in user prompt
    mock_call.assert_called_once()
    prompt_arg = mock_call.call_args[1]["prompt"]
    assert "Program 1: Factorial of a Number" in prompt_arg
    assert "Program 1: Factorial" in prompt_arg


def test_clean_ocr_text_for_llm_converts_html_tables():
    from extractor import _clean_ocr_text_for_llm
    raw_html = (
        "<div style='text-align: center;'>Variables</div>\n"
        "<table border=1 style='margin: auto;'><tr><td>Variable</td><td>Purpose</td></tr>"
        "<tr><td>n</td><td>Number</td></tr></table>"
    )
    cleaned = _clean_ocr_text_for_llm(raw_html)
    assert "| Variable | Purpose |" in cleaned
    assert "| --- | --- |" in cleaned
    assert "| n | Number |" in cleaned
    assert "<table" not in cleaned
    assert "<div" not in cleaned


def test_build_extraction_result_handles_list_and_aliases():
    from extractor import _build_extraction_result
    raw_llm_list = {
        "objective": "Practice basic C logic",
        "programs": [
            {
                "name": "Even or Odd",
                "problem": "Check even or odd",
                "logic": "Modulus 2",
                "variables": [{"name": "n", "purpose": "input number"}],
                "observations": "Remainder is 0 for even"
            },
            {
                "name": "Positive, Negative, or Zero",
                "problem": "Check sign",
                "logic": "Compare with 0",
                "variables": [{"name": "val", "role": "input value"}],
                "observations": "Zero is neither"
            }
        ]
    }
    res = _build_extraction_result(raw_llm_list, expected_count=2)
    assert res.status == "success"
    assert res.objective_of_lab == "Practice basic C logic"
    assert res.detected_programs == ["P1", "P2"]
    assert res.programs["P1"].problem_understanding == "Check even or odd"
    assert res.programs["P1"].important_variables[0].variable == "n"
    assert res.programs["P2"].important_variables[0].variable == "val"


@patch("extractor.requests.post")
def test_call_ollama_passes_num_ctx(mock_post):
    from extractor import _call_ollama
    mock_resp = mock_post.return_value
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "{}"}}

    _call_ollama(prompt="test", system_prompt="sys", num_ctx=16384)
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["options"]["num_ctx"] == 16384

