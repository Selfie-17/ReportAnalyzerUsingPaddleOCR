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
