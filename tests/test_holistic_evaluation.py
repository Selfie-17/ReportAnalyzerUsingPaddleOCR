"""
tests/test_holistic_evaluation.py
Comprehensive test suite verifying the Unified 5-Dimension Evaluation Engine,
conservative R1..Rn report extraction, semantic question matching,
static analysis (no execution claims), and deterministic score arithmetic.
"""

import pytest
import json
from typing import Dict, Any, List

from schemas import (
    ReportProgramEntry,
    ReportExtractionResult,
    ExtractionResult,
    StudentCodeSnippet,
    StudentCodeCollection,
    QuestionMatch,
    HolisticEvaluationResult,
    AssignedQuestion,
    VariableItem,
)
from unittest.mock import patch
from extractor import _build_extraction_result
from verifier import (
    match_report_programs_to_questions,
    evaluate_holistic_student,
    ingest_student_code_from_json_or_files,
    DEFAULT_MATCHED_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
)


@pytest.fixture(autouse=True)
def mock_ollama_offline():
    """Ensures test suite executes deterministically and rapidly using static fallback evaluation."""
    with patch("verifier._call_ollama", return_value=(False, "", "Simulated offline for unit testing")):
        yield


@pytest.fixture
def sample_week4_questions() -> List[AssignedQuestion]:
    return [
        AssignedQuestion(question_number=1, question_text="Check whether a given number is prime."),
        AssignedQuestion(question_number=2, question_text="Check whether a string is a palindrome."),
        AssignedQuestion(question_number=3, question_text="Find length of a string without using strlen()."),
        AssignedQuestion(question_number=4, question_text="Find sum of natural numbers using recursion."),
        AssignedQuestion(question_number=5, question_text="Find factorial of a number using recursion."),
        AssignedQuestion(question_number=6, question_text="Swap two numbers using call by value."),
        AssignedQuestion(question_number=7, question_text="Swap two numbers using call by reference."),
        AssignedQuestion(question_number=8, question_text="Print Armstrong numbers within a range."),
        AssignedQuestion(question_number=9, question_text="Print a diamond pattern."),
        AssignedQuestion(question_number=10, question_text="Generate Fibonacci series using recursion."),
        AssignedQuestion(question_number=11, question_text="Sort strings in lexicographical order."),
        AssignedQuestion(question_number=12, question_text="Reverse a string."),
        AssignedQuestion(question_number=13, question_text="Find the frequency of each character in a string."),
        AssignedQuestion(question_number=14, question_text="Evaluate recursive function f(n) = 2f(n-1)."),
        AssignedQuestion(question_number=15, question_text="Count number of 1s in binary representation."),
        AssignedQuestion(question_number=16, question_text="Print values using two recursive calls."),
        AssignedQuestion(question_number=17, question_text="Evaluate recursive function with start and end bounds.")
    ]


# ============================================================
# TEST 1: 17 code files + 4 report programs
# ============================================================
def test_01_seventeen_code_files_with_four_report_programs(sample_week4_questions):
    """
    Verifies that a student with 17 code files and only 4 report programs
    produces 17 code snippets and 4 report entries, with ZERO missing programs penalty.
    """
    code_snippets = [
        StudentCodeSnippet(problem_number=i, problem_id=f"P{i}", source_file=f"p{i}.c", source_code=f"int main() {{ return {i}; }}")
        for i in range(1, 18)
    ]
    code_coll = StudentCodeCollection(student_id="N241003", week="week-04", problems=code_snippets)
    assert len(code_coll.problems) == 17

    raw_extraction = {
        "objective_of_lab": "Learn recursion and strings",
        "detected_programs": [
            {"program_title": "Find Length of String", "problem_understanding": "Count characters"},
            {"program_title": "Factorial using recursion", "problem_understanding": "Multiply n * fact(n-1)"},
            {"program_title": "Armstrong in range", "problem_understanding": "Sum of powers of digits"},
            {"program_title": "Character frequency", "problem_understanding": "Count occurrences of characters"}
        ]
    }
    extracted = _build_extraction_result(raw_extraction)
    report_entries = extracted.get_report_entries()

    assert len(report_entries) == 4
    assert extracted.missing_programs == []  # No missing program placeholders!

    # Evaluate
    result = evaluate_holistic_student(
        report_text="Sample text",
        extracted_report=extracted,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    # Check D3 did not penalize for missing 13 programs
    d3_score = [d for d in result.dimensions if d.dimension == "D3"][0].score
    assert d3_score >= 1.5


# ============================================================
# TEST 2: Program title without number
# ============================================================
def test_02_program_title_without_number():
    """
    Verifies that a program title with no explicit number assigns R1
    with student_written_program_number = None.
    """
    raw_data = {
        "detected_programs": [
            {
                "program_title": "Find Length of a String Without Using strlen",
                "problem_understanding": "Calculate length without library function",
                "logic_approach": "Traverse until null terminator"
            }
        ]
    }
    ext = _build_extraction_result(raw_data)
    entries = ext.get_report_entries()
    assert len(entries) == 1
    assert entries[0].report_program_id == "R1"
    assert entries[0].student_written_program_number is None


# ============================================================
# TEST 3: Explicit program number preserved without forcing official ID
# ============================================================
def test_03_explicit_program_number_preserved():
    """
    Verifies that if student wrote 'Program 7: Find Factorial',
    student_written_program_number is '7', but report_program_id is 'R1'.
    """
    raw_data = {
        "detected_programs": [
            {
                "program_title": "Program 7: Find Factorial",
                "student_written_program_number": "7",
                "problem_understanding": "Compute factorial using recursion",
                "logic_approach": "Multiply by recursive call"
            }
        ]
    }
    ext = _build_extraction_result(raw_data)
    entries = ext.get_report_entries()
    assert len(entries) == 1
    assert entries[0].report_program_id == "R1"
    assert entries[0].student_written_program_number == "7"


# ============================================================
# TEST 4: Grouped Level heading does not duplicate programs
# ============================================================
def test_04_grouped_level_heading_deduplication():
    """
    Verifies that identical grouped text (e.g. copied Level 1 summary)
    is deduplicated and does not create fake duplicate program entries.
    """
    copied_summary = "Level 1: Even/odd, positive/negative, and character classification programs."
    raw_data = {
        "detected_programs": [
            {"program_title": "P1", "problem_understanding": copied_summary},
            {"program_title": "P2", "problem_understanding": copied_summary}
        ]
    }
    ext = _build_extraction_result(raw_data)
    entries = ext.get_report_entries()
    # Deduplication guardrail neutralizes copied text
    for e in entries:
        assert e.problem_understanding is None


# ============================================================
# TEST 5: Missing variables returns empty list
# ============================================================
def test_05_missing_variables_returns_empty_list():
    """
    Verifies that missing variables section yields an empty list rather than invented variables.
    """
    raw_data = {
        "detected_programs": [
            {
                "program_title": "Prime Number",
                "problem_understanding": "Check prime",
                "important_variables": []
            }
        ]
    }
    ext = _build_extraction_result(raw_data)
    entries = ext.get_report_entries()
    assert len(entries) == 1
    assert entries[0].important_variables == []


# ============================================================
# TEST 6: Semantic question matching matches related titles
# ============================================================
def test_06_semantic_question_matching(sample_week4_questions):
    """
    Verifies that 'Find length of string' semantically matches official question P3.
    """
    report_entry = ReportProgramEntry(
        report_program_id="R1",
        program_title="Find Length of a String Without using strlen",
        problem_understanding="Traverse characters until null character 0",
        logic_approach="Counter incremented for each character"
    )
    ext = ReportExtractionResult(detected_programs=[report_entry])
    matches = match_report_programs_to_questions(ext, sample_week4_questions)

    assert len(matches) == 1
    assert matches[0].matched_problem_id == "P3"
    assert matches[0].match_status == "matched"
    assert matches[0].match_confidence >= DEFAULT_MATCHED_THRESHOLD


# ============================================================
# TEST 7: Low-confidence question matching flagged as needs_review
# ============================================================
def test_07_low_confidence_question_matching(sample_week4_questions):
    """
    Verifies that a vague or ambiguous title is flagged as needs_review or unmatched.
    """
    report_entry = ReportProgramEntry(
        report_program_id="R1",
        program_title="Calculate values using math",
        problem_understanding="Performs some calculations",
        logic_approach="Used basic formula"
    )
    ext = ReportExtractionResult(detected_programs=[report_entry])
    matches = match_report_programs_to_questions(ext, sample_week4_questions)

    assert len(matches) == 1
    assert matches[0].match_status in ("needs_review", "unmatched")
    assert matches[0].match_confidence < DEFAULT_MATCHED_THRESHOLD


# ============================================================
# TEST 8: Correct code + weak report handled independently
# ============================================================
def test_08_correct_code_with_weak_report(sample_week4_questions):
    """
    Verifies that high code validity with a sparse report awards high D1/D2 and low D3.
    """
    code_coll = StudentCodeCollection(
        student_id="N100",
        problems=[
            StudentCodeSnippet(
                problem_number=1,
                problem_id="P1",
                source_file="prime.c",
                source_code="#include <stdio.h>\nint is_prime(int n) {\n    if (n <= 1) return 0;\n    for (int i = 2; i * i <= n; i++) {\n        if (n % i == 0) return 0;\n    }\n    return 1;\n}\nint main() {\n    printf(\"%d\", is_prime(7));\n    return 0;\n}"
            )
        ]
    )
    # Report has minimal substance
    weak_report = ReportExtractionResult(
        objective_of_lab=None,
        detected_programs=[
            ReportProgramEntry(
                report_program_id="R1",
                program_title="P1",
                problem_understanding="Did program 1",
                logic_approach=None,
                important_variables=[],
                what_i_observed=None
            )
        ]
    )

    result = evaluate_holistic_student(
        report_text="weak",
        extracted_report=weak_report,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    d_scores = {d.dimension: d.score for d in result.dimensions}
    assert d_scores["D1"] >= 1.5  # High static syntax validity
    assert d_scores["D3"] <= 1.5  # Weak report


# ============================================================
# TEST 9: Weak code + strong report handled independently
# ============================================================
def test_09_weak_code_with_strong_report(sample_week4_questions):
    """
    Verifies that poor/incomplete code with an exhaustive report awards low D1/D2 and high D3.
    """
    # Incomplete code
    weak_code = StudentCodeCollection(
        student_id="N101",
        problems=[
            StudentCodeSnippet(
                problem_number=1,
                problem_id="P1",
                source_file="p1.c",
                source_code="broken code without syntax"
            )
        ]
    )
    # Strong report
    strong_report = ReportExtractionResult(
        objective_of_lab="Comprehensive exploration of recursive functions, stack frames, and string manipulations.",
        detected_programs=[
            ReportProgramEntry(
                report_program_id="R1",
                program_title="Prime Number Verification",
                problem_understanding="Determine whether an integer has any positive divisors other than 1 and itself.",
                logic_approach="Iterate from 2 up to the square root of n and check remainder using modulus.",
                important_variables=[VariableItem(variable="n", purpose="Input number"), VariableItem(variable="i", purpose="Loop divisor")],
                what_i_observed="Observed that composite numbers always have at least one factor <= sqrt(n)."
            ),
            ReportProgramEntry(
                report_program_id="R2",
                program_title="Factorial Using Recursion",
                problem_understanding="Multiply all integers down to 1.",
                logic_approach="Base case returns 1 when n == 0; recursive step multiplies n * fact(n-1).",
                important_variables=[VariableItem(variable="n", purpose="Input value")],
                what_i_observed="Each function call pushes an activation frame onto the execution stack."
            ),
            ReportProgramEntry(
                report_program_id="R3",
                program_title="String Length Without strlen",
                problem_understanding="Count characters in string.",
                logic_approach="Traverse array until null character is found.",
                important_variables=[VariableItem(variable="str", purpose="Input string")],
                what_i_observed="Null character marks the termination of standard C strings."
            )
        ]
    )

    result = evaluate_holistic_student(
        report_text="strong",
        extracted_report=strong_report,
        student_code=weak_code,
        question_context=sample_week4_questions
    )

    d_scores = {d.dimension: d.score for d in result.dimensions}
    assert d_scores["D1"] <= 1.2  # Code is invalid
    assert d_scores["D3"] >= 1.5  # Strong report


# ============================================================
# TEST 10: Report claim contradicts code (Inconsistent in D4)
# ============================================================
def test_10_report_claim_contradicts_code(sample_week4_questions):
    """
    Verifies that claiming tail recursion when code uses non-tail recursion is flagged in D4.
    """
    code_coll = StudentCodeCollection(
        student_id="N102",
        problems=[
            StudentCodeSnippet(
                problem_number=5,
                problem_id="P5",
                source_file="fact.c",
                # Non-tail recursive! Multiplication follows recursive call
                source_code="#include <stdio.h>\nint fact(int n) {\n    if (n <= 1) return 1;\n    return n * fact(n - 1);\n}\nint main() { return 0; }"
            )
        ]
    )
    report = ReportExtractionResult(
        detected_programs=[
            ReportProgramEntry(
                report_program_id="R1",
                program_title="Factorial using recursion",
                problem_understanding="Calculates factorial",
                logic_approach="I used tail recursion where the recursive call is the final action.",
                what_i_observed="Tail recursion is very efficient."
            )
        ]
    )
    matches = [
        QuestionMatch(
            report_program_id="R1",
            matched_problem_id="P5",
            matched_problem_title="Find factorial using recursion",
            match_status="matched",
            match_confidence=0.95
        )
    ]

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions,
        matches=matches
    )

    # Check that inconsistency was detected
    inconsistent = [c for c in result.code_report_consistency if c.claim_type == "inconsistent"]
    assert len(inconsistent) >= 1
    assert "tail recursion" in inconsistent[0].report_claim.lower() or "tail recursion" in inconsistent[0].assessment.lower()


# ============================================================
# TEST 11: Report claim supported by code verified in D4
# ============================================================
def test_11_report_claim_supported_by_code(sample_week4_questions):
    """
    Verifies that a report claiming nested loops matching code is marked supported.
    """
    code_coll = StudentCodeCollection(
        student_id="N103",
        problems=[
            StudentCodeSnippet(
                problem_number=13,
                problem_id="P13",
                source_file="freq.c",
                source_code="#include <stdio.h>\nint main() {\n    char str[] = \"hello\";\n    for (int i = 0; str[i]; i++) {\n        for (int j = i + 1; str[j]; j++) {\n            // compare\n        }\n    }\n    return 0;\n}"
            )
        ]
    )
    report = ReportExtractionResult(
        detected_programs=[
            ReportProgramEntry(
                report_program_id="R1",
                program_title="Character frequency in string",
                problem_understanding="Count frequency",
                logic_approach="Used nested loop to compare each character against subsequent characters.",
                what_i_observed="Nested loop identifies all duplicate characters."
            )
        ]
    )
    matches = [
        QuestionMatch(
            report_program_id="R1",
            matched_problem_id="P13",
            matched_problem_title="Find frequency of each character",
            match_status="matched",
            match_confidence=0.95
        )
    ]

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions,
        matches=matches
    )

    supported = [c for c in result.code_report_consistency if c.claim_type == "supported"]
    assert len(supported) >= 1


# ============================================================
# TEST 12: Interesting / innovative code detected across ALL files in D5
# ============================================================
def test_12_interesting_logic_detected_in_undocumented_code(sample_week4_questions):
    """
    Verifies that an interesting approach in p10.c (Fibonacci tree recursion)
    is detected as D5 evidence even when p10 is NOT documented in the report!
    """
    code_coll = StudentCodeCollection(
        student_id="N104",
        problems=[
            StudentCodeSnippet(
                problem_number=1,
                problem_id="P1",
                source_file="p1.c",
                source_code="int main() { return 0; }"
            ),
            StudentCodeSnippet(
                problem_number=10,
                problem_id="P10",
                problem_title="Fibonacci Series using Recursion",
                source_file="p10.c",
                # Tree recursion with 2 recursive branches
                source_code="#include <stdio.h>\nint fib(int n) {\n    if (n <= 1) return n;\n    return fib(n - 1) + fib(n - 2);\n}\nint main() { return fib(5); }"
            )
        ]
    )
    # Report documents only P1, NOT P10
    report = ReportExtractionResult(
        detected_programs=[
            ReportProgramEntry(
                report_program_id="R1",
                program_title="Check Prime",
                problem_understanding="Checks prime numbers",
                logic_approach="Loop up to n/2"
            )
        ]
    )

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    # D5 should identify P10 interesting logic
    assert len(result.interesting_logic) >= 1
    found_p10 = any(item.problem in ("P10", "10") for item in result.interesting_logic)
    assert found_p10
    assert result.novelty_score_20 >= 10.0  # Earns novelty credit!


# ============================================================
# TEST 13: Long but unnecessarily complicated code not given high novelty
# ============================================================
def test_13_long_verbose_code_not_overrewarded(sample_week4_questions):
    """
    Verifies that bloated, repetitive code does NOT automatically receive high novelty.
    """
    # 50 lines of redundant variable assignments and empty comments
    bloated_code = "#include <stdio.h>\nint main() {\n" + "\n".join([f"    int var_{i} = {i}; // dummy variable" for i in range(50)]) + "\n    return 0;\n}"
    code_coll = StudentCodeCollection(
        student_id="N105",
        problems=[StudentCodeSnippet(problem_number=1, problem_id="P1", source_file="p1.c", source_code=bloated_code)]
    )
    report = ReportExtractionResult(
        detected_programs=[
            ReportProgramEntry(report_program_id="R1", program_title="Dummy Program", problem_understanding="Long code")
        ]
    )

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    d5_score = [d for d in result.dimensions if d.dimension == "D5"][0].score
    assert d5_score <= 1.3  # Not over-rewarded!


# ============================================================
# TEST 14: Short but elegant code can receive high novelty
# ============================================================
def test_14_short_elegant_code_can_receive_high_novelty(sample_week4_questions):
    """
    Verifies that concise, elegant recursive logic receives strong novelty credit.
    """
    elegant_code = "#include <stdio.h>\nint gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }\nint main() { return gcd(48, 18); }"
    code_coll = StudentCodeCollection(
        student_id="N106",
        problems=[StudentCodeSnippet(problem_number=1, problem_id="P1", source_file="gcd.c", source_code=elegant_code)]
    )
    report = ReportExtractionResult(
        detected_programs=[
            ReportProgramEntry(report_program_id="R1", program_title="Euclidean GCD", problem_understanding="Find greatest common divisor")
        ]
    )

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    d5_score = [d for d in result.dimensions if d.dimension == "D5"][0].score
    assert d5_score >= 1.0


# ============================================================
# TEST 15: Score totals always equal 10 and 100 correctly
# ============================================================
def test_15_deterministic_score_totals_and_weights(sample_week4_questions):
    """
    Verifies that score arithmetic is completely deterministic in Python:
    - total_score_10 = D1 + D2 + D3 + D4 + D5 (<= 10.0)
    - total_score_100 = total_score_10 * 10 (<= 100.0)
    - programming_score_40 = (D1 + D2) * 10
    - report_score_20 = D3 * 10
    - conceptual_score_20 = D4 * 10
    - novelty_score_20 = D5 * 10
    """
    code_coll = StudentCodeCollection(
        student_id="N241003",
        problems=[
            StudentCodeSnippet(problem_number=1, problem_id="P1", source_file="p1.c", source_code="int main() { return 0; }"),
            StudentCodeSnippet(problem_number=2, problem_id="P2", source_file="p2.c", source_code="int fact(int n) { return n<=1?1:n*fact(n-1); }")
        ]
    )
    report = ReportExtractionResult(
        objective_of_lab="Study recursion",
        detected_programs=[
            ReportProgramEntry(report_program_id="R1", program_title="Factorial", problem_understanding="Factorial problem", logic_approach="Recursive multiply")
        ]
    )

    result = evaluate_holistic_student(
        report_text="Sample",
        extracted_report=report,
        student_code=code_coll,
        question_context=sample_week4_questions
    )

    d_scores = {d.dimension: d.score for d in result.dimensions}
    expected_10 = round(sum(d_scores.values()), 2)
    expected_100 = round(expected_10 * 10, 1)
    expected_prog_40 = round((d_scores["D1"] + d_scores["D2"]) * 10, 1)
    expected_rep_20 = round(d_scores["D3"] * 10, 1)
    expected_conc_20 = round(d_scores["D4"] * 10, 1)
    expected_nov_20 = round(d_scores["D5"] * 10, 1)

    assert result.total_score_10 == expected_10
    assert result.total_score_100 == expected_100
    assert result.programming_score_40 == expected_prog_40
    assert result.report_score_20 == expected_rep_20
    assert result.conceptual_score_20 == expected_conc_20
    assert result.novelty_score_20 == expected_nov_20

    # Verify each dimension max is 2.0
    for d in result.dimensions:
        assert 0.0 <= d.score <= 2.0
