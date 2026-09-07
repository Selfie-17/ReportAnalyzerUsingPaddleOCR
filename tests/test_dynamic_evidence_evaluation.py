"""
tests/test_dynamic_evidence_evaluation.py
Comprehensive validation suite for hallucination resistance, dynamic questions,
dynamic instruction manuals, title-only detection, OCR error tolerance,
and Python evidence grounding validation.

Covers Test Scenarios A, B, C, D, E, F as required by the specification.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from schemas import (
    AssignedQuestion,
    RequirementEvaluation,
    QuestionEvaluation,
    ObjectiveEvaluation,
    DynamicEvaluationResult,
)
from verifier import (
    parse_assigned_questions,
    parse_instruction_manual,
    is_evidence_grounded,
    is_generic_observation,
    extract_grounded_source_span,
    process_and_ground_evidence,
    build_dynamic_verification_messages,
    validate_and_ground_evaluation,
    render_verification_markdown,
    evaluate_observation_report,
    OFFICIAL_INSTRUCTION_MANUAL,
    WEEK1_13_PROGRAMS_PRESET,
    DEFAULT_MANUAL_QUESTIONS_PRESET,
)


# ============================================================
# 1. PARSER UNIT TESTS
# ============================================================

def test_parse_assigned_questions_arbitrary_counts():
    # 3 questions
    q3 = """1. Write a Java program to reverse a string.
2. Explain exception handling.
3. Implement binary search."""
    parsed3 = parse_assigned_questions(q3)
    assert len(parsed3) == 3
    assert parsed3[0].question_number == 1
    assert "reverse a string" in parsed3[0].question_text
    assert parsed3[2].question_number == 3
    assert "binary search" in parsed3[2].question_text

    # 13 questions
    parsed13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    assert len(parsed13) == 13
    assert parsed13[0].question_number == 1
    assert parsed13[12].question_number == 13

    # JSON input
    json_q = json.dumps([
        {"question_number": 1, "question_text": "Task A"},
        {"question_number": 2, "question_text": "Task B"}
    ])
    parsed_json = parse_assigned_questions(json_q)
    assert len(parsed_json) == 2
    assert parsed_json[0].question_text == "Task A"

    # Empty / whitespace
    assert parse_assigned_questions("") == []
    assert parse_assigned_questions(None) == []


def test_parse_instruction_manual_dynamic_sections():
    # Default manual
    def_man = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    assert "Problem Understanding" in def_man["per_question_requirements"]
    assert "Logic / Approach Used" in def_man["per_question_requirements"]
    assert "Important Variables and Their Purpose" in def_man["per_question_requirements"]
    assert "What I Observed" in def_man["per_question_requirements"]

    # Completely different manual (Software Engineering / Research)
    custom_man = """# Software Engineering Lab Manual
## Objective
Explain overall lab purpose.
## Methodology
Explain class structure and algorithms.
## Test Cases
Describe edge test inputs and expected results.
## Discussion
Discuss computational complexity and tradeoffs.
"""
    parsed_custom = parse_instruction_manual(custom_man)
    assert parsed_custom["objective_requirement"] == "Objective"
    reqs = parsed_custom["per_question_requirements"]
    assert "Methodology" in reqs
    assert "Test Cases" in reqs
    assert "Discussion" in reqs
    assert "Problem Understanding" not in reqs


# ============================================================
# 2. EVIDENCE GROUNDING UNIT TESTS
# ============================================================

def test_evidence_grounding_verification():
    ocr_text = """
    ## Program 1: Even or Odd
    User enters an integer and it is stored in variable n.
    Checks whether number is divisible by 2 using modulus operator %.
    If n % 2 == 0 prints even otherwise prints odd.
    """

    # Exact snippet
    grounded, ratio = is_evidence_grounded("User enters an integer and it is stored in variable n.", ocr_text)
    assert grounded is True
    assert ratio >= 0.8

    # Paraphrased / OCR variation
    grounded2, ratio2 = is_evidence_grounded("Checks whether number is divisible by 2 using modulus", ocr_text)
    assert grounded2 is True
    assert ratio2 >= 0.8

    # Fabricated / Hallucinated sentence NOT in OCR text
    hallucinated = "The student implemented Euclidean greatest common divisor using recursion and bitwise shifts."
    grounded3, ratio3 = is_evidence_grounded(hallucinated, ocr_text)
    assert grounded3 is False
    assert ratio3 < 0.4


# ============================================================
# 3. TEST SCENARIO A: CURRENT 13-QUESTION C LABORATORY
# ============================================================

def test_scenario_a_13_question_c_laboratory():
    """
    Verifies that for the 13 C questions, the system:
    - Parses exactly 13 questions
    - Evaluates all 13 without hardcoding
    - Detects covered vs incomplete vs missing questions
    - Produces factual coverage percentage without fabricated grades
    """
    questions = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    assert len(questions) == 13

    # Sample report where P1 is fully explained, P2 is listed, and P3..P13 are missing
    ocr_text = """
    # Week 1 Observation Report
    Objective: To learn basic C programming control statements and arithmetic operators.
    Experiments covered:
    1. Even or odd
    2. Positive, negative or zero

    Program 1: Even or odd
    Problem Understanding: Checks whether a given integer is even or odd.
    Logic: Uses modulus operator n % 2 == 0. If remainder is 0, prints even, else prints odd.
    Important Variables:
    Variable n: stores input integer.
    Observation: Modulus by 2 produces remainder 0 for even numbers.
    """

    # Mock Qwen response following the dynamic schema
    mock_llm_json = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "To learn basic C programming control statements and arithmetic operators.",
            "evaluation": "Student clearly states the programming concepts practiced.",
            "confidence": "HIGH"
        },
        "questions": [
            {
                "question_number": 1,
                "question_text": questions[0].question_text,
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Even or odd",
                "confidence": "HIGH",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "Checks whether a given integer is even or odd.",
                        "evaluation": "Explains the input and expected output.",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": "Uses modulus operator n % 2 == 0.",
                        "evaluation": "Explains the remainder check logic.",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "PRESENT",
                        "evidence": "Variable n: stores input integer.",
                        "evaluation": "Describes input variable n.",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "PRESENT",
                        "evidence": "Modulus by 2 produces remainder 0 for even numbers.",
                        "evaluation": "Provides a specific mathematical observation.",
                        "confidence": "HIGH"
                    }
                ]
            },
            {
                "question_number": 2,
                "question_text": questions[1].question_text,
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "2. Positive, negative or zero",
                "confidence": "HIGH",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    # 1. Check all 13 questions are present
    assert len(eval_result.questions) == 13
    assert eval_result.summary["total_assigned"] == 13
    assert eval_result.summary["found_and_covered"] == 1
    assert eval_result.summary["found_but_incomplete"] == 1
    assert eval_result.summary["not_found"] == 11
    assert eval_result.summary["coverage_percentage"] == round((1 / 13) * 100, 1)

    # Render Markdown report
    md = render_verification_markdown(eval_result)
    assert "# 📊 Laboratory Observation Report Verification Report" in md
    assert "Total Assigned Questions:** 13" in md
    assert "Fully Addressed & Explained:** 1 / 13" in md
    assert "Mentioned / Incomplete (Title Only):** 1 / 13" in md
    assert "Not Found in Report:** 11 / 13" in md
    assert "9.5 / 10.0" not in md  # No invented numerical scores
    assert "Grade: A" not in md     # No invented letter grades


# ============================================================
# 4. TEST SCENARIO B: COMPLETELY DIFFERENT QUESTION SET (JAVA)
# ============================================================

def test_scenario_b_different_question_set_and_manual():
    """
    Verifies that the system seamlessly adapts to a completely different subject
    (Java OOP, 3 questions) and a completely different manual (Methodology, Test Cases, Discussion)
    without any code changes or C-specific artifacts.
    """
    java_questions_text = """1. Write a Java program to reverse a string using recursion.
2. Explain custom exception handling with an example.
3. Implement binary search on a sorted integer array."""

    custom_manual = """# Java Programming Laboratory Guide
## Objective
State the software design and programming objectives.
## Methodology
Explain the design methodology and class structure.
## Test Cases
Describe sample test cases and edge conditions tested.
## Discussion
Discuss computational complexity and memory tradeoffs.
"""

    ocr_text = """
    # Java Lab Report
    Objective: Implement OOP algorithms and test cases in Java.
    1. String Reversal:
    Methodology: Used a recursive function that takes the substring from index 1 and appends the first character to the end.
    Test Cases: Tested with 'hello' giving 'olleh', empty string returning empty string.
    Discussion: Recursion uses O(N) call stack space.
    """

    questions = parse_assigned_questions(java_questions_text)
    manual_reqs = parse_instruction_manual(custom_manual)

    assert len(questions) == 3
    assert "Methodology" in manual_reqs["per_question_requirements"]
    assert "Test Cases" in manual_reqs["per_question_requirements"]
    assert "Discussion" in manual_reqs["per_question_requirements"]

    mock_llm_json = {
        "objective": {
            "requirement": "Objective",
            "status": "PRESENT",
            "evidence": "Implement OOP algorithms and test cases in Java.",
            "evaluation": "Clear objective stated.",
            "confidence": "HIGH"
        },
        "questions": [
            {
                "question_number": 1,
                "question_text": questions[0].question_text,
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "1. String Reversal",
                "confidence": "HIGH",
                "requirements": [
                    {
                        "requirement": "Methodology",
                        "status": "PRESENT",
                        "evidence": "Used a recursive function that takes the substring from index 1 and appends the first character to the end.",
                        "evaluation": "Explains recursive methodology.",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Test Cases",
                        "status": "PRESENT",
                        "evidence": "Tested with 'hello' giving 'olleh', empty string returning empty string.",
                        "evaluation": "Documents input and expected output.",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Discussion",
                        "status": "PRESENT",
                        "evidence": "Recursion uses O(N) call stack space.",
                        "evaluation": "Analyzes call stack overhead.",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert len(eval_result.questions) == 3
    assert eval_result.summary["total_assigned"] == 3
    assert eval_result.summary["found_and_covered"] == 1
    assert eval_result.summary["not_found"] == 2

    # Render report
    md = render_verification_markdown(eval_result)
    assert "Methodology" in md
    assert "Test Cases" in md
    assert "Discussion" in md
    # Should not mention C or Problem Understanding
    assert "C Programming" not in md
    assert "Problem Understanding" not in md
    assert "Important Variables and Their Purpose" not in md


# ============================================================
# 5. TEST SCENARIO C: REPORT WITH ONLY PROGRAM TITLES PRESENT
# ============================================================

def test_scenario_c_title_only_report():
    """
    When a student report only contains program titles (e.g. In an index or 'Experiments covered' list),
    the system MUST classify them as FOUND_BUT_INCOMPLETE, all requirements MUST be MISSING,
    and the LLM must NEVER generate solutions.
    """
    questions_text = """1. Check whether a number is prime.
2. Calculate factorial of a number.
3. Check palindrome number."""
    questions = parse_assigned_questions(questions_text)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    ## Week 1 Laboratory Report
    Apparatus: Laptop, GCC compiler.
    Experiments covered:
    * Prime number
    * Factorial of a number
    * Palindrome number
    Conclusion: All experiments executed successfully.
    """

    # If the LLM incorrectly tried to mark FOUND_AND_COVERED with no evidence,
    # Python post-validation MUST downgrade it to FOUND_BUT_INCOMPLETE!
    raw_llm_with_attempted_overclaim = {
        "objective": None,
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check whether a number is prime.",
                "match_status": "FOUND_AND_COVERED",  # Overclaim by LLM!
                "matched_heading": "* Prime number",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            },
            {
                "question_number": 2,
                "question_text": "Calculate factorial of a number.",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "* Factorial of a number",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            },
            {
                "question_number": 3,
                "question_text": "Check palindrome number.",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "* Palindrome number",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=raw_llm_with_attempted_overclaim,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    # Question 1 MUST have been downgraded from FOUND_AND_COVERED to FOUND_BUT_INCOMPLETE
    assert eval_result.questions[0].match_status == "FOUND_BUT_INCOMPLETE"
    assert eval_result.questions[1].match_status == "FOUND_BUT_INCOMPLETE"
    assert eval_result.questions[2].match_status == "FOUND_BUT_INCOMPLETE"

    # Fully covered count must be ZERO
    assert eval_result.summary["found_and_covered"] == 0
    assert eval_result.summary["found_but_incomplete"] == 3
    assert eval_result.summary["coverage_percentage"] == 0.0

    # Ensure all requirements are MISSING and evidence is None
    for q in eval_result.questions:
        for r in q.requirements:
            assert r.status == "MISSING"
            assert r.evidence is None


# ============================================================
# 6. TEST SCENARIO D: ONE PROGRAM DETAILED, OTHERS ONLY LISTED
# ============================================================

def test_scenario_d_one_detailed_others_listed():
    """
    Verifies that when one program has detailed evidence and others are only listed,
    the system correctly classifies only the first as FOUND_AND_COVERED and others as FOUND_BUT_INCOMPLETE.
    """
    questions_text = """1. Even or odd
2. Leap year
3. Memory allocation"""
    questions = parse_assigned_questions(questions_text)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    Experiments performed:
    - Even or odd
    - Leap year
    - Memory allocation

    Program 1: Even or odd
    Problem Understanding: The program takes an integer n and checks whether it is divisible by 2.
    Logic: We use the remainder operator n % 2. If the remainder is 0, it is even; otherwise it is odd.
    Important Variables:
    | Variable | Purpose |
    | n | Input number |
    Observation: Modulo operation by 2 yields remainder 0 for even numbers and 1 for odd numbers.
    """

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Even or odd",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Even or odd",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "The program takes an integer n and checks whether it is divisible by 2.",
                        "evaluation": "Clear explanation of input and expected output."
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": "We use the remainder operator n % 2. If the remainder is 0, it is even; otherwise it is odd.",
                        "evaluation": "Explains condition checking and branching."
                    },
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "PRESENT",
                        "evidence": "n | Input number",
                        "evaluation": "Documents input variable n."
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "PRESENT",
                        "evidence": "Modulo operation by 2 yields remainder 0 for even numbers and 1 for odd numbers.",
                        "evaluation": "Meaningful mathematical observation."
                    }
                ]
            },
            {
                "question_number": 2,
                "question_text": "Leap year",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "- Leap year",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            },
            {
                "question_number": 3,
                "question_text": "Memory allocation",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "- Memory allocation",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables and Their Purpose", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None}
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_result.summary["found_and_covered"] == 1
    assert eval_result.summary["found_but_incomplete"] == 2
    assert eval_result.summary["not_found"] == 0
    assert eval_result.summary["coverage_percentage"] == 33.3

    assert eval_result.questions[0].match_status == "FOUND_AND_COVERED"
    assert eval_result.questions[1].match_status == "FOUND_BUT_INCOMPLETE"
    assert eval_result.questions[2].match_status == "FOUND_BUT_INCOMPLETE"


# ============================================================
# 7. TEST SCENARIO E: GENERIC OBSERVATIONS
# ============================================================

def test_scenario_e_generic_observation_flagging():
    """
    Verifies that generic statements like 'All programs compiled and executed successfully'
    are classified as PARTIAL or flagged by the evaluator rather than counting as a valid program observation.
    """
    ocr_text = """
    Program 1: Prime number
    Problem Understanding: Checks if number is prime.
    Observations:
    All programs were compiled and executed successfully without any errors. The output matched expected results.
    """
    questions = parse_assigned_questions("1. Check prime number.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check prime number.",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Prime number",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "Checks if number is prime.",
                        "evaluation": "Valid problem description."
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "MISSING",
                        "evidence": None
                    },
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "MISSING",
                        "evidence": None
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "PARTIAL",
                        "evidence": "All programs were compiled and executed successfully without any errors.",
                        "evaluation": "Generic execution statement; does not describe program behavior."
                    }
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    obs_req = [r for r in eval_result.questions[0].requirements if "observed" in r.requirement.lower()][0]
    assert obs_req.status == "PARTIAL"
    assert obs_req.evidence is not None


# ============================================================
# 8. TEST SCENARIO F: OCR CONTAINING SPELLING ERRORS & NOISE
# ============================================================

def test_scenario_f_ocr_spelling_errors_and_noise():
    """
    Verifies that evidence grounding is robust against common OCR noise (spelling mistakes,
    broken characters, merged tokens) without inventing new substantive claims.
    """
    ocr_text = """
    ## Objeclive:
    To leam basic programing using input, outpul and conditonal stalements.
    Exp:
    1. Euen or odd numbr
    logic usd: checks whethr the numbr is plausble by 2. modulus operalor % is usd.
    Control flow: If n%2==0 prnts even, othr wise prnts odd.
    """
    questions = parse_assigned_questions("1. Write a C program to check whether a given number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    # Student evidence contains the noisy text
    noisy_evidence = "checks whethr the numbr is plausble by 2. modulus operalor % is usd."
    is_grounded, ratio = is_evidence_grounded(noisy_evidence, ocr_text)
    assert is_grounded is True
    assert ratio >= 0.8

    mock_llm_json = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "To leam basic programing using input, outpul and conditonal stalements.",
            "evaluation": "Covers programming concepts despite OCR spelling errors.",
            "confidence": "MEDIUM"
        },
        "questions": [
            {
                "question_number": 1,
                "question_text": questions[0].question_text,
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "1. Euen or odd numbr",
                "confidence": "MEDIUM",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "MISSING",
                        "evidence": None
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": noisy_evidence,
                        "evaluation": "Student explains remainder divisibility check.",
                        "confidence": "MEDIUM"
                    },
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "MISSING",
                        "evidence": None
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "MISSING",
                        "evidence": None
                    }
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_result.questions[0].match_status == "FOUND_AND_COVERED"
    logic_req = [r for r in eval_result.questions[0].requirements if "logic" in r.requirement.lower()][0]
    assert logic_req.status == "PRESENT"
    assert logic_req.grounded is True
    assert logic_req.confidence == "MEDIUM"


# ============================================================
# 9. UNGROUNDED / FABRICATED EVIDENCE DETECTION TEST
# ============================================================

def test_hallucination_detection_and_demotion():
    """
    Verifies that if the LLM hallucinates an explanation that NEVER appears in the OCR text,
    Python post-validation detects it, marks grounded=False, and demotes status to MISSING.
    """
    ocr_text = "Experiments: 1. Prime number\nNothing else written here."
    questions = parse_assigned_questions("1. Check whether a number is prime.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    # LLM hallucinates a complete algorithm not in OCR
    fabricated_evidence = "The student used the Sieve of Eratosthenes to find prime factors with array indexing."

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check whether a number is prime.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": fabricated_evidence,
                        "evaluation": "Algorithm described in detail."
                    }
                ]
            }
        ]
    }

    eval_result = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    # Logic requirement MUST be demoted to MISSING because evidence is ungrounded
    logic_req = [r for r in eval_result.questions[0].requirements if "logic" in r.requirement.lower()][0]
    assert logic_req.status == "MISSING"
    assert logic_req.grounded is False
    assert logic_req.evidence is None
    assert "[Flagged: Evidence not found in OCR text]" in logic_req.evaluation

    # Question 1 must be demoted to FOUND_BUT_INCOMPLETE because all requirements are now MISSING!
    assert eval_result.questions[0].match_status == "FOUND_BUT_INCOMPLETE"


# ============================================================
# 10. ADVERSARIAL GROUNDING & EVIDENCE INTEGRITY TESTS
# ============================================================

def test_adversarial_test_1_expanded_sentence_rejected_as_verbatim_span():
    """
    TEST 1:
    OCR: "The program uses modulus operator."
    LLM evidence: "The program uses modulus operator to correctly identify even numbers."
    Expected: Evidence must NOT be accepted as a verbatim grounded span.
    """
    ocr_text = "The program uses modulus operator."
    llm_evidence = "The program uses modulus operator to correctly identify even numbers."

    is_grounded, ratio = is_evidence_grounded(llm_evidence, ocr_text)
    # The full sentence must NOT be accepted as a verbatim grounded span
    assert is_grounded is False
    assert ratio < 0.75


def test_adversarial_test_2_checks_whether_divisible_and_prints_rejected():
    """
    TEST 2:
    OCR: "number divisible by 2"
    LLM evidence: "The student checks whether the number is divisible by 2 and prints even otherwise odd."
    Expected: Reject/demote unless the complete evidence is actually present.
    """
    ocr_text = "number divisible by 2"
    llm_evidence = "The student checks whether the number is divisible by 2 and prints even otherwise odd."

    is_grounded, ratio = is_evidence_grounded(llm_evidence, ocr_text)
    assert is_grounded is False
    assert ratio < 0.50

    # Also verify that in validate_and_ground_evaluation, this is rejected/demoted
    questions = parse_assigned_questions("1. Check whether a number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check whether a number is even or odd.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": llm_evidence,
                        "evaluation": "Full logic described."
                    }
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(mock_llm_json, ocr_text, questions, manual_reqs)
    logic_req = [r for r in eval_res.questions[0].requirements if "logic" in r.requirement.lower()][0]
    # Complete evidence was not present, so requirement must be demoted or trimmed
    assert logic_req.status == "MISSING" or logic_req.evidence != llm_evidence


def test_adversarial_test_3_leap_year_unsupported_algorithm_missing():
    """
    TEST 3:
    OCR: "leap year"
    LLM evidence: "The year is divisible by 400 or divisible by 4 but not 100."
    Expected: MISSING / ungrounded.
    """
    ocr_text = "Program 1: leap year\nEnd of experiment."
    llm_evidence = "The year is divisible by 400 or divisible by 4 but not 100."

    is_grounded, ratio = is_evidence_grounded(llm_evidence, ocr_text)
    assert is_grounded is False
    assert ratio < 0.30

    questions = parse_assigned_questions("1. Check whether a given year is a leap year.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check whether a given year is a leap year.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": llm_evidence,
                        "evaluation": "Divisibility algorithm given."
                    }
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(mock_llm_json, ocr_text, questions, manual_reqs)
    logic_req = [r for r in eval_res.questions[0].requirements if "logic" in r.requirement.lower()][0]
    assert logic_req.status == "MISSING"
    assert logic_req.grounded is False
    assert logic_req.evidence is None


def test_adversarial_test_4_only_source_fragment_accepted_not_expanded_sentence():
    """
    TEST 4:
    OCR: "uses temporary variable"
    LLM evidence: "The student swaps the values using a temporary variable."
    Expected: Only accept the source fragment "uses temporary variable", not the
    expanded generated sentence.
    """
    ocr_text = "In this program the student uses temporary variable to hold the value."
    llm_evidence = "The student swaps the values using a temporary variable."

    # The full sentence itself is NOT in OCR
    full_grounded, _ = is_evidence_grounded(llm_evidence, ocr_text)
    assert full_grounded is False

    # But extract_grounded_source_span extracts ONLY the pure source fragment
    extracted_span = extract_grounded_source_span(llm_evidence, ocr_text)
    assert extracted_span is not None
    assert "temporary variable" in extracted_span
    # Verify the hallucinated prefix "The student swaps the values" is NOT in the extracted span
    assert "The student swaps" not in extracted_span

    # Verify that in validate_and_ground_evaluation, evidence contains only the source fragment
    questions = parse_assigned_questions("1. Swap two numbers.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Swap two numbers.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": llm_evidence,
                        "evaluation": "Describes using temporary variable."
                    }
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(mock_llm_json, ocr_text, questions, manual_reqs)
    logic_req = [r for r in eval_res.questions[0].requirements if "logic" in r.requirement.lower()][0]
    assert logic_req.status == "PRESENT"
    assert logic_req.evidence != llm_evidence
    assert "The student swaps" not in logic_req.evidence
    assert "temporary variable" in logic_req.evidence


def test_adversarial_test_5_generic_observation_rejected_as_program_evidence():
    """
    TEST 5:
    OCR contains a generic observation:
    "All programs executed successfully."
    LLM evidence:
    "The Even/Odd program correctly determines whether the number is even."
    Expected: Reject as program-specific evidence.
    """
    ocr_text = "All programs executed successfully."
    llm_evidence = "The Even/Odd program correctly determines whether the number is even."

    # 1. Hallucinated program claim is completely ungrounded in OCR
    is_grounded, ratio = is_evidence_grounded(llm_evidence, ocr_text)
    assert is_grounded is False
    assert ratio < 0.25

    # 2. Even if Qwen directly cites "All programs executed successfully", is_generic_observation flags it
    assert is_generic_observation(ocr_text) is True
    assert is_generic_observation("The program executed successfully") is True
    assert is_generic_observation("I got the correct output") is True
    assert is_generic_observation("Outputs were verified") is True

    # Meaningful observation is NOT flagged as generic
    assert is_generic_observation("I observed that n % 2 gives remainder 0 for even numbers.") is False


def test_question_matching_vs_section_evidence_separation():
    """
    Requirement 8:
    Do not confuse successful question matching with successful section evidence.
    Question: "Write a program to check whether a number is even or odd."
    Student: "Even or odd"
    Expected:
      question matching = successful (FOUND_BUT_INCOMPLETE)
      Problem Understanding = MISSING
      Logic = MISSING
      Variables = MISSING
      Observation = MISSING
    """
    ocr_text = "Index of Programs:\n1. Even or odd\n2. Factorial"
    questions = parse_assigned_questions("1. Write a program to check whether a number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Write a program to check whether a number is even or odd.",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "1. Even or odd",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "MISSING", "evidence": None},
                    {"requirement": "Logic / Approach Used", "status": "MISSING", "evidence": None},
                    {"requirement": "Important Variables", "status": "MISSING", "evidence": None},
                    {"requirement": "What I Observed", "status": "MISSING", "evidence": None},
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(mock_llm_json, ocr_text, questions, manual_reqs)
    q1 = eval_res.questions[0]
    assert q1.match_status == "FOUND_BUT_INCOMPLETE"
    assert q1.matched_heading == "1. Even or odd"
    for r in q1.requirements:
        assert r.status == "MISSING"
        assert r.evidence is None


def test_multi_span_evidence_support():
    """
    Requirement 5:
    If a requirement is supported by multiple separate OCR fragments,
    support multiple evidence spans rather than merging them into a new sentence.
    """
    ocr_text = """
    Program 1: Even or Odd
    The program checks divisibility using modulus operator %.
    The condition tested is n % 2 == 0.
    """
    spans = ["modulus operator %", "n % 2 == 0"]

    # Grounding check handles list of spans
    is_gr, ratio = is_evidence_grounded(spans, ocr_text)
    assert is_gr is True
    assert ratio >= 0.9

    questions = parse_assigned_questions("1. Check whether a number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Check whether a number is even or odd.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": spans,
                        "evaluation": "Operator and condition described.",
                        "reasoning": "Both fragments directly show the logic."
                    }
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(mock_llm_json, ocr_text, questions, manual_reqs)
    logic_req = [r for r in eval_res.questions[0].requirements if "logic" in r.requirement.lower()][0]
    assert logic_req.status == "PRESENT"
    assert isinstance(logic_req.evidence, list)
    assert len(logic_req.evidence) == 2
    assert "modulus operator %" in logic_req.evidence
    assert "n % 2 == 0" in logic_req.evidence

    # Verify Markdown rendering formats multi-span evidence properly
    md_output = render_verification_markdown(eval_res)
    assert "Student Evidence (Verbatim OCR Spans):" in md_output
    assert "\"modulus operator %\"" in md_output
    assert "\"n % 2 == 0\"" in md_output


def test_three_way_separation_evidence_evaluation_reasoning():
    """
    Requirement 2:
    Maintain three separate concepts:
    EVIDENCE: Actual text/span from student OCR.
    EVALUATION: Whether that evidence satisfies requirement.
    REASONING: Why classified as PRESENT/PARTIAL/MISSING.
    """
    ocr_text = "Checks divisibility using modulus operator %."
    req = RequirementEvaluation(
        requirement="Logic / Approach Used",
        status="PRESENT",
        evidence="modulus operator %",
        evaluation="The report explains the operator used to check divisibility.",
        reasoning="This directly addresses the required logic.",
        confidence="HIGH",
        grounded=True
    )
    assert req.evidence == "modulus operator %"
    assert "explains the operator" in req.evaluation
    assert "directly addresses" in req.reasoning
    # Evidence does NOT contain evaluation or reasoning text
    assert req.evaluation not in req.evidence
    assert req.reasoning not in req.evidence


# ============================================================
# 11. TARGETED QUESTION MATCHING ACCURACY TESTS (10 TESTS)
# ============================================================

def test_qmatch_1_exact_heading_match():
    """
    Test 1: Exact heading match.
    Assigned questions are matched decisively via explicit heading.
    """
    questions = parse_assigned_questions("""1. Calculate factorial of a number.
2. Check whether a number is prime.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    # Program 1: Calculate factorial of a number
    Problem Understanding: Multiplies integers from 1 to n.
    Logic: Uses a loop fact = fact * i.
    Important Variables: fact: accumulator variable.
    Observation: Factorial increases rapidly with n.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},  # Pure Python matching & post-validation
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    q1 = eval_res.questions[0]
    q2 = eval_res.questions[1]

    assert q1.match_status == "FOUND_AND_COVERED"
    assert "Calculate factorial of a number" in q1.matched_heading
    assert q1.match_confidence == "HIGH"
    assert q2.match_status == "NOT_FOUND"


def test_qmatch_2_semantic_match_different_wording():
    """
    Test 2: Semantic match with different wording.
    Student describes even/odd without using words 'even' or 'odd' in heading.
    Preserves exact OCR fragment as evidence, not generated explanation.
    """
    questions = parse_assigned_questions("1. Write a program to check whether a number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    ## Experiment 1: Number Property Analysis
    Problem Understanding: The value is divided by 2 and remainder is checked.
    Logic: Uses remainder operator to determine if remainder equals zero.
    Important Variables: num: entered value.
    Observation: Modulus by 2 separates values into two categories.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    q1 = eval_res.questions[0]
    assert q1.match_status == "FOUND_AND_COVERED"
    assert q1.match_confidence in ("HIGH", "MEDIUM")
    # Grounded evidence is preserved
    assert q1.match_evidence is not None
    # Crucial: OCR text fragment preserved, not hallucinated text
    assert is_evidence_grounded(q1.match_evidence[0], ocr_text)[0] is True


def test_qmatch_3_similar_questions_not_confused():
    """
    Test 3: Similar questions that should not be confused.
    Factorial and Fibonacci are both numerical loop algorithms.
    Student describes Fibonacci logic; system MUST NOT associate with Factorial.
    """
    questions = parse_assigned_questions("""1. Calculate factorial of a number.
2. Generate Fibonacci series.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    ## Numerical Experiment
    Problem Understanding: In this procedure previous two values are added to obtain next term.
    Logic: f0 + f1 gives f2, then update first two terms.
    Important Variables: f0, f1: store previous two values.
    Observation: Every term depends on sum of previous two.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    q1 = eval_res.questions[0]  # Factorial
    q2 = eval_res.questions[1]  # Fibonacci

    # Must match Fibonacci, NOT Factorial
    assert q2.match_status == "FOUND_AND_COVERED"
    assert q1.match_status == "NOT_FOUND"


def test_qmatch_4_ambiguous_short_ocr_content():
    """
    Test 4: Ambiguous short OCR content.
    When student content is too brief or ambiguous between similar questions:
    DO NOT GUESS. Return LOW_CONFIDENCE / NOT_FOUND.
    """
    questions = parse_assigned_questions("""1. Find the largest of three numbers.
2. Find the smallest of three numbers.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = "three numbers"

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    # Neither should be forcefully matched
    assert eval_res.questions[0].match_status == "NOT_FOUND"
    assert eval_res.questions[1].match_status == "NOT_FOUND"
    assert eval_res.questions[0].match_confidence == "LOW"
    assert eval_res.questions[1].match_confidence == "LOW"


def test_qmatch_5_ocr_spelling_errors():
    """
    Test 5: OCR spelling errors.
    Tolerates minor OCR distortions: factoria1 -> factorial, palindrom -> palindrome.
    """
    questions = parse_assigned_questions("""1. Write a program to calculate factorial of a number.
2. Check if a string is a palindrome.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    # Program 1: factoria1 of a number
    Problem Understanding: Computes factoria1 of input n.
    Logic: fact = fact * i.
    Important Variables: fact: result.
    Observation: Multiplies all terms.

    # Program 2: palindrom check
    Problem Understanding: Checks if string is palindrom.
    Logic: Compares characters from left and right.
    Important Variables: str: input string.
    Observation: Original equals reversed.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_res.questions[0].match_status == "FOUND_AND_COVERED"
    assert eval_res.questions[1].match_status == "FOUND_AND_COVERED"


def test_qmatch_6_same_keyword_in_multiple_questions():
    """
    Test 6: Same keyword appearing in multiple questions.
    Two questions share 'digits':
    Q1: 'Calculate sum of digits of a given number.'
    Q2: 'Reverse digits of a number to check palindrome.'
    Student only completes Q1. Q2 must NOT be falsely triggered.
    """
    questions = parse_assigned_questions("""1. Calculate sum of digits of a given number.
2. Reverse digits of a number to check palindrome.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    Program 1: Sum of digits
    Problem Understanding: Calculates sum of digits extracted from n.
    Logic: sum += n % 10 and n /= 10 in while loop.
    Important Variables: sum: stores accumulated sum of digits.
    Observation: Each digit is stripped and added.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_res.questions[0].match_status == "FOUND_AND_COVERED"
    assert eval_res.questions[1].match_status == "NOT_FOUND"


def test_qmatch_7_matching_followed_by_missing_section_detection():
    """
    Test 7: Correct question matching followed by missing section detection.
    Question is matched by heading, but sections are completely absent.
    Must be FOUND_BUT_INCOMPLETE, NOT FOUND_AND_COVERED.
    """
    questions = parse_assigned_questions("1. Write a program to check whether a given year is a leap year.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = "Program 1: Leap Year Program"

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    q1 = eval_res.questions[0]
    assert q1.match_status == "FOUND_BUT_INCOMPLETE"
    assert q1.matched_heading == "Program 1: Leap Year Program"
    for r in q1.requirements:
        assert r.status == "MISSING"
        assert r.evidence is None


def test_qmatch_8_semantic_match_followed_by_evidence_grounding_rejection():
    """
    Test 8: Semantic match followed by evidence-grounding rejection.
    Even when question matching associates content with a question,
    hallucinated/ungrounded evidence MUST be rejected by the grounding gate.
    """
    questions = parse_assigned_questions("1. Write a program to check whether a number is even or odd.")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    Program 1: Number Parity Check
    Problem Understanding: remainder is checked with 2.
    """

    # Mock Qwen trying to generate an ungrounded expansion
    mock_llm_json = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Write a program to check whether a number is even or odd.",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Number Parity Check",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "The student checks whether the number is even or odd by dividing by 2 and inspecting remainder.",
                        "evaluation": "Clear explanation."
                    }
                ]
            }
        ]
    }

    eval_res = validate_and_ground_evaluation(
        raw_data=mock_llm_json,
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    prob_req = [r for r in eval_res.questions[0].requirements if "problem" in r.requirement.lower()][0]
    # The hallucinated expansion MUST not pass as grounded evidence
    assert prob_req.status == "MISSING" or prob_req.evidence != "The student checks whether the number is even or odd by dividing by 2 and inspecting remainder."


def test_qmatch_9_multiple_questions_on_one_page():
    """
    Test 9: Multiple questions on one page.
    Locality engine preserves distinct blocks for each program on the same page.
    """
    questions = parse_assigned_questions("""1. Check whether a number is even or odd.
2. Check whether a number is positive, negative or zero.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    # Laboratory Sheet - Page 1
    Program 1: Even or odd
    Problem Understanding: Checks if number is even or odd.
    Logic: n % 2 == 0.
    Important Variables: n: input value.
    Observation: Remainder 0 indicates even.

    Program 2: Positive or negative
    Problem Understanding: Checks if number is positive, negative or zero.
    Logic: if n > 0 positive, else if n < 0 negative, else zero.
    Important Variables: n: input value.
    Observation: Compares value against zero.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_res.questions[0].match_status == "FOUND_AND_COVERED"
    assert eval_res.questions[1].match_status == "FOUND_AND_COVERED"
    assert eval_res.summary["found_and_covered"] == 2


def test_qmatch_10_questions_in_different_order_from_assignment():
    """
    Test 10: Questions appearing in a different order from the assignment.
    Assigned: Q1 Even/Odd, Q2 Factorial, Q3 Fibonacci.
    Student writes: Fibonacci -> Factorial -> Even/Odd.
    System matches accurately regardless of sequence.
    """
    questions = parse_assigned_questions("""1. Check whether a number is even or odd.
2. Calculate factorial of a number.
3. Generate Fibonacci series.""")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    ocr_text = """
    ## Program: Fibonacci series
    Problem Understanding: Generates Fibonacci sequence.
    Logic: f0 + f1 produces next term.
    Important Variables: f0, f1: initial terms.
    Observation: Sum of previous two terms.

    ## Program: Factorial
    Problem Understanding: Calculates factorial of n.
    Logic: fact = fact * i in loop.
    Important Variables: fact: product.
    Observation: Multiplies 1 to n.

    ## Program: Even or odd
    Problem Understanding: Determines parity of integer.
    Logic: n % 2 == 0.
    Important Variables: n: input.
    Observation: Remainder 0 for even.
    """

    eval_res = validate_and_ground_evaluation(
        raw_data={},
        ocr_text=ocr_text,
        questions=questions,
        manual_requirements=manual_reqs
    )

    assert eval_res.questions[0].match_status == "FOUND_AND_COVERED"
    assert "Even or odd" in eval_res.questions[0].matched_heading
    assert eval_res.questions[1].match_status == "FOUND_AND_COVERED"
    assert "Factorial" in eval_res.questions[1].matched_heading
    assert eval_res.questions[2].match_status == "FOUND_AND_COVERED"
    assert "Fibonacci series" in eval_res.questions[2].matched_heading
    assert eval_res.summary["found_and_covered"] == 3

