"""
tests/test_final_scoring_layer.py
Comprehensive verification suite for the Final Scoring Layer of the
Observation Report Verification system.

Covers all 20 required test scenarios:
1. Objective present -> receives score.
2. Objective missing -> 0.
3. Example sections in manual are ignored.
4. Exact question with all four sections -> 8/8.
5. Question title only -> 0/8.
6. Problem Understanding only -> 2/8.
7. Logic only -> 2/8.
8. Variables only -> 2/8.
9. Meaningful observation only -> 2/8.
10. Generic "executed successfully" observation -> 0 for observation.
11. Partial criterion -> 1 mark.
12. Dynamic question count of 3.
13. Dynamic question count of 5.
14. Dynamic question count of 13.
15. Evidence missing -> score cannot be > 0.
16. Hallucinated/generated content not present in OCR -> 0.
17. Reordered questions.
18. Multiple questions on same page.
19. OCR spelling errors.
20. Ambiguous question matching.
"""

import os
import json
import pytest

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
    calculate_criterion_score,
    calculate_objective_score,
    generate_overall_assessment,
    validate_and_ground_evaluation,
    render_verification_markdown,
    parse_evaluation_scores,
    is_generic_observation,
    format_extraction_for_evaluation,
    OFFICIAL_INSTRUCTION_MANUAL,
    WEEK1_13_PROGRAMS_PRESET,
)


# ============================================================
# 1. OBJECTIVE PRESENT -> RECEIVES SCORE
# ============================================================

def test_01_objective_present_receives_score():
    ocr_text = """
    ## Objective of the Lab
    To understand arithmetic operations, loops, and conditions in C programming.
    """
    raw_data = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "understand arithmetic operations, loops, and conditions in C programming",
            "evaluation": "Clear objective stating core concepts.",
            "reasoning": "Concepts and skills are clearly stated.",
            "confidence": "HIGH"
        },
        "questions": []
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [], manual_reqs)

    assert res.objective is not None
    assert res.objective.status == "PRESENT"
    assert res.objective.score == 2
    assert res.objective.max_score == 2
    assert res.summary["objective_score"] == 2
    assert res.summary["objective_max"] == 2


# ============================================================
# 2. OBJECTIVE MISSING -> 0
# ============================================================

def test_02_objective_missing_receives_zero():
    ocr_text = """
    ## Program 1: Factors
    Input n and find all numbers dividing n.
    """
    raw_data = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "MISSING",
            "evidence": None,
            "evaluation": "No objective statement found.",
            "reasoning": "Section omitted.",
            "confidence": "HIGH"
        },
        "questions": []
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [], manual_reqs)

    assert res.objective is not None
    assert res.objective.status == "MISSING"
    assert res.objective.score == 0
    assert res.objective.max_score == 2
    assert res.summary["objective_score"] == 0


# ============================================================
# 3. EXAMPLE SECTIONS IN MANUAL ARE IGNORED
# ============================================================

def test_03_example_sections_in_manual_are_ignored():
    # The official manual contains examples:
    # Example: Factorial Program, Example: Prime Number Program, Example: Palindrome Program,
    # Example: Factors, Example: Fibonacci Series
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    per_q = manual_reqs["per_question_requirements"]

    forbidden_examples = [
        "Example: Factorial Program",
        "Example: Prime Number Program",
        "Example: Palindrome Program",
        "Example: Factors",
        "Example: Factorial",
        "Example: Palindrome",
        "Example: Prime Number",
        "Example: Fibonacci Series",
        "Example",
        "Program 1: Factors of a Number"
    ]

    for fb in forbidden_examples:
        assert fb not in per_q, f"Forbidden example heading '{fb}' found in requirements!"

    # Exactly the 4 required criteria should be present
    expected_criteria = [
        "Problem Understanding",
        "Logic / Approach Used",
        "Important Variables and Their Purpose",
        "What I Observed"
    ]
    assert per_q == expected_criteria

    # Ensure validate_and_ground_evaluation also filters them out
    ocr_text = "Objective: Test\nProgram 1: Factors\nProblem: Find factors"
    q1 = AssignedQuestion(question_number=1, question_text="Find factors")
    res = validate_and_ground_evaluation({}, ocr_text, [q1], manual_reqs)
    req_names = [r.requirement for r in res.questions[0].requirements]
    for fb in forbidden_examples:
        assert fb not in req_names


# ============================================================
# 4. EXACT QUESTION WITH ALL FOUR SECTIONS -> 8/8
# ============================================================

def test_04_exact_question_with_all_four_sections_receives_8_of_8():
    ocr_text = """
    ## Program 1: Factorial of a Number
    ### Problem Understanding
    The program takes an integer n and computes the product of 1 to n.
    ### Logic / Approach Used
    Initialize fact to 1. Loop i from 1 to n multiplying fact by i.
    ### Important Variables and Their Purpose
    n stores input number, fact stores running factorial result.
    ### What I Observed
    The factorial grows rapidly and the previous result is required for the next term.
    """
    raw_data = {
        "objective": None,
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factorial of a Number",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Factorial of a Number",
                "confidence": "HIGH",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "takes an integer n and computes the product of 1 to n",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": "Initialize fact to 1. Loop i from 1 to n multiplying fact by i",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "PRESENT",
                        "evidence": "n stores input number, fact stores running factorial result",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "PRESENT",
                        "evidence": "factorial grows rapidly and the previous result is required for the next term",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factorial of a Number")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.match_status == "FOUND_AND_COVERED"
    assert q_eval.score == 8
    assert q_eval.max_score == 8
    for r in q_eval.requirements:
        assert r.score == 2
        assert r.max_score == 2


# ============================================================
# 5. QUESTION TITLE ONLY -> 0/8
# ============================================================

def test_05_question_title_only_receives_0_of_8():
    ocr_text = """
    List of Experiments:
    1. Program 1: Factorial of a Number
    2. Program 2: Prime Number
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factorial of a Number",
                "match_status": "FOUND_BUT_INCOMPLETE",
                "matched_heading": "Program 1: Factorial of a Number",
                "confidence": "HIGH",
                "requirements": []
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factorial of a Number")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.match_status == "FOUND_BUT_INCOMPLETE"
    assert q_eval.score == 0
    assert q_eval.max_score == 8
    for r in q_eval.requirements:
        assert r.score == 0


# ============================================================
# 6. PROBLEM UNDERSTANDING ONLY -> 2/8
# ============================================================

def test_06_problem_understanding_only_receives_2_of_8():
    ocr_text = """
    ## Program 1: Factorial of a Number
    ### Problem Understanding
    The program takes an integer n and computes the product of 1 to n.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factorial of a Number",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "takes an integer n and computes the product of 1 to n",
                        "confidence": "HIGH"
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
                        "status": "MISSING",
                        "evidence": None
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factorial of a Number")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.score == 2
    assert q_eval.max_score == 8
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["Problem Understanding"] == 2
    assert req_map["Logic / Approach Used"] == 0
    assert req_map["Important Variables and Their Purpose"] == 0
    assert req_map["What I Observed"] == 0


# ============================================================
# 7. LOGIC ONLY -> 2/8
# ============================================================

def test_07_logic_only_receives_2_of_8():
    ocr_text = """
    ## Program 1: Prime Number
    ### Logic / Approach Used
    Loop from 2 to n-1 checking if n % i == 0. If remainder is zero then flag not prime.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Prime Number",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": "Loop from 2 to n-1 checking if n % i == 0",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Prime Number")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.score == 2
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["Logic / Approach Used"] == 2
    assert req_map["Problem Understanding"] == 0


# ============================================================
# 8. VARIABLES ONLY -> 2/8
# ============================================================

def test_08_variables_only_receives_2_of_8():
    ocr_text = """
    ## Program 1: Palindrome
    The variables used are num for the input value and rev for storing the reversed digits.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Palindrome",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Important Variables and Their Purpose",
                        "status": "PRESENT",
                        "evidence": "num for the input value and rev for storing the reversed digits",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Palindrome")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.score == 2
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["Important Variables and Their Purpose"] == 2


# ============================================================
# 9. MEANINGFUL OBSERVATION ONLY -> 2/8
# ============================================================

def test_09_meaningful_observation_only_receives_2_of_8():
    ocr_text = """
    ## Program 1: Factors
    ### What I Observed
    I observed that numbers greater than n/2 cannot be factors except for n itself.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factors",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "What I Observed",
                        "status": "PRESENT",
                        "evidence": "numbers greater than n/2 cannot be factors except for n itself",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factors")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    assert q_eval.score == 2
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["What I Observed"] == 2


# ============================================================
# 10. GENERIC "EXECUTED SUCCESSFULLY" OBSERVATION -> 0 FOR OBSERVATION
# ============================================================

def test_10_generic_observation_receives_zero():
    ocr_text = """
    ## Program 1: Factors
    ### Problem Understanding
    Identify all integers that divide n without remainder.
    ### What I Observed
    The program executed successfully and I got the correct output.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factors",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "Identify all integers that divide n without remainder",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "What I Observed",
                        "status": "PRESENT",
                        "evidence": "The program executed successfully and I got the correct output",
                        "confidence": "LOW"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factors")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["Problem Understanding"] == 2
    assert req_map["What I Observed"] == 0  # Generic observation receives 0 marks!


# ============================================================
# 11. PARTIAL CRITERION -> 1 MARK
# ============================================================

def test_11_partial_criterion_receives_1_mark():
    ocr_text = """
    ## Program 1: Fibonacci
    ### Problem Understanding
    Generate numbers.
    ### Logic / Approach Used
    Add previous numbers in loop.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Fibonacci",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PARTIAL",
                        "evidence": "Generate numbers",
                        "confidence": "MEDIUM"
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PARTIAL",
                        "evidence": "Add previous numbers in loop",
                        "confidence": "MEDIUM"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Fibonacci")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    req_map = {r.requirement: r.score for r in q_eval.requirements}
    assert req_map["Problem Understanding"] == 1
    assert req_map["Logic / Approach Used"] == 1
    assert q_eval.score == 2


# ============================================================
# 12. DYNAMIC QUESTION COUNT OF 3
# ============================================================

def test_12_dynamic_question_count_of_3():
    questions = [
        AssignedQuestion(question_number=1, question_text="Factors"),
        AssignedQuestion(question_number=2, question_text="Factorial"),
        AssignedQuestion(question_number=3, question_text="Prime")
    ]
    ocr_text = "Objective: Loops in C.\nProgram 1: Factors\nProblem: Divisors."
    raw_data = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "Loops in C",
            "confidence": "HIGH"
        },
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factors",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Divisors"}
                ]
            }
        ]
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, questions, manual_reqs)

    # total_max = 2 + 3 * 8 = 26
    assert res.summary["total_assigned"] == 3
    assert res.summary["total_max"] == 26
    # Objective = 2, Q1 = 2 (Problem Understanding), Q2 = 0, Q3 = 0 -> total = 4
    assert res.summary["total_obtained"] == 4
    # final_score = round((4 / 26) * 10, 2) = 1.54
    assert res.summary["final_score"] == round((4 / 26) * 10, 2)


# ============================================================
# 13. DYNAMIC QUESTION COUNT OF 5
# ============================================================

def test_13_dynamic_question_count_of_5():
    questions = [
        AssignedQuestion(question_number=i, question_text=f"Task {i}")
        for i in range(1, 6)
    ]
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation({}, "Sample text", questions, manual_reqs)

    # total_max = 2 + 5 * 8 = 42
    assert res.summary["total_assigned"] == 5
    assert res.summary["total_max"] == 42
    assert res.summary["total_obtained"] == 0
    assert res.summary["final_score"] == 0.0


# ============================================================
# 14. DYNAMIC QUESTION COUNT OF 13 (AND 106-MARKS NORMALIZATION)
# ============================================================

def test_14_dynamic_question_count_of_13():
    q13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    assert len(q13) == 13

    # Construct simulation matching the user's example in the prompt:
    # Objective = 2 / 2
    # Problem Understanding = 20 / 26
    # Logic = 18 / 26
    # Variables = 15 / 26
    # Observation = 17 / 26
    # Total Obtained = 72 / 106
    # Final Score = 6.79 / 10
    total_max = 2 + 13 * 8
    assert total_max == 106

    obtained = 72
    final_score = round((obtained / total_max) * 10, 2)
    assert final_score == 6.79

    # Verify rendering of table with this exact scoring model
    eval_result = DynamicEvaluationResult(
        objective=ObjectiveEvaluation(
            requirement="Objective of the Lab",
            status="PRESENT",
            score=2,
            max_score=2,
            evidence="Learn C programming loops",
            grounded=True
        ),
        questions=[
            QuestionEvaluation(
                question_number=i,
                question_text=f"Program {i}",
                match_status="FOUND_AND_COVERED",
                score=6 if i <= 9 else 4,
                max_score=8,
                requirements=[
                    RequirementEvaluation(requirement="Problem Understanding", status="PRESENT", score=2, max_score=2, evidence="text", grounded=True),
                    RequirementEvaluation(requirement="Logic / Approach Used", status="PRESENT", score=2, max_score=2, evidence="text", grounded=True),
                    RequirementEvaluation(requirement="Important Variables and Their Purpose", status="PARTIAL", score=1, max_score=2, evidence="text", grounded=True),
                    RequirementEvaluation(requirement="What I Observed", status="PRESENT" if i <= 9 else "MISSING", score=2 if i <= 9 else 0, max_score=2, evidence="text" if i <= 9 else None, grounded=True),
                ]
            )
            for i in range(1, 14)
        ],
        summary={
            "total_assigned": 13,
            "total_obtained": 72,
            "total_max": 106,
            "final_score": 6.79,
            "overall_assessment": "Good understanding with some incomplete explanations.",
            "criteria_totals": {
                "Problem Understanding": {"obtained": 20, "max": 26},
                "Logic / Approach Used": {"obtained": 18, "max": 26},
                "Important Variables and Their Purpose": {"obtained": 15, "max": 26},
                "What I Observed": {"obtained": 17, "max": 26},
            }
        }
    )

    md = render_verification_markdown(eval_result)
    assert "## 📊 Final Score" in md
    assert "| **Total** | **72 / 106** |" in md
    assert "| **Final Score** | **6.79 / 10** |" in md
    assert "Good understanding with some incomplete explanations." in md

    # Check parser
    parsed_metrics = parse_evaluation_scores(md)
    assert parsed_metrics["total_score"] == 6.79
    assert parsed_metrics["max_score"] == 10.0
    assert "72 / 106" in md


# ============================================================
# 15. EVIDENCE MISSING -> SCORE CANNOT BE > 0
# ============================================================

def test_15_evidence_missing_score_cannot_be_gt_zero():
    ocr_text = "Program 1: Factors\nSome description without required section."
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factors",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": None  # Evidence is missing!
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factors")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    prob_req = [r for r in q_eval.requirements if r.requirement == "Problem Understanding"][0]
    assert prob_req.score == 0
    assert prob_req.status == "MISSING"


# ============================================================
# 16. HALLUCINATED/GENERATED CONTENT NOT PRESENT IN OCR -> 0
# ============================================================

def test_16_hallucinated_content_not_in_ocr_receives_zero():
    ocr_text = """
    ## Program 1: Leap Year
    User enters year.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Leap Year",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        # Completely invented text not in OCR
                        "evidence": "A year is leap if divisible by 400 or divisible by 4 but not by 100",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Leap Year")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    logic_req = [r for r in q_eval.requirements if "Logic" in r.requirement][0]
    assert logic_req.grounded is False
    assert logic_req.status == "MISSING"
    assert logic_req.score == 0


# ============================================================
# 17. REORDERED QUESTIONS
# ============================================================

def test_17_reordered_questions():
    questions = [
        AssignedQuestion(question_number=1, question_text="Factors of a Number"),
        AssignedQuestion(question_number=2, question_text="Prime Number"),
        AssignedQuestion(question_number=3, question_text="Palindrome Number"),
    ]
    # Student answered Program 3 first, then Program 1, then Program 2
    ocr_text = """
    ## Program 3: Palindrome Number
    ### Problem Understanding
    Reverses digits of n and checks equality with original number.
    
    ## Program 1: Factors of a Number
    ### Problem Understanding
    Finds all numbers dividing n without remainder.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 3,
                "question_text": "Palindrome Number",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 3: Palindrome Number",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Reverses digits of n and checks equality"}
                ]
            },
            {
                "question_number": 1,
                "question_text": "Factors of a Number",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Program 1: Factors of a Number",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Finds all numbers dividing n without remainder"}
                ]
            },
            {
                "question_number": 2,
                "question_text": "Prime Number",
                "match_status": "NOT_FOUND",
                "requirements": []
            }
        ]
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, questions, manual_reqs)

    # Questions are ordered by question_number 1, 2, 3
    assert res.questions[0].question_number == 1
    assert res.questions[0].score == 2
    assert res.questions[1].question_number == 2
    assert res.questions[1].score == 0
    assert res.questions[2].question_number == 3
    assert res.questions[2].score == 2


# ============================================================
# 18. MULTIPLE QUESTIONS ON SAME PAGE
# ============================================================

def test_18_multiple_questions_on_same_page():
    ocr_text = """
    Page 1:
    ## Program 1: Factors
    ### Problem Understanding
    Find all divisors of n.

    ## Program 2: Factorial
    ### Problem Understanding
    Calculate product of 1 to n.
    """
    questions = [
        AssignedQuestion(question_number=1, question_text="Factors"),
        AssignedQuestion(question_number=2, question_text="Factorial")
    ]
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factors",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Find all divisors of n"}
                ]
            },
            {
                "question_number": 2,
                "question_text": "Factorial",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": "Calculate product of 1 to n"}
                ]
            }
        ]
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, questions, manual_reqs)

    assert res.questions[0].score == 2
    assert res.questions[1].score == 2
    assert res.summary["total_obtained"] == 4


# ============================================================
# 19. OCR SPELLING ERRORS (TOLERANT GROUNDING)
# ============================================================

def test_19_ocr_spelling_errors():
    # OCR has slight distortions
    ocr_text = """
    ## Prgrm 1: Factroial of a Numbr
    ### Problm Undrstandng
    Computes prdct of 1 to n for givn integerr n.
    """
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Factorial of a Number",
                "match_status": "FOUND_AND_COVERED",
                "matched_heading": "Prgrm 1: Factroial of a Numbr",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "Computes prdct of 1 to n for givn integerr n",
                        "confidence": "HIGH"
                    }
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Factorial of a Number")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1], manual_reqs)

    q_eval = res.questions[0]
    prob_req = [r for r in q_eval.requirements if "Problem" in r.requirement][0]
    assert prob_req.grounded is True
    assert prob_req.score == 2


# ============================================================
# 20. AMBIGUOUS QUESTION MATCHING
# ============================================================

def test_20_ambiguous_question_matching():
    # OCR only has generic "Print number"
    ocr_text = "Print number"
    questions = [
        AssignedQuestion(question_number=1, question_text="Print prime numbers"),
        AssignedQuestion(question_number=2, question_text="Print perfect numbers")
    ]
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Print prime numbers",
                "match_status": "NOT_FOUND",
                "requirements": []
            },
            {
                "question_number": 2,
                "question_text": "Print perfect numbers",
                "match_status": "NOT_FOUND",
                "requirements": []
            }
        ]
    }
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, questions, manual_reqs)

    assert res.questions[0].match_status == "NOT_FOUND"
    assert res.questions[0].score == 0
    assert res.questions[1].match_status == "NOT_FOUND"
    assert res.questions[1].score == 0
    assert res.summary["total_obtained"] == 0


# ============================================================
# 21. REGRESSION: GENERIC OBSERVATION SCORES ZERO
# ============================================================

def test_generic_observation_scores_zero():
    generic_samples = [
        "All programs were compiled and executed successfully without any errors.",
        "All programs executed successfully.",
        "The program executed successfully and I got the correct output.",
        "expected output was obtained",
        "matched the expected results",
        "program worked correctly",
        "compiled and executed successfully",
        "all programs ran successfully without errors",
    ]
    for sample in generic_samples:
        assert is_generic_observation(sample) is True
        score = calculate_criterion_score(
            requirement_name="What I Observed",
            status="PRESENT",
            evidence=sample,
            grounded=True,
            is_generic=True
        )
        assert score == 0, f"Generic observation '{sample}' must score 0, got {score}"


# ============================================================
# 22. REGRESSION: EMPTY VARIABLES SCORES ZERO
# ============================================================

def test_empty_variables_scores_zero():
    assert calculate_criterion_score("Important Variables and Their Purpose", "PRESENT", "[]", True) == 0
    assert calculate_criterion_score("Important Variables and Their Purpose", "PRESENT", "", False) == 0
    assert calculate_criterion_score("Important Variables and Their Purpose", "PRESENT", "None", True) == 0
    assert calculate_criterion_score("Important Variables and Their Purpose", "MISSING", None, False) == 0
    assert calculate_criterion_score("Important Variables and Their Purpose", "PRESENT", "n", True, raw_variables_list=[]) == 0


# ============================================================
# 23. REGRESSION: VARIABLES NAME ONLY SCORES ONE
# ============================================================

def test_variables_name_only_scores_one():
    # Table with variable name but empty / '-' purpose
    table_name_only = "| Variable | Purpose |\n| :--- | :--- |\n| `n` | - |"
    score_table = calculate_criterion_score(
        requirement_name="Important Variables and Their Purpose",
        status="PRESENT",
        evidence=table_name_only,
        grounded=True
    )
    assert score_table == 1

    # Text with variable name but no purpose
    score_text = calculate_criterion_score(
        requirement_name="Important Variables and Their Purpose",
        status="PRESENT",
        evidence="Variables used: num, temp",
        grounded=True
    )
    assert score_text == 1


# ============================================================
# 24. REGRESSION: VARIABLES NAME AND PURPOSE SCORES TWO
# ============================================================

def test_variables_name_and_purpose_scores_two():
    table_complete = "| Variable | Purpose |\n| :--- | :--- |\n| `n` | Stores the input integer to check |"
    score_table = calculate_criterion_score(
        requirement_name="Important Variables and Their Purpose",
        status="PRESENT",
        evidence=table_complete,
        grounded=True
    )
    assert score_table == 2

    score_colon = calculate_criterion_score(
        requirement_name="Important Variables and Their Purpose",
        status="PRESENT",
        evidence="n: Stores the input number from user",
        grounded=True
    )
    assert score_colon == 2


# ============================================================
# 25. REGRESSION: SUPERFICIAL LOGIC SCORES ONE
# ============================================================

def test_superficial_logic_scores_one():
    superficial_examples = [
        "Uses if-else.",
        "using switch statement",
        "by using loops",
        "ternary operator is used",
    ]
    for sample in superficial_examples:
        score = calculate_criterion_score(
            requirement_name="Logic / Approach Used",
            status="PRESENT",
            evidence=sample,
            grounded=True
        )
        assert score == 1, f"Superficial logic '{sample}' must score 1, got {score}"


# ============================================================
# 26. REGRESSION: STEP-BY-STEP LOGIC SCORES TWO
# ============================================================

def test_step_by_step_logic_scores_two():
    logic_explanation = "Loop from 2 to n-1 checking if n % i == 0. If remainder is zero then flag not prime."
    score = calculate_criterion_score(
        requirement_name="Logic / Approach Used",
        status="PRESENT",
        evidence=logic_explanation,
        grounded=True
    )
    assert score == 2


# ============================================================
# 27. REGRESSION: BRIEF UNDERSTANDING SCORES ONE
# ============================================================

def test_brief_understanding_scores_one():
    score = calculate_criterion_score(
        requirement_name="Problem Understanding",
        status="PARTIAL",
        evidence="Check even or odd.",
        grounded=True
    )
    assert score == 1


# ============================================================
# 28. REGRESSION: COMPLETE UNDERSTANDING SCORES TWO
# ============================================================

def test_complete_understanding_scores_two():
    full_understanding = "The program accepts an integer n as input and determines whether it is divisible by 2 to produce output stating even or odd."
    score = calculate_criterion_score(
        requirement_name="Problem Understanding",
        status="PRESENT",
        evidence=full_understanding,
        grounded=True
    )
    assert score == 2


# ============================================================
# 29. REGRESSION: SUPERFICIAL OBJECTIVE SCORES ONE
# ============================================================

def test_superficial_objective_scores_one():
    score = calculate_objective_score(
        status="PARTIAL",
        evidence="To practice C programs",
        grounded=True
    )
    assert score == 1


# ============================================================
# 30. REGRESSION: COMPREHENSIVE OBJECTIVE SCORES TWO
# ============================================================

def test_comprehensive_objective_scores_two():
    comprehensive_obj = "The objective of this laboratory exercise is to understand the use of loops, conditional statements, arithmetic operations, and iterative logic in solving basic numerical problems using C programming."
    score = calculate_objective_score(
        status="PRESENT",
        evidence=comprehensive_obj,
        grounded=True
    )
    assert score == 2


# ============================================================
# 31. REGRESSION: DUPLICATE SPANS ACROSS QUESTIONS SCORE ZERO
# ============================================================

def test_duplicate_spans_score_zero():
    ocr_text = """
    ## Program 1: Even or Odd
    In this level we look at the understanding of these programs.
    
    ## Program 2: Positive or Negative
    In this level we look at the understanding of these programs.
    """
    dup_span = "In this level we look at the understanding of these programs."
    raw_data = {
        "questions": [
            {
                "question_number": 1,
                "question_text": "Even or Odd",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": dup_span}
                ]
            },
            {
                "question_number": 2,
                "question_text": "Positive or Negative",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": dup_span}
                ]
            }
        ]
    }
    q1 = AssignedQuestion(question_number=1, question_text="Even or Odd")
    q2 = AssignedQuestion(question_number=2, question_text="Positive or Negative")
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    res = validate_and_ground_evaluation(raw_data, ocr_text, [q1, q2], manual_reqs)

    # Program 1 gets credit for the first instance
    assert res.questions[0].requirements[0].score == 2
    # Program 2 gets flagged as duplicate and scores 0!
    assert res.questions[1].requirements[0].score == 0
    assert res.questions[1].requirements[0].status == "MISSING"
    assert "Duplicate evidence" in (res.questions[1].requirements[0].evaluation or "")


# ============================================================
# 32. REGRESSION: FINAL SCORE NORMALIZATION OUT OF 10 FORMULA
# ============================================================

def test_final_score_normalized_out_of_ten():
    # Verify formula round((obtained / total_max) * 10, 1)
    total_q = 13
    total_max = 2 + (total_q * 8)
    assert total_max == 106

    # Test cases:
    # 6 marks (e.g. N240157 Even/Odd only + objective)
    assert round((6 / 106) * 10, 1) == 0.6
    # 25 marks (e.g. N240046 partial descriptions without variables/obs)
    assert round((25 / 106) * 10, 1) == 2.4
    # 72 marks (good report)
    assert round((72 / 106) * 10, 1) == 6.8
    # 106 marks (perfect report)
    assert round((106 / 106) * 10, 1) == 10.0


# ============================================================
# 33. ACCEPTANCE TEST: REAL STUDENT N240157
# ============================================================

def test_acceptance_real_student_n240157():
    """
    N240157:
    - OCR contains only objective + Even/Odd program description.
    - Observation is generic: 'All programs were compiled and executed successfully without any errors...'
    - Expected: Observation scores 0, only 1 question covered, final score <= 1.5 / 10.
    """
    json_path = os.path.join("output", "sections", "week-01", "SEC1", "students", "N240157.json")
    assert os.path.exists(json_path), f"File {json_path} must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    ocr_text = student_data["ocr"]["text"]
    q13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    # Simulate evaluation input with LLM extraction
    raw_data = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "To learn basic programming using input, output, operators and conditional statements",
            "confidence": "HIGH"
        },
        "questions": [
            {
                "question_number": 1,
                "question_text": "Write a C program to check whether a given number is even or odd.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {
                        "requirement": "Problem Understanding",
                        "status": "PRESENT",
                        "evidence": "user enters one Integer number and these numbers are stored in variable n",
                        "confidence": "HIGH"
                    },
                    {
                        "requirement": "Logic / Approach Used",
                        "status": "PRESENT",
                        "evidence": "checks whether the number is plausible by 2. modulus operator % is used",
                        "confidence": "HIGH"
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
                        "confidence": "LOW"
                    }
                ]
            }
        ]
    }

    res = validate_and_ground_evaluation(raw_data, ocr_text, q13, manual_reqs)

    # Observation must receive 0 marks!
    p1_obs = [r for r in res.questions[0].requirements if "observed" in r.requirement.lower()][0]
    assert p1_obs.score == 0

    # Only Program 1 is found and covered; remaining 12 programs are NOT_FOUND
    covered = sum(1 for q in res.questions if q.match_status == "FOUND_AND_COVERED")
    assert covered == 1

    # Total score out of 10 must be <= 1.5 (was wrongly 10/10 previously)
    final_score = res.summary["final_score_out_of_10"]
    assert final_score <= 1.5, f"N240157 final score must be <= 1.5/10, got {final_score}"


# ============================================================
# 34. ACCEPTANCE TEST: REAL STUDENT N240046
# ============================================================

def test_acceptance_real_student_n240046():
    """
    N240046:
    - Extraction has important_variables = [] for all 13 programs.
    - OCR conclusion is generic: 'All the programs were implemented, compiled, executed successfully...'
    - Expected: Variables score = 0 across all programs, Observation score = 0, final score <= 3.5 / 10.
    """
    json_path = os.path.join("output", "sections", "week-01", "SEC1", "students", "N240046.json")
    assert os.path.exists(json_path), f"File {json_path} must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    ext = student_data["extraction"]
    progs = ext.get("programs", {})
    # Verify that in extraction, all important_variables are empty
    for p_key, p_val in progs.items():
        assert p_val.get("important_variables") == [], f"Expected empty variables for {p_key}"

    # Format extraction for evaluation
    eval_text = format_extraction_for_evaluation(ext, fallback_text=student_data["ocr"]["text"])
    q13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    # Run validation
    res = validate_and_ground_evaluation({}, eval_text, q13, manual_reqs)

    # Variables score must be 0 for all questions!
    criteria_totals = res.summary["criteria_totals"]
    vars_obtained = criteria_totals.get("Important Variables and Their Purpose", {}).get("obtained", 0)
    assert vars_obtained == 0, f"Variables score must be 0 for N240046, got {vars_obtained}"

    # Final score must be <= 3.5 / 10
    final_score = res.summary["final_score_out_of_10"]
    assert final_score <= 3.5, f"N240046 final score must be <= 3.5/10, got {final_score}"


# ============================================================
# 35. ACCEPTANCE TEST: REAL STUDENT N240081
# ============================================================

def test_acceptance_real_student_n240081():
    """
    N240081:
    - OCR contains Level-1, Level-2, Level-3 grouped descriptions.
    - Extraction deduplication clears repeated text, and verifier duplicate suppression
      ensures duplicate evidence spans receive 0 points.
    """
    json_path = os.path.join("output", "sections", "week-01", "SEC1", "students", "N240081.json")
    assert os.path.exists(json_path), f"File {json_path} must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    ocr_text = student_data["ocr"]["text"]
    q13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)

    # Simulate raw LLM output that tried to copy Level-3 text to multiple programs
    level3_text = "In this level, we look at the understanding of these programs in the Menu program with error handling. Largest among three numbers, Grade point to letter grade."
    raw_data = {
        "objective": {
            "requirement": "Objective of the Lab",
            "status": "PRESENT",
            "evidence": "learn the use of variables and different data types",
            "confidence": "HIGH"
        },
        "questions": [
            {
                "question_number": 11,
                "question_text": "Write an extended menu-based C calculator that handles invalid menu choices and division by zero.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": level3_text}
                ]
            },
            {
                "question_number": 12,
                "question_text": "Write a C program to find the largest of three numbers using the ternary operator.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": level3_text}
                ]
            },
            {
                "question_number": 13,
                "question_text": "Write a C program to convert a grade point to a letter grade using a switch statement.",
                "match_status": "FOUND_AND_COVERED",
                "requirements": [
                    {"requirement": "Problem Understanding", "status": "PRESENT", "evidence": level3_text}
                ]
            }
        ]
    }

    res = validate_and_ground_evaluation(raw_data, ocr_text, q13, manual_reqs)

    # Q11 gets credit for the first instance
    q11 = [q for q in res.questions if q.question_number == 11][0]
    q11_prob = [r for r in q11.requirements if "problem" in r.requirement.lower()][0]
    assert q11_prob.score == 2

    # Q12 and Q13 duplicate Q11's evidence and MUST score 0!
    q12 = [q for q in res.questions if q.question_number == 12][0]
    q12_prob = [r for r in q12.requirements if "problem" in r.requirement.lower()][0]
    assert q12_prob.score == 0
    assert q12_prob.status == "MISSING"

    q13_eval = [q for q in res.questions if q.question_number == 13][0]
    q13_prob = [r for r in q13_eval.requirements if "problem" in r.requirement.lower()][0]
    assert q13_prob.score == 0
    assert q13_prob.status == "MISSING"

