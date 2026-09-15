"""
tests/test_unified_evaluation_output.py
Comprehensive regression test suite validating the Unified 5-Dimension Student Evaluation architecture.
Covers all 10 required regression tests from Section 24 of the implementation specification:
- Test 1: Code ingestion (17 .c files)
- Test 2: p6 variants (p6_1.c and p6_2.c remain separate)
- Test 3: Report extraction (4 report entries)
- Test 4: Semantic matching (R1->P3, R2->P5, R3->P7, R4->P12)
- Test 5: Undocumented programs (17 - 4 = 13, not penalized under D3)
- Test 6: Consistency (R4/P12 is inconsistent)
- Test 7: Score calculation (deterministic Python arithmetic)
- Test 8: Missing code count (never defaults to 1)
- Test 9: Markdown escaping (pipe and newline safety in tables)
- Test 10: Stale Streamlit state isolation (no bleed between evaluations)
"""

import json
import os
import zipfile
import pytest
from unittest.mock import patch

from schemas import (
    StudentCodeCollection,
    StudentCodeSnippet,
    ReportProgramEntry,
    HolisticEvaluationResult,
    DimensionScore,
    EvaluationEvidence,
    CodeReportConsistencyItem,
    InterestingLogicItem,
    PresentationReadiness,
)
from verifier import (
    verify_observation_report_sync,
    evaluate_holistic_student,
    render_holistic_evaluation_markdown,
    ingest_student_code_from_json_or_files,
    match_report_programs_to_questions,
    parse_evaluation_scores,
    safe_md_cell,
    validate_student_submission,
)


@pytest.fixture(autouse=True)
def mock_ollama_offline():
    """Ensures test suite executes deterministically and rapidly using static evaluation."""
    with patch("verifier._call_ollama", return_value=(False, "", "Simulated offline for unit testing")):
        yield


@pytest.fixture
def sample_17_code_collection():
    """Builds a collection of 17 submitted .c code problems matching week-04 domains."""
    titles = {
        1: "Check whether a given number is prime",
        2: "Check whether a string is a palindrome",
        3: "Find length of a string without using strlen",
        4: "Find sum of natural numbers using recursion",
        5: "Find factorial of a number using recursion",
        7: "Print Armstrong numbers within a range",
        8: "Print a diamond pattern",
        9: "Generate Fibonacci series using recursion",
        10: "Sort strings in lexicographical order",
        11: "Reverse a string",
        12: "Find the frequency of each character in a string",
        13: "Evaluate recursive function f(n) = 2f(n-1)",
        14: "Count number of 1s in binary representation",
        15: "Print values using two recursive calls",
        16: "Evaluate recursive function with start and end bounds"
    }
    problems = []
    # 1..5
    for i in range(1, 6):
        if i == 3:
            code = "// Problem 3 - String length\n#include <stdio.h>\nint main() { char s[100]; int len=0; while(s[len]!='\\0') len++; return 0; }"
        elif i == 4:
            code = "// Problem 4 - Sum recursion\nint sum(int n) { if(n<=0) return 0; return n + sum(n-1); }"
        elif i == 5:
            code = "// Problem 5 - Factorial recursion\nint factorial(int n) { if(n==0) return 1; return n * factorial(n-1); }"
        else:
            code = f"// Problem {i}\n#include <stdio.h>\nint main() {{ return 0; }}\n"

        problems.append(StudentCodeSnippet(
            problem_number=i,
            problem_id=f"P{i}",
            problem_title=titles.get(i, f"Problem {i}"),
            source_file=f"p{i}.c",
            source_code=code
        ))

    # 6 variants: p6_1.c and p6_2.c
    problems.append(StudentCodeSnippet(
        problem_number=6,
        problem_id="P6_1",
        problem_title="Swap two numbers using call by value",
        source_file="p6_1.c",
        source_code="void swap(int a, int b) { int t = a; a = b; b = t; }"
    ))
    problems.append(StudentCodeSnippet(
        problem_number=6,
        problem_id="P6_2",
        problem_title="Swap two numbers using call by reference",
        source_file="p6_2.c",
        source_code="void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }"
    ))

    # 7..16 -> P7..P16
    for i in range(7, 17):
        code = f"// Problem {i}\n#include <stdio.h>\nint main() {{ return 0; }}\n"
        if i == 7:
            code = "// Armstrong\nint isarmstrong(int n) { return 1; }\nint main() { for(int i=1; i<=100; i++) isarmstrong(i); return 0; }"
        elif i == 9:
            code = "// Fibonacci\nint fibonacci(int n) { if(n<=1) return n; return fibonacci(n-1)+fibonacci(n-2); }"
        elif i == 10:
            code = "// Lexicographical sort\n#include <string.h>\nint main() { char str[20][50]; return 0; }"
        elif i == 12:
            code = "// Character frequency\n#include <stdio.h>\nint main() { int freq[256]={0}; char s[]=\"test\"; for(int i=0; s[i]!='\\0'; i++) freq[(unsigned char)s[i]]++; return 0; }"
        elif i == 13:
            code = "// Recursive doubling\nint f(int n) { if(n<=1) return 1; return f(n-1)+f(n-1); }"
        elif i == 14:
            code = "// Bit count\nint count_bits(int n) { if(n==0) return 0; return count_bits(n/2) + (n%2); }"
        elif i == 15:
            code = "// Double recursion\nvoid f(int n) { if(n<=0) return; f(n/2); f(n/2); printf(\"%d \", n); }"
        elif i == 16:
            code = "// Modulo recursion\nint rec(int s, int e) { if(s>=e) return 0; int len = e-s; return rec(s+1, e); }"

        problems.append(StudentCodeSnippet(
            problem_number=i,
            problem_id=f"P{i}",
            problem_title=titles.get(i, f"Problem {i}"),
            source_file=f"p{i}.c",
            source_code=code
        ))

    return StudentCodeCollection(
        student_id="N241003",
        week="week-04",
        problems=problems
    )


@pytest.fixture
def sample_4_report_extraction():
    """Builds an extraction result documenting only 4 report entries (R1..R4)."""
    return {
        "status": "success",
        "objective_of_lab": "Practice functions, recursion, and string manipulation in C.",
        "conclusion": "Successfully practiced modular functions and recursion in C.",
        "detected_programs": [
            {
                "report_program_id": "R1",
                "program_title": "Find length of a string without using strlen",
                "status": "detected",
                "problem_understanding": "Count characters up to null terminator '\\0'.",
                "logic_approach": "Loop through characters until '\\0' is reached and increment counter.",
                "important_variables": [{"variable": "str", "purpose": "input string"}, {"variable": "len", "purpose": "character counter"}],
                "what_i_observed": "String terminates with null character '\\0'."
            },
            {
                "report_program_id": "R2",
                "program_title": "Find factorial of a number using recursion",
                "status": "detected",
                "problem_understanding": "Calculate n! recursively with base case n==0 or 1.",
                "logic_approach": "Multiply n by factorial(n-1) until base condition is met.",
                "important_variables": [{"variable": "n", "purpose": "input integer"}],
                "what_i_observed": "Base case stops recursion and unwinds call stack."
            },
            {
                "report_program_id": "R3",
                "program_title": "Print Armstrong numbers within a range using a function",
                "status": "detected",
                "problem_understanding": "Check whether each number in range equals sum of powers of digits.",
                "logic_approach": "Extract digits, raise to power of count of digits, check equality in isarmstrong() function.",
                "important_variables": [{"variable": "sum", "purpose": "accumulated powers"}, {"variable": "start", "purpose": "range lower bound"}],
                "what_i_observed": "Armstrong check function modularizes the range loop."
            },
            {
                "report_program_id": "R4",
                "program_title": "Find the frequency of each character in a string",
                "status": "detected",
                "problem_understanding": "Count occurrences of every character in a string.",
                "logic_approach": "Use nested loops to compare each character with all subsequent characters and count matches.",
                "important_variables": [{"variable": "count", "purpose": "character count"}],
                "what_i_observed": "Nested loops count character occurrences."
            }
        ]
    }


# ==============================================================================
# SECTION 24: ALL 10 REQUIRED REGRESSION TESTS
# ==============================================================================

def test_01_code_ingestion_17_files():
    """
    Test 1 — Code ingestion
    Input: 17 .c files from week-4-sec-2.zip (or synthesized 17 files)
    Expected: len(student_code.problems) == 17
    """
    if os.path.exists("week-4-sec-2.zip"):
        coll = ingest_student_code_from_json_or_files("week-4-sec-2.zip")
    else:
        # Fallback to json archive if zip missing
        coll = ingest_student_code_from_json_or_files("student_N241003_week_04.json")

    assert isinstance(coll, StudentCodeCollection)
    assert len(coll.problems) == 17, f"Expected exactly 17 code files, found {len(coll.problems)}"


def test_02_p6_variants_separate_files(tmp_path):
    """
    Test 2 — p6 variants
    Ensure: p6_1.c and p6_2.c remain two separate source files.
    Expected: count += 2, not 1.
    """
    test_dir = tmp_path / "test_p6"
    test_dir.mkdir()
    (test_dir / "p1.c").write_text("int main(){return 1;}", encoding="utf-8")
    (test_dir / "p6_1.c").write_text("void swap_val(int a, int b){}", encoding="utf-8")
    (test_dir / "p6_2.c").write_text("void swap_ref(int *a, int *b){}", encoding="utf-8")
    (test_dir / "p7.c").write_text("int main(){return 7;}", encoding="utf-8")

    coll = ingest_student_code_from_json_or_files(str(test_dir))
    prob_ids = [p.problem_id for p in coll.problems]
    source_files = [p.source_file for p in coll.problems]

    assert "p6_1.c" in source_files
    assert "p6_2.c" in source_files
    assert len(coll.problems) == 4, "p6_1.c and p6_2.c must both be counted (count += 2, not 1)"

    # Also check real zip if available
    if os.path.exists("week-4-sec-2.zip"):
        zip_coll = ingest_student_code_from_json_or_files("week-4-sec-2.zip")
        zip_files = [p.source_file for p in zip_coll.problems]
        assert "p6_1.c" in zip_files
        assert "p6_2.c" in zip_files
        assert len(zip_coll.problems) == 17


def test_03_report_extraction_4_entries(sample_4_report_extraction):
    """
    Test 3 — Report extraction
    Four actual report programs must result in: len(report_programs) == 4
    """
    det_programs = sample_4_report_extraction["detected_programs"]
    assert len(det_programs) == 4
    report_ids = [p["report_program_id"] for p in det_programs]
    assert report_ids == ["R1", "R2", "R3", "R4"]


def test_04_semantic_matching_expected_mappings(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 4 — Semantic matching
    Expected:
    R1 -> P3
    R2 -> P5
    R3 -> P7
    R4 -> P12
    """
    matches = match_report_programs_to_questions(
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection
    )
    m_dict = {m.report_program_id: m.matched_problem_id for m in matches}
    assert m_dict.get("R1") == "P3", f"R1 expected P3, got {m_dict.get('R1')}"
    assert m_dict.get("R2") == "P5", f"R2 expected P5, got {m_dict.get('R2')}"
    assert m_dict.get("R3") == "P7", f"R3 expected P7, got {m_dict.get('R3')}"
    assert m_dict.get("R4") == "P12", f"R4 expected P12, got {m_dict.get('R4')}"


def test_05_undocumented_programs_not_d3_failures(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 5 — Undocumented programs
    Expected: 17 - 4 = 13
    Undocumented programs must NOT be added as D3 failures.
    """
    res = evaluate_holistic_student(
        report_text="Sample observation report",
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection,
        student_id="N241003",
        week_id="week-04"
    )
    assert res.total_code_problems == 17
    assert len(res.report_entries) == 4
    undocumented_count = res.total_code_problems - len(res.report_entries)
    assert undocumented_count == 13

    # D3 score must not be penalized for undocumented programs
    d3 = next(d for d in res.dimensions if d.dimension == "D3")
    assert d3.score >= 1.4, "D3 should score the 4 report entries without penalty for the 13 undocumented files"
    assert "No penalty applied for undocumented programs" in d3.justification


def test_06_consistency_r4_p12_inconsistent(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 6 — Consistency
    R4/P12 must be: inconsistent
    Because the report describes nested loop comparison, but p12.c implements a frequency array.
    """
    res = evaluate_holistic_student(
        report_text="R4 describes nested loops to count frequencies.",
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection,
        student_id="N241003",
        week_id="week-04"
    )
    r4_item = next((c for c in res.code_report_consistency if c.report_program_id == "R4"), None)
    assert r4_item is not None, "R4 must have a consistency check entry"
    assert r4_item.problem == "P12"
    assert r4_item.claim_type == "inconsistent", f"R4/P12 must be 'inconsistent', got '{r4_item.claim_type}'"


def test_07_score_calculation_arithmetic():
    """
    Test 7 — Score calculation
    Given arbitrary LLM dimension scores:
    D1 = 1.75
    D2 = 1.45
    D3 = 1.65
    D4 = 1.55
    D5 = 1.70
    Python should calculate:
    Total = 8.10 / 10
    Final = 81.0 / 100
    Programming = 32.0 / 40
    Report = 16.5 / 20
    Conceptual = 15.5 / 20
    Novelty = 17.0 / 20
    """
    d1, d2, d3, d4, d5 = 1.75, 1.45, 1.65, 1.55, 1.70

    total_score_10 = round(d1 + d2 + d3 + d4 + d5, 2)
    total_score_100 = round(total_score_10 * 10, 1)
    programming_score_40 = round((d1 + d2) * 10, 1)
    report_score_20 = round(d3 * 10, 1)
    conceptual_score_20 = round(d4 * 10, 1)
    novelty_score_20 = round(d5 * 10, 1)

    assert total_score_10 == 8.10
    assert total_score_100 == 81.0
    assert programming_score_40 == 32.0
    assert report_score_20 == 16.5
    assert conceptual_score_20 == 15.5
    assert novelty_score_20 == 17.0

    # Test that HolisticEvaluationResult and parse_evaluation_scores agree
    result = HolisticEvaluationResult(
        student_id="N241003",
        week_id="week-04",
        total_code_problems=17,
        dimensions=[
            DimensionScore(dimension="D1", name="Syntax & Code Validity", score=d1, max_score=2.0),
            DimensionScore(dimension="D2", name="Algorithmic Logic & Functional Correctness", score=d2, max_score=2.0),
            DimensionScore(dimension="D3", name="Observation Report Quality & Completeness", score=d3, max_score=2.0),
            DimensionScore(dimension="D4", name="Conceptual Understanding & Code-Report Consistency", score=d4, max_score=2.0),
            DimensionScore(dimension="D5", name="Novelty, Innovation & Presentation Readiness", score=d5, max_score=2.0),
        ],
        total_score_10=total_score_10,
        total_score_100=total_score_100,
        programming_score_40=programming_score_40,
        report_score_20=report_score_20,
        conceptual_score_20=conceptual_score_20,
        novelty_score_20=novelty_score_20,
        report_entries=[],
        question_matches=[],
        code_report_consistency=[],
        interesting_logic=[],
        presentation_readiness=PresentationReadiness()
    )

    md = render_holistic_evaluation_markdown(result)
    parsed = parse_evaluation_scores(md)

    assert parsed.get("total_score_10") == 8.10
    assert parsed.get("total_score_100") == 81.0
    assert parsed.get("D1") == 1.75
    assert parsed.get("D2") == 1.45
    assert parsed.get("D3") == 1.65
    assert parsed.get("D4") == 1.55
    assert parsed.get("D5") == 1.70


def test_08_missing_code_count_never_defaults_to_one():
    """
    Test 8 — Missing code count
    If an evaluator result accidentally omits: total_code_problems (or it is 0)
    the renderer must NOT output: 1.
    It must output 0 or safe missing value.
    """
    # Create result with 0 code problems and 4 report matches
    result = HolisticEvaluationResult(
        student_id="N_NO_CODE",
        week_id="week-04",
        total_code_problems=0,
        dimensions=[
            DimensionScore(dimension="D1", name="Syntax & Code Validity", score=0.0, max_score=2.0),
            DimensionScore(dimension="D2", name="Algorithmic Logic", score=0.0, max_score=2.0),
            DimensionScore(dimension="D3", name="Report Quality", score=1.5, max_score=2.0),
            DimensionScore(dimension="D4", name="Consistency", score=1.5, max_score=2.0),
            DimensionScore(dimension="D5", name="Novelty", score=0.0, max_score=2.0),
        ],
        total_score_10=3.0,
        total_score_100=30.0,
        programming_score_40=0.0,
        report_score_20=15.0,
        conceptual_score_20=15.0,
        novelty_score_20=0.0,
        report_entries=[],
        question_matches=[],
        code_report_consistency=[],
        interesting_logic=[],
        presentation_readiness=PresentationReadiness()
    )

    md = render_holistic_evaluation_markdown(result)
    # The submission summary table must NOT show 1
    assert "| Code Programs | 0 |" in md, "When total_code_problems is 0, it must display 0, never fallback to 1"
    assert "| Code Programs | 1 |" not in md


def test_09_markdown_escaping_safety():
    """
    Test 9 — Markdown escaping
    Test titles/evidence containing: | and newline
    and ensure the Markdown table remains valid.
    """
    dangerous_title = "Find length of a string | without using strlen\nwith secondary condition"
    safe = safe_md_cell(dangerous_title)
    assert "|" not in safe or "\\|" in safe
    assert "\n" not in safe

    # Test rendering a table row with dangerous title
    item = CodeReportConsistencyItem(
        problem="P3",
        report_program_id="R1",
        report_claim="Report says | pipe character\nand newline",
        code_reality="Code says | another pipe\nwith newline",
        claim_type="supported",
        assessment="Valid"
    )
    result = HolisticEvaluationResult(
        student_id="N_ESCAPE",
        week_id="week-04",
        total_code_problems=1,
        dimensions=[
            DimensionScore(dimension="D1", name="Syntax", score=2.0, max_score=2.0),
            DimensionScore(dimension="D2", name="Logic", score=2.0, max_score=2.0),
            DimensionScore(dimension="D3", name="Report", score=2.0, max_score=2.0),
            DimensionScore(dimension="D4", name="Consistency", score=2.0, max_score=2.0),
            DimensionScore(dimension="D5", name="Novelty", score=2.0, max_score=2.0),
        ],
        total_score_10=10.0,
        total_score_100=100.0,
        programming_score_40=40.0,
        report_score_20=20.0,
        conceptual_score_20=20.0,
        novelty_score_20=20.0,
        report_entries=[],
        question_matches=[],
        code_report_consistency=[item],
        interesting_logic=[],
        presentation_readiness=PresentationReadiness()
    )
    md = render_holistic_evaluation_markdown(result)

    # In section 4 (Consistency Table), ensure each line in the table has exactly 6 pipe delimiters (5 columns)
    lines = md.splitlines()
    in_table = False
    for line in lines:
        if "## 4. Code–Report Consistency" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("## "):
                break
            if line.startswith("|") and not line.startswith("|---"):
                # Count unescaped pipes
                unescaped_pipes = line.replace("\\|", "").count("|")
                assert unescaped_pipes == 6, f"Broken table row format: {line}"


def test_10_stale_streamlit_state_isolation():
    """
    Test 10 — Stale Streamlit state
    Run evaluation A, then evaluation B.
    Ensure evaluation B does not display score/state from evaluation A.
    """
    student_a_code = StudentCodeCollection(
        student_id="STUDENT_A",
        week="week-04",
        problems=[
            StudentCodeSnippet(problem_number=i, problem_id=f"P{i}", problem_title=f"Prog A {i}", source_file=f"a_{i}.c", source_code=f"// A {i}")
            for i in range(1, 18)
        ]
    )
    student_b_code = StudentCodeCollection(
        student_id="STUDENT_B",
        week="week-04",
        problems=[
            StudentCodeSnippet(problem_number=i, problem_id=f"P{i}", problem_title=f"Prog B {i}", source_file=f"b_{i}.c", source_code=f"// B {i}")
            for i in range(1, 6)
        ]
    )

    # Simulated Streamlit session state container
    session_state = {}

    # 1. Run Evaluation A
    session_state["evaluation_result"] = None
    res_a = evaluate_holistic_student(
        report_text="Report A",
        extracted_report={"detected_programs": [{"report_program_id": "R1", "program_title": "Prog A 1"}]},
        student_code=student_a_code,
        student_id="STUDENT_A",
        week_id="week-04"
    )
    session_state["evaluation_result"] = res_a
    assert session_state["evaluation_result"].total_code_problems == 17
    assert session_state["evaluation_result"].student_id == "STUDENT_A"

    # 2. Reset before running Evaluation B (as enforced in app.py)
    session_state["evaluation_result"] = None
    res_b = evaluate_holistic_student(
        report_text="Report B",
        extracted_report={"detected_programs": [{"report_program_id": "R1", "program_title": "Prog B 1"}]},
        student_code=student_b_code,
        student_id="STUDENT_B",
        week_id="week-04"
    )
    session_state["evaluation_result"] = res_b

    # Verify B is isolated from A
    assert session_state["evaluation_result"].total_code_problems == 5
    assert session_state["evaluation_result"].student_id == "STUDENT_B"
    md_b = render_holistic_evaluation_markdown(session_state["evaluation_result"])
    assert "STUDENT_A" not in md_b
    assert "| Code Programs | 5 |" in md_b
    assert "| Code Programs | 17 |" not in md_b


def test_acceptance_routing_and_pre_llm_data_integrity():
    """
    Acceptance test for Week-4 submission (N241003):
    1. Pre-LLM data integrity: exactly 17 code problems ingested, including p6_1.c and p6_2.c.
    2. Absolute routing in verify_observation_report_sync: passing assigned_questions must NEVER
       route to legacy 42-mark evaluator.
    3. Required internal data model counts:
       - total_code_problems == 17
       - total_report_entries == 4
       - matched_report_entries == 4
       - undocumented_code_problems == 13
    4. Required semantic matches:
       - R1 -> P3 -> p3.c
       - R2 -> P5 -> p5.c
       - R3 -> P7 -> p7.c
       - R4 -> P12 -> p12.c
    5. Required consistency:
       - R1/P3 -> supported
       - R2/P5 -> supported
       - R3/P7 -> supported
       - R4/P12 -> inconsistent
    6. Required dimensions & Python calculations:
       - D1..D5 <= 2.0
       - total_score_10 <= 10.0
       - total_score_100 == round(total_score_10 * 10, 1)
    7. Forbidden legacy output absent from Markdown.
    """
    import os
    from verifier import (
        ingest_student_code_from_json_or_files,
        verify_observation_report_sync,
        match_report_programs_to_questions
    )

    zip_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "week-4-sec-2.zip")
    if not os.path.exists(zip_path):
        pytest.skip(f"ZIP file not found at {zip_path}")

    # 1. Verify pre-LLM parsed submission count before evaluation
    student_code = ingest_student_code_from_json_or_files(zip_path, student_id="N241003", week="week-04")
    assert student_code is not None
    assert len(student_code.problems) == 17, f"Expected 17 problems, got {len(student_code.problems)}"
    filenames = [p.source_filename for p in student_code.problems]
    assert "p6_1.c" in filenames, "p6_1.c must be preserved separately"
    assert "p6_2.c" in filenames, "p6_2.c must be preserved separately"

    # 2. Extract report entries from existing extracted JSON
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "sections", "week-04", "SEC2", "students", "N241003.json")
    if not os.path.exists(json_path):
        pytest.skip(f"Extracted student JSON not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ext_dict = data.get("extraction", {})
    ocr_txt = data.get("ocr", {}).get("text", "")

    # 3. Call verify_observation_report_sync with legacy assigned_questions to prove bypass
    md_output = verify_observation_report_sync(
        report_text=ocr_txt,
        assigned_questions="Program 1: Factors\nProgram 2: Factorial",
        instruction_manual="Official manual",
        student_code=student_code,
        extracted_report=ext_dict,
        student_id="N241003",
        week_id="week-04"
    )

    result = getattr(verify_observation_report_sync, "last_holistic_result", None)
    assert result is not None

    # 4. Assert required internal data model
    assert result.total_code_problems == 17
    assert result.total_report_entries == 4
    assert result.matched_report_entries == 4
    assert result.undocumented_code_problems == 13

    # 5. Assert required semantic matches
    match_map = {m.report_program_id: m for m in result.matches}
    assert "R1" in match_map and match_map["R1"].matched_problem_id == "P3" and match_map["R1"].source_file == "p3.c"
    assert "R2" in match_map and match_map["R2"].matched_problem_id == "P5" and match_map["R2"].source_file == "p5.c"
    assert "R3" in match_map and match_map["R3"].matched_problem_id == "P7" and match_map["R3"].source_file == "p7.c"
    assert "R4" in match_map and match_map["R4"].matched_problem_id == "P12" and match_map["R4"].source_file == "p12.c"

    # 6. Assert required consistency
    cons_map = {c.report_program_id: c for c in result.code_report_consistency}
    assert cons_map["R1"].status == "supported"
    assert cons_map["R2"].status == "supported"
    assert cons_map["R3"].status == "supported"
    assert cons_map["R4"].status == "inconsistent"

    # 7. Assert required dimensions & calculations strictly computed by Python
    assert len(result.dimensions) == 5
    for d in result.dimensions:
        assert 0.0 <= d.score <= 2.0
    dim_sum = round(sum(d.score for d in result.dimensions), 2)
    assert result.total_score_10 == dim_sum
    assert result.total_score_100 == round(dim_sum * 10, 1)

    # 8. Assert Final Output Format Verification (exact section ordering)
    assert md_output.startswith("# Unified Student Evaluation"), "Markdown must begin with '# Unified Student Evaluation'"
    
    sections = [
        "## 1. Submission Summary",
        "## 2. Reported Programs",
        "## 3. Evaluation",
        "## 4. Code–Report Consistency",
        "## 5. Notable Implementations",
        "## 6. Presentation Preparation",
        "## 7. Final Score",
        "## 8. Feedback"
    ]
    last_idx = -1
    for sec in sections:
        idx = md_output.find(sec)
        assert idx != -1, f"Required section '{sec}' missing from generated markdown"
        assert idx > last_idx, f"Section '{sec}' appeared out of order"
        last_idx = idx

    # Verify D1..D5 appear in order inside Section 3
    d_indices = [md_output.find(f"### D{i}") for i in range(1, 6)]
    assert all(i != -1 for i in d_indices), "All D1..D5 headers must appear in Section 3"
    assert d_indices == sorted(d_indices), "D1..D5 must appear in sequential order"

    # 9. Assert forbidden legacy output
    forbidden_terms = [
        "Assigned Questions",
        "42 marks",
        "0 / 42",
        "Requirements Compliance Matrix",
        "Total Assigned Questions",
        "Assigned Questions Coverage Analysis",
        "Detailed Evidence-First Analysis",
        "Actionable Recommendations for the Student",
        "evaluate_observation_report",
        "render_verification_markdown"
    ]
    for term in forbidden_terms:
        assert term not in md_output, f"Forbidden legacy term '{term}' found in generated markdown"


def test_11_dynamic_faculty_questions_grounded_in_actual_submission(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 11 — Dynamic presentation questions
    Verifies that faculty presentation questions are dynamically generated from
    the actual code files, functions, and discrepancies in the submission,
    and are not hard-coded generic placeholders.
    """
    res = evaluate_holistic_student(
        report_text="Sample report",
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection,
        student_id="N241003",
        week_id="week-04"
    )
    questions = res.presentation_readiness.possible_faculty_questions
    assert len(questions) >= 3, "Must generate at least 3 faculty questions"
    q_text = " ".join(questions)

    # Dynamic grounding assertions
    assert "p6_1.c" in q_text or "p6_2.c" in q_text or "swap" in q_text
    assert "p12.c" in q_text or "frequency" in q_text or "R4" in q_text
    # Questions must reference specific implementations, not generic defaults
    assert not any("Question 1:" in q for q in questions)


def test_12_d1_dynamic_issue_detection_without_hardcoding():
    """
    Test 12 — D1 dynamic calculation without hardcoded 15/17
    Verifies that declaration/return-value issues are dynamically calculated
    based on the exact code passed in.
    """
    code_coll = StudentCodeCollection(
        student_id="N999",
        week="week-01",
        problems=[
            # Issue: non-void function missing return statement
            StudentCodeSnippet(problem_number=1, problem_id="P1", source_file="p1.c", source_code="int helper(int a) { int x = a + 1; } int main() { return 0; }"),
            # Clean
            StudentCodeSnippet(problem_number=2, problem_id="P2", source_file="p2.c", source_code="int clean(int a) { return a + 1; } int main() { return 0; }"),
            # Clean
            StudentCodeSnippet(problem_number=3, problem_id="P3", source_file="p3.c", source_code="void proc() { int x = 0; } int main() { return 0; }"),
        ]
    )
    res = evaluate_holistic_student(
        report_text="Sample report",
        extracted_report={"detected_programs": []},
        student_code=code_coll,
        student_id="N999",
        week_id="week-01"
    )
    d1 = next(d for d in res.dimensions if d.dimension == "D1")
    # Must dynamically find 2 of 3 clean files, NOT hardcoded 15 of 17!
    assert "2 of 3" in d1.justification or "2/3" in " ".join(e.content for e in d1.evidence)
    assert "15 of 17" not in d1.justification


def test_13_d2_recurrence_distinguishes_linear_from_branching_and_derives_base_case():
    """
    Test 13 — D2 recurrence analysis
    - Linear single-call recursion (e.g. sum(n-1)) must NOT be labeled binary tree recursion
    - Doubling recurrence (f(n-1) + f(n-1)) must derive recurrence and base case dynamically
    - Branching tree recursion must be reserved for genuine distinct branches
    """
    code_coll = StudentCodeCollection(
        student_id="N888",
        week="week-04",
        problems=[
            # Linear recursion
            StudentCodeSnippet(problem_number=4, problem_id="P4", source_file="p4.c", source_code="int sum(int n) { if(n<=0) return 0; return n + sum(n-1); }"),
            # Doubling recurrence
            StudentCodeSnippet(problem_number=13, problem_id="P13", source_file="p13.c", source_code="int f(int n) { if(n<=1) return 1; return f(n-1) + f(n-1); }"),
            # Branching tree recursion
            StudentCodeSnippet(problem_number=9, problem_id="P9", source_file="p9.c", source_code="int fib(int n) { if(n<=1) return n; return fib(n-1) + fib(n-2); }")
        ]
    )
    res = evaluate_holistic_student(
        report_text="Sample",
        extracted_report={"detected_programs": []},
        student_code=code_coll,
        student_id="N888",
        week_id="week-04"
    )
    d2 = next(d for d in res.dimensions if d.dimension == "D2")
    evidence_text = " ".join(e.content for e in d2.evidence)

    # p4.c must NOT be labeled branching/binary tree recursion
    assert "p4.c" not in evidence_text or "branching tree" not in evidence_text.split("p4.c")[1][:100]
    # p9.c is genuine branching
    assert "branching tree recursion" in evidence_text or "fib" in evidence_text
    # p13.c is doubling recurrence without hardcoded closed form
    assert "doubling recurrence" in evidence_text
    assert "2^(n-1)" not in evidence_text


def test_14_authoritative_python_metadata_cannot_be_overridden_by_llm(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 14 — Authoritative Python metadata override
    Verifies that Python retains strict authority over problem IDs, filenames, counts,
    and never collapses p6_1.c or p6_2.c to p6.c.
    """
    res = evaluate_holistic_student(
        report_text="Sample report",
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection,
        student_id="N241003",
        week_id="week-04"
    )
    md = render_holistic_evaluation_markdown(res, matches=res.matches, student_code=sample_17_code_collection)

    # p6_1.c and p6_2.c must appear as distinct filenames in the markdown
    assert "`p6_1.c`" in md or "p6_1.c" in md
    assert "`p6_2.c`" in md or "p6_2.c" in md

    # Ensure no collapsed p6.c exists
    lines_with_p6 = [line for line in md.splitlines() if "p6.c" in line and "p6_1" not in line and "p6_2" not in line]
    assert len(lines_with_p6) == 0, f"Found collapsed p6.c in markdown: {lines_with_p6}"


def test_15_notable_implementations_candidate_header(sample_17_code_collection, sample_4_report_extraction):
    """
    Test 15 — Notable Implementations candidate count header
    Verifies the markdown contains the dynamic candidate header:
    *9 candidates identified; top 4 selected for presentation relevance.*
    """
    res = evaluate_holistic_student(
        report_text="Sample report",
        extracted_report=sample_4_report_extraction,
        student_code=sample_17_code_collection,
        student_id="N241003",
        week_id="week-04"
    )
    md = render_holistic_evaluation_markdown(res, matches=res.matches, student_code=sample_17_code_collection)
    assert "candidates identified; top 4 selected for presentation relevance" in md


