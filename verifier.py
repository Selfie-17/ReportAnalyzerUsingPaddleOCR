"""
verifier.py
Dynamic, Hallucination-Resistant Observation Report Verification Engine.
Evaluates student laboratory observation reports against runtime-supplied
assigned questions and runtime-supplied instruction manuals.

Core Architectural Principles:
1. Assigned Questions are runtime data (never hardcoded, arbitrary count).
2. Instruction Manual requirements are runtime data (never hardcoded sections).
3. Student OCR is the sole evidence of what was written (never invent solutions).
4. Evidence is separated from evaluation and validated against OCR text in Python.
5. All counting, metrics, and Markdown formatting are performed deterministically in Python.
"""

import json
import re
import difflib
import requests
import sys
from typing import Generator, Dict, Any, Optional, List, Tuple, Union

def safe_print(*args, **kwargs):
    """
    Safely prints text handling Windows cp1252 / charmap encoding restrictions.
    """
    try:
        print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeError):
        enc = sys.stdout.encoding or "utf-8"
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        msg = sep.join(str(a) for a in args)
        try:
            sys.stdout.buffer.write(msg.encode(enc, errors="replace") + end.encode(enc, errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            pass


from schemas import (
    AssignedQuestion,
    RequirementEvaluation,
    QuestionEvaluation,
    ObjectiveEvaluation,
    DynamicEvaluationResult,
)


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"

# ============================================================
# RUNTIME FALLBACK PRESETS (FOR TESTING / CONVENIENCE ONLY)
# ============================================================

DEFAULT_MANUAL_QUESTIONS_PRESET = """Program 1: Factors of a Number
Program 2: Factorial of a Number
Program 3: Palindrome Number
Program 4: Prime Number
Program 5: Fibonacci Series"""

WEEK1_13_PROGRAMS_PRESET = """1. Write a C program to check whether a given number is even or odd.
2. Write a C program to determine whether a given number is positive, negative, or zero.
3. Write a C program to check whether an entered character is an uppercase letter, lowercase letter, digit, or special character.
4. Write a C program to check whether a given year is a leap year or not.
5. Write a C program to display the memory allocation required for different C data types using the sizeof() operator.
6. Write a menu-based C program to perform addition, subtraction, multiplication, division, modulus, and power using a switch statement.
7. Write a C program to swap two numbers without using a third/temporary variable.
8. Write a C program to swap two numbers using a temporary variable.
9. Write a C program to check whether a given number is a perfect square without using the sqrt() library function.
10. Write a C program to calculate a student's grade based on marks using if-else statements.
11. Write an extended menu-based C calculator that handles invalid menu choices and division by zero.
12. Write a C program to find the largest of three numbers using the ternary operator.
13. Write a C program to convert a grade point to a letter grade using a switch statement."""

OFFICIAL_INSTRUCTION_MANUAL = """# Instruction Manual for Writing the Observation Report

## Purpose of the Observation Report
The observation report is not meant to reproduce the complete C program or copy theory from the laboratory manual. Its purpose is to show **what you understood while performing the experiment**.
For every laboratory session, write the observation report in your own words using the sections given below.

---

## 1. Objective of the Lab
Write the overall purpose of the laboratory exercise.
Your objective should explain:
* what programming concepts you practiced,
* what type of problems you solved, and
* what skills the experiment was intended to develop.
Do not simply copy the list of program questions.

### Example
> The objective of this laboratory exercise is to understand the use of loops, conditional statements, arithmetic operations, and iterative logic in solving basic numerical problems using C programming.

The objective is written **once for the complete laboratory session**.

---

## 2. Problem Understanding
Write this section **separately for every program**.
Explain in simple words:
* what input is given to the program,
* what the program is expected to determine or calculate, and
* what output should be produced.

Do not write the C code in this section.

### Example: Factorial Program
> The program accepts a positive integer as input and calculates the product of all integers from 1 up to that number. For example, if the input is 5, the expected result is 5 * 4 * 3 * 2 * 1 = 120.

### Example: Prime Number Program
> The program accepts an integer and checks whether the number has any factors other than 1 and itself. If no such factor exists, the number is identified as prime.

Try to explain the problem **in your own words**.

---

## 3. Logic / Approach Used
Write this section **for every program**.
Explain how you solved the problem step by step.
Your explanation should mention:
* where the repetition or loop is required,
* what condition is checked,
* how the values change during execution, and
* how the final result is obtained.

Do not copy the program statement by statement. Explain the **idea behind the program**.

### Example: Factorial Program
> First, the factorial variable is initialized to 1. A loop is then used from 1 up to the given number. During every iteration, the current value is multiplied with the factorial variable. After the loop finishes, the factorial variable contains the required result.

### Example: Palindrome Program
> The original number is stored separately. The last digit is repeatedly obtained using the modulus operator. These digits are used to construct the reversed number. After all the digits are processed, the reversed number is compared with the original number. If both are equal, the number is a palindrome.

Your explanation should make it possible for another student to understand the solution **even without seeing your C program**.

---

## 4. Important Variables and Their Purpose
For every program, identify the important variables used in your code and explain why each variable is required.
Use a small table.

### Example
| Variable | Purpose |
| -------- | ------- |
| `n`      | Stores the input number |
| `i`      | Controls the loop |
| `fact`   | Stores the factorial calculated so far |

For a palindrome program, the table may look like:
| Variable   | Purpose |
| ---------- | ------- |
| `n`        | Stores the input number |
| `original` | Stores the original number for final comparison |
| `digit`    | Stores the last digit extracted from the number |
| `reverse`  | Stores the reversed number |

Do not list every variable unnecessarily. Mention only the variables that are important for understanding the program.

---

## 5. What I Observed
Write what you noticed while executing the program.
This section should contain an **actual observation about the behaviour or logic of the program**.
Avoid statements such as:
> The program executed successfully.
or
> I got the correct output.
These statements alone do not explain what you learned from the experiment.
Instead, describe something meaningful that you observed.

### Example: Factors
> I observed that a number is printed as a factor only when the remainder obtained using `n % i` is zero.

### Example: Factorial
> I observed that the factorial value changes in every iteration and the result from the previous iteration is required for the next multiplication.

### Example: Palindrome
> I observed that the original number must be saved before reversing it because the value of the input number changes while its digits are being extracted.

### Example: Prime Number
> I observed that if the number is exactly divisible by any number other than 1 and itself, it cannot be prime.

### Example: Fibonacci Series
> I observed that every new Fibonacci term is obtained using the previous two terms. Therefore, the values of the two previous terms must be updated after every iteration.

Try to write **at least one meaningful observation for every program**.

---

# Recommended Report Structure
For a laboratory containing five programs, arrange your observation report as follows:
## Objective of the Lab
Write the common objective for the complete experiment.
---
## Program 1: Factors of a Number
### Problem Understanding
### Logic / Approach Used
### Important Variables and Their Purpose
### What I Observed
---
(Repeated for subsequent programs: Factorial, Palindrome, Prime, Fibonacci, etc.)

---

# Important Instructions to Students
* Write the observation report **after performing the programs in the laboratory**.
* Write the explanation in **your own words**.
* Do not copy the complete program into the observation report.
* Do not copy definitions directly from the laboratory manual or from another student's report.
* Explain the **reasoning behind the program**, not merely the syntax used.
* Keep the explanation short, clear, and technically correct.
* Use proper variable names from the program when explaining their purpose.
* Write at least one meaningful observation for every program.
* The observation should show **what you understood from executing the program**.
* Different students may write different observations even when they perform the same program. This is acceptable if the observation is logically and technically correct.

## Remember
A good observation report should answer five questions:
1. **What was I trying to learn?**
2. **What problem was I solving?**
3. **How did I solve it?**
4. **What role did the important variables play?**
5. **What did I understand or notice while executing the program?**

The observation report will be evaluated based on your **understanding of the program and your ability to explain it clearly**, rather than on the length of the report.
"""


# ============================================================
# 1. STRICT SOURCE SEPARATION & SANITIZATION
# ============================================================

def sanitize_student_ocr_text(
    raw_text: str,
    filename: Optional[str] = None,
    assigned_count: Optional[int] = None,
    manual_char_count: Optional[int] = None,
    debug: bool = True
) -> str:
    """
    Guarantees strict source separation:
    - Retains ONLY legitimate student OCR text extracted from the uploaded observation report.
    - Strips any appended evaluator test cases, harness notes, or 'Expected qualitative behavior' sections.
    - Strips any previously generated verification Markdown reports or final score tables.
    - Strips any Python source code or test cases (e.g. def add, def subtract, test_cases = [...]).
    - Strips any leaked evaluator prompt instructions or XML wrappers.
    - Logs [INPUT DEBUG] metrics.
    - Enforces source separation assertions.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    cleaned = str(raw_text)

    # 1. Strip evaluator test case harness / notes / expected qualitative behavior appendix
    cleaned = re.split(
        r'\n+\s*(?:#+\s*)?(?:Intentional\s+Test\s+Cases\s+for\s+the\s+Evaluator|Expected\s+qualitative\s+behavior:)',
        cleaned,
        flags=re.IGNORECASE
    )[0]

    # 2. Strip any previously generated evaluation markdown reports or tables
    cleaned = re.split(
        r'\n+\s*#+\s*📊\s*(?:Laboratory\s+)?Observation\s+Report',
        cleaned,
        flags=re.IGNORECASE
    )[0]
    cleaned = re.split(
        r'\n+\s*##+\s*📊\s*Final\s+Score',
        cleaned,
        flags=re.IGNORECASE
    )[0]
    cleaned = re.split(
        r'\n+\s*##+\s*(?:📋\s*)?Assigned\s+Questions\s+Coverage\s+Analysis',
        cleaned,
        flags=re.IGNORECASE
    )[0]
    cleaned = re.split(
        r'\n+\s*##+\s*(?:📋\s*)?Requirements\s+Compliance\s+Matrix',
        cleaned,
        flags=re.IGNORECASE
    )[0]

    # 3. Strip any Python functions or test case blocks
    cleaned = re.sub(
        r'(?:\n|^)\s*def\s+(?:add|subtract|multiply|divide|test_\w+)\s*\(.*?\):.*?(?=\n\s*(?:#|def|\w+\s*=|\d+\.|\Z))',
        '',
        cleaned,
        flags=re.DOTALL
    )
    cleaned = re.sub(
        r'(?:\n|^)\s*test_cases\s*=\s*\[.*?\]',
        '',
        cleaned,
        flags=re.DOTALL
    )

    # 4. Strip prompt XML tags if leaked
    for tag in [
        "<STUDENT_REPORT_OCR>", "</STUDENT_REPORT_OCR>",
        "<ASSIGNED_QUESTIONS>", "</ASSIGNED_QUESTIONS>",
        "<INSTRUCTION_MANUAL_REQUIREMENTS>", "</INSTRUCTION_MANUAL_REQUIREMENTS>"
    ]:
        cleaned = cleaned.replace(tag, "")

    cleaned = cleaned.strip()

    # Log [INPUT DEBUG]
    if debug:
        safe_print("[INPUT DEBUG]")
        if filename:
            safe_print(f"PDF filename: {filename}")
        safe_print(f"OCR character count: {len(raw_text)}")
        safe_print(f"OCR first 500 characters:\n{raw_text[:500]}")
        safe_print(f"OCR last 500 characters:\n{raw_text[-500:]}")
        if assigned_count is not None:
            safe_print(f"Assigned question count: {assigned_count}")
        if manual_char_count is not None:
            safe_print(f"Instruction manual character count: {manual_char_count}")
        safe_print(f"Evaluation source character count: {len(cleaned)}")

    # Strict Assertions
    assert "## 📊 Final Score" not in cleaned, "Generated final score markdown must not be in evaluation source"
    assert "# 📊 Laboratory Observation Report" not in cleaned, "Generated report markdown must not be in evaluation source"
    assert "def add(" not in cleaned, "Evaluator python test code must not be in evaluation source"
    assert "def subtract(" not in cleaned, "Evaluator python test code must not be in evaluation source"
    assert "def multiply(" not in cleaned, "Evaluator python test code must not be in evaluation source"
    assert "test_cases =" not in cleaned, "Test cases must not be in evaluation source"
    assert "Expected qualitative behavior" not in cleaned, "Expected qualitative behavior must not be in evaluation source"

    return cleaned


# ============================================================
# 1. DYNAMIC PARSERS (QUESTIONS & INSTRUCTION MANUAL)
# ============================================================

def parse_assigned_questions(raw_text: Optional[str]) -> List[AssignedQuestion]:
    """
    Parses arbitrary assigned questions from text input into structured AssignedQuestion objects.
    Supports:
      - Numbered lists: '1. Write a C program...', 'Question 1: ...', 'Program 1: ...', '1) ...'
      - Markdown lists: '- 1. ...', '* Question 1: ...'
      - JSON strings: '[{"question_number": 1, "question_text": "..."}]'
      - Plain unnumbered non-empty lines (auto-numbered 1..N)
    Never assumes a fixed count (works with 3, 5, 13, 20, etc.).
    """
    if not raw_text or not str(raw_text).strip():
        return []

    text = str(raw_text).strip()

    # Case 1: JSON list of objects
    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                parsed = []
                for idx, item in enumerate(data, start=1):
                    if isinstance(item, dict):
                        q_num = item.get("question_number", idx)
                        q_txt = item.get("question_text", "")
                        if q_txt:
                            parsed.append(AssignedQuestion(question_number=int(q_num), question_text=str(q_txt).strip()))
                    elif isinstance(item, str) and item.strip():
                        parsed.append(AssignedQuestion(question_number=idx, question_text=item.strip()))
                if parsed:
                    return parsed
        except Exception:
            pass

    # Case 2: Line-by-line parsing
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed_questions: List[AssignedQuestion] = []
    
    # Regex to detect numbered items
    pattern = re.compile(
        r"^(?:[-*•]\s*)?(?:(?:Question|Program|Problem|Task|Q|Level\s*\d+[\s\:]*\d*)\s*)?(\d+)[\.\:\)\-]\s*(.*)$",
        re.IGNORECASE
    )

    fallback_lines: List[str] = []

    for line in lines:
        # Ignore markdown headers that look like section titles rather than questions
        if re.match(r"^#{1,3}\s+(?:Assigned|Lab|Programs|Questions|Table of)", line, re.IGNORECASE):
            continue
        if re.match(r"^<.*?>$", line):
            continue

        m = pattern.match(line)
        if m:
            q_num = int(m.group(1))
            q_txt = m.group(2).strip()
            if q_txt:
                parsed_questions.append(AssignedQuestion(
                    question_number=q_num,
                    question_text=q_txt
                ))
            else:
                fallback_lines.append(line)
        else:
            fallback_lines.append(line)

    # If line numbers were matched and cover the questions, return sorted by question number
    if parsed_questions:
        # Sort and re-index sequentially if gaps or duplicates occur
        parsed_questions.sort(key=lambda q: q.question_number)
        # Normalize numbering to 1..N
        normalized = []
        for idx, q in enumerate(parsed_questions, start=1):
            normalized.append(AssignedQuestion(
                question_number=idx,
                question_text=q.question_text
            ))
        return normalized

    # Fallback: Treat each non-header line as an assigned question
    res = []
    for idx, f_line in enumerate(fallback_lines, start=1):
        clean_text = re.sub(r"^[-*•\d\.\:\)\s]+", "", f_line).strip()
        if clean_text and len(clean_text) > 3:
            res.append(AssignedQuestion(
                question_number=idx,
                question_text=clean_text
            ))
    return res


def parse_instruction_manual(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Dynamically extracts evaluation criteria and requirements from the runtime instruction manual.
    Extracts:
      - objective_requirement: (e.g. 'Objective of the Lab' or custom name, if present)
      - per_question_requirements: List of required sections per question
      - guidelines: Dict mapping requirement name to guidelines/penalties
    Adapts dynamically to any manual structure (e.g. C lab vs Java vs Research methodology).
    """
    text = (raw_text or "").strip()
    if not text:
        text = OFFICIAL_INSTRUCTION_MANUAL

    lines = text.splitlines()
    sections: Dict[str, str] = {}
    current_heading = "Header"
    current_content: List[str] = []

    # Meta-sections whose contents and sub-headings should NOT be parsed as criteria
    meta_section_keywords = [
        "purpose of the observation report",
        "instruction manual",
        "lab manual",
        "laboratory manual",
        "manual",
        "recommended report structure",
        "important instructions",
        "remember",
        "evaluation rules",
        "scoring rubric",
        "apparatus",
        "procedure",
        "header"
    ]

    active_meta_h1 = False
    active_meta_h2 = False
    clean_headings: List[str] = []

    for line in lines:
        h_match = re.match(r"^(#{1,3})\s*(?:(?:\d+)[\.\:\)]\s*)?([^\n#]+)", line)
        if h_match:
            hashes = h_match.group(1)
            raw_title = h_match.group(2).strip()

            if current_heading and current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = raw_title
            current_content = []

            title_lower = raw_title.lower()

            if len(hashes) == 1:
                active_meta_h1 = any(k in title_lower for k in [
                    "recommended report structure",
                    "important instructions",
                    "guidelines",
                    "general instructions",
                    "structure",
                    "appendix"
                ])
                active_meta_h2 = False
            elif len(hashes) == 2:
                active_meta_h2 = any(k in title_lower for k in meta_section_keywords)

            # Skip headings inside a meta section or matching meta keywords
            if active_meta_h1 or active_meta_h2 or any(k in title_lower for k in meta_section_keywords):
                continue

            # Strict Example Filter: Examples inside manual must NEVER be requirements
            if re.search(r'\b(?:example|examples|sample|e\.?g\.?|illustration)\b', title_lower):
                continue

            clean_headings.append(raw_title)
        else:
            current_content.append(line)

    if current_heading and current_content:
        sections[current_heading] = "\n".join(current_content).strip()

    # Detect session-wide objective requirement
    objective_req = None
    per_question_reqs = []

    for h in clean_headings:
        if any(k in h.lower() for k in ["objective", "aim", "overall purpose"]):
            if not objective_req:
                objective_req = h
        else:
            per_question_reqs.append(h)

    # Fallback if no specific per-question headings found
    if not per_question_reqs:
        # Check if bullet points specify requirements
        bullet_reqs = []
        for line in lines:
            bm = re.match(r"^[-*]\s+\*\*?([A-Za-z\s/]+)\*\*?", line)
            if bm:
                b_name = bm.group(1).strip()
                if (
                    len(b_name) > 3
                    and not any(k in b_name.lower() for k in meta_section_keywords)
                    and not any(k in b_name.lower() for k in ["objective", "aim"])
                    and not re.search(r'\b(?:example|examples|sample|e\.?g\.?)\b', b_name.lower())
                ):
                    if b_name not in bullet_reqs:
                        bullet_reqs.append(b_name)
        if bullet_reqs:
            per_question_reqs = bullet_reqs
        else:
            # Default fallback requirements
            per_question_reqs = [
                "Problem Understanding",
                "Logic / Approach Used",
                "Important Variables and Their Purpose",
                "What I Observed"
            ]

    # Clean per_question_reqs names
    clean_per_q = []
    for r in per_question_reqs:
        c_name = re.sub(r"^(?:\d+[\.\)]\s*)", "", r).strip()
        if (
            c_name
            and c_name not in clean_per_q
            and not re.search(r'\b(?:example|examples|sample|e\.?g\.?)\b', c_name.lower())
        ):
            clean_per_q.append(c_name)

    return {
        "objective_requirement": objective_req or "Objective of the Lab",
        "per_question_requirements": clean_per_q,
        "sections": sections,
        "raw_text": text
    }


# ============================================================
# 2. EVIDENCE GROUNDING VERIFIER (PYTHON POST-VALIDATION)
# ============================================================

def _normalize_text_for_search(text: str) -> str:
    """Collapses whitespace, lowercases, and removes punctuation for robust substring search."""
    t = (text or "").lower()
    t = re.sub(r'[`*_#~>\[\]\(\)\{\}|\\:;.,!?"\'\-\/]', ' ', t)
    return " ".join(t.split())


def is_generic_observation(text: Optional[str]) -> bool:
    """
    Detects whether an observation or statement is generic/boilerplate rather than
    describing specific program behaviour.
    Examples:
      - 'All programs were compiled and executed successfully without any errors'
      - 'All programs executed successfully'
      - 'I got the correct output'
      - 'expected output was obtained'
      - 'matched the expected results'
      - 'program worked correctly'
      - 'All programs compiled without errors'
      - 'Outputs were verified'
    """
    if not text:
        return False
    norm = _normalize_text_for_search(str(text))
    if not norm:
        return False
    generic_patterns = [
        r"all programs? (?:were )?(?:compiled|executed|ran|completed)",
        r"(?:all )?(?:programs?|code) (?:were )?(?:compiled and )?executed successfully",
        r"(?:i|we) got (?:the )?(?:correct|expected) output",
        r"(?:output|outputs|result|results) (?:was|were)? ?(?:verified|correct|checked|matched|obtained)",
        r"expected output (?:was )?obtained",
        r"matched (?:the )?expected results?",
        r"program worked correctly",
        r"(?:compiled and )?executed without (?:any )?(?:error|errors|issue|issues)",
        r"without (?:any )?(?:syntax )?(?:error|errors)",
        r"all (?:experiments|programs|outputs) (?:are |were )?(?:verified|successful)",
        r"successfully executed",
        r"compiled successfully",
        r"compiled and executed successfully",
        r"compiled and run successfully",
        r"run successfully",
        r"no syntax errors?",
        r"executed properly",
        r"verified successfully",
    ]
    for pat in generic_patterns:
        if re.search(pat, norm):
            return True
    return False


COMMON_FILLER_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "and", "or", "but", "so", "if", "then", "else", "as", "it", "its",
    "this", "that", "these", "those", "student", "students", "program", "programs",
    "we", "i", "he", "she", "they", "my", "his", "her", "their", "user", "users"
}


def is_meaningful_span(span: str) -> bool:
    """Checks if a span contains at least one meaningful domain/technical term (not purely stopword filler)."""
    tokens = [t for t in _normalize_text_for_search(span).split() if t]
    non_filler = [t for t in tokens if t not in COMMON_FILLER_WORDS and len(t) >= 2]
    return len(non_filler) >= 1


def count_non_filler(span: str) -> int:
    tokens = [t for t in _normalize_text_for_search(span).split() if t]
    return sum(1 for t in tokens if t not in COMMON_FILLER_WORDS and len(t) >= 2)


def extract_grounded_source_span(
    candidate: str,
    ocr_text: str,
    min_words: int = 2
) -> Optional[str]:
    """
    Extracts the longest contiguous sub-span from candidate that exists
    as an exact or normalized substring in the OCR text.
    Strips away hallucinated model prefixes ('The student states that...', 'We can see...')
    and generated explanations ('...to determine if even').
    Returns None if no meaningful span (>= min_words) exists in the OCR.
    """
    if not candidate or not ocr_text:
        return None

    cand_norm = _normalize_text_for_search(candidate)
    ocr_norm = _normalize_text_for_search(ocr_text)

    # 1. If the candidate as a whole is already an exact normalized substring
    if cand_norm in ocr_norm:
        return candidate.strip()

    # 2. Find longest contiguous word sequence from candidate that exists in ocr_norm
    cand_words = candidate.strip().split()
    if len(cand_words) < min_words:
        # Check single word if length >= 4
        if len(cand_words) == 1 and len(cand_words[0]) >= 4:
            w_norm = _normalize_text_for_search(cand_words[0])
            if w_norm and w_norm in ocr_norm and is_meaningful_span(cand_words[0]):
                return cand_words[0]
        return None

    valid_matches = []
    for window_size in range(len(cand_words), min_words - 1, -1):
        for start_idx in range(len(cand_words) - window_size + 1):
            sub_words = cand_words[start_idx:start_idx + window_size]
            sub_str = " ".join(sub_words)
            sub_norm = _normalize_text_for_search(sub_str)

            if sub_norm in ocr_norm:
                cleaned_span = re.sub(r"^[\s,;:.!?'\"`\-]+|[\s,;:.!?'\"`\-]+$", "", sub_str)
                if len(cleaned_span.split()) >= min_words and is_meaningful_span(cleaned_span):
                    valid_matches.append((
                        len(cleaned_span.split()),
                        count_non_filler(cleaned_span),
                        len(cleaned_span),
                        cleaned_span
                    ))

    if valid_matches:
        # Sort prioritizing highest word count, then highest count of non-filler words
        valid_matches.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return valid_matches[0][3]

    # 3. Fuzzy window check for short candidates (<= 5 words)
    if len(cand_words) <= 5 and is_meaningful_span(candidate):
        ocr_words = ocr_norm.split()
        cand_norm_words = cand_norm.split()
        cw_len = len(cand_norm_words)
        for i in range(len(ocr_words) - cw_len + 1):
            window = " ".join(ocr_words[i:i + cw_len])
            if difflib.SequenceMatcher(None, cand_norm, window).ratio() >= 0.85:
                return window

    return None


def is_evidence_grounded(
    evidence: Optional[Union[str, List[str]]],
    ocr_text: str
) -> Tuple[bool, float]:
    """
    Validates whether the student evidence claimed by the LLM is actually present in the OCR text.
    Returns:
      (is_grounded: bool, confidence_ratio: float)
    Strict Grounding Hierarchy:
      1. Exact substring match in OCR text
      2. Normalized substring match (whitespace, punctuation, lowercase collapsed)
      3. OCR-tolerant fuzzy span match (localized window of matching length, ratio >= 0.80)
    Rejects long model-generated sentences containing ungrounded semantic claims,
    even if individual words appear in the OCR.
    """
    if evidence is None:
        return True, 1.0

    if isinstance(evidence, list):
        if not evidence:
            return True, 1.0
        results = [is_evidence_grounded(item, ocr_text) for item in evidence]
        all_grounded = all(r[0] for r in results)
        avg_ratio = sum(r[1] for r in results) / len(results) if results else 1.0
        return all_grounded, avg_ratio

    ev_str = str(evidence).strip()
    if not ev_str or ev_str.lower() in ("null", "none"):
        return True, 1.0

    # 1. Exact substring
    if ev_str in ocr_text:
        return True, 1.0

    # 2. Normalized substring match
    ev_clean = _normalize_text_for_search(ev_str)
    ocr_clean = _normalize_text_for_search(ocr_text)

    if not ev_clean:
        return True, 1.0

    if ev_clean in ocr_clean:
        return True, 1.0

    # 3. Localized windowed fuzzy match
    ev_words = ev_clean.split()
    ocr_words = ocr_clean.split()
    n_ev = len(ev_words)

    if n_ev == 0 or len(ocr_words) == 0:
        return False, 0.0

    # Single word check
    if n_ev == 1:
        w = ev_words[0]
        if w in ocr_clean:
            return True, 1.0
        for ow in ocr_words:
            if difflib.SequenceMatcher(None, w, ow).ratio() >= 0.80:
                return True, 0.85
        return False, 0.0

    # Windowed alignment: compare evidence ONLY against OCR windows of approximately the same word length (+/- 2 words)
    min_k = max(1, n_ev - 2)
    max_k = min(len(ocr_words), n_ev + 2)

    best_ratio = 0.0
    for k in range(min_k, max_k + 1):
        for i in range(len(ocr_words) - k + 1):
            window_str = " ".join(ocr_words[i:i + k])
            ratio = difflib.SequenceMatcher(None, ev_clean, window_str).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
            if best_ratio >= 0.82:
                return True, best_ratio

    if best_ratio >= 0.80:
        return True, best_ratio

    # Otherwise reject: complete span is not grounded in OCR
    return False, best_ratio


def process_and_ground_evidence(
    raw_ev: Any,
    ocr_text: str
) -> Tuple[Optional[Union[str, List[str]]], bool, float, bool]:
    """
    Processes raw evidence from LLM:
    Returns (cleaned_evidence, is_grounded, confidence_ratio, is_generic)
    """
    if raw_ev is None or str(raw_ev).strip().lower() in ("none", "null", ""):
        return None, True, 1.0, False

    if isinstance(raw_ev, list):
        cleaned_spans: List[str] = []
        all_spans_grounded = True
        has_generic = False
        ratios: List[float] = []

        for item in raw_ev:
            item_str = str(item).strip()
            if not item_str or item_str.lower() in ("null", "none"):
                continue
            if is_generic_observation(item_str):
                has_generic = True
            is_gr, ratio = is_evidence_grounded(item_str, ocr_text)
            ratios.append(ratio)
            if not is_gr:
                src_span = extract_grounded_source_span(item_str, ocr_text)
                if src_span:
                    cleaned_spans.append(src_span)
                else:
                    all_spans_grounded = False
            else:
                cleaned_spans.append(item_str)

        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        res_spans: Optional[Union[str, List[str]]] = cleaned_spans if cleaned_spans else None
        return res_spans, all_spans_grounded, avg_ratio, has_generic

    ev_str = str(raw_ev).strip()
    is_gen = is_generic_observation(ev_str)
    is_gr, ratio = is_evidence_grounded(ev_str, ocr_text)

    if is_gr:
        return ev_str, True, ratio, is_gen
    else:
        # Check if Qwen wrapped a pure source fragment with generated explanation
        src_span = extract_grounded_source_span(ev_str, ocr_text)
        if src_span and len(src_span.split()) >= 2:
            return src_span, True, 1.0, is_gen
        else:
            return ev_str, False, ratio, is_gen


# ============================================================
# 2B. LAYERED QUESTION MATCHING ENGINE (LOCALITY + SEMANTICS + NO GUESSING)
# ============================================================

def segment_report_blocks(ocr_text: str) -> List[Dict[str, Any]]:
    """
    Segments raw OCR text into localized blocks delimited by headings, program markers, or page breaks.
    Preserves document order and section locality.
    """
    if not ocr_text or not str(ocr_text).strip():
        return []

    lines = ocr_text.splitlines()
    blocks: List[Dict[str, Any]] = []
    current_heading_raw = None
    current_heading_text = None
    current_num = None
    current_content: List[str] = []

    heading_pattern = re.compile(
        r"^(?:#{1,4}\s*|\*{0,2})(?:(?:Program|Programme|Experiment|Exp|Question|Prog|P|Q)[\.\s\:\-]*\s*(\d+)[\.\s\:\-]*|(\d+)[\.\)\:\-]\s+)(.*)$",
        re.IGNORECASE
    )
    alt_heading_pattern = re.compile(
        r"^(?:#{1,3}\s+)(.+)$"
    )
    bullet_pattern = re.compile(
        r"^(?:[\*\-•]\s+)(.+)$"
    )

    for line in lines:
        line_s = line.strip()
        if not line_s:
            current_content.append(line)
            continue

        m = heading_pattern.match(line_s)
        m_alt = alt_heading_pattern.match(line_s) if not m else None
        m_bul = bullet_pattern.match(line_s) if (not m and not m_alt) else None

        is_heading = False
        num = None
        title = ""

        if m:
            is_heading = True
            num_str = m.group(1) or m.group(2)
            num = int(num_str) if num_str else None
            title = (m.group(3) or "").strip()
        elif m_alt:
            h_candidate = m_alt.group(1).strip()
            sub_sections = [
                "problem understanding", "logic", "algorithm", "variables",
                "what i observed", "observation", "output", "aim", "objective",
                "table of contents", "index", "procedure"
            ]
            if not any(sub in h_candidate.lower() for sub in sub_sections):
                is_heading = True
                title = h_candidate
        elif m_bul:
            cand = m_bul.group(1).strip()
            non_heading_starters = (
                "note", "apparatus", "conclusion", "result", "requirement",
                "int ", "float ", "char ", "double ", "void ", "variable ", "var ",
                "step ", "case ", "if ", "for ", "while ", "return ", "printf"
            )
            if len(cand) <= 80 and not any(cand.lower().startswith(x) for x in non_heading_starters):
                is_heading = True
                title = cand

        if is_heading:
            if current_heading_raw or current_content:
                blocks.append({
                    "block_index": len(blocks),
                    "heading_raw": current_heading_raw,
                    "heading_text": current_heading_text or "",
                    "explicit_number": current_num,
                    "content": "\n".join(current_content).strip(),
                    "is_preamble": current_heading_raw is None
                })
            current_heading_raw = line_s
            current_heading_text = title or line_s
            current_num = num
            current_content = []
        else:
            current_content.append(line)

    if current_heading_raw or current_content:
        blocks.append({
            "block_index": len(blocks),
            "heading_raw": current_heading_raw,
            "heading_text": current_heading_text or "",
            "explicit_number": current_num,
            "content": "\n".join(current_content).strip(),
            "is_preamble": current_heading_raw is None
        })

    if not blocks and ocr_text.strip():
        blocks.append({
            "block_index": 0,
            "heading_raw": None,
            "heading_text": "",
            "explicit_number": None,
            "content": ocr_text.strip(),
            "is_preamble": False
        })

    # Mark preamble: only the first block if it has no heading AND is followed by program blocks,
    # or if it explicitly contains lab preamble markers
    for idx, b in enumerate(blocks):
        c_lower = b.get("content", "").lower()
        has_preamble_kw = any(kw in c_lower for kw in ("objective", "index of programs", "experiments covered"))
        if idx == 0 and len(blocks) > 1 and b.get("heading_raw") is None:
            b["is_preamble"] = True
        elif has_preamble_kw and b.get("heading_raw") is None:
            b["is_preamble"] = True
        else:
            b["is_preamble"] = False

    return blocks


SEMANTIC_CONCEPT_ASSOCIATIONS: Dict[str, Dict[str, Any]] = {
    "even_odd": {
        "keywords": ["even", "odd", "parity"],
        "logic_patterns": [
            r"divided by 2",
            r"divisible by 2",
            r"remainder (?:is )?checked",
            r"remainder (?:with|using) 2",
            r"% 2",
            r"modulus (?:operator )?2",
            r"n % 2 == 0"
        ]
    },
    "fibonacci": {
        "keywords": ["fibonacci", "series"],
        "logic_patterns": [
            r"previous two (?:values|numbers|terms) (?:are )?added",
            r"sum of (?:the )?previous two",
            r"next term is (?:obtained|calculated|found)",
            r"f0 \+ f1",
            r"a \+ b",
            r"first two terms"
        ]
    },
    "factorial": {
        "keywords": ["factorial", "fact"],
        "logic_patterns": [
            r"multiply (?:numbers )?from 1 to n",
            r"previous (?:iteration )?multiplication",
            r"product of (?:all )?integers",
            r"fact \*= i",
            r"fact = fact \* i"
        ]
    },
    "prime": {
        "keywords": ["prime"],
        "logic_patterns": [
            r"divisible only by 1 and itself",
            r"no other factors",
            r"divisible by (?:any )?number from 2 to",
            r"factor count (?:is )?2",
            r"count == 2"
        ]
    },
    "palindrome": {
        "keywords": ["palindrome"],
        "logic_patterns": [
            r"reverse (?:of )?(?:the )?(?:number|digits)",
            r"same (?:from )?(?:both )?left and right",
            r"original (?:number )?equals (?:the )?reversed",
            r"rev == original",
            r"extract(?:ing)? (?:the )?digits"
        ]
    },
    "swap": {
        "keywords": ["swap", "interchange", "exchange"],
        "logic_patterns": [
            r"temporary variable",
            r"temp = a",
            r"without (?:using )?(?:a )?temporary variable",
            r"swapping (?:the )?values"
        ]
    },
    "factors": {
        "keywords": ["factors", "divisor"],
        "logic_patterns": [
            r"n % i == 0",
            r"remainder obtained (?:using|is) zero",
            r"divides (?:the number )?completely"
        ]
    },
    "sum_digits": {
        "keywords": ["sum of digits", "sum digits"],
        "logic_patterns": [
            r"sum \+= digit",
            r"sum = sum \+",
            r"n % 10.*sum",
            r"extract(?:ing)? digit and add"
        ]
    },
    "stack": {
        "keywords": ["stack", "lifo"],
        "logic_patterns": [
            r"push and pop",
            r"last in first out",
            r"top of (?:the )?stack",
            r"top = top->next"
        ]
    },
    "bst": {
        "keywords": ["binary search tree", "bst"],
        "logic_patterns": [
            r"left child",
            r"right child",
            r"insert(?:ing)? into (?:the )?tree",
            r"root->left",
            r"root->right"
        ]
    },
    "dijkstra": {
        "keywords": ["dijkstra", "shortest path"],
        "logic_patterns": [
            r"shortest path",
            r"minimum distance",
            r"visited (?:vertices|nodes|set)",
            r"adjacency matrix"
        ]
    }
}


def _fuzzy_token_in_text(token: str, text_tokens: List[str], threshold: float = 0.85) -> bool:
    """Tolerates minor OCR spelling errors (e.g. factoria1 -> factorial, palindrom -> palindrome)."""
    t_clean = token.lower()
    for tt in text_tokens:
        if tt == t_clean:
            return True
        if abs(len(tt) - len(t_clean)) <= 2 and len(t_clean) >= 4:
            if difflib.SequenceMatcher(None, t_clean, tt).ratio() >= threshold:
                return True
    return False


def match_assigned_questions_to_blocks(
    questions: List[AssignedQuestion],
    blocks: List[Dict[str, Any]],
    ocr_text: str
) -> Dict[int, Dict[str, Any]]:
    """
    Executes layered question matching adhering to:
      Level 1: Explicit question/program heading match (OCR tolerant).
      Level 2: Strong semantic / concept match in nearby content.
      Level 3: Cross-block distinction & disambiguation.
      Level 4: Ambiguity guardrail (no guessing, returns LOW_CONFIDENCE / AMBIGUOUS).
    Preserves section locality and strict evidence grounding.
    """
    results: Dict[int, Dict[str, Any]] = {}

    GENERIC_LAB_WORDS = {
        "write", "program", "programs", "programme", "check", "given", "calculate", "using", "find",
        "determine", "whether", "generate", "implement", "print", "display",
        "input", "output", "user", "enter", "entered", "read",
        "result", "results", "c", "code", "codes", "file",
        "session", "experiment", "experiments", "lab", "laboratory", "exercise", "exercises", "following",
        "perform", "take", "takes", "taking", "show", "shows", "learn", "learning",
        "statement", "statements", "condition", "conditions", "control", "basic",
        "operator", "operators", "arithmetic", "logical", "relational", "simple",
        "table", "contents", "index", "objective", "aim"
    }

    # Step 1: Pre-extract distinctive tokens, active concepts, and compute IDF-like uniqueness weights
    question_profiles = {}
    token_doc_counts: Dict[str, int] = {}

    for q in questions:
        q_norm = _normalize_text_for_search(q.question_text)
        q_tokens = [
            t for t in q_norm.split()
            if len(t) >= 3 and t not in COMMON_FILLER_WORDS and t not in GENERIC_LAB_WORDS
        ]
        q_tokens_unique = list(dict.fromkeys(q_tokens))
        for t in q_tokens_unique:
            token_doc_counts[t] = token_doc_counts.get(t, 0) + 1

        concepts = []
        for c_key, c_val in SEMANTIC_CONCEPT_ASSOCIATIONS.items():
            if any(kw in q_norm for kw in c_val["keywords"]):
                concepts.append(c_key)

        question_profiles[q.question_number] = {
            "tokens": q_tokens_unique,
            "concepts": concepts,
            "question": q
        }

    # Assign weights: tokens unique to 1 question get weight 2.0; shared tokens get discounted
    for q_num, prof in question_profiles.items():
        weights = {}
        for t in prof["tokens"]:
            cnt = token_doc_counts.get(t, 1)
            weights[t] = 2.0 if cnt == 1 else (1.0 / cnt)
        prof["weights"] = weights

    # Step 2: Score all (question, block) pairs with locality preservation
    block_scores: Dict[int, List[Dict[str, Any]]] = {b["block_index"]: [] for b in blocks}

    for b in blocks:
        # Preamble / document header blocks do not contain question answers
        if b.get("is_preamble"):
            continue

        b_idx = b["block_index"]
        h_text = b.get("heading_text", "")
        h_raw = b.get("heading_raw", "")
        b_content = b.get("content", "")
        h_tokens = _normalize_text_for_search(h_text).split()
        c_tokens = _normalize_text_for_search(b_content).split()
        c_norm = _normalize_text_for_search(b_content)

        for q in questions:
            q_num = q.question_number
            prof = question_profiles[q_num]
            q_tokens = prof["tokens"]
            weights = prof["weights"]
            concepts = prof["concepts"]

            heading_score = 0.0
            semantic_score = 0.0
            ev_spans: List[str] = []

            # --- LEVEL 1: Heading Match ---
            # 1a. Explicit sequence number match (e.g. Program 1 == Q1)
            if b.get("explicit_number") == q_num:
                conflicting = False
                for other_q in questions:
                    if other_q.question_number != q_num:
                        other_prof = question_profiles[other_q.question_number]
                        for ot in other_prof["tokens"]:
                            if other_prof["weights"].get(ot, 1.0) >= 1.5:
                                if _fuzzy_token_in_text(ot, h_tokens, 0.85):
                                    conflicting = True
                                    break
                if not conflicting:
                    heading_score += 4.0
                    if h_raw:
                        ev_spans.append(h_raw.strip())

            # 1b. Distinctive topic tokens in heading (OCR tolerant)
            for qt in q_tokens:
                if _fuzzy_token_in_text(qt, h_tokens, 0.85):
                    w = weights.get(qt, 1.0)
                    heading_score += 4.0 * w
                    if h_raw and h_raw.strip() not in ev_spans:
                        ev_spans.append(h_raw.strip())

            # --- LEVEL 2: Semantic / Nearby Content Match ---
            # 2a. Content tokens
            content_token_matches = 0
            for qt in q_tokens:
                if _fuzzy_token_in_text(qt, c_tokens, 0.85):
                    w = weights.get(qt, 1.0)
                    semantic_score += 1.5 * w
                    content_token_matches += 1
                    for line in b_content.splitlines():
                        if qt in _normalize_text_for_search(line):
                            if line.strip() not in ev_spans:
                                ev_spans.append(line.strip())
                            break

            # 2b. Domain concept logic patterns
            concept_logic_matched = False
            for c_key in concepts:
                c_data = SEMANTIC_CONCEPT_ASSOCIATIONS.get(c_key, {})
                for pat in c_data.get("logic_patterns", []):
                    if re.search(pat, c_norm):
                        semantic_score += 4.0
                        concept_logic_matched = True
                        for line in b_content.splitlines():
                            if re.search(pat, _normalize_text_for_search(line)):
                                if line.strip() not in ev_spans:
                                    ev_spans.append(line.strip())
                                break

            # 2c. Penalty if content matches logic of an unrelated concept
            for other_q in questions:
                if other_q.question_number != q_num:
                    for other_c in question_profiles[other_q.question_number]["concepts"]:
                        if other_c not in concepts:
                            other_data = SEMANTIC_CONCEPT_ASSOCIATIONS.get(other_c, {})
                            for pat in other_data.get("logic_patterns", []):
                                if re.search(pat, c_norm):
                                    semantic_score -= 3.0

            # Level 2 Precision Gate: If there is no heading match, content match MUST be strong!
            if heading_score == 0.0 and not concept_logic_matched and content_token_matches < 2:
                semantic_score = 0.0
                ev_spans = []

            total_score = heading_score + semantic_score
            if total_score > 0:
                block_scores[b_idx].append({
                    "question_number": q_num,
                    "total_score": total_score,
                    "heading_score": heading_score,
                    "semantic_score": semantic_score,
                    "ev_spans": ev_spans,
                    "block": b
                })

    # Step 3: Resolve block assignments & identify ambiguous competitions (Level 4)
    assigned_blocks: Dict[int, List[Dict[str, Any]]] = {q.question_number: [] for q in questions}
    ambiguous_questions = set()

    for b in blocks:
        b_idx = b["block_index"]
        cands = block_scores.get(b_idx, [])
        if not cands:
            continue

        cands.sort(key=lambda x: x["total_score"], reverse=True)
        top = cands[0]
        runner_up = cands[1] if len(cands) > 1 else None

        # Threshold check: minimal evidence required
        if top["total_score"] < 2.5:
            if top["total_score"] > 0:
                for c in cands:
                    if c["total_score"] > 0:
                        ambiguous_questions.add(c["question_number"])
            continue

        # Ambiguity check between top two candidates (Level 4)
        if runner_up and runner_up["total_score"] >= 2.0 and (top["total_score"] - runner_up["total_score"]) < 1.5:
            ambiguous_questions.add(top["question_number"])
            ambiguous_questions.add(runner_up["question_number"])
            continue

        # Clean winner for this block
        assigned_blocks[top["question_number"]].append(top)

    # Step 4: Assemble Question Results
    for q in questions:
        q_num = q.question_number
        b_matches = assigned_blocks.get(q_num, [])
        is_ambig = q_num in ambiguous_questions and not b_matches

        if b_matches:
            b_matches.sort(key=lambda x: (x["heading_score"] >= 4.0, x["total_score"]), reverse=True)
            best = b_matches[0]
            best_b = best["block"]
            h_score = best["heading_score"]
            t_score = best["total_score"]
            matched_heading = best_b.get("heading_raw")

            all_spans = []
            for bm in b_matches:
                for sp in bm["ev_spans"]:
                    if sp not in all_spans:
                        all_spans.append(sp)

            if h_score >= 4.0:
                match_conf = "HIGH"
                is_matched = True
            elif t_score >= 4.0:
                match_conf = "HIGH"
                is_matched = True
            elif t_score >= 2.5:
                match_conf = "MEDIUM"
                is_matched = True
            else:
                match_conf = "LOW"
                is_matched = False
                matched_heading = None

            verified_ev = [sp for sp in all_spans if is_evidence_grounded(sp, ocr_text)[0]]

            results[q_num] = {
                "matched": is_matched,
                "match_status": "FOUND_AND_COVERED" if is_matched else "NOT_FOUND",
                "match_confidence": match_conf,
                "matched_heading": matched_heading,
                "match_evidence": verified_ev[:3] if verified_ev else None,
                "matched_block": best_b if is_matched else None,
                "ambiguous": False,
                "score": t_score
            }
        else:
            results[q_num] = {
                "matched": False,
                "match_status": "NOT_FOUND",
                "match_confidence": "LOW" if is_ambig else "HIGH",
                "matched_heading": None,
                "match_evidence": None,
                "matched_block": None,
                "ambiguous": is_ambig,
                "score": 0.0
            }

    return results


# ============================================================
# 3. PROMPT GENERATOR FOR QWEN (PURE JSON FORMAT)
# ============================================================

VERIFICATION_SYSTEM_PROMPT = """You are a strict, objective academic laboratory report evaluation auditor.
Evaluate student evidence in an OCR report against assigned lab questions and Instruction Manual requirements."""


def build_verification_messages(
    report_text: str,
    assigned_questions: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Backward-compatible message builder for legacy tests and scripts.
    """
    sys_prompt = f"""You are an expert Laboratory Observation Report Evaluator.
Your task is to strictly evaluate, grade, and verify a student's observation report against the official submission guide ("Instruction Manual for Writing the Observation Report") and the assigned lab questions provided by the instructor.

### OFFICIAL SUBMISSION GUIDE & REFERENCE MANUAL:
{OFFICIAL_INSTRUCTION_MANUAL}

### EVALUATION RULES & SCORING RUBRIC (Total: 10.0 points):
1. OBJECTIVE OF THE LAB (Max 2.0 points):
   - Written ONCE for the complete lab session.
   - Explains: programming concepts practiced, problem types solved, and skills intended to develop.
   - VIOLATION: Must NOT simply copy the list of program questions.
2. PROBLEM UNDERSTANDING (Max 2.0 points)
3. LOGIC / APPROACH USED (Max 2.0 points)
4. IMPORTANT VARIABLES AND THEIR PURPOSE (Max 2.0 points)
5. WHAT I OBSERVED (Max 2.0 points):
   - Contains actual meaningful observations about program behaviour.

### PROHIBITED OUTPUT:
- NEVER write Python source code, functions, or implementation snippets (e.g. def add, def subtract, def multiply).
- NEVER generate test case arrays or test scripts (e.g. test_cases = [...]).
- DO NOT invent missing code or algorithms.
- Strictly evaluate the student's written text against the 5 criteria.
"""
    if assigned_questions and assigned_questions.strip():
        usr_prompt = (
            "Please evaluate and verify the following student observation report against the "
            "official Instruction Manual submission guide and the assigned lab questions below:\n\n"
            "### 📝 ASSIGNED LAB QUESTIONS / PROBLEMS (Basis for Student Report):\n"
            f"{assigned_questions.strip()}\n\n"
            "### 📄 EXTRACTED STUDENT OBSERVATION REPORT:\n"
            f"```markdown\n{report_text.strip()}\n```\n"
        )
    else:
        usr_prompt = (
            "Please evaluate and verify the following student observation report against the "
            "official Instruction Manual submission guide:\n\n"
            "### 📄 EXTRACTED STUDENT OBSERVATION REPORT:\n"
            f"```markdown\n{report_text.strip()}\n```\n"
        )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt}
    ]


def build_dynamic_verification_messages(
    report_text: str,
    questions: List[AssignedQuestion],
    manual_requirements: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Constructs the system and user messages for Ollama / Qwen.
    Instructs Qwen strictly as an evidence classifier without answer generation.
    Enforces strict pure JSON output.
    """
    obj_req = manual_requirements.get("objective_requirement", "Objective of the Lab")
    per_q_reqs = manual_requirements.get("per_question_requirements", [])

    system_prompt = f"""You are a strict, objective academic laboratory report evaluation auditor.
Your job is to classify student evidence in an OCR report against assigned lab questions and the Instruction Manual requirements.

CRITICAL NON-HALLUCINATION & EVIDENCE RULES:
1. THE STUDENT OCR IS THE ONLY EVIDENCE of what was actually written.
2. NEVER INVENT ANSWERS: If the student did not write something, mark it as MISSING. Never generate missing algorithms, solutions, variables, outputs, or observations using general knowledge.
3. TITLE-ONLY IS INCOMPLETE:
   Mentioning a question title or experiment name in a list or heading without explaining it is NOT an explanation.
   - If a question is mentioned ONLY by title/name without substantive explanation:
     classify match_status = "FOUND_BUT_INCOMPLETE"
     classify all requirements as status = "MISSING" with evidence = null.
   - Only classify match_status = "FOUND_AND_COVERED" if the student provides actual explanations for the question's requirements.
   - If no reliable evidence exists for the question, classify match_status = "NOT_FOUND" with all requirements MISSING.
4. STRICT 3-WAY SEPARATION (EVIDENCE vs EVALUATION vs REASONING):
   - "evidence": MUST be verbatim excerpts/quotes copied directly from the student OCR text.
     Can be a single verbatim string, or a JSON list of strings if multiple distinct fragments support the requirement (e.g. ["modulus operator %", "n % 2 == 0"]).
     NEVER generate a new sentence explaining what the student meant!
     If status is MISSING, evidence MUST be null.
   - "evaluation": Objective statement explaining whether the evidence satisfies the requirement.
   - "reasoning": Why you selected PRESENT, PARTIAL, or MISSING.
   - "evidence" MUST NEVER contain information that only exists in evaluation or reasoning!
5. NO FABRICATED SCORING: Do NOT generate numerical scores (e.g. 9.5/10), letter grades (A, B, C), Pass/Fail, or academic integrity claims.
6. GENERIC OBSERVATIONS: Statements like "All programs compiled and executed successfully" or "I got correct output" are generic and do NOT describe program-specific behavior. Mark them as PARTIAL or MISSING.
7. CONFIDENCE: Set confidence to "HIGH" if evidence is clear, "MEDIUM" if minor OCR distortion, or "LOW" if ambiguous.
8. PROHIBITED CONTENT: NEVER output Python source code, functions (e.g. def add, def subtract), test scripts, test case lists (e.g. test_cases = [...]), or evaluator implementation code.
9. ONLY VALID JSON: Your response must be 100% valid JSON matching the REQUIRED OUTPUT SCHEMA, with no text before or after the JSON.

REQUIRED OUTPUT SCHEMA (JSON ONLY):
{{
  "objective": {{
    "requirement": "{obj_req}",
    "status": "PRESENT|PARTIAL|MISSING",
    "evidence": "Direct excerpt from OCR or [\\"excerpt 1\\", \\"excerpt 2\\"] or null",
    "evaluation": "Brief objective assessment or null",
    "reasoning": "Reason for status or null",
    "confidence": "HIGH|MEDIUM|LOW"
  }},
  "questions": [
    {{
      "question_number": 1,
      "question_text": "Full text of question 1",
      "match_status": "FOUND_AND_COVERED|FOUND_BUT_INCOMPLETE|NOT_FOUND",
      "matched_heading": "Heading or title text from student report or null",
      "confidence": "HIGH|MEDIUM|LOW",
      "requirements": [
        {{
          "requirement": "Requirement name",
          "status": "PRESENT|PARTIAL|MISSING",
          "evidence": "Direct excerpt from OCR or [\\"excerpt 1\\", \\"excerpt 2\\"] or null",
          "evaluation": "Brief assessment of how evidence satisfies requirement",
          "reasoning": "Why classified as PRESENT/PARTIAL/MISSING",
          "confidence": "HIGH|MEDIUM|LOW"
        }}
      ]
    }}
  ]
}}
Output valid JSON only. Do not wrap in commentary outside JSON.
"""

    # Build formatted questions list for user prompt
    q_lines = [f"{q.question_number}. {q.question_text}" for q in questions]
    questions_block = "\n".join(q_lines)

    req_lines = [f"- {r}" for r in per_q_reqs]
    reqs_block = "\n".join(req_lines)

    user_prompt = f"""<ASSIGNED_QUESTIONS>
{questions_block}
</ASSIGNED_QUESTIONS>

<INSTRUCTION_MANUAL_REQUIREMENTS>
Session-level Requirement: {obj_req}
Per-Question Required Sections:
{reqs_block}
</INSTRUCTION_MANUAL_REQUIREMENTS>

<STUDENT_REPORT_OCR>
{report_text.strip()}
</STUDENT_REPORT_OCR>

Evaluate all {len(questions)} assigned questions. For each question, evaluate each of the {len(per_q_reqs)} per-question requirements.
Output valid JSON strictly following the schema.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def extract_sections_from_block_content(
    content: str,
    per_q_req_names: List[str],
    ocr_text: str
) -> List[RequirementEvaluation]:
    """
    Extracts standard required sections directly from block content when LLM output is absent.
    Enforces strict evidence grounding against the raw OCR text.
    """
    reqs: List[RequirementEvaluation] = []
    lines = content.splitlines()

    for r_name in per_q_req_names:
        r_norm = r_name.lower()
        found_ev = None

        patterns = []
        if "problem" in r_norm or "understanding" in r_norm:
            patterns = [r"^(?:problem\s*understanding|problem|aim|objective)\s*[\:\-]\s*(.*)"]
        elif "logic" in r_norm or "approach" in r_norm:
            patterns = [r"^(?:logic\s*(?:\/\s*approach\s*used)?|approach|algorithm)\s*[\:\-]\s*(.*)"]
        elif "variable" in r_norm:
            patterns = [r"^(?:important\s*variables(?:\s*and\s*their\s*purpose)?|variables?)\s*[\:\-]\s*(.*)"]
        elif "observe" in r_norm or "observation" in r_norm:
            patterns = [r"^(?:what\s*i\s*observed|observations?)\s*[\:\-]\s*(.*)"]
        else:
            first_word = r_norm.split()[0] if r_norm.split() else r_norm
            patterns = [rf"^{re.escape(first_word)}[^\:\n]*[\:\-]\s*(.*)"]

        for idx, line in enumerate(lines):
            line_s = line.strip()
            # Strip markdown bolding or headers
            line_clean = re.sub(r"^#{1,4}\s*", "", line_s)
            line_clean = re.sub(r"^\*{1,2}\s*", "", line_clean)
            for pat in patterns:
                m = re.match(pat, line_clean, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if val:
                        found_ev = val
                    elif idx + 1 < len(lines):
                        next_line = lines[idx + 1].strip()
                        if next_line and not any(kw in next_line.lower() for kw in ("logic", "variable", "observ", "problem")):
                            found_ev = next_line
                    break
            if found_ev:
                break

        if found_ev:
            cleaned_ev, is_gr, ratio, is_gen = process_and_ground_evidence(found_ev, ocr_text)
            if is_gr and not is_gen:
                reqs.append(RequirementEvaluation(
                    requirement=r_name,
                    status="PRESENT",
                    score=2,
                    max_score=2,
                    evidence=cleaned_ev,
                    evaluation=f"Direct evidence found in section '{r_name}'.",
                    reasoning="Section present in student report.",
                    confidence="HIGH",
                    grounded=True
                ))
                continue

        reqs.append(RequirementEvaluation(
            requirement=r_name,
            status="MISSING",
            score=0,
            max_score=2,
            evidence=None,
            evaluation="No evidence found in student report.",
            reasoning="Section absent from student report.",
            confidence="HIGH",
            grounded=True
        ))

    return reqs


def calculate_criterion_score(
    requirement_name: str,
    status: str,
    evidence: Optional[Union[str, List[str]]],
    grounded: bool,
    is_generic: bool = False,
    question_status: str = "FOUND_AND_COVERED",
    raw_variables_list: Optional[List[Any]] = None
) -> int:
    """
    Computes deterministic score (0, 1, or 2) for a criterion strictly in Python:
      2 = PRESENT / COMPLETE (with grounded evidence)
      1 = PARTIAL / INCOMPLETE (with grounded evidence)
      0 = MISSING / UNGROUNDED / GENERIC

    Evidence-grounded scoring rules:
    - Every non-zero score MUST have grounded evidence.
    - If question is FOUND_BUT_INCOMPLETE (title-only) or NOT_FOUND, score is 0.
    - If evidence is missing, empty, or ungrounded (grounded=False), score is 0.
    - If status is MISSING, score is 0.

    Criterion-specific rules:
    1. Important Variables and Their Purpose:
       - 0 if no grounded variable or if raw_variables_list == [] or empty.
       - 1 if variable name present without purpose (or purpose is '-' or empty).
       - 2 if variable name AND meaningful purpose are grounded in evidence.
    2. Problem Understanding:
       - 0 if missing / title-only.
       - 1 if brief 1-line description / partial.
       - 2 if grounded evidence covers input + determination + output (or substantive explanation).
    3. Logic / Approach Used:
       - 0 if missing / title-only.
       - 1 if superficial construct naming ("uses if-else", "uses switch", "uses loop") without explaining step-by-step logic.
       - 2 if step-by-step logic explained.
    4. What I Observed:
       - 0 if generic observation statement ("All programs were compiled and executed successfully", "got correct output", etc.).
       - 1 if brief program-specific observation / partial.
       - 2 if detailed program-specific observation.
    """
    if question_status in ("FOUND_BUT_INCOMPLETE", "NOT_FOUND"):
        return 0
    if not evidence or not grounded or status == "MISSING":
        return 0

    ev_str = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
    clean_ev = ev_str.strip()
    if not clean_ev or clean_ev.lower() in ("[]", "none", "null", "-", "n/a", "no evidence found in student report."):
        return 0

    norm_name = (requirement_name or "").lower()

    # 1. Observation Criterion
    if "observ" in norm_name:
        if is_generic or is_generic_observation(clean_ev):
            return 0
        if len(clean_ev) < 5:
            return 0
        if status == "PRESENT":
            return 2
        return 1

    # 2. Variables Criterion
    if "variable" in norm_name:
        if raw_variables_list is not None and len(raw_variables_list) == 0:
            return 0
        if clean_ev.lower() in ("[]", "none", "no variables", "no important variables", "n/a", "-"):
            return 0

        # Extract markdown table rows: | var | purpose |
        table_matches = re.findall(r'\|\s*`?([a-zA-Z_]\w*)`?\s*\|\s*([^|\n]+)\s*\|', clean_ev)
        has_var_name = False
        substantive_purpose = False

        if table_matches:
            for v_name, p_text in table_matches:
                v_clean = v_name.strip().lower()
                p_clean = p_text.strip().lower()
                if v_clean not in ("variable", "name", ":---", "---"):
                    has_var_name = True
                    if len(p_clean) > 2 and p_clean not in ("-", "purpose", "n/a", "none", ":---", "---"):
                        substantive_purpose = True
        # Strip generic labels like "Variables used:", "Important variables:"
        clean_var_text = re.sub(r'^(?:important\s+)?variables?(?:\s+used)?\s*:\s*', '', clean_ev, flags=re.IGNORECASE).strip()

        # If it's purely a list of variable names (e.g. "num, temp" or "a, b, c")
        if re.match(r'^`?[a-zA-Z_]\w*`?(?:\s*,\s*`?[a-zA-Z_]\w*`?)*$', clean_var_text):
            has_var_name = bool(re.search(r'\b[a-zA-Z_]\w*\b', clean_var_text))
            substantive_purpose = False
        else:
            colon_m = re.findall(r'\b([a-zA-Z_]\w*)\s*:\s*([^,\n]+)', clean_var_text)
            if colon_m:
                for v_name, p_text in colon_m:
                    v_clean = v_name.strip().lower()
                    p_clean = p_text.strip().lower()
                    if v_clean not in ("variable", "variables", "note", "used"):
                        has_var_name = True
                        if len(p_clean) > 2 and p_clean not in ("-", "n/a", "none"):
                            substantive_purpose = True
            else:
                has_var_name = bool(re.search(r'\b[a-zA-Z_]\w*\b', clean_var_text))
                has_purpose_keywords = any(kw in clean_var_text.lower() for kw in ["store", "hold", "count", "represent", "track", "control", "input", "output", "temp", "flag", "result", "used to", "used for", "keep", "for the", "value"])
                if has_purpose_keywords and len(clean_var_text) > 15:
                    substantive_purpose = True

        if not has_var_name:
            return 0
        if substantive_purpose and status == "PRESENT":
            return 2
        elif substantive_purpose or has_var_name:
            return 1
        return 0

    # 3. Problem Understanding Criterion
    if "problem" in norm_name or "understanding" in norm_name:
        if len(clean_ev) < 5:
            return 0
        if status == "PRESENT":
            return 2
        return 1

    # 4. Logic / Approach Criterion
    if "logic" in norm_name or "approach" in norm_name:
        if len(clean_ev) < 5:
            return 0
        superficial_patterns = [
            r"^(?:it\s+)?uses?\s+(?:if\s*[-–]?\s*else|switch(?:\s+statement)?|loops?|ternary(?:\s+operator)?|for\s+loop|while\s+loop)\.?$",
            r"^(?:using|by\s+using)\s+(?:if\s*[-–]?\s*else|switch(?:\s+statement)?|loops?|ternary(?:\s+operator)?|for\s+loop|while\s+loop)\.?$",
            r"^(?:if\s*[-–]?\s*else|switch(?:\s+statement|\s+case)?|ternary(?:\s+operator)?)\s+(?:is\s+used|construct)\.?$",
        ]
        is_superficial = any(re.search(p, clean_ev, re.IGNORECASE) for p in superficial_patterns)
        if is_superficial:
            return 1
        if status == "PRESENT":
            return 2
        return 1

    # Default fallback for any other criterion
    if status == "PRESENT":
        return 2
    elif status == "PARTIAL":
        return 1
    return 0


def calculate_objective_score(
    status: str,
    evidence: Optional[Union[str, List[str]]],
    grounded: bool,
    is_generic: bool = False
) -> int:
    """
    Evaluates objective once for the complete laboratory (max 2 marks).
      2 = PRESENT / COMPLETE with grounded student evidence representing the lab
      1 = PARTIAL (incomplete or mostly generic, but with grounded evidence)
      0 = MISSING / no grounded evidence / generic observation
    """
    if not evidence or not grounded or status == "MISSING":
        return 0
    ev_str = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
    clean_ev = ev_str.strip()
    if not clean_ev or clean_ev.lower() in ("[]", "none", "null", "-", "n/a", "no evidence found in student report."):
        return 0
    if is_generic or is_generic_observation(clean_ev):
        return 0

    if status == "PRESENT":
        return 2
    elif status == "PARTIAL":
        return 1
    return 0


def generate_overall_assessment(
    total_obtained: float,
    total_max: float,
    final_score: float,
    questions: List[QuestionEvaluation],
    objective: Optional[ObjectiveEvaluation]
) -> str:
    """
    Multi-factor assessment based on:
    score + evidence quality + missing criteria + generic observations.
    """
    total_q = len(questions)
    if total_q == 0 or total_max == 0:
        return "Insufficient evidence to evaluate the report reliably."

    covered_count = sum(1 for q in questions if q.match_status == "FOUND_AND_COVERED")
    incomplete_count = sum(1 for q in questions if q.match_status == "FOUND_BUT_INCOMPLETE")
    not_found_count = sum(1 for q in questions if q.match_status == "NOT_FOUND")
    score_pct = (total_obtained / total_max) * 100.0 if total_max > 0 else 0.0

    # 1. Predominantly title-only listings
    if incomplete_count >= max(2, int(total_q * 0.6)) or (covered_count == 0 and incomplete_count > 0):
        return "Limited evidence of understanding because most programs contain only titles."

    # 2. Insufficient evidence / 0 score / all not found
    if total_obtained == 0 or score_pct < 15.0 or (covered_count == 0 and not_found_count == total_q):
        return "Insufficient evidence to evaluate the report reliably."

    # 3. High comprehension across all assigned programs
    if score_pct >= 85.0 and covered_count == total_q:
        return "Excellent understanding demonstrated across the laboratory."

    # 4. Good understanding with some incomplete explanations
    if score_pct >= 60.0:
        return "Good understanding with some incomplete explanations."

    # 5. Basic understanding with missing sections
    return "Basic understanding demonstrated, but several required sections are missing."


# ============================================================
# 4. PYTHON POST-VALIDATION & DETERMINISTIC METRICS
# ============================================================

def validate_and_ground_evaluation(
    raw_data: Union[Dict[str, Any], str],
    ocr_text: str,
    questions: List[AssignedQuestion],
    manual_requirements: Dict[str, Any]
) -> DynamicEvaluationResult:
    """
    Validates Qwen's JSON output with Pydantic, validates evidence grounding against OCR text,
    enforces non-hallucination rules, and computes deterministic coverage metrics in Python.
    """
    obj_req_name = manual_requirements.get("objective_requirement", "Objective of the Lab")
    per_q_req_names = manual_requirements.get("per_question_requirements", [])

    # Strict Example Filter Guardrail: ensure examples never become requirements
    per_q_req_names = [
        r for r in per_q_req_names
        if not re.search(r'\b(?:example|examples|sample|e\.?g\.?|illustration)\b', r, re.IGNORECASE)
    ]

    # Step 1: Clean & Parse JSON if string
    if isinstance(raw_data, str):
        cleaned = raw_data.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        try:
            parsed_dict = json.loads(cleaned)
        except Exception:
            parsed_dict = {}
    elif isinstance(raw_data, dict):
        parsed_dict = raw_data
    else:
        parsed_dict = {}

    # Step 2: Validate Objective
    raw_obj = parsed_dict.get("objective") or {}
    obj_eval: Optional[ObjectiveEvaluation] = None
    if isinstance(raw_obj, dict) and raw_obj:
        obj_status = str(raw_obj.get("status", "MISSING")).upper()
        if obj_status not in ("PRESENT", "PARTIAL", "MISSING"):
            obj_status = "MISSING"

        raw_obj_ev = raw_obj.get("evidence")
        cleaned_obj_ev, obj_grounded, obj_ratio, obj_is_gen = process_and_ground_evidence(raw_obj_ev, ocr_text)

        # Objective Grounding Check
        if obj_status in ("PRESENT", "PARTIAL"):
            if not cleaned_obj_ev or not obj_grounded:
                obj_status = "MISSING"
                cleaned_obj_ev = None
                obj_grounded = False
            elif obj_is_gen:
                obj_status = "PARTIAL"

        if obj_status == "MISSING":
            cleaned_obj_ev = None

        obj_score = calculate_objective_score(
            status=obj_status,
            evidence=cleaned_obj_ev,
            grounded=obj_grounded,
            is_generic=obj_is_gen
        )

        obj_eval = ObjectiveEvaluation(
            requirement=str(raw_obj.get("requirement", obj_req_name)),
            status=obj_status,
            score=obj_score,
            max_score=2,
            evidence=cleaned_obj_ev,
            evaluation=raw_obj.get("evaluation"),
            reasoning=raw_obj.get("reasoning"),
            confidence=raw_obj.get("confidence", "HIGH") if obj_status != "MISSING" else "HIGH",
            grounded=obj_grounded
        )
    else:
        obj_eval = ObjectiveEvaluation(
            requirement=obj_req_name,
            status="MISSING",
            score=0,
            max_score=2,
            evidence=None,
            evaluation="No objective statement found in student report.",
            reasoning="Omitted from report.",
            confidence="HIGH",
            grounded=True
        )

    # Step 2b: Segment OCR into localized blocks and run Layered Question Matching
    ocr_blocks = segment_report_blocks(ocr_text)
    matching_map = match_assigned_questions_to_blocks(questions, ocr_blocks, ocr_text)

    # Step 3: Process Questions and ensure all assigned questions 1..N are accounted for
    raw_questions = parsed_dict.get("questions", [])
    raw_q_by_num: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_questions, list):
        for item in raw_questions:
            if isinstance(item, dict):
                q_num = item.get("question_number")
                if q_num is not None:
                    try:
                        raw_q_by_num[int(q_num)] = item
                    except Exception:
                        pass

    validated_questions: List[QuestionEvaluation] = []
    seen_evidence_spans: Dict[str, int] = {}

    for q in questions:
        raw_q = raw_q_by_num.get(q.question_number)
        q_match = matching_map.get(q.question_number, {})

        if not raw_q:
            # Check if layered matching engine found evidence for this question
            if q_match.get("matched") and not q_match.get("ambiguous"):
                matched_heading = q_match.get("matched_heading")
                match_conf = q_match.get("match_confidence", "MEDIUM")
                match_ev = q_match.get("match_evidence")
                block = q_match.get("matched_block")

                if block and block.get("content"):
                    extracted_reqs = extract_sections_from_block_content(
                        block["content"], per_q_req_names, ocr_text
                    )
                    present_cnt = sum(1 for r in extracted_reqs if r.status in ("PRESENT", "PARTIAL") and r.evidence and r.grounded)
                else:
                    extracted_reqs = [
                        RequirementEvaluation(
                            requirement=r_name,
                            status="MISSING",
                            score=0,
                            max_score=2,
                            evidence=None,
                            evaluation="No section explanation provided.",
                            reasoning="Omitted from report.",
                            confidence="HIGH",
                            grounded=True
                        )
                        for r_name in per_q_req_names
                    ]
                    present_cnt = 0

                final_status = "FOUND_AND_COVERED" if present_cnt > 0 else "FOUND_BUT_INCOMPLETE"
                for r_item in extracted_reqs:
                    if r_item.evidence and r_item.grounded:
                        # Check duplicate span against earlier questions
                        ev_str_check = " ".join(r_item.evidence) if isinstance(r_item.evidence, list) else str(r_item.evidence)
                        norm_ev = _normalize_text_for_search(ev_str_check)
                        if len(norm_ev) > 20:
                            for seen_span, seen_q in seen_evidence_spans.items():
                                if seen_q != q.question_number and (seen_span in norm_ev or norm_ev in seen_span or difflib.SequenceMatcher(None, seen_span, norm_ev).ratio() > 0.75):
                                    r_item.status = "MISSING"
                                    r_item.grounded = False
                                    r_item.evidence = None
                                    r_item.evaluation = f"[Flagged: Duplicate evidence copied from Program {seen_q}]"
                                    break
                            if r_item.grounded and r_item.evidence:
                                seen_evidence_spans[norm_ev] = q.question_number

                    if final_status in ("FOUND_BUT_INCOMPLETE", "NOT_FOUND"):
                        r_item.score = 0
                    else:
                        r_item.score = calculate_criterion_score(
                            requirement_name=r_item.requirement,
                            status=r_item.status,
                            evidence=r_item.evidence,
                            grounded=r_item.grounded,
                            is_generic=is_generic_observation(str(r_item.evidence)),
                            question_status=final_status
                        )
                    r_item.max_score = 2

                q_score = sum(r.score for r in extracted_reqs)
                q_max_score = sum(r.max_score for r in extracted_reqs) if extracted_reqs else 8

                validated_questions.append(QuestionEvaluation(
                    question_number=q.question_number,
                    question_text=q.question_text,
                    match_status=final_status,
                    match_confidence=match_conf,
                    matched_heading=matched_heading,
                    match_evidence=match_ev,
                    confidence=match_conf,
                    score=q_score,
                    max_score=q_max_score,
                    requirements=extracted_reqs
                ))
            else:
                default_reqs = [
                    RequirementEvaluation(
                        requirement=r_name,
                        status="MISSING",
                        score=0,
                        max_score=2,
                        evidence=None,
                        evaluation="No evidence found in student report.",
                        reasoning="Omitted from report.",
                        confidence="HIGH",
                        grounded=True
                    )
                    for r_name in per_q_req_names
                ]
                validated_questions.append(QuestionEvaluation(
                    question_number=q.question_number,
                    question_text=q.question_text,
                    match_status="NOT_FOUND",
                    match_confidence="LOW" if q_match.get("ambiguous") else "HIGH",
                    matched_heading=None,
                    match_evidence=None,
                    confidence="LOW" if q_match.get("ambiguous") else "HIGH",
                    score=0,
                    max_score=len(default_reqs) * 2 if default_reqs else 8,
                    requirements=default_reqs
                ))
            continue

        # Parse match_status from LLM
        raw_match = str(raw_q.get("match_status", "NOT_FOUND")).upper()
        if raw_match not in ("FOUND_AND_COVERED", "FOUND_BUT_INCOMPLETE", "NOT_FOUND"):
            raw_match = "NOT_FOUND"

        matched_heading = q_match.get("matched_heading") or raw_q.get("matched_heading")
        if matched_heading and str(matched_heading).lower() in ("null", "none", ""):
            matched_heading = None

        match_conf = q_match.get("match_confidence") or raw_q.get("confidence", "HIGH")
        if q_match.get("ambiguous"):
            match_conf = "LOW"

        match_ev = q_match.get("match_evidence")

        raw_reqs_list = raw_q.get("requirements", [])
        raw_reqs_by_name: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_reqs_list, list):
            for r_item in raw_reqs_list:
                if isinstance(r_item, dict) and "requirement" in r_item:
                    raw_reqs_by_name[str(r_item["requirement"]).strip().lower()] = r_item

        # Validate each required section from the Instruction Manual
        validated_reqs: List[RequirementEvaluation] = []
        present_count = 0

        for r_name in per_q_req_names:
            # Find match in raw_reqs
            matched_raw_req = raw_reqs_by_name.get(r_name.lower())
            if not matched_raw_req:
                # Partial name matching (e.g. "logic" in "logic / approach used")
                for k, v in raw_reqs_by_name.items():
                    if k in r_name.lower() or r_name.lower() in k:
                        matched_raw_req = v
                        break

            if not matched_raw_req:
                validated_reqs.append(RequirementEvaluation(
                    requirement=r_name,
                    status="MISSING",
                    score=0,
                    max_score=2,
                    evidence=None,
                    evaluation="No evidence found in student report.",
                    reasoning="Section absent from student report.",
                    confidence="HIGH",
                    grounded=True
                ))
                continue

            r_status = str(matched_raw_req.get("status", "MISSING")).upper()
            if r_status not in ("PRESENT", "PARTIAL", "MISSING"):
                r_status = "MISSING"

            raw_ev = matched_raw_req.get("evidence")
            cleaned_ev, is_grounded, match_ratio, is_gen = process_and_ground_evidence(raw_ev, ocr_text)
            r_eval = matched_raw_req.get("evaluation")
            r_reason = matched_raw_req.get("reasoning")
            confidence = matched_raw_req.get("confidence", "HIGH")

            # Handle Generic Observations (e.g. "All programs executed successfully")
            if is_gen and r_status in ("PRESENT", "PARTIAL"):
                r_eval = f"[Flagged: Generic observation ('{cleaned_ev}') does not provide program-specific evidence] {r_eval or ''}".strip()
                if any(k in r_name.lower() for k in ["logic", "variable", "approach", "algorithm"]):
                    r_status = "MISSING"
                    cleaned_ev = None
                else:
                    r_status = "PARTIAL"
                    confidence = "LOW"

            # Cross-question duplicate evidence tracking:
            if is_grounded and cleaned_ev and r_status in ("PRESENT", "PARTIAL"):
                ev_str_check = " ".join(cleaned_ev) if isinstance(cleaned_ev, list) else str(cleaned_ev)
                norm_ev = _normalize_text_for_search(ev_str_check)
                if len(norm_ev) > 20:
                    is_dup = False
                    for seen_span, seen_q in seen_evidence_spans.items():
                        if seen_q != q.question_number and (seen_span in norm_ev or norm_ev in seen_span or difflib.SequenceMatcher(None, seen_span, norm_ev).ratio() > 0.75):
                            is_dup = True
                            r_status = "MISSING"
                            is_grounded = False
                            cleaned_ev = None
                            r_eval = f"[Flagged: Duplicate evidence copied from Program {seen_q}] {r_eval or ''}".strip()
                            confidence = "LOW"
                            break
                    if not is_dup:
                        seen_evidence_spans[norm_ev] = q.question_number

            # Enforce rule: if status is PRESENT or PARTIAL, evidence is required and must be grounded
            if r_status in ("PRESENT", "PARTIAL"):
                if not cleaned_ev:
                    r_status = "MISSING"
                elif not is_grounded:
                    # Demote ungrounded / fabricated evidence!
                    r_eval = f"[Flagged: Evidence not found in OCR text] {r_eval or ''}".strip()
                    r_status = "MISSING"
                    cleaned_ev = None
                    validated_reqs.append(RequirementEvaluation(
                        requirement=r_name,
                        status="MISSING",
                        score=0,
                        max_score=2,
                        evidence=None,
                        evaluation=r_eval,
                        reasoning=r_reason,
                        confidence="LOW",
                        grounded=False
                    ))
                    continue

            if r_status == "MISSING":
                cleaned_ev = None

            if r_status in ("PRESENT", "PARTIAL") and cleaned_ev and is_grounded:
                present_count += 1

            validated_reqs.append(RequirementEvaluation(
                requirement=r_name,
                status=r_status,
                score=0,  # Will be calculated deterministically below
                max_score=2,
                evidence=cleaned_ev,
                evaluation=r_eval,
                reasoning=r_reason,
                confidence=confidence if r_status != "MISSING" else "HIGH",
                grounded=is_grounded
            ))

        # Enforce Program Title Rule and Ambiguity Guardrail:
        # A question may ONLY be FOUND_AND_COVERED if required sections have actual supporting evidence
        if present_count == 0:
            if q_match.get("ambiguous"):
                # Level 4 Guardrail: ambiguous match with no section evidence -> do NOT force match
                final_match_status = "NOT_FOUND"
                match_conf = "LOW"
            elif matched_heading or q_match.get("matched") or raw_match in ("FOUND_AND_COVERED", "FOUND_BUT_INCOMPLETE"):
                final_match_status = "FOUND_BUT_INCOMPLETE"
            else:
                final_match_status = "NOT_FOUND"
        else:
            final_match_status = "FOUND_AND_COVERED"

        # Deterministic scoring for requirements
        for r_item in validated_reqs:
            if final_match_status in ("FOUND_BUT_INCOMPLETE", "NOT_FOUND"):
                r_item.score = 0
            else:
                r_item.score = calculate_criterion_score(
                    requirement_name=r_item.requirement,
                    status=r_item.status,
                    evidence=r_item.evidence,
                    grounded=r_item.grounded,
                    is_generic=is_generic_observation(str(r_item.evidence)),
                    question_status=final_match_status
                )
            r_item.max_score = 2

        q_score = sum(r.score for r in validated_reqs)
        q_max_score = sum(r.max_score for r in validated_reqs) if validated_reqs else 8

        # Log per-program debug block
        safe_print(f"\n{'='*50}")
        safe_print(f"QUESTION: Program {q.question_number}: {q.question_text}")
        safe_print(f"MATCH STATUS: {final_match_status}")
        raw_excerpt = (match_ev[0] if isinstance(match_ev, list) and match_ev else (str(match_ev) if match_ev else (matched_heading or "None")))[:200]
        safe_print(f"RAW OCR EVIDENCE: {raw_excerpt}")
        safe_print("GROUNDED EVIDENCE:")
        for r_item in validated_reqs:
            r_label = "understanding" if any(k in r_item.requirement.lower() for k in ("understanding", "problem")) else (
                "logic" if "logic" in r_item.requirement.lower() else (
                    "variables" if "variable" in r_item.requirement.lower() else (
                        "observation" if "observ" in r_item.requirement.lower() else r_item.requirement
                    )
                )
            )
            ev_disp = str(r_item.evidence)[:100] if r_item.evidence else "None"
            safe_print(f"  - {r_label}: {ev_disp}")
        safe_print("CRITERION SCORES:")
        for r_item in validated_reqs:
            r_label = "understanding" if any(k in r_item.requirement.lower() for k in ("understanding", "problem")) else (
                "logic" if "logic" in r_item.requirement.lower() else (
                    "variables" if "variable" in r_item.requirement.lower() else (
                        "observation" if "observ" in r_item.requirement.lower() else r_item.requirement
                    )
                )
            )
            safe_print(f"  - {r_label}: {r_item.score}/{r_item.max_score}")
        safe_print(f"{'='*50}")

        validated_questions.append(QuestionEvaluation(
            question_number=q.question_number,
            question_text=q.question_text,
            match_status=final_match_status,
            match_confidence=match_conf,
            matched_heading=matched_heading,
            match_evidence=match_ev,
            confidence=match_conf,
            score=q_score,
            max_score=q_max_score,
            requirements=validated_reqs
        ))

    # Step 4: Deterministic Summary & Coverage Calculation
    total_assigned = len(questions)
    covered_count = sum(1 for q in validated_questions if q.match_status == "FOUND_AND_COVERED")
    incomplete_count = sum(1 for q in validated_questions if q.match_status == "FOUND_BUT_INCOMPLETE")
    not_found_count = sum(1 for q in validated_questions if q.match_status == "NOT_FOUND")
    coverage_pct = round((covered_count / total_assigned) * 100, 1) if total_assigned > 0 else 0.0

    # Evidence grounding metrics
    all_evidence_count = 0
    grounded_evidence_count = 0

    def tally_evidence(ev_val, is_gr):
        nonlocal all_evidence_count, grounded_evidence_count
        if not ev_val:
            return
        if isinstance(ev_val, list):
            for _ in ev_val:
                all_evidence_count += 1
                if is_gr:
                    grounded_evidence_count += 1
        else:
            all_evidence_count += 1
            if is_gr:
                grounded_evidence_count += 1

    if obj_eval and obj_eval.evidence:
        tally_evidence(obj_eval.evidence, obj_eval.grounded)

    for q in validated_questions:
        for r in q.requirements:
            if r.evidence:
                tally_evidence(r.evidence, r.grounded)

    grounding_rate = (
        round((grounded_evidence_count / all_evidence_count) * 100, 1)
        if all_evidence_count > 0 else 100.0
    )

    # Calculate Total Marks and Dynamic Normalization
    obj_score = obj_eval.score if obj_eval else 0
    obj_max = obj_eval.max_score if obj_eval else 2
    per_q_max = 8
    total_max = obj_max + (total_assigned * per_q_max)
    obtained_marks = obj_score + sum(q.score for q in validated_questions)
    final_score_2dec = round((obtained_marks / total_max) * 10, 2) if total_max > 0 else 0.0
    final_score_1dec = round((obtained_marks / total_max) * 10, 1) if total_max > 0 else 0.0

    overall_assessment = generate_overall_assessment(
        total_obtained=obtained_marks,
        total_max=total_max,
        final_score=final_score_1dec,
        questions=validated_questions,
        objective=obj_eval
    )

    criteria_totals: Dict[str, Dict[str, int]] = {}
    for q in validated_questions:
        for r in q.requirements:
            if r.requirement not in criteria_totals:
                criteria_totals[r.requirement] = {"obtained": 0, "max": 0}
            criteria_totals[r.requirement]["obtained"] += r.score
            criteria_totals[r.requirement]["max"] += r.max_score

    safe_print(f"\n[FINAL EVALUATION METRICS]")
    safe_print(f"Total Assigned Questions: {total_assigned}")
    safe_print(f"Found and Covered: {covered_count}, Incomplete: {incomplete_count}, Not Found: {not_found_count}")
    safe_print(f"Objective Score: {obj_score}/{obj_max}")
    safe_print(f"Total Marks: {obtained_marks}/{total_max}")
    safe_print(f"Final Score: {final_score_1dec}/10.0\n")

    summary = {
        "total_assigned": total_assigned,
        "found_and_covered": covered_count,
        "found_but_incomplete": incomplete_count,
        "not_found": not_found_count,
        "coverage_percentage": coverage_pct,
        "total_evidence_snippets": all_evidence_count,
        "grounded_evidence_snippets": grounded_evidence_count,
        "evidence_grounding_rate": grounding_rate,
        "objective_status": obj_eval.status if obj_eval else "NOT_SPECIFIED",
        "objective_score": obj_score,
        "objective_max": obj_max,
        "total_obtained": obtained_marks,
        "total_max": total_max,
        "final_score": final_score_2dec,
        "final_score_out_of_10": final_score_1dec,
        "overall_assessment": overall_assessment,
        "criteria_totals": criteria_totals
    }

    return DynamicEvaluationResult(
        objective=obj_eval,
        questions=validated_questions,
        summary=summary
    )


# ============================================================
# 5. DETERMINISTIC MARKDOWN REPORT RENDERER (NO LLM FORMATTING)
# ============================================================

def render_verification_markdown(eval_result: DynamicEvaluationResult) -> str:
    """
    Renders the DynamicEvaluationResult into a clean, comprehensive, professional Markdown report.
    Zero hallucination in tables, counts, scores, or structure: 100% computed in Python.
    """
    summary = eval_result.summary or {}
    total_q = summary.get("total_assigned", len(eval_result.questions))
    covered = summary.get("found_and_covered", 0)
    incomplete = summary.get("found_but_incomplete", 0)
    not_found = summary.get("not_found", 0)
    cov_pct = summary.get("coverage_percentage", 0.0)
    grounding_rate = summary.get("evidence_grounding_rate", 100.0)

    lines = [
        "# 📊 Laboratory Observation Report Verification Report",
        "",
        "> **Evaluation Mode:** Evidence-grounded with post-generation validation and hallucination resistance.",
        "",
        "## 📈 Overall Coverage & Evidence Summary",
        f"- **Total Assigned Questions:** {total_q}",
        f"- **Fully Addressed & Explained:** {covered} / {total_q} ({cov_pct}%)",
        f"- **Mentioned / Incomplete (Title Only):** {incomplete} / {total_q}",
        f"- **Not Found in Report:** {not_found} / {total_q}",
        f"- **OCR Evidence Grounding Rate:** {grounding_rate}%",
        ""
    ]

    # Objective Summary
    if eval_result.objective:
        obj = eval_result.objective
        obj_icon = "✓" if obj.status == "PRESENT" else ("⚠️" if obj.status == "PARTIAL" else "❌")
        lines.append(f"- **{obj.requirement}:** {obj_icon} {obj.status}")
        lines.append("")

    # TABLE 1: Assigned Questions Coverage Analysis
    lines.extend([
        "## 📋 Assigned Questions Coverage Analysis",
        "| # | Assigned Question / Program | Found in Report? | Report Section Heading | Coverage Status | Confidence |",
        "| :--- | :--- | :---: | :--- | :---: | :---: |"
    ])

    for q in eval_result.questions:
        if q.match_status == "FOUND_AND_COVERED":
            found_str = "Yes"
            status_str = "Fully Addressed"
        elif q.match_status == "FOUND_BUT_INCOMPLETE":
            found_str = "Partial"
            status_str = "Mentioned / Incomplete"
        else:
            found_str = "No"
            status_str = "Not Found"

        heading_str = q.matched_heading if q.matched_heading else "—"
        # Escape markdown table pipes
        safe_q_text = q.question_text.replace("|", "\\|")
        safe_heading = heading_str.replace("|", "\\|")

        lines.append(
            f"| {q.question_number} | {safe_q_text} | {found_str} | {safe_heading} | {status_str} | {q.confidence} |"
        )
    lines.append("")

    # TABLE 2: Requirements Compliance Matrix
    # Get distinct requirement names
    req_names: List[str] = []
    for q in eval_result.questions:
        for r in q.requirements:
            if r.requirement not in req_names:
                req_names.append(r.requirement)

    if req_names:
        header_row = "| # | Program / Question | " + " | ".join(req_names) + " |"
        sep_row = "| :--- | :--- | " + " | ".join([":---:"] * len(req_names)) + " |"
        lines.extend([
            "## 📋 Requirements Compliance Matrix",
            header_row,
            sep_row
        ])

        for q in eval_result.questions:
            row_cols = [str(q.question_number), f"Q{q.question_number}"]
            req_map = {r.requirement: r.status for r in q.requirements}
            for r_name in req_names:
                st = req_map.get(r_name, "MISSING")
                if st == "PRESENT":
                    badge = "✓ Present"
                elif st == "PARTIAL":
                    badge = "⚠️ Partial"
                else:
                    badge = "— Missing"
                row_cols.append(badge)
            lines.append("| " + " | ".join(row_cols) + " |")
        lines.append("")

    # SECTION 3: Detailed Evidence-First Analysis
    lines.extend([
        "## 🔍 Detailed Evidence-First Analysis",
        ""
    ])

    if eval_result.objective:
        obj = eval_result.objective
        lines.extend([
            f"### 🎯 {obj.requirement}",
            f"- **Status:** `{obj.status}` (Confidence: {obj.confidence})",
        ])
        if obj.evidence:
            if isinstance(obj.evidence, list):
                lines.append("- **Student Evidence (Verbatim OCR Spans):**")
                for sp in obj.evidence:
                    lines.append(f"  - \"{sp}\"")
            else:
                lines.append(f"- **Student Evidence:** \"{obj.evidence}\"")
        else:
            lines.append("- **Student Evidence:** _No objective statement found in student report._")

        if obj.evaluation:
            lines.append(f"- **Evaluator Assessment:** {obj.evaluation}")
        if obj.reasoning:
            lines.append(f"- **Evaluator Reasoning:** {obj.reasoning}")
        lines.append("")

    for q in eval_result.questions:
        lines.append(f"### Question {q.question_number}: {q.question_text}")
        lines.append("")
        lines.append("| Criterion | Score |")
        lines.append("|---|---:|")
        for r in q.requirements:
            lines.append(f"| {r.requirement} | {r.score} / {r.max_score} |")
        lines.append(f"| **Program Score** | **{q.score} / {q.max_score}** |")
        lines.append("")
        lines.append(f"- **Coverage Status:** `{q.match_status}` (Match Confidence: {q.match_confidence})")
        if q.matched_heading:
            lines.append(f"- **Identified Report Heading:** `{q.matched_heading}`")
        if q.match_evidence:
            assoc_evidence = ", ".join(f'"{e}"' for e in q.match_evidence)
            lines.append(f"- **Association Evidence:** {assoc_evidence}")

        if q.match_status == "NOT_FOUND":
            lines.append("- _No evidence found in student report for this assigned question._")
            lines.append("")
            continue

        if q.match_status == "FOUND_BUT_INCOMPLETE":
            lines.append("- ⚠️ _This question was mentioned only as a title or in an experiment list. Detailed section explanations were not provided._")

        lines.append("")
        for r in q.requirements:
            r_icon = "✓" if r.status == "PRESENT" else ("⚠️" if r.status == "PARTIAL" else "❌")
            lines.append(f"#### {r_icon} {r.requirement} ({r.score} / {r.max_score} marks)")
            lines.append(f"- **Status:** `{r.status}`")
            if r.evidence:
                if isinstance(r.evidence, list):
                    lines.append("- **Student Evidence (Verbatim OCR Spans):**")
                    for sp in r.evidence:
                        lines.append(f"  - \"{sp}\"")
                else:
                    lines.append(f"- **Student Evidence:** \"{r.evidence}\"")
            else:
                lines.append("- **Student Evidence:** _None provided (missing)_")
            if r.evaluation:
                lines.append(f"- **Evaluation:** {r.evaluation}")
            if r.reasoning:
                lines.append(f"- **Evaluator Reasoning:** {r.reasoning}")
            lines.append("")

    # SECTION 4: Actionable Recommendations
    lines.extend([
        "## 🎯 Actionable Recommendations for the Student"
    ])

    recs = []
    if eval_result.objective and eval_result.objective.status in ("MISSING", "PARTIAL"):
        recs.append("Include a clear, overall **Objective of the Lab** explaining the key programming concepts and skills practiced, rather than listing program titles.")

    if incomplete > 0:
        recs.append(f"Provide complete explanations for the **{incomplete} program(s)** that were merely listed by title but lacked section details.")

    if not_found > 0:
        recs.append(f"Ensure all assigned laboratory questions are attempted and documented. Currently, **{not_found} assigned question(s)** have no evidence in the report.")

    # Check for recurring missing requirements
    missing_by_req: Dict[str, int] = {}
    for q in eval_result.questions:
        for r in q.requirements:
            if r.status == "MISSING":
                missing_by_req[r.requirement] = missing_by_req.get(r.requirement, 0) + 1

    for r_name, m_cnt in missing_by_req.items():
        if m_cnt >= max(2, total_q // 2):
            recs.append(f"Consistently document **{r_name}** for every program. It was missing in {m_cnt} program(s).")

    if not recs:
        recs.append("Excellent work! All assigned programs and instruction manual requirements are thoroughly documented with evidence.")

    for idx, rec in enumerate(recs, start=1):
        lines.append(f"{idx}. {rec}")

    # SECTION 5: Final Score
    total_obtained = summary.get("total_obtained", 0)
    total_max_marks = summary.get("total_max", (2 + total_q * 8))
    final_score_val = summary.get("final_score", 0.0)
    overall_assessment_text = summary.get("overall_assessment", "")
    if not overall_assessment_text:
        overall_assessment_text = generate_overall_assessment(
            total_obtained, total_max_marks, final_score_val, eval_result.questions, eval_result.objective
        )

    criteria_totals_map = summary.get("criteria_totals", {})

    lines.extend([
        "",
        "## 📊 Final Score",
        "",
        "### Overall Evaluation",
        "",
        "| Criterion | Score |",
        "|---|---:|"
    ])

    obj_score_val = summary.get("objective_score", (eval_result.objective.score if eval_result.objective else 0))
    obj_max_val = summary.get("objective_max", (eval_result.objective.max_score if eval_result.objective else 2))
    obj_name = eval_result.objective.requirement if eval_result.objective else "Objective of the Lab"
    lines.append(f"| {obj_name} | {obj_score_val} / {obj_max_val} |")

    for r_name in req_names:
        c_stats = criteria_totals_map.get(r_name)
        if c_stats:
            c_obt = c_stats["obtained"]
            c_max = c_stats["max"]
        else:
            c_obt = sum(r.score for q in eval_result.questions for r in q.requirements if r.requirement == r_name)
            c_max = sum(r.max_score for q in eval_result.questions for r in q.requirements if r.requirement == r_name)
        lines.append(f"| {r_name} | {c_obt} / {c_max} |")

    lines.append(f"| **Total** | **{total_obtained} / {total_max_marks}** |")
    lines.append(f"| **Final Score** | **{final_score_val:.2f} / 10** |")
    lines.extend([
        "",
        "### Overall Assessment",
        "",
        overall_assessment_text
    ])

    return "\n".join(lines).strip()


# ============================================================
# 6. VERIFICATION INVOCATION (STREAMING & SYNC ADAPTERS)
# ============================================================

def evaluate_observation_report(
    report_text: str,
    assigned_questions: Optional[Union[str, List[AssignedQuestion]]] = None,
    instruction_manual: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: int = 180,
    filename: Optional[str] = None
) -> DynamicEvaluationResult:
    """
    Core dynamic evaluation pipeline:
    1. Parses runtime assigned questions and instruction manual.
    2. Strictly sanitizes report_text to guarantee complete source separation.
    3. Constructs strict JSON prompt for Qwen.
    4. Calls Ollama with format="json".
    5. Validates JSON, checks evidence grounding against sanitized OCR text.
    6. Returns fully validated DynamicEvaluationResult with deterministic metrics.
    """
    # Parse questions
    if isinstance(assigned_questions, list):
        questions = assigned_questions
    else:
        questions = parse_assigned_questions(assigned_questions)

    # Fallback to default manual questions if none provided
    if not questions:
        questions = parse_assigned_questions(DEFAULT_MANUAL_QUESTIONS_PRESET)

    # Parse manual requirements
    manual_reqs = parse_instruction_manual(instruction_manual)

    # Enforce strict source separation: sanitize student OCR text
    sanitized_report = sanitize_student_ocr_text(
        raw_text=report_text,
        filename=filename,
        assigned_count=len(questions),
        manual_char_count=len(instruction_manual or ""),
        debug=True
    )

    # Build prompt messages
    messages = build_dynamic_verification_messages(
        report_text=sanitized_report,
        questions=questions,
        manual_requirements=manual_reqs
    )

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_ctx": 16384,
        }
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        res_json = resp.json()
        raw_content = res_json.get("message", {}).get("content", "{}")
    except Exception:
        raw_content = "{}"

    # Validate, ground evidence in Python, and compute deterministic summary
    eval_result = validate_and_ground_evaluation(
        raw_data=raw_content,
        ocr_text=sanitized_report,
        questions=questions,
        manual_requirements=manual_reqs
    )

    return eval_result


def stream_observation_verification(
    report_text: str,
    assigned_questions: Optional[Union[str, List[AssignedQuestion]]] = None,
    instruction_manual: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1
) -> Generator[str, None, None]:
    """
    Streams the verification analysis. Runs dynamic evidence-grounded evaluation
    and yields the deterministic markdown report.
    Yields string chunks to be consumed by Streamlit's st.write_stream().
    """
    try:
        eval_result = evaluate_observation_report(
            report_text=report_text,
            assigned_questions=assigned_questions,
            instruction_manual=instruction_manual,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        md_text = render_verification_markdown(eval_result)
        # Yield in paragraph chunks for clean streaming UX
        for line in md_text.splitlines(keepends=True):
            yield line
    except requests.exceptions.ConnectionError:
        yield f"\n\n❌ **Error**: Cannot connect to Ollama at `{base_url}`. Please ensure Ollama is running (`ollama serve`)."
    except Exception as e:
        yield f"\n\n❌ **Error during verification**: {str(e)}"


def verify_observation_report_sync(
    report_text: str,
    assigned_questions: Optional[Union[str, List[AssignedQuestion]]] = None,
    instruction_manual: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1
) -> str:
    """
    Non-streaming synchronous evaluation helper for CLI tests or batch scripts.
    Calls evaluate_observation_report and returns complete deterministic Markdown report.
    """
    eval_result = evaluate_observation_report(
        report_text=report_text,
        assigned_questions=assigned_questions,
        instruction_manual=instruction_manual,
        base_url=base_url,
        model=model,
        temperature=temperature
    )
    return render_verification_markdown(eval_result)


def check_ollama_status(base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Checks if the Ollama server is running and if the requested model is available."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=4)
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"Ollama returned HTTP status {resp.status_code}",
                "models": [],
                "model_found": False
            }

        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        model_found = any(
            model == m or m.startswith(f"{model}:") or model.startswith(f"{m}:")
            for m in models
        )
        return {
            "ok": True,
            "error": None,
            "models": models,
            "model_found": model_found,
            "active_model": model if model_found else (models[0] if models else None)
        }
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": f"Cannot connect to Ollama at {base_url}. Please ensure Ollama is running.",
            "models": [],
            "model_found": False
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "models": [],
            "model_found": False
        }


def format_extraction_for_evaluation(
    extraction_result: Any,
    fallback_text: str = ""
) -> str:
    """
    Formats an ExtractionResult or dict into structured markdown text
    aligned with the 5 core sections of the Instruction Manual for rubric evaluation.
    """
    if not extraction_result:
        return fallback_text.strip()

    if hasattr(extraction_result, "objective_of_lab"):
        obj = extraction_result.objective_of_lab or ""
        progs = getattr(extraction_result, "programs", {}) or {}
    elif isinstance(extraction_result, dict):
        obj = extraction_result.get("objective_of_lab", "")
        progs = extraction_result.get("programs", {}) or {}
    else:
        return fallback_text.strip()

    formatted_lines = [
        "# Laboratory Observation Report",
        f"## 1. Objective of the Lab\n{obj or 'Not provided'}\n",
    ]

    def _prog_sort_key(item):
        m = re.search(r"\d+", str(item[0]))
        return int(m.group(0)) if m else 999

    for p_key, p_val in sorted(progs.items(), key=_prog_sort_key):
        if hasattr(p_val, "status"):
            st = p_val.status
            prob = p_val.problem_understanding
            logic = p_val.logic_approach
            vars_list = p_val.important_variables or []
            obs = p_val.what_i_observed
        elif isinstance(p_val, dict):
            st = p_val.get("status")
            prob = p_val.get("problem_understanding")
            logic = p_val.get("logic_approach")
            vars_list = p_val.get("important_variables", []) or []
            obs = p_val.get("what_i_observed")
        else:
            continue

        if st == "detected":
            formatted_lines.append(f"## Program {p_key}")
            if prob:
                formatted_lines.append(f"### Problem Understanding\n{prob}\n")
            if logic:
                formatted_lines.append(f"### Logic / Approach Used\n{logic}\n")
            if vars_list:
                formatted_lines.append("### Important Variables and Their Purpose")
                formatted_lines.append("| Variable | Purpose |")
                formatted_lines.append("| :--- | :--- |")
                for v in vars_list:
                    var_name = getattr(v, "variable", None) if hasattr(v, "variable") else (v.get("variable") if isinstance(v, dict) else str(v))
                    var_purpose = getattr(v, "purpose", None) if hasattr(v, "purpose") else (v.get("purpose") if isinstance(v, dict) else "")
                    if var_name:
                        formatted_lines.append(f"| `{var_name}` | {var_purpose or '-'} |")
                formatted_lines.append("")
            if obs:
                formatted_lines.append(f"### What I Observed\n{obs}\n")

    text = "\n".join(formatted_lines).strip()
    if len(text) > 40:
        return text
    return fallback_text.strip()


def parse_evaluation_scores(evaluation_text: Optional[str]) -> Dict[str, Any]:
    """
    Parses key scoring metrics from an evaluation report markdown.
    Extracts total score out of 10, criteria scores, assessment, and question coverage.
    Compatible with both the final scoring layer format and legacy reports.
    """
    if not evaluation_text or not isinstance(evaluation_text, str) or not evaluation_text.strip():
        return {
            "total_score": None,
            "max_score": None,
            "score_display": "—",
            "grade": "—",
            "status": "—",
            "objective": "—",
            "problem_understanding": "—",
            "logic_approach": "—",
            "variables_table": "—",
            "what_i_observed": "—",
            "questions_covered": "—"
        }

    # 1. Final Score / Total Score parsing
    final_score_m = re.search(r'\|\s*\*\*?Final Score\*\*?\s*\|\s*\*\*?([0-9\.]+)\s*/\s*([0-9\.]+)\*\*?', evaluation_text)
    if final_score_m:
        total_score = float(final_score_m.group(1))
        max_score = float(final_score_m.group(2))
        score_display = f"{total_score:.2f} / {max_score:.1f}"
    else:
        total_m = re.search(r'\|\s*\*\*?Total\*\*?\s*\|\s*\*\*?([0-9\.]+)\s*/\s*([0-9\.]+)\*\*?', evaluation_text)
        if total_m:
            total_score = float(total_m.group(1))
            max_score = float(total_m.group(2))
            score_display = f"{total_score:.1f} / {max_score:.1f}"
        else:
            score_m = re.search(r'\*\*Total Score:\*\*\s*\[?([0-9\.]+)\s*/\s*([0-9\.]+)\]?', evaluation_text)
            total_score = float(score_m.group(1)) if score_m else None
            max_score = float(score_m.group(2)) if score_m else None
            score_display = f"{total_score:.1f} / {max_score:.1f}" if total_score is not None else "—"

    # 2. Criteria Scores from Overall Evaluation table
    def find_criterion_score(pattern: str) -> str:
        m = re.search(rf'\|\s*(?:\d+\.?\s*)?\*?{pattern}\*?[^|\n]*\|\s*([0-9\.]+\s*/\s*[0-9\.]+)', evaluation_text, re.IGNORECASE)
        return m.group(1).strip() if m else "—"

    obj = find_criterion_score(r'Objective(?: of the Lab)?')
    prob = find_criterion_score(r'Problem Understanding')
    logic = find_criterion_score(r'Logic\s*(?:/\s*Approach Used)?')
    vars_tbl = find_criterion_score(r'Important Variables(?: and Their Purpose)?')
    obs = find_criterion_score(r'What I Observed')

    # 3. Overall Assessment / Status
    assess_m = re.search(r'###\s*Overall Assessment\s*\n+([^\n#]+)', evaluation_text)
    if assess_m:
        status_val = assess_m.group(1).strip()
    else:
        status_m = re.search(r'\*\*Status:\*\*\s*\[?([A-Za-z\s]+)\]?', evaluation_text)
        status_val = status_m.group(1).strip() if status_m else "—"

    # 4. Grade derivation or extraction
    grade_m = re.search(r'\*\*Grade:\*\*\s*\[?([A-Za-z+-]+)\]?', evaluation_text)
    if grade_m:
        grade = grade_m.group(1).strip().upper()
    elif total_score is not None and max_score is not None and max_score > 0:
        pct = (total_score / max_score) * 100
        if pct >= 85:
            grade = "A"
        elif pct >= 70:
            grade = "B"
        elif pct >= 55:
            grade = "C"
        elif pct >= 40:
            grade = "D"
        else:
            grade = "F"
    else:
        grade = "—"

    # 5. Question Coverage
    cov_dyn_m = re.search(r'\*\*Fully Addressed & Explained:\*\*\s*(\d+\s*/\s*\d+)', evaluation_text)
    if cov_dyn_m:
        q_cov = cov_dyn_m.group(1).strip()
    else:
        cov_table_m = re.search(r'##[^\n]*Assigned Questions Coverage Analysis.*?(?=##|\Z)', evaluation_text, re.DOTALL | re.IGNORECASE)
        if cov_table_m:
            cov_section = cov_table_m.group(0)
            table_rows = [l.strip() for l in cov_section.splitlines() if l.strip().startswith('|') and not re.match(r'\|\s*:?-+', l.strip())]
            data_rows = [r for r in table_rows[1:] if not any(kw in r.lower() for kw in ["assigned question", "coverage status"])]
            if data_rows:
                yes_count = sum(1 for r in data_rows if re.search(r'\|\s*(?:Yes|Pass|Fully Addressed)\s*\|', r, re.IGNORECASE))
                q_cov = f"{yes_count} / {len(data_rows)}"
            else:
                q_cov = "—"
        else:
            q_cov = "—"

    return {
        "total_score": total_score,
        "max_score": max_score,
        "score_display": score_display,
        "grade": grade,
        "status": status_val,
        "objective": obj,
        "problem_understanding": prob,
        "logic_approach": logic,
        "variables_table": vars_tbl,
        "what_i_observed": obs,
        "questions_covered": q_cov
    }
