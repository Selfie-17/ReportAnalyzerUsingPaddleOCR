"""
verifier.py
Observation Report Verification Engine using Ollama and Qwen 2.5 Coder 3B.
Evaluates student laboratory observation reports against the official Instruction Manual
and user-specified assigned lab questions / problem statements.
"""

import json
import re
import requests
from typing import Generator, Dict, Any, Optional, Tuple


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"

# Official 5-program example assignment from the Instruction Manual
DEFAULT_MANUAL_QUESTIONS_PRESET = """Program 1: Factors of a Number
Program 2: Factorial of a Number
Program 3: Palindrome Number
Program 4: Prime Number
Program 5: Fibonacci Series"""

# Week 1 Lab 13-Program Assignment Preset
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

# Official Instruction Manual for Writing the Observation Report
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

VERIFICATION_SYSTEM_PROMPT = f"""You are an expert C Programming Laboratory Observation Report Evaluator.
Your task is to strictly evaluate, grade, and verify a student's observation report against the official submission guide ("Instruction Manual for Writing the Observation Report") and the assigned lab questions provided by the instructor.

### OFFICIAL SUBMISSION GUIDE & REFERENCE MANUAL:
{OFFICIAL_INSTRUCTION_MANUAL}

---

### EVALUATION RULES & SCORING RUBRIC (Total: 10.0 points):

1. OBJECTIVE OF THE LAB (Max 2.0 points):
   - Written ONCE for the complete lab session.
   - Explains: programming concepts practiced, problem types solved, and skills intended to develop.
   - VIOLATION: Must NOT simply copy the list of program questions.

2. PROBLEM UNDERSTANDING (Max 2.0 points):
   - Provided separately for every program in the session.
   - Explains in student's own words: (a) input given, (b) what needs to be calculated/determined, (c) output produced.
   - VIOLATION: Must NOT contain C code statements.

3. LOGIC / APPROACH USED (Max 2.0 points):
   - Provided separately for every program.
   - Explains the reasoning/idea step-by-step: where loop/repetition is required, condition checked, how values change, how final result is obtained.
   - VIOLATION: Must NOT copy C syntax or full code blocks. Another student should understand the logic without seeing C code.

4. IMPORTANT VARIABLES AND THEIR PURPOSE (Max 2.0 points):
   - Uses a table format (| Variable | Purpose |).
   - Identifies important variables and gives clear, meaningful descriptions of their roles.
   - Does NOT list trivial variables unnecessarily. Uses proper variable names.

5. WHAT I OBSERVED (Max 2.0 points):
   - Contains actual meaningful observations about program behaviour, values changing across iterations, condition handling, or logic.
   - STRICT PENALTY: Heavily penalize generic non-observations like "The program executed successfully" or "I got the correct output".

6. ASSIGNED QUESTIONS ALIGNMENT:
   - Compare the report against the ASSIGNED LAB QUESTIONS / PROBLEMS provided by the instructor.
   - Verify that all assigned problems are addressed in the report.
   - Flag any missing assigned questions, misplaced questions, or unassigned extra programs.

7. CODE-COPYING & INTEGRITY CHECK:
   - Flag any full C code copied into the text. The report is an understanding report, not a code reproduction.

8. 5 CORE QUESTIONS CONSISTENCY CHECK:
   Check whether the overall report answers:
   (1) What was I trying to learn?
   (2) What problem was I solving?
   (3) How did I solve it?
   (4) What role did the important variables play?
   (5) What did I understand or notice while executing the program?

### SCORING SYSTEM:
Each of the 5 sections is scored from 0.0 to 2.0 points:
- 1. Objective: 0.0 to 2.0
- 2. Problem Understanding: 0.0 to 2.0
- 3. Logic / Approach Used: 0.0 to 2.0
- 4. Important Variables Table: 0.0 to 2.0
- 5. What I Observed: 0.0 to 2.0
TOTAL SCORE = Sum of all 5 sections (0.0 to 10.0). Do NOT assign more than 2.0 for any individual section score!

### REQUIRED OUTPUT FORMAT:

Format your entire response in clear, professional Markdown with these exact sections:

# 📊 Observation Report Verification Report

## Overall Evaluation
- **Total Score:** [X.X / 10.0]
- **Grade:** [A (9.0-10.0) / B (8.0-8.9) / C (6.5-7.9) / D (5.0-6.4) / F (<5.0)]
- **Status:** [Approved / Needs Revision / Rejected]

## 📋 Assigned Questions Coverage Analysis
| # | Assigned Question / Program | Found in Report? | Report Section Heading | Coverage Status | Notes |
| :--- | :--- | :---: | :--- | :---: | :--- |
| 1 | [Assigned Question 1] | [Yes / Partial / No] | [e.g. Program 1: ...] | [Fully Addressed / Partially Addressed / Missing / Mismatched] | [Brief comment] |

## 📋 Section-by-Section Score Breakdown
| Section | Score (out of 2.0) | Max | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| 1. Objective of the Lab | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |
| 2. Problem Understanding | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |
| 3. Logic / Approach Used | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |
| 4. Important Variables Table | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |
| 5. What I Observed | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |

## 🔍 Detailed Analysis
### 1. Objective of the Lab
[Detailed evaluation with praise or specific gaps]

### 2. Problem Understanding
[Evaluation per program; check if student's own words and no C code]

### 3. Logic / Approach Used
[Evaluation per program; check conceptual reasoning vs code dumping]

### 4. Important Variables & Table
[Evaluation of table presence and variable descriptions]

### 5. What I Observed
[Critique of observation quality; flag generic statements]

## ⚠️ Academic & Integrity Checks
- **Assigned Questions Alignment:** [All assigned questions addressed / Missing X questions / Discrepancies noted]
- **Code Copying Violation:** [None detected / Detected in Program X (describe)]
- **5 Core Questions Addressed:** [Yes / Partial (missing question #X) / No]

## 💡 Key Strengths
- [Strength 1]
- [Strength 2]

## 🎯 Actionable Recommendations for the Student
1. [Specific improvement 1]
2. [Specific improvement 2]
3. [Specific improvement 3]
"""


def build_verification_messages(
    report_text: str,
    assigned_questions: Optional[str] = None
) -> list:
    """
    Builds the system and user messages for Ollama verification.
    Integrates the official Instruction Manual and any instructor-provided assigned lab questions.
    """
    if assigned_questions and assigned_questions.strip():
        user_prompt = (
            "Please evaluate and verify the following student observation report against the "
            "official Instruction Manual submission guide and the assigned lab questions below:\n\n"
            "### 📝 ASSIGNED LAB QUESTIONS / PROBLEMS (Basis for Student Report):\n"
            f"{assigned_questions.strip()}\n\n"
            "### 📄 EXTRACTED STUDENT OBSERVATION REPORT:\n"
            f"```markdown\n{report_text.strip()}\n```\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT copy or reprint the student observation report.\n"
            "2. You MUST evaluate all 5 core rubric questions (each scored 0.0 to 2.0):\n"
            "   - 1. Objective of the Lab (Max 2.0)\n"
            "   - 2. Problem Understanding (Max 2.0)\n"
            "   - 3. Logic / Approach Used (Max 2.0)\n"
            "   - 4. Important Variables Table (Max 2.0)\n"
            "   - 5. What I Observed (Max 2.0)\n"
            "   Total Score is the sum of these 5 section scores (0.0 to 10.0).\n"
            "3. You MUST include BOTH markdown tables in your output:\n"
            "   - Table 1: '## 📋 Assigned Questions Coverage Analysis' covering every assigned question.\n"
            "   - Table 2: '## 📋 Section-by-Section Score Breakdown' containing all 5 criteria rows with exact scores out of 2.0.\n"
            "4. You MUST include '## 🔍 Detailed Analysis' with subsections for all 5 core questions.\n\n"
            "Output your entire evaluation in this EXACT structure:\n\n"
            "# 📊 Observation Report Verification Report\n\n"
            "## Overall Evaluation\n"
            "- **Total Score:** [X.X / 10.0]\n"
            "- **Grade:** [Grade: A / B / C / D / F]\n"
            "- **Status:** [Approved / Needs Revision / Rejected]\n\n"
            "## 📋 Assigned Questions Coverage Analysis\n"
            "| # | Assigned Question / Program | Found in Report? | Report Section Heading | Coverage Status | Notes |\n"
            "| :--- | :--- | :---: | :--- | :---: | :--- |\n\n"
            "## 📋 Section-by-Section Score Breakdown\n"
            "| Section | Score (out of 2.0) | Max | Status | Notes |\n"
            "| :--- | :---: | :---: | :--- | :---: | :--- |\n"
            "| 1. Objective of the Lab | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 2. Problem Understanding | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 3. Logic / Approach Used | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 4. Important Variables Table | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 5. What I Observed | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n\n"
            "## 🔍 Detailed Analysis\n"
            "### 1. Objective of the Lab\n...\n"
            "### 2. Problem Understanding\n...\n"
            "### 3. Logic / Approach Used\n...\n"
            "### 4. Important Variables Table\n...\n"
            "### 5. What I Observed\n...\n\n"
            "## ⚠️ Academic & Integrity Checks\n"
            "- **Assigned Questions Alignment:** ...\n"
            "- **Code Copying Violation:** ...\n"
            "- **5 Core Questions Addressed:** ...\n\n"
            "## 💡 Key Strengths\n...\n\n"
            "## 🎯 Actionable Recommendations for the Student\n..."
        )
    else:
        user_prompt = (
            "Please evaluate and verify the following student observation report against the "
            "official Instruction Manual submission guide:\n\n"
            "### 📄 EXTRACTED STUDENT OBSERVATION REPORT:\n"
            f"```markdown\n{report_text.strip()}\n```\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT copy or reprint the student observation report.\n"
            "2. You MUST evaluate all 5 core rubric questions (each scored 0.0 to 2.0):\n"
            "   - 1. Objective of the Lab (Max 2.0)\n"
            "   - 2. Problem Understanding (Max 2.0)\n"
            "   - 3. Logic / Approach Used (Max 2.0)\n"
            "   - 4. Important Variables Table (Max 2.0)\n"
            "   - 5. What I Observed (Max 2.0)\n"
            "   Total Score is the sum of these 5 section scores (0.0 to 10.0).\n"
            "3. You MUST include '## 📋 Section-by-Section Score Breakdown' containing all 5 criteria rows with exact scores out of 2.0.\n"
            "4. You MUST include '## 🔍 Detailed Analysis' with subsections for all 5 core questions.\n\n"
            "Output your entire evaluation in this EXACT structure:\n\n"
            "# 📊 Observation Report Verification Report\n\n"
            "## Overall Evaluation\n"
            "- **Total Score:** [X.X / 10.0]\n"
            "- **Grade:** [Grade: A / B / C / D / F]\n"
            "- **Status:** [Approved / Needs Revision / Rejected]\n\n"
            "## 📋 Section-by-Section Score Breakdown\n"
            "| Section | Score (out of 2.0) | Max | Status | Notes |\n"
            "| :--- | :---: | :---: | :--- | :---: | :--- |\n"
            "| 1. Objective of the Lab | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 2. Problem Understanding | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 3. Logic / Approach Used | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 4. Important Variables Table | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n"
            "| 5. What I Observed | X.X | 2.0 | [Pass / Needs Work / Missing] | Brief remark |\n\n"
            "## 🔍 Detailed Analysis\n"
            "### 1. Objective of the Lab\n...\n"
            "### 2. Problem Understanding\n...\n"
            "### 3. Logic / Approach Used\n...\n"
            "### 4. Important Variables Table\n...\n"
            "### 5. What I Observed\n...\n\n"
            "## ⚠️ Academic & Integrity Checks\n"
            "- **Code Copying Violation:** ...\n"
            "- **5 Core Questions Addressed:** ...\n\n"
            "## 💡 Key Strengths\n...\n\n"
            "## 🎯 Actionable Recommendations for the Student\n..."
        )

    return [
        {
            "role": "system",
            "content": VERIFICATION_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]


def check_ollama_status(base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Checks if the Ollama server is running and if the requested model is available.
    """
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

        # Match model name (handle :latest or specific tags like :3b)
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


def stream_observation_verification(
    report_text: str,
    assigned_questions: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Streams the verification analysis token by token from Ollama.
    Compares the student report against the assigned lab questions and the Instruction Manual.
    Yields string chunks to be consumed by Streamlit's st.write_stream().
    """
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    messages = build_verification_messages(report_text, assigned_questions)

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
        }
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            stream=True,
            timeout=(5, 120)
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                message = data.get("message", {})
                chunk = message.get("content", "")
                if chunk:
                    yield chunk
            except json.JSONDecodeError:
                continue

    except requests.exceptions.RequestException as e:
        yield f"\n\n❌ **Error during verification streaming**: {str(e)}"


def verify_observation_report_sync(
    report_text: str,
    assigned_questions: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> str:
    """
    Non-streaming synchronous verification helper for CLI tests or batch scripts.
    """
    chunks = list(stream_observation_verification(
        report_text=report_text,
        assigned_questions=assigned_questions,
        base_url=base_url,
        model=model,
        temperature=temperature
    ))
    return "".join(chunks)


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

    for p_key, p_val in sorted(progs.items()):
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
    Parses key scoring metrics from a Qwen rubric evaluation report markdown.
    Extracts total score, grade, status, 5 section criteria scores, and question coverage.
    """
    if not evaluation_text or not isinstance(evaluation_text, str) or not evaluation_text.strip():
        return {
            "total_score": None,
            "max_score": 10.0,
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

    # Total Score & Max
    score_m = re.search(r'\*\*Total Score:\*\*\s*\[?([0-9\.]+)\s*/\s*([0-9\.]+)\]?', evaluation_text)
    total_score = float(score_m.group(1)) if score_m else None
    max_score = float(score_m.group(2)) if score_m else 10.0
    score_display = f"{total_score:.1f} / {max_score:.1f}" if total_score is not None else "—"

    # Grade
    grade_m = re.search(r'\*\*Grade:\*\*\s*\[?([A-Za-z+-]+)\]?', evaluation_text)
    if not grade_m:
        grade_m = re.search(r'###\s*Final Grade:\s*\[?([A-Za-z+-]+)\]?', evaluation_text, re.IGNORECASE)
    grade = grade_m.group(1).strip().upper() if grade_m else "—"

    # Status / Verdict
    status_m = re.search(r'\*\*Status:\*\*\s*\[?([A-Za-z\s]+)\]?', evaluation_text)
    status_val = status_m.group(1).strip() if status_m else "—"

    # Criteria scores
    def find_criterion(pattern: str) -> str:
        m = re.search(pattern, evaluation_text, re.IGNORECASE)
        return m.group(1).strip() if m else "—"

    obj = find_criterion(r'\|\s*(?:1\.?\s*)?\*?Objective[^\n|]*\|\s*([0-9\.]+)')
    prob = find_criterion(r'\|\s*(?:2\.?\s*)?\*?Problem[^\n|]*\|\s*([0-9\.]+)')
    logic = find_criterion(r'\|\s*(?:3\.?\s*)?\*?Logic[^\n|]*\|\s*([0-9\.]+)')
    vars_tbl = find_criterion(r'\|\s*(?:4\.?\s*)?\*?Important Variables[^\n|]*\|\s*([0-9\.]+)')
    obs = find_criterion(r'\|\s*(?:5\.?\s*)?\*?What I Observed[^\n|]*\|\s*([0-9\.]+)')

    # Questions coverage: count from Assigned Questions Coverage Analysis table
    cov_table_m = re.search(r'##[^\n]*Assigned Questions Coverage Analysis.*?(?=##|\Z)', evaluation_text, re.DOTALL | re.IGNORECASE)
    if cov_table_m:
        cov_section = cov_table_m.group(0)
        table_rows = [line.strip() for line in cov_section.splitlines() if line.strip().startswith('|') and not re.match(r'\|\s*:?-+', line.strip())]
        data_rows = [r for r in table_rows[1:] if not any(kw in r.lower() for kw in ["assigned question", "coverage status"])]
        if data_rows:
            yes_count = sum(1 for r in data_rows if re.search(r'\|\s*(?:Yes|Pass|Fully Addressed)\s*\|', r, re.IGNORECASE))
            q_cov = f"{yes_count} / {len(data_rows)}"
        else:
            yes_matches = len(re.findall(r'\|\s*Yes\s*\|', evaluation_text, re.IGNORECASE))
            no_matches = len(re.findall(r'\|\s*No\s*\|', evaluation_text, re.IGNORECASE))
            total_q = yes_matches + no_matches
            q_cov = f"{yes_matches} / {total_q}" if total_q > 0 else "—"
    else:
        yes_matches = len(re.findall(r'\|\s*Yes\s*\|', evaluation_text, re.IGNORECASE))
        no_matches = len(re.findall(r'\|\s*No\s*\|', evaluation_text, re.IGNORECASE))
        total_q = yes_matches + no_matches
        q_cov = f"{yes_matches} / {total_q}" if total_q > 0 else "—"

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

