"""
verifier.py
Observation Report Verification Engine using Ollama and Qwen 2.5 Coder 3B.
Evaluates student laboratory observation reports against the official Instruction Manual.
"""

import json
import requests
from typing import Generator, Dict, Any


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"

# Official Instruction Manual prompt template
VERIFICATION_SYSTEM_PROMPT = """You are an expert C Programming Laboratory Observation Report Evaluator.
Your task is to strictly evaluate and verify a student's observation report based on the official "Instruction Manual for Writing the Observation Report".

### OFFICIAL RULES & RUBRIC (Total: 10.0 points):

1. OBJECTIVE OF THE LAB (Max 2.0 points):
   - Written ONCE for the complete lab session.
   - Explains: programming concepts practiced, problem types solved, and skills intended to develop.
   - MUST NOT simply copy the list of program questions.

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
   - Does NOT list trivial variables unnecessarily.

5. WHAT I OBSERVED (Max 2.0 points):
   - Contains actual meaningful observations about program behaviour, values changing across iterations, condition handling, or logic.
   - STRICT PENALTY: Heavily penalize generic non-observations like "The program executed successfully" or "I got the correct output".

6. CODE-COPYING & INTEGRITY CHECK:
   - Flag any full C code copied into the text. The report is an understanding report, not a code reproduction.

7. 5 CORE QUESTIONS CONSISTENCY CHECK:
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
TOTAL SCORE = Sum of all 5 sections (0.0 to 10.0). Do NOT write more than 2.0 for any individual section score!

### REQUIRED OUTPUT FORMAT:

Format your entire response in clear, professional Markdown with these exact sections:

# 📊 Observation Report Verification Report

## Overall Evaluation
- **Total Score:** [X.X / 10.0]
- **Grade:** [A (9.0-10.0) / B (8.0-8.9) / C (6.5-7.9) / D (5.0-6.4) / F (<5.0)]
- **Status:** [Approved / Needs Revision / Rejected]

## 📋 Section-by-Section Score Breakdown
| Section | Score (out of 2.0) | Max | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| 1. Objective of the Lab | X.X | 2.0 | [Pass/Needs Work/Missing] | Brief remark |
| 2. Problem Understanding | X.X | 2.0 | [Pass/Needs Work/Missing] | Brief remark |
| 3. Logic / Approach Used | X.X | 2.0 | [Pass/Needs Work/Missing] | Brief remark |
| 4. Important Variables Table | X.X | 2.0 | [Pass/Needs Work/Missing] | Brief remark |
| 5. What I Observed | X.X | 2.0 | [Pass/Needs Work/Missing] | Brief remark |

## 🔍 Detailed Analysis
### 1. Objective of the Lab
[Detailed evaluation with praise or specific gaps]

### 2. Problem Understanding
[Evaluation per program; check if own words and no C code]

### 3. Logic / Approach Used
[Evaluation per program; check conceptual reasoning vs code dumping]

### 4. Important Variables & Table
[Evaluation of table presence and variable descriptions]

### 5. What I Observed
[Critique of observation quality; flag generic statements]

## ⚠️ Academic & Integrity Checks
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
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Streams the verification analysis token by token from Ollama.
    Yields string chunks to be consumed by Streamlit's st.write_stream().
    """
    endpoint = f"{base_url.rstrip('/')}/api/chat"

    messages = [
        {
            "role": "system",
            "content": VERIFICATION_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                "Please verify the following extracted observation report against the "
                "Instruction Manual and output the complete evaluation:\n\n"
                f"```markdown\n{report_text.strip()}\n```"
            )
        }
    ]

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
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> str:
    """
    Non-streaming synchronous verification helper for CLI tests or batch scripts.
    """
    chunks = list(stream_observation_verification(report_text, base_url, model, temperature))
    return "".join(chunks)
