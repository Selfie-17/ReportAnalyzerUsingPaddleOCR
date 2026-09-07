"""
extractor.py
Structured Observation Report Extraction Engine using Ollama and Qwen 2.5 Coder 3B.
Converts raw OCR text into strictly validated canonical Pydantic JSON without grading or hallucination.
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List, Tuple
from schemas import ExtractionResult, ProgramDetails, VariableItem

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"

EXTRACTION_SYSTEM_PROMPT = """You are an accurate, strictly conservative academic report extraction engine.
Your sole job is to parse the student's laboratory observation report OCR text and extract its structured content into pure JSON.

CRITICAL EXTRACTION RULES (STRICT NON-HALLUCINATION POLICY):
1. EXTRACT WHAT IS ACTUALLY THERE: Extract only the content present in the OCR text. Do NOT improve, rewrite, infer, summarize with outside knowledge, or invent academic explanations.
2. MISSING CONTENT MUST BE NULL: If a section (Objective, Problem Understanding, Logic/Approach, What I Observed) is not present or cannot be read, you MUST set its value to null. NEVER invent observations (e.g., do NOT invent "I observed the program executed successfully" if not written).
3. VARIABLES: If no variables table or list is present for a program, return an empty array []. Never infer or invent variable names or purposes.
4. NO GROUPED TEXT DUPLICATION:
   - If the student's report groups programs into levels or lists (e.g., "Level 1: Even/odd, positive/negative...", "Level 2...", "Level 3..."), do NOT duplicate or copy that level summary into multiple programs.
   - Each program must contain only standalone, program-specific explanation written specifically for that program.
   - If a program is merely mentioned in a list or grouped title without its own individual explanation, mark it with "status": "not_detected" and set its fields to null.
5. PROGRAM IDENTIFICATION:
   - Identify programs from headings like: "P1", "Program 1", "Problem 1", "1.", "1)", "Question 1", etc.
   - Do NOT treat "Level 1" or "Level 2" as "Program 1" or "Program 2".
   - Map programs to canonical keys "P1" through "PN".
   - If a program (e.g. P4) is NOT present in the OCR text, mark it with "status": "not_detected" and null fields.
   - If a program IS present with its own section, mark it with "status": "detected".
   - DO NOT fabricate missing programs simply to complete the assigned list.
6. OBJECTIVE: Extract the single overall lab objective into "objective_of_lab". If missing, set to null.

JSON OUTPUT STRUCTURE SCHEMA:
{
  "objective_of_lab": "Overall lab objective text or null",
  "programs": {
    "P1": {
      "status": "detected" or "not_detected",
      "problem_understanding": "extracted text or null",
      "logic_approach": "extracted text or null",
      "important_variables": [
        {"variable": "var_name", "purpose": "description"}
      ],
      "what_i_observed": "extracted text or null"
    },
    "P2": {
      "status": "detected" or "not_detected",
      "problem_understanding": null,
      "logic_approach": null,
      "important_variables": [],
      "what_i_observed": null
    }
  }
}
Output valid JSON only. Do not include markdown commentary or extra keys.
"""


def _clean_json_response(raw_text: str) -> str:
    """Removes markdown code fences and isolates the outer JSON object."""
    cleaned = raw_text.strip()
    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Find first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    return cleaned


def _call_ollama(
    prompt: str,
    system_prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: int = 180
) -> Tuple[bool, str, Optional[str]]:
    """Calls Ollama chat endpoint with JSON format enforcement."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        }
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return False, "", f"Ollama HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return True, content, None
    except requests.exceptions.ConnectionError:
        return False, "", f"Cannot connect to Ollama at {base_url}. Ensure Ollama is running."
    except requests.exceptions.Timeout:
        return False, "", f"Ollama request timed out after {timeout} seconds."
    except Exception as e:
        return False, "", str(e)


def assign_source_pages(
    extraction_result: ExtractionResult,
    page_breakdown: List[Dict[str, Any]]
) -> ExtractionResult:
    """
    Associates source page numbers to each detected program using ground-truth
    OCR page breakdown, ensuring page numbers are never hallucinated by the LLM.
    """
    if not page_breakdown:
        return extraction_result

    for prog_key, prog_detail in extraction_result.programs.items():
        if prog_detail.status != "detected":
            prog_detail.source_pages = []
            continue

        matched_pages = set()
        prog_num = re.sub(r"[^\d]", "", prog_key)
        # Search patterns for program header
        patterns = [
            re.compile(rf"\b{re.escape(prog_key)}\b", re.IGNORECASE),
            re.compile(rf"\bProgram\s*{prog_num}\b", re.IGNORECASE),
            re.compile(rf"\bProblem\s*{prog_num}\b", re.IGNORECASE),
            re.compile(rf"\bQuestion\s*{prog_num}\b", re.IGNORECASE),
            re.compile(rf"(?:^|\n)\s*{prog_num}[\.\)]\s+", re.IGNORECASE),
            re.compile(rf"Level\s*\d+\s*:\s*(?:Compulsory|Medium|Advanced)?\s*(?:Programs?)?\s*{prog_num}[\.\)]", re.IGNORECASE)
        ]

        # Extract snippet phrases from understanding or logic to check multi-page spans
        content_snippets = []
        for text_source in (prog_detail.problem_understanding, prog_detail.logic_approach, prog_detail.what_i_observed):
            if text_source and len(text_source.strip()) > 15:
                # Take words from beginning of text
                words = text_source.strip().split()[:6]
                if len(words) >= 3:
                    content_snippets.append(" ".join(words).lower())

        for page_info in page_breakdown:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")
            if not page_text:
                continue

            # Check header match
            if any(p.search(page_text) for p in patterns):
                matched_pages.add(page_num)

            # Check snippet match for continuation pages
            lower_page = page_text.lower()
            for snippet in content_snippets:
                if snippet in lower_page:
                    matched_pages.add(page_num)

        if matched_pages:
            prog_detail.source_pages = sorted(list(matched_pages))
        elif len(page_breakdown) == 1:
            prog_detail.source_pages = [1]
        else:
            prog_detail.source_pages = []

    return extraction_result


def extract_observation_report(
    report_text: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    page_breakdown: Optional[List[Dict[str, Any]]] = None,
    assigned_questions: Optional[str] = None
) -> ExtractionResult:
    """
    Sends OCR text to Ollama/Qwen, requests strict structured JSON,
    validates with Pydantic, performs controlled retry on failure,
    and maps ground-truth source pages.
    """
    text_clean = (report_text or "").strip()
    if not text_clean:
        return ExtractionResult(
            status="failed",
            errors=["Empty report text provided for extraction"]
        )

    if assigned_questions and assigned_questions.strip():
        user_prompt = (
            "The student was assigned the following lab programs/questions:\n"
            f"```text\n{assigned_questions.strip()}\n```\n\n"
            "Extract the structured observation report from the following OCR text:\n\n"
            f"```markdown\n{text_clean}\n```\n\n"
            "Identify the programs (P1, P2, etc.) matching the student's report. "
            "Remember: output strictly valid JSON matching the schema. Do not invent missing sections or programs."
        )
    else:
        user_prompt = (
            "Extract the structured observation report from the following OCR text:\n\n"
            f"```markdown\n{text_clean}\n```\n\n"
            "Remember: output strictly valid JSON matching the schema. Do not invent missing sections."
        )

    success, raw_content, error = _call_ollama(
        prompt=user_prompt,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        base_url=base_url,
        model=model,
        temperature=temperature
    )

    if not success:
        return ExtractionResult(
            status="failed",
            errors=[f"LLM request failed: {error}"]
        )

    # Attempt 1: Parse and validate JSON
    parsed_dict, parse_err = _parse_and_validate_json(raw_content)

    # If parsing or validation failed, attempt 1 controlled retry
    if parse_err is not None:
        retry_prompt = (
            f"The previous output failed validation with error: {parse_err}\n\n"
            "Please fix the error and output strictly valid JSON adhering to the schema:\n"
            f"{user_prompt}"
        )
        r_success, r_raw_content, r_error = _call_ollama(
            prompt=retry_prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        if r_success:
            parsed_dict, retry_parse_err = _parse_and_validate_json(r_raw_content)
            if retry_parse_err is not None:
                return ExtractionResult(
                    status="failed",
                    errors=[f"Initial validation failed: {parse_err}", f"Retry validation failed: {retry_parse_err}"]
                )
        else:
            return ExtractionResult(
                status="failed",
                errors=[f"Initial validation failed: {parse_err}", f"Retry request failed: {r_error}"]
            )

    expected_count = None
    if assigned_questions and assigned_questions.strip():
        from verifier import parse_assigned_questions
        parsed_q = parse_assigned_questions(assigned_questions)
        if parsed_q:
            expected_count = len(parsed_q)

    # Build Pydantic ExtractionResult
    result = _build_extraction_result(parsed_dict, expected_count=expected_count)

    # Assign source pages using reliable OCR page breakdown
    if page_breakdown:
        result = assign_source_pages(result, page_breakdown)

    return result


def _parse_and_validate_json(raw_content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Cleans text, parses JSON, and verifies basic structure."""
    cleaned = _clean_json_response(raw_content)
    if not cleaned:
        return None, "Empty response from LLM or no JSON braces found"

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)}"

    if not isinstance(data, dict):
        return None, f"Expected JSON object (dict), got {type(data).__name__}"

    return data, None


def _build_extraction_result(data: Dict[str, Any], expected_count: Optional[int] = None) -> ExtractionResult:
    """Normalizes raw dictionary into validated ExtractionResult."""
    objective = data.get("objective_of_lab")
    if objective is not None:
        objective = str(objective).strip()
        if objective.lower() in ("null", "none", ""):
            objective = None

    raw_programs = data.get("programs", {})
    if not isinstance(raw_programs, dict):
        raw_programs = {}

    programs: Dict[str, ProgramDetails] = {}
    detected: List[str] = []
    missing: List[str] = []

    # Collect any programs under aliases (e.g. "Program 1" -> "P1")
    normalized_incoming: Dict[str, Dict[str, Any]] = {}
    max_prog_num = expected_count if expected_count is not None else 10
    for k, v in raw_programs.items():
        if not isinstance(v, dict):
            continue
        m = re.search(r"\d+", str(k))
        if m:
            p_num = int(m.group(0))
            max_prog_num = max(max_prog_num, p_num)
            std_key = f"P{p_num}"
            normalized_incoming[std_key] = v

    # Ensure canonical keys cover all expected programs (e.g. P1..P3, P1..P10, P1..P13)
    canonical_keys = [f"P{i}" for i in range(1, max_prog_num + 1)]

    for key in canonical_keys:
        item = normalized_incoming.get(key)
        if item is None:
            programs[key] = ProgramDetails(status="not_detected")
            missing.append(key)
            continue

        raw_status = str(item.get("status", "")).lower()
        has_content = any([
            item.get("problem_understanding"),
            item.get("logic_approach"),
            item.get("what_i_observed"),
            item.get("important_variables")
        ])

        if raw_status == "detected" or has_content:
            status = "detected"
            detected.append(key)
        else:
            status = "not_detected"
            missing.append(key)

        # Parse variables
        raw_vars = item.get("important_variables", [])
        clean_vars: List[VariableItem] = []
        if isinstance(raw_vars, list):
            for v in raw_vars:
                if isinstance(v, dict) and "variable" in v and "purpose" in v:
                    clean_vars.append(VariableItem(
                        variable=str(v["variable"]).strip(),
                        purpose=str(v["purpose"]).strip()
                    ))

        def _clean_str(val: Any) -> Optional[str]:
            if val is None:
                return None
            s = str(val).strip()
            if s.lower() in ("null", "none", ""):
                return None
            return s

        programs[key] = ProgramDetails(
            status=status,
            source_pages=[],
            problem_understanding=_clean_str(item.get("problem_understanding")),
            logic_approach=_clean_str(item.get("logic_approach")),
            important_variables=clean_vars,
            what_i_observed=_clean_str(item.get("what_i_observed"))
        )

    # Cross-Program Deduplication Guardrail:
    # Detects and neutralizes grouped descriptions (e.g. Level-1, Level-2) copied across multiple programs
    text_usage: Dict[str, List[Tuple[str, str]]] = {}
    for p_key, p_detail in programs.items():
        if p_detail.status != "detected":
            continue
        for field_name in ("problem_understanding", "logic_approach", "what_i_observed"):
            val = getattr(p_detail, field_name)
            if val and len(val.strip()) > 20:
                norm_val = re.sub(r"\s+", " ", val.strip().lower())
                text_usage.setdefault(norm_val, []).append((p_key, field_name))

    for norm_text, occurrences in text_usage.items():
        if len(occurrences) > 1:
            for p_key, field_name in occurrences:
                setattr(programs[p_key], field_name, None)
            for p_key, _ in occurrences:
                p_obj = programs[p_key]
                has_substance = any([
                    p_obj.problem_understanding,
                    p_obj.logic_approach,
                    p_obj.what_i_observed,
                    p_obj.important_variables
                ])
                if not has_substance:
                    p_obj.status = "not_detected"
                    if p_key in detected:
                        detected.remove(p_key)
                    if p_key not in missing:
                        missing.append(p_key)

    detected.sort(key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0)
    missing.sort(key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0)

    overall_status = "success" if detected else ("partial" if objective else "failed")

    return ExtractionResult(
        status=overall_status,
        objective_of_lab=objective,
        programs=programs,
        detected_programs=detected,
        missing_programs=missing,
        errors=[]
    )
