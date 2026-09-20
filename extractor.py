"""
extractor.py
Structured Observation Report Extraction Engine using Ollama and Qwen 2.5 Coder 3B.
Converts raw OCR text into strictly validated canonical Pydantic JSON without grading or hallucination.
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List, Tuple
from schemas import (
    ExtractionResult,
    ProgramDetails,
    VariableItem,
    ReportProgramEntry,
    ReportExtractionResult,
    ObservationReport,
    ObservationReportEntry,
    ObservationReportVariable,
    ObservationReportDocument,
    ObservationReportObjective,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"


def clean_latex_and_ocr(text: str) -> str:
    """Strips LaTeX markup, OCR noise, and math markers from extracted text."""
    if not text:
        return ""
    t = re.sub(r'\\underline\{\s*\\text\{([^}]+)\}\s*\}', r'\1', text)
    t = re.sub(r'\\underline\{([^}]+)\}', r'\1', t)
    t = re.sub(r'\\text\{([^}]+)\}', r'\1', t)
    t = re.sub(r'[\$_\{\}\\]', '', t)
    t = re.sub(r'\bunderline([a-zA-Z]+)', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


def clean_program_title(title_raw: str) -> str:
    """Extracts and standardizes the program title from an OCR line."""
    if not title_raw:
        return "Untitled Program"
    t = re.sub(r'Problem\s+Understanding.*', '', title_raw, flags=re.IGNORECASE)
    t = clean_latex_and_ocr(t)
    t = t.strip(' :.-')
    if re.match(r'^pind\s+', t, re.IGNORECASE):
        t = "Find " + t[5:]
    elif re.match(r'^bind\s+', t, re.IGNORECASE):
        t = "Find " + t[5:]
    return t.strip() or "Untitled Program"


def parse_observation_report_deterministic(
    ocr_text: str,
    page_breakdown: Optional[List[Dict[str, Any]]] = None,
    student_id: str = "",
    week: str = "week-04",
    filename: str = "observation_report.pdf"
) -> ObservationReport:
    """
    Bulletproof deterministic structural parser for handwritten observation reports.
    Recognizes heading aliases, noisy handwriting OCR, LaTeX annotations, and multi-page flows.
    Guarantees that documented programs are NEVER lost even when LLM is unavailable or outputs malformed JSON.
    """
    if not ocr_text or not ocr_text.strip():
        return ObservationReport(
            schema_version="1.0",
            student_id=student_id,
            week=week,
            document=ObservationReportDocument(filename=filename, page_count=0),
            objective=ObservationReportObjective(text="", confidence=0.0),
            entries=[],
            document_conclusion=None,
            extraction_status="empty_report",
            extraction_warnings=["OCR text is empty or missing"]
        )

    normalized = ocr_text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 1: Lab Objective
    obj_match = re.search(
        r'(?:^|\n)(?:#*\s*)?(?:Objective|Objectives|Aim)\s*[\d:.-]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Program|Experiment|Problem|Title)\s*[:\d\-.]|\Z)',
        normalized,
        re.IGNORECASE | re.DOTALL
    )
    objective_text = ""
    if obj_match:
        obj_candidate = obj_match.group(1).strip()
        obj_candidate = re.sub(r'<!--.*?-->', '', obj_candidate).strip()
        objective_text = clean_latex_and_ocr(obj_candidate)

    # Step 2: Lab Conclusion
    conc_match = re.search(
        r'(?:^|\n)(?:#*\s*)?(?:Overall\s+Conclusion|Conclusion)\s*[:\d\-.]*\s*\n*(.*?)(?=\Z)',
        normalized,
        re.IGNORECASE | re.DOTALL
    )
    document_conclusion = None
    if conc_match:
        conc_text = conc_match.group(1).strip()
        conc_text = re.sub(r'<!--.*?-->', '', conc_text).strip()
        document_conclusion = clean_latex_and_ocr(conc_text) or None

    # Step 3: Identify program blocks
    prog_pattern = re.compile(
        r'(?:^|\n)(?:#*\s*)?(?:Program|Experiment|Title)\s*[:\d\-.]*\s*([^\n]+)',
        re.IGNORECASE
    )
    prog_matches = list(prog_pattern.finditer(normalized))
    entries: List[ObservationReportEntry] = []

    def get_pages_for_span(start_pos: int, end_pos: int) -> List[int]:
        if not page_breakdown:
            return [1]
        pages = []
        sub = normalized[start_pos:end_pos]
        for p_info in page_breakdown:
            p_num = p_info.get("page", 1)
            p_txt = p_info.get("text", "")
            words = [w for w in sub.split() if len(w) > 4][:5]
            if words and any(w in p_txt for w in words):
                pages.append(p_num)
        return pages or [1]

    for idx, match in enumerate(prog_matches):
        r_id = f"R{idx + 1}"
        raw_title_line = match.group(1).strip()
        title = clean_program_title(raw_title_line)

        start_content = match.end()
        if idx + 1 < len(prog_matches):
            end_content = prog_matches[idx + 1].start()
        else:
            if conc_match and conc_match.start() > start_content:
                end_content = conc_match.start()
            else:
                end_content = len(normalized)

        block_text = normalized[start_content:end_content]
        clean_block = re.sub(r'<!--.*?-->', '', block_text)
        clean_block = re.sub(r'---\s*', '', clean_block)

        # Handle heading concatenated into title line
        if re.search(r'Problem\s+Understanding', raw_title_line, re.IGNORECASE):
            clean_block = "Problem Understanding:\n" + clean_block

        # 1. Problem Understanding
        pu_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Problem\s+Understanding|Understanding\s+the\s+Problem|Problem)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Logic\s+Used|Logic|Algorithm\s+Used|Algorithm|Important\s+Variables|Variables|What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        problem_understanding = clean_latex_and_ocr(pu_match.group(1).strip()) if pu_match else None

        # 2. Logic Used
        logic_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Logic\s+Used|Algorithm\s+Used|Logic|Algorithm)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Important\s+Variables|Import\s+Variables|Variables\s+and\s+their\s+Purpose|Variables|What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        logic_used = clean_latex_and_ocr(logic_match.group(1).strip()) if logic_match else None

        # 3. Important Variables
        vars_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Important\s+Variables\s+and\s+their\s+Purpose[s]?|Import\s+Variables\s+and\s+their\s+Purpose[s]?|Important\s+Variables|Variables\s+and\s+their\s+Purpose[s]?|Variables)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        important_variables: List[ObservationReportVariable] = []
        if vars_match:
            vars_raw = vars_match.group(1).strip()
            var_lines = [vl.strip() for vl in vars_raw.splitlines() if vl.strip() and not vl.lower().startswith("variable")]
            for vl in var_lines:
                vl = clean_latex_and_ocr(vl)
                vm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*|\*?[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:->|:|stores|storage|storo|holds|represents|counts|traverses|selects|compares|is)\s*(.*)', vl, re.IGNORECASE)
                if vm:
                    v_name = vm.group(1).strip()
                    second_part = vm.group(2).strip()
                    action_word = vl[len(v_name):len(vl)-len(second_part)].strip()
                    full_purp = f"{action_word} {second_part}".strip()
                    important_variables.append(ObservationReportVariable(
                        name=v_name,
                        purpose=full_purp or "Variable used in program"
                    ))
                elif len(vl.split()) >= 2:
                    parts = vl.split(None, 1)
                    if len(parts[0]) <= 15:
                        important_variables.append(ObservationReportVariable(
                            name=parts[0],
                            purpose=parts[1]
                        ))

        # 4. What I Observed
        obs_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:What\s+[a-zA-Z\s]+observed|What\s+I\s+used|What\s+observed|Observations?)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        observations = clean_latex_and_ocr(obs_match.group(1).strip()) if obs_match else None

        # 5. Program Entry Conclusion
        entry_conc_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Program\s+Conclusion|Conclusion)\s*[:\d\-.]*\s*\n*(.*?)(?=\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        entry_conclusion = clean_latex_and_ocr(entry_conc_match.group(1).strip()) if (entry_conc_match and idx < len(prog_matches)-1) else None

        pages = get_pages_for_span(match.start(), end_content)
        sec_conf = {
            "program_title": 0.95 if title else 0.0,
            "problem_understanding": 0.92 if problem_understanding else 0.0,
            "logic_used": 0.90 if logic_used else 0.0,
            "important_variables": 0.88 if important_variables else 0.0,
            "observations": 0.91 if observations else 0.0,
            "conclusion": 0.85 if entry_conclusion else 0.0
        }

        # 3-state section status differentiation
        def _get_sec_status(content: Optional[str], header_pat: str) -> str:
            if content and len(content.strip()) > 0:
                return "detected"
            if re.search(header_pat, clean_block, re.IGNORECASE):
                return "ocr_detection_failed"
            return "not_provided"

        sec_status = {
            "program_title": "detected" if title and title != "Untitled Program" else "not_provided",
            "problem_understanding": _get_sec_status(problem_understanding, r'Problem\s+Understanding|Understanding\s+the\s+Problem'),
            "logic_used": _get_sec_status(logic_used, r'Logic\s+Used|Algorithm\s+Used|Logic|Algorithm'),
            "important_variables": "detected" if important_variables else _get_sec_status(None, r'Variables|Important\s+Variables'),
            "observations": _get_sec_status(observations, r'What\s+[a-zA-Z\s]+observed|What\s+I\s+used|Observations?'),
            "conclusion": _get_sec_status(entry_conclusion, r'Program\s+Conclusion|Conclusion') if idx < len(prog_matches)-1 else "not_provided"
        }

        entries.append(ObservationReportEntry(
            report_id=r_id,
            program_title=title or f"Program {r_id}",
            problem_understanding=problem_understanding,
            logic_used=logic_used,
            important_variables=important_variables,
            observations=observations,
            conclusion=entry_conclusion,
            page_numbers=pages,
            section_confidence=sec_conf,
            section_status=sec_status,
            extraction_status="detected"
        ))

    if entries:
        ext_status = "success"
        warnings = []
    elif objective_text:
        ext_status = "partial"
        warnings = ["Objective detected, but no structured program entries could be extracted from OCR text."]
    else:
        ext_status = "failed"
        warnings = ["OCR text contains unparseable or garbled content with 0 recognized programs and no objective."]

    return ObservationReport(
        schema_version="1.0",
        student_id=student_id,
        week=week,
        document=ObservationReportDocument(
            filename=filename,
            page_count=len(page_breakdown) if page_breakdown else 1
        ),
        objective=ObservationReportObjective(
            text=objective_text,
            confidence=0.95 if objective_text else 0.0
        ),
        entries=entries,
        document_conclusion=document_conclusion,
        extraction_status=ext_status,
        extraction_warnings=warnings
    )


EXTRACTION_SYSTEM_PROMPT = """You are an accurate, strictly conservative academic report extraction engine.
Your sole job is to parse the student's laboratory observation report OCR text and extract its structured content into pure JSON.

CRITICAL EXTRACTION RULES (STRICT NON-HALLUCINATION POLICY):
1. EXTRACT WHAT IS ACTUALLY THERE: Extract ONLY the content present in the OCR text. Do NOT improve, rewrite, infer, summarize with outside knowledge, or invent academic explanations.
2. MISSING CONTENT MUST BE NULL: If a section (Objective, Problem Understanding, Logic/Approach, What I Observed, Conclusion) is not present or cannot be read, you MUST set its value to null. NEVER invent observations (e.g. do NOT invent "I observed the program executed successfully" if not written).
3. VARIABLES: If no variables table or list is present for a program, return an empty array []. Never infer or invent variable names or purposes.
4. NO GROUPED TEXT DUPLICATION:
   - If the student's report groups programs into levels or lists (e.g. "Level 1: Even/odd, positive/negative...", "Level 2...", "Easy Level...", "Medium Level..."), do NOT duplicate or copy that level summary into multiple programs.
   - Never treat "Level 1", "Level 2", "Easy Level" as program numbers or program entries.
   - Each program must contain only standalone, program-specific explanation written specifically for that program.
   - If a program is merely mentioned in a list or grouped title without its own individual explanation, do NOT create a program entry for it.
5. REPORT ENTRY IDENTIFICATION (R1, R2, R3...):
   - Designate each program documented in the report as "R1", "R2", "R3"... where R means "Report Entry".
   - The first detected report program is R1, the second is R2, etc. Do NOT assume R1 is official P1!
   - Extract "student_written_program_number": if the student wrote "Program 7: Find Factorial", set "student_written_program_number": "7". If no number was written, set null.
   - Extract "program_title": the actual student-written program title.
   - DO NOT create placeholders for unwritten or missing programs. If a student documented only 4 programs, output ONLY 4 items in detected_programs.
6. REPORT-LEVEL CONTENT:
   - Extract the overall lab objective into "objective_of_lab" (or null if absent).
   - Extract the overall lab conclusion into "conclusion" (or null if absent).
   - Do NOT copy the objective or conclusion into individual program entries.

JSON OUTPUT STRUCTURE SCHEMA:
{
  "objective_of_lab": "Overall lab objective text or null",
  "detected_programs": [
    {
      "report_program_id": "R1",
      "student_written_program_number": null,
      "program_title": "student-written program title",
      "status": "detected",
      "problem_understanding": "extracted student text or null",
      "logic_approach": "extracted student text or null",
      "important_variables": [
        {"variable": "actual variable", "purpose": "actual student-written purpose"}
      ],
      "what_i_observed": "student-written observation or null"
    }
  ],
  "conclusion": "student-written conclusion or null"
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


def _clean_ocr_text_for_llm(raw_text: str) -> str:
    """
    Simplifies verbose HTML tables and tags emitted by PaddleOCR into clean,
    token-efficient Markdown tables for optimal LLM context efficiency.
    """
    if not raw_text:
        return ""
    text = raw_text

    # Replace div wrappers with heading / clean text
    text = re.sub(r'<div[^>]*>(.*?)</div>', r'### \1', text, flags=re.IGNORECASE | re.DOTALL)

    # Convert HTML tables to standard Markdown tables
    def _convert_table(match):
        table_html = match.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.IGNORECASE | re.DOTALL)
        if not rows:
            return ""
        md_lines = []
        for r_idx, row in enumerate(rows):
            cols = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.IGNORECASE | re.DOTALL)
            cols = [re.sub(r'<[^>]+>', '', c).strip() for c in cols]
            if cols:
                md_lines.append("| " + " | ".join(cols) + " |")
                if r_idx == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        return "\n" + "\n".join(md_lines) + "\n"

    text = re.sub(r'<table[^>]*>.*?</table>', _convert_table, text, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining orphan HTML formatting tags
    text = re.sub(r'</?(?:span|p|font|b|i|u|center)[^>]*>', '', text, flags=re.IGNORECASE)
    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _call_ollama(
    prompt: str,
    system_prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: int = 180,
    num_ctx: int = 16384
) -> Tuple[bool, str, Optional[str]]:
    """Calls Ollama chat endpoint with JSON format enforcement and expanded context window."""
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
            "num_ctx": num_ctx,
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
    Associates source page numbers to each detected report program using ground-truth
    OCR page breakdown, ensuring page numbers are never hallucinated by the LLM.
    """
    if not page_breakdown:
        return extraction_result

    # Process ReportProgramEntry items in detected_programs
    entries = extraction_result.get_report_entries()
    for entry in entries:
        matched_pages = set()
        patterns = [
            re.compile(rf"\b{re.escape(entry.report_program_id)}\b", re.IGNORECASE),
        ]
        if entry.program_title:
            patterns.append(re.compile(rf"\b{re.escape(entry.program_title[:30])}\b", re.IGNORECASE))
        if entry.student_written_program_number:
            num = entry.student_written_program_number
            patterns.extend([
                re.compile(rf"\bProgram\s*{num}\b", re.IGNORECASE),
                re.compile(rf"\bProblem\s*{num}\b", re.IGNORECASE),
                re.compile(rf"\bQuestion\s*{num}\b", re.IGNORECASE),
                re.compile(rf"(?:^|\n)\s*{num}[\.\)]\s+", re.IGNORECASE)
            ])

        content_snippets = []
        for text_source in (entry.problem_understanding, entry.logic_approach, entry.what_i_observed):
            if text_source and len(text_source.strip()) > 15:
                words = text_source.strip().split()[:6]
                if len(words) >= 3:
                    content_snippets.append(" ".join(words).lower())

        for page_info in page_breakdown:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")
            if not page_text:
                continue

            if any(p.search(page_text) for p in patterns):
                matched_pages.add(page_num)

            lower_page = page_text.lower()
            for snippet in content_snippets:
                if snippet in lower_page:
                    matched_pages.add(page_num)

        if matched_pages:
            entry.source_pages = sorted(list(matched_pages))
        elif len(page_breakdown) == 1:
            entry.source_pages = [1]
        else:
            entry.source_pages = []

    # Also keep legacy programs dictionary in sync if populated
    for prog_key, prog_detail in extraction_result.programs.items():
        if prog_detail.status != "detected":
            prog_detail.source_pages = []
            continue
        for entry in entries:
            if (entry.report_program_id.replace("R", "P") == prog_key or
                (entry.student_written_program_number and entry.student_written_program_number == re.sub(r"[^\d]", "", prog_key))):
                prog_detail.source_pages = entry.source_pages
                break

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

    # Convert verbose HTML table tags into compact Markdown to conserve LLM tokens
    llm_clean_text = _clean_ocr_text_for_llm(text_clean)

    if assigned_questions and assigned_questions.strip():
        user_prompt = (
            "The student was assigned the following lab programs/questions:\n"
            f"```text\n{assigned_questions.strip()}\n```\n\n"
            "Extract the structured observation report from the following OCR text:\n\n"
            f"```markdown\n{llm_clean_text}\n```\n\n"
            "Identify the programs (P1, P2, etc.) matching the student's report. "
            "Remember: output strictly valid JSON matching the schema. Do not invent missing sections or programs."
        )
    else:
        user_prompt = (
            "Extract the structured observation report from the following OCR text:\n\n"
            f"```markdown\n{llm_clean_text}\n```\n\n"
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
        # Fallback to deterministic structural extraction
        det_obs = parse_observation_report_deterministic(
            ocr_text=report_text,
            page_breakdown=page_breakdown
        )
        if det_obs.entries:
            return det_obs.to_extraction_result()
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
                det_obs = parse_observation_report_deterministic(
                    ocr_text=report_text,
                    page_breakdown=page_breakdown
                )
                if det_obs.entries:
                    return det_obs.to_extraction_result()
                return ExtractionResult(
                    status="failed",
                    errors=[f"Initial validation failed: {parse_err}", f"Retry validation failed: {retry_parse_err}"]
                )
        else:
            det_obs = parse_observation_report_deterministic(
                ocr_text=report_text,
                page_breakdown=page_breakdown
            )
            if det_obs.entries:
                return det_obs.to_extraction_result()
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

    # If LLM produced 0 detected programs but deterministic parsing finds entries in the OCR text,
    # use the deterministic entries so documented programs are never silently lost!
    if not result.detected_programs:
        det_obs = parse_observation_report_deterministic(
            ocr_text=report_text,
            page_breakdown=page_breakdown
        )
        if det_obs.entries:
            return det_obs.to_extraction_result()

    # Assign source pages using reliable OCR page breakdown
    if page_breakdown:
        result = assign_source_pages(result, page_breakdown)

    return result


def extract_observation_report_stage1(
    report_text: str,
    page_breakdown: Optional[List[Dict[str, Any]]] = None,
    student_id: str = "",
    week: str = "week-04",
    filename: str = "observation_report.pdf",
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    assigned_questions: Optional[str] = None
) -> ObservationReport:
    """
    Extracts canonical Stage 1 ObservationReport JSON from handwritten OCR text.
    Combines deterministic structural extraction with LLM extraction, guaranteeing
    that all handwritten report sections (Objective, Program, Problem Understanding,
    Logic Used, Important Variables, What I Observed, Conclusion) are preserved.
    """
    det_obs = parse_observation_report_deterministic(
        ocr_text=report_text,
        page_breakdown=page_breakdown,
        student_id=student_id,
        week=week,
        filename=filename
    )
    # If deterministic parser extracted documented programs, it is authoritative on structure
    if det_obs.entries:
        return det_obs

    # Otherwise try LLM extraction as secondary fallback
    try:
        ext_res = extract_observation_report(
            report_text=report_text,
            base_url=base_url,
            model=model,
            temperature=temperature,
            page_breakdown=page_breakdown,
            assigned_questions=assigned_questions
        )
        return ObservationReport.from_extraction_result(
            ext=ext_res,
            student_id=student_id,
            week=week,
            filename=filename,
            page_count=len(page_breakdown) if page_breakdown else 1
        )
    except Exception as e:
        det_obs.extraction_warnings.append(f"LLM extraction error: {str(e)}")
        return det_obs



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
    """
    Normalizes raw dictionary into validated ExtractionResult.
    Produces detected_programs as a list of ReportProgramEntry (R1, R2, ...)
    representing strictly what the student documented.
    """
    objective = (
        data.get("objective_of_lab")
        or data.get("objective")
        or data.get("lab_objective")
        or data.get("overall_objective")
    )
    if objective is not None:
        objective = str(objective).strip()
        if objective.lower() in ("null", "none", ""):
            objective = None

    conclusion = (
        data.get("conclusion")
        or data.get("overall_conclusion")
        or data.get("lab_conclusion")
    )
    if conclusion is not None:
        conclusion = str(conclusion).strip()
        if conclusion.lower() in ("null", "none", ""):
            conclusion = None

    def _clean_str(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if s.lower() in ("null", "none", ""):
            return None
        return s

    # Collect raw program items from either detected_programs or programs
    raw_programs = data.get("detected_programs")
    if raw_programs is None or (isinstance(raw_programs, list) and all(isinstance(x, str) for x in raw_programs)):
        raw_programs = data.get("programs", [])

    items_to_process: List[Dict[str, Any]] = []
    if isinstance(raw_programs, list):
        for item in raw_programs:
            if isinstance(item, dict):
                items_to_process.append(item)
    elif isinstance(raw_programs, dict):
        for k, v in raw_programs.items():
            if isinstance(v, dict):
                v_copy = dict(v)
                if "report_program_id" not in v_copy:
                    v_copy["report_program_id"] = k.replace("P", "R") if k.startswith("P") else k
                if "program_title" not in v_copy:
                    v_copy["program_title"] = v_copy.get("title") or f"Program {k}"
                items_to_process.append(v_copy)

    detected_entries: List[ReportProgramEntry] = []
    r_counter = 1

    for raw_item in items_to_process:
        raw_status = str(raw_item.get("status", "detected")).lower()
        prob = _clean_str(raw_item.get("problem_understanding") or raw_item.get("problem") or raw_item.get("understanding"))
        logic = _clean_str(raw_item.get("logic_approach") or raw_item.get("logic") or raw_item.get("approach"))
        obs = _clean_str(raw_item.get("what_i_observed") or raw_item.get("observations") or raw_item.get("observation"))
        vars_raw = raw_item.get("important_variables") or raw_item.get("variables") or []

        has_substance = any([prob, logic, obs, vars_raw])
        if raw_status == "not_detected" and not has_substance:
            continue

        r_id = f"R{r_counter}"
        r_counter += 1

        written_num = raw_item.get("student_written_program_number")
        if not written_num:
            title_str = str(raw_item.get("program_title") or raw_item.get("title") or "")
            m = re.search(r"(?:Program|Problem|Question|P)\s*(\d+)", title_str, re.IGNORECASE)
            if m:
                written_num = m.group(1)

        title = raw_item.get("program_title") or raw_item.get("title")
        if not title:
            title = f"Report Entry {r_id}"

        clean_vars: List[VariableItem] = []
        if isinstance(vars_raw, list):
            for v in vars_raw:
                if isinstance(v, dict):
                    v_name = v.get("variable") or v.get("name") or v.get("var")
                    v_purpose = v.get("purpose") or v.get("description") or v.get("role")
                    if v_name is not None and v_purpose is not None:
                        clean_vars.append(VariableItem(
                            variable=str(v_name).strip(),
                            purpose=str(v_purpose).strip()
                        ))

        source_pages = raw_item.get("source_pages", [])
        if not isinstance(source_pages, list):
            source_pages = [source_pages] if source_pages else []

        entry = ReportProgramEntry(
            report_program_id=r_id,
            student_written_program_number=str(written_num) if written_num else None,
            program_title=str(title).strip(),
            status="detected",
            problem_understanding=prob,
            logic_approach=logic,
            important_variables=clean_vars,
            what_i_observed=obs,
            source_pages=source_pages
        )
        detected_entries.append(entry)

    # Cross-Program Deduplication Guardrail:
    # Detects and neutralizes grouped descriptions (e.g. Level-1, Level-2) copied across multiple programs
    text_usage: Dict[str, List[Tuple[int, str]]] = {}
    for idx, entry in enumerate(detected_entries):
        for field_name in ("problem_understanding", "logic_approach", "what_i_observed"):
            val = getattr(entry, field_name)
            if val and len(val.strip()) > 20:
                norm_val = re.sub(r"\s+", " ", val.strip().lower())
                text_usage.setdefault(norm_val, []).append((idx, field_name))

    for norm_text, occurrences in text_usage.items():
        if len(occurrences) > 1:
            for idx, field_name in occurrences:
                setattr(detected_entries[idx], field_name, None)

    # Retain entries that still have substantive content
    valid_entries: List[ReportProgramEntry] = []
    for entry in detected_entries:
        has_substance = any([
            entry.problem_understanding,
            entry.logic_approach,
            entry.what_i_observed,
            entry.important_variables
        ])
        if has_substance or entry.program_title:
            valid_entries.append(entry)

    # Legacy programs dictionary for backward compatibility
    legacy_programs: Dict[str, ProgramDetails] = {}
    missing_list: List[str] = []

    is_legacy_dict = isinstance(data.get("programs"), dict) and "detected_programs" not in data
    if is_legacy_dict:
        raw_dict = data.get("programs", {})
        total_p = expected_count if expected_count else (10 if any(k.startswith("P") for k in raw_dict) else len(raw_dict))
        for i in range(1, total_p + 1):
            pk = f"P{i}"
            v_entry = next((e for e in valid_entries if e.student_written_program_number == str(i) or e.report_program_id == f"R{i}"), None)
            if v_entry:
                legacy_programs[pk] = ProgramDetails(
                    status="detected",
                    source_pages=v_entry.source_pages,
                    problem_understanding=v_entry.problem_understanding,
                    logic_approach=v_entry.logic_approach,
                    important_variables=v_entry.important_variables,
                    what_i_observed=v_entry.what_i_observed
                )
            else:
                legacy_programs[pk] = ProgramDetails(status="not_detected")
                missing_list.append(pk)
    else:
        for i, entry in enumerate(valid_entries):
            p_key = f"P{i+1}"
            legacy_programs[p_key] = ProgramDetails(
                status="detected",
                source_pages=entry.source_pages,
                problem_understanding=entry.problem_understanding,
                logic_approach=entry.logic_approach,
                important_variables=entry.important_variables,
                what_i_observed=entry.what_i_observed
            )

    overall_status = "success" if valid_entries else ("partial" if objective else "failed")
    errors = []
    if overall_status == "failed":
        errors.append("No valid laboratory programs or lab objective could be extracted from the document.")

    return ExtractionResult(
        status=overall_status,
        objective_of_lab=objective,
        conclusion=conclusion,
        detected_programs=valid_entries,
        programs=legacy_programs,
        missing_programs=missing_list,
        errors=errors
    )
