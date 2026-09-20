import json
import re
from typing import Dict, Any, List, Optional

def clean_latex_and_ocr(text: str) -> str:
    t = re.sub(r'\\underline\{\s*\\text\{([^}]+)\}\s*\}', r'\1', text)
    t = re.sub(r'\\underline\{([^}]+)\}', r'\1', text)
    t = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    t = re.sub(r'[\$_\{\}\\]', '', t)
    t = re.sub(r'\bunderline([a-zA-Z]+)', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()

def clean_program_title(title_raw: str) -> str:
    t = re.sub(r'Problem\s+Understanding.*', '', title_raw, flags=re.IGNORECASE)
    t = clean_latex_and_ocr(t)
    t = t.strip(' :.-')
    if re.match(r'^pind\s+', t, re.IGNORECASE):
        t = "Find " + t[5:]
    elif re.match(r'^bind\s+', t, re.IGNORECASE):
        t = "Find " + t[5:]
    return t.strip()


def parse_observation_report_deterministic(
    ocr_text: str,
    page_breakdown: Optional[List[Dict[str, Any]]] = None,
    student_id: str = "",
    week: str = "week-04",
    filename: str = "observation_report.pdf"
) -> Dict[str, Any]:
    if not ocr_text or not ocr_text.strip():
        return {
            "schema_version": "1.0",
            "student_id": student_id,
            "week": week,
            "document": {"filename": filename, "page_count": 0},
            "objective": {"text": "", "confidence": 0.0},
            "entries": [],
            "document_conclusion": "",
            "extraction_warnings": ["OCR text is empty or missing"]
        }

    # Normalize newlines
    normalized = ocr_text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 1: Overall Objective
    # Look for Objective at the beginning before the first Program
    obj_match = re.search(
        r'(?:^|\n)(?:#*\s*)?(?:Objective|Objectives|Aim)\s*[\d:.-]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Program|Experiment|Problem|Title)\s*[:\d\-.]|\Z)',
        normalized,
        re.IGNORECASE | re.DOTALL
    )
    objective_text = ""
    if obj_match:
        obj_candidate = obj_match.group(1).strip()
        # Filter out page comment markers
        obj_candidate = re.sub(r'<!--.*?-->', '', obj_candidate).strip()
        objective_text = obj_candidate

    # Step 2: Overall Conclusion
    # Look for Conclusion at the end
    conc_match = re.search(
        r'(?:^|\n)(?:#*\s*)?(?:Overall\s+Conclusion|Conclusion)\s*[:\d\-.]*\s*\n*(.*?)(?=\Z)',
        normalized,
        re.IGNORECASE | re.DOTALL
    )
    document_conclusion = ""
    if conc_match:
        document_conclusion = conc_match.group(1).strip()
        document_conclusion = re.sub(r'<!--.*?-->', '', document_conclusion).strip()

    # Step 3: Identify program blocks
    # A program starts with Program: / Experiment: / Problem:
    # Pattern to find program headings
    prog_pattern = re.compile(
        r'(?:^|\n)(?:#*\s*)?(?:Program|Experiment|Title)\s*[:\d\-.]*\s*([^\n]+)',
        re.IGNORECASE
    )

    prog_matches = list(prog_pattern.finditer(normalized))
    entries = []

    # Map text positions to pages if page_breakdown is provided
    def get_pages_for_span(start_pos: int, end_pos: int) -> List[int]:
        if not page_breakdown:
            return [1]
        pages = []
        # Find which pages contain the text
        sub = normalized[start_pos:end_pos]
        for p_info in page_breakdown:
            p_num = p_info.get("page", 1)
            p_txt = p_info.get("text", "")
            # Check if any significant slice of sub is in p_txt
            words = [w for w in sub.split() if len(w) > 4][:5]
            if words and any(w in p_txt for w in words):
                pages.append(p_num)
        return pages or [1]

    for idx, match in enumerate(prog_matches):
        r_id = f"R{idx + 1}"
        raw_title_line = match.group(1).strip()
        title = clean_program_title(raw_title_line)

        # Content span between this program heading and the next program heading (or conclusion / EOF)
        start_content = match.end()
        if idx + 1 < len(prog_matches):
            end_content = prog_matches[idx + 1].start()
        else:
            # Until conclusion or end of string
            if conc_match and conc_match.start() > start_content:
                end_content = conc_match.start()
            else:
                end_content = len(normalized)

        block_text = normalized[start_content:end_content]
        # Clean page markers from block text
        clean_block = re.sub(r'<!--.*?-->', '', block_text)
        clean_block = re.sub(r'---\s*', '', clean_block)

        # Check if the title line itself contained "Problem Understanding"
        if re.search(r'Problem\s+Understanding', raw_title_line, re.IGNORECASE):
            clean_block = "Problem Understanding:\n" + clean_block

        # Section aliases
        # 1. Problem Understanding
        pu_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Problem\s+Understanding|Understanding\s+the\s+Problem|Problem)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Logic\s+Used|Logic|Algorithm\s+Used|Algorithm|Important\s+Variables|Variables|What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        problem_understanding = ""
        if pu_match:
            problem_understanding = pu_match.group(1).strip()


        # 2. Logic Used
        logic_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Logic\s+Used|Algorithm\s+Used|Logic|Algorithm)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Important\s+Variables|Import\s+Variables|Variables\s+and\s+their\s+Purpose|Variables|What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        logic_used = logic_match.group(1).strip() if logic_match else ""

        # 3. Important Variables
        vars_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Important\s+Variables\s+and\s+their\s+Purpose[s]?|Import\s+Variables\s+and\s+their\s+Purpose[s]?|Important\s+Variables|Variables\s+and\s+their\s+Purpose[s]?|Variables)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:What\s+[a-zA-Z\s]+observed|What\s+[a-zA-Z\s]+used|Observation|Observations|Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        important_variables = []
        if vars_match:
            vars_raw = vars_match.group(1).strip()
            # Parse variables lines
            var_lines = [vl.strip() for vl in vars_raw.splitlines() if vl.strip() and not vl.lower().startswith("variable")]
            for vl in var_lines:
                vl = clean_latex_and_ocr(vl)
                vm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*|\*?[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:->|:|stores|storage|storo|holds|represents|counts|traverses|selects|compares|is)\s*(.*)', vl, re.IGNORECASE)
                if vm:
                    v_name = vm.group(1).strip()
                    second_part = vm.group(2).strip()
                    action_word = vl[len(v_name):len(vl)-len(second_part)].strip()
                    full_purp = f"{action_word} {second_part}".strip()
                    important_variables.append({
                        "name": v_name,
                        "purpose": full_purp or "Variable used in program"
                    })
                elif len(vl.split()) >= 2:
                    parts = vl.split(None, 1)
                    if len(parts[0]) <= 15:
                        important_variables.append({
                            "name": parts[0],
                            "purpose": parts[1]
                        })

        # 4. What I Observed (aliases: What I observed, What D observed, what I used, what observed, Observation, Observations)
        obs_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:What\s+[a-zA-Z\s]+observed|What\s+I\s+used|What\s+observed|Observations?)\s*[:\d\-.]*\s*\n*(.*?)(?=\n(?:#*\s*)?(?:Conclusion)|\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        observations = obs_match.group(1).strip() if obs_match else ""

        # 5. Entry conclusion if present
        entry_conc_match = re.search(
            r'(?:^|\n)(?:#*\s*)?(?:Program\s+Conclusion|Conclusion)\s*[:\d\-.]*\s*\n*(.*?)(?=\Z)',
            clean_block,
            re.IGNORECASE | re.DOTALL
        )
        entry_conclusion = entry_conc_match.group(1).strip() if (entry_conc_match and idx < len(prog_matches)-1) else ""

        pages = get_pages_for_span(match.start(), end_content)

        sec_conf = {
            "program_title": 0.95 if title else 0.0,
            "problem_understanding": 0.92 if problem_understanding else 0.0,
            "logic_used": 0.90 if logic_used else 0.0,
            "important_variables": 0.88 if important_variables else 0.0,
            "observations": 0.91 if observations else 0.0,
            "conclusion": 0.85 if entry_conclusion else 0.0
        }

        entries.append({
            "report_id": r_id,
            "program_title": title or f"Program {r_id}",
            "problem_understanding": problem_understanding or None,
            "logic_used": logic_used or None,
            "important_variables": important_variables,
            "observations": observations or None,
            "conclusion": entry_conclusion or None,
            "page_numbers": pages,
            "section_confidence": sec_conf,
            "extraction_status": "detected"
        })

    return {
        "schema_version": "1.0",
        "student_id": student_id,
        "week": week,
        "document": {
            "filename": filename,
            "page_count": len(page_breakdown) if page_breakdown else 1
        },
        "objective": {
            "text": objective_text,
            "confidence": 0.95 if objective_text else 0.0
        },
        "entries": entries,
        "document_conclusion": document_conclusion or None,
        "extraction_warnings": []
    }

# Run test on N241003
with open('output/sections/week-04/SEC2/students/N241003.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ocr_txt = data['ocr']['text']
pb = data['ocr']['page_breakdown']
res = parse_observation_report_deterministic(ocr_txt, page_breakdown=pb, student_id="N241003", week="week-04")

print(f"Extracted {len(res['entries'])} entries:")
print("Objective:", res['objective']['text'][:80] + "...")
for e in res['entries']:
    print(f"\n--- [{e['report_id']}] {e['program_title']} ---")
    print("  Problem Understanding:", (e['problem_understanding'] or "")[:60] + "...")
    print("  Logic Used:", (e['logic_used'] or "")[:60] + "...")
    print("  Variables:", len(e['important_variables']), [v['name'] for v in e['important_variables']])
    print("  Observations:", (e['observations'] or "")[:60] + "...")
print("\nConclusion:", (res['document_conclusion'] or "")[:80] + "...")
