import os
import json
import re
import time
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, ".")
from google import genai
from google.genai import types

def parse_robust_json(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Sanitize invalid backslash escapes
    sanitized = re.sub(r'\\(?![/"\\bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
    try:
        return json.loads(sanitized, strict=False)
    except Exception:
        pass

    try:
        sanitized2 = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', sanitized)
        return json.loads(sanitized2, strict=False)
    except Exception as e:
        print(f"JSON Parse error: {e}")
        return None

def test_gemini_eval():
    from verifier import (
        find_student_code,
        extract_report_entries_normalized,
        match_report_programs_to_questions,
        render_holistic_evaluation_markdown,
        HolisticEvaluationResult,
        DimensionScore,
        CodeReportConsistencyItem,
        InterestingLogicItem,
        PresentationReadiness,
    )

    sid = "N240035"
    json_path = os.path.join(r"output/sections/week-04/SEC2/students", f"{sid}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        sdata = json.load(f)

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    models = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.5-flash-lite"]
    
    # Simple prompt test with schema
    prompt = """You are an academic evaluator. Evaluate student N240035.
Return pure JSON with keys:
{
  "submission_summary": {"total_code_problems": 12, "total_report_entries": 0, "matched_report_entries": 0, "undocumented_code_problems": 12},
  "dimensions": [
    {"dimension": "D1", "name": "Syntax & Code Validity", "score": 1.5, "max_score": 2.0, "justification": "Good code syntax in C", "evidence": [{"content": "Verified standard headers."}]},
    {"dimension": "D2", "name": "Algorithmic Logic & Functional Correctness", "score": 1.8, "max_score": 2.0, "justification": "Correct logic", "evidence": [{"content": "Recursive and iterative solutions present."}]},
    {"dimension": "D3", "name": "Observation Report Quality & Completeness", "score": 1.0, "max_score": 2.0, "justification": "Missing report entries", "evidence": [{"content": "No report entries documented."}]},
    {"dimension": "D4", "name": "Conceptual Understanding & Consistency", "score": 1.5, "max_score": 2.0, "justification": "Solid grasp", "evidence": [{"content": "Consistent code logic."}]},
    {"dimension": "D5", "name": "Novelty, Innovation & Presentation Readiness", "score": 1.6, "max_score": 2.0, "justification": "Presentation ready", "evidence": [{"content": "Interesting implementations."}]}
  ],
  "code_report_consistency": [],
  "interesting_logic": [
    {"problem": "P4_1", "source_file": "4.10.c", "title": "2D String Sorting", "interesting_logic": "Uses strcmp and nested loops", "why_interesting": "Demonstrates multi-string array management", "presentation_potential": "HIGH"}
  ],
  "presentation_readiness": {
    "interesting_topics": ["2D String Sorting in C", "Recursive Tree Branching"],
    "recommended_explanation_points": ["Explain sorting invariant", "Explain termination condition"],
    "possible_faculty_questions": ["How do pointers work with 2D string arrays?", "What is time complexity of bubble sort?"],
    "strong_areas": ["Modular code structure"],
    "weak_areas": ["Observation report completeness"]
  },
  "strengths": ["Clear modular layout across all files."],
  "improvement_areas": ["Submit handwritten observation report entries."],
  "overall_summary": "Solid programming effort across 12 C files with great potential for presentation."
}
"""
    parsed = None
    for m in models:
        try:
            print(f"Trying Gemini model {m}...")
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
                config=config
            )
            parsed = parse_robust_json(resp.text)
            if parsed and "dimensions" in parsed:
                print(f"SUCCESS with {m}!")
                break
        except Exception as e:
            print(f"Model {m} failed: {e}")
            time.sleep(2)

    print("Parsed JSON dimensions count:", len(parsed.get("dimensions", [])))
    print("Interesting logic count:", len(parsed.get("interesting_logic", [])))
    print("Faculty questions count:", len(parsed.get("presentation_readiness", {}).get("possible_faculty_questions", [])))

test_gemini_eval()
