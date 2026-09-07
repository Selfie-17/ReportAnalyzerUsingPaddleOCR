"""
test_13_questions_pipeline.py
End-to-end test script:
1. Generates questions_13.json for the 13 assigned laboratory questions.
2. Extracts the student report text from Test_Observation_Report_13_Programs.pdf.
3. Uses Qwen 2.5 Coder 3B via Ollama to extract structured JSON for all 13 programs -> extracted_13_questions_report.json.
4. Uses Qwen 2.5 Coder 3B via Ollama to evaluate the report against the 13 questions and manual requirements -> evaluation_13_questions_result.json & markdown.
5. Inspects and prints how marks are calculated per criterion, generic observation penalties, and overall marks out of 106 and 10.
"""

import os
import json
import time
import pymupdf as fitz
from verifier import (
    WEEK1_13_PROGRAMS_PRESET,
    OFFICIAL_INSTRUCTION_MANUAL,
    parse_assigned_questions,
    sanitize_student_ocr_text,
    evaluate_observation_report,
    render_verification_markdown,
)
from extractor import extract_observation_report

def main():
    print("=" * 70)
    print("STEP 1: GENERATE JSON FOR THE 13 QUESTIONS")
    print("=" * 70)
    
    assigned_q = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    print(f"Parsed {len(assigned_q)} assigned questions.")
    
    questions_json_data = {
        "total_questions": len(assigned_q),
        "lab_title": "C Programming Week 1 Laboratory",
        "questions": [
            {
                "question_number": q.question_number,
                "title": f"Program {q.question_number}",
                "description": q.question_text
            }
            for q in assigned_q
        ]
    }
    
    with open("questions_13.json", "w", encoding="utf-8") as f:
        json.dump(questions_json_data, f, indent=2, ensure_ascii=False)
    print("Saved assigned questions to 'questions_13.json'.\n")

    print("=" * 70)
    print("STEP 2: EXTRACT RAW OCR TEXT FROM 13-PROGRAM PDF")
    print("=" * 70)
    pdf_path = r"C:\Users\kampa\Downloads\Test_Observation_Report_13_Programs.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    doc = fitz.open(pdf_path)
    page_breakdown = []
    full_text_list = []
    for idx, page in enumerate(doc):
        t = page.get_text()
        full_text_list.append(t)
        page_breakdown.append({"page": idx + 1, "text": t})
    doc.close()
    
    raw_pdf_text = "\n\n".join(full_text_list)
    print(f"Total raw text length: {len(raw_pdf_text)} chars across {len(page_breakdown)} pages.")
    
    # Sanitize to strictly separate student report from evaluation test instructions on Page 5
    sanitized_text = sanitize_student_ocr_text(
        raw_pdf_text,
        filename="Test_Observation_Report_13_Programs.pdf",
        assigned_count=13,
        debug=True
    )
    print(f"Sanitized student report length: {len(sanitized_text)} chars.\n")

    print("=" * 70)
    print("STEP 3: RUN QWEN STRUCTURED EXTRACTION (P1..P13)")
    print("=" * 70)
    t0 = time.time()
    ext_result = extract_observation_report(
        report_text=sanitized_text,
        page_breakdown=page_breakdown,
        assigned_questions=WEEK1_13_PROGRAMS_PRESET
    )
    t_ext = time.time() - t0
    print(f"Extraction completed in {t_ext:.2f}s.")
    print(f"Extraction status: {ext_result.status}")
    print(f"Objective detected: {ext_result.objective_of_lab is not None}")
    print(f"Detected programs ({len(ext_result.detected_programs)}): {ext_result.detected_programs}")
    print(f"Missing programs ({len(ext_result.missing_programs)}): {ext_result.missing_programs}")
    
    with open("extracted_13_questions_report.json", "w", encoding="utf-8") as f:
        json.dump(ext_result.model_dump(), f, indent=2, ensure_ascii=False)
    print("Saved structured extraction to 'extracted_13_questions_report.json'.\n")

    print("=" * 70)
    print("STEP 4: RUN QWEN DYNAMIC EVALUATION & MARK GENERATION")
    print("=" * 70)
    t1 = time.time()
    eval_result = evaluate_observation_report(
        report_text=sanitized_text,
        assigned_questions=assigned_q,
        instruction_manual=OFFICIAL_INSTRUCTION_MANUAL,
        filename="Test_Observation_Report_13_Programs.pdf"
    )
    t_eval = time.time() - t1
    print(f"Evaluation completed in {t_eval:.2f}s.\n")

    # Render markdown report
    markdown_report = render_verification_markdown(eval_result)
    with open("evaluation_13_questions_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print("Saved Markdown report to 'evaluation_13_questions_report.md'.")
    
    with open("evaluation_13_questions_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_result.model_dump(), f, indent=2, ensure_ascii=False)
    print("Saved Evaluation JSON result to 'evaluation_13_questions_result.json'.\n")

    print("=" * 70)
    print("STEP 5: DETAILED MARKS BREAKDOWN")
    print("=" * 70)
    print("SUMMARY METRICS:")
    for k, v in eval_result.summary.items():
        if k != "criteria_totals":
            print(f"  {k}: {v}")
    
    print("\nCRITERIA TOTALS:")
    for crit, stats in eval_result.summary.get("criteria_totals", {}).items():
        print(f"  - {crit}: {stats.get('obtained')} / {stats.get('max')}")

    print("\nOBJECTIVE SCORE:")
    if eval_result.objective:
        print(f"  Status: {eval_result.objective.status}")
        print(f"  Score: {eval_result.objective.score} / {eval_result.objective.max_score}")
        print(f"  Grounded: {eval_result.objective.grounded}")
        print(f"  Evidence: {eval_result.objective.evidence}")

    print("\nPER-QUESTION SCORES (1 to 13):")
    for q in eval_result.questions:
        print(f"  Q{q.question_number} ({q.question_text[:35]}...): Score = {q.score}/{q.max_score} [{q.match_status}]")
        for r in q.requirements:
            flag = ""
            if "[Flagged" in (r.evaluation or ""):
                flag = " [FLAGGED]"
            print(f"    * {r.requirement[:25]}: {r.score}/{r.max_score} ({r.status}) - Grounded={r.grounded}{flag}")

if __name__ == "__main__":
    main()
