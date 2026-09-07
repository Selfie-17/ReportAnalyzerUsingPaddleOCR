"""
run_evaluation_on_extracted.py
Runs the dynamic evaluation pipeline on the extracted student text
from extracted_13_questions_report.json for all 13 questions.
"""

import json
import time
from verifier import (
    WEEK1_13_PROGRAMS_PRESET,
    OFFICIAL_INSTRUCTION_MANUAL,
    parse_assigned_questions,
    format_extraction_for_evaluation,
    evaluate_observation_report,
    render_verification_markdown,
)

def main():
    print("=" * 75)
    print("STEP 1: LOAD EXTRACTED 13-QUESTION STUDENT TEXT")
    print("=" * 75)
    with open("extracted_13_questions_report.json", "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    # Format into canonical student observation report markdown text
    formatted_student_text = format_extraction_for_evaluation(extracted_data)
    print(f"Formatted student report text length: {len(formatted_student_text)} chars.")
    print("First 350 chars:\n" + formatted_student_text[:350])
    print("...\nLast 350 chars:\n" + formatted_student_text[-350:])
    
    # Save the formatted student text as a reference artifact
    with open("student_extracted_text_for_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(formatted_student_text)
    print("\nSaved formatted text to 'student_extracted_text_for_evaluation.txt'.\n")

    print("=" * 75)
    print("STEP 2: RUN QWEN DYNAMIC EVALUATION ON EXTRACTED TEXT")
    print("=" * 75)
    assigned_q = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    print(f"Evaluating against {len(assigned_q)} assigned questions...")
    
    t0 = time.time()
    eval_result = evaluate_observation_report(
        report_text=formatted_student_text,
        assigned_questions=assigned_q,
        instruction_manual=OFFICIAL_INSTRUCTION_MANUAL,
        filename="extracted_13_questions_report.json"
    )
    t_eval = time.time() - t0
    print(f"Evaluation completed in {t_eval:.2f}s.\n")

    print("=" * 75)
    print("STEP 3: SAVE EVALUATION ARTIFACTS")
    print("=" * 75)
    # Render and save Markdown report
    markdown_report = render_verification_markdown(eval_result)
    with open("extracted_text_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print("Saved Markdown report to 'extracted_text_evaluation_report.md'.")

    with open("extracted_text_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_result.model_dump(), f, indent=2, ensure_ascii=False)
    print("Saved JSON result to 'extracted_text_evaluation_result.json'.\n")

    print("=" * 75)
    print("STEP 4: DETAILED MARKS & SCORING BREAKDOWN")
    print("=" * 75)
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
        print(f"  Q{q.question_number} ({q.question_text[:40]}...): Score = {q.score}/{q.max_score} [{q.match_status}]")
        for r in q.requirements:
            flag = ""
            if "[Flagged" in (r.evaluation or ""):
                flag = " [FLAGGED]"
            print(f"    * {r.requirement[:25]}: {r.score}/{r.max_score} ({r.status}) - Grounded={r.grounded}{flag}")

if __name__ == "__main__":
    main()
