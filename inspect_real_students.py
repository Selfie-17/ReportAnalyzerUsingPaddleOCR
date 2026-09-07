import json
import os
from verifier import (
    validate_and_ground_evaluation,
    parse_assigned_questions,
    parse_instruction_manual,
    format_extraction_for_evaluation,
    render_verification_markdown,
    WEEK1_13_PROGRAMS_PRESET,
    OFFICIAL_INSTRUCTION_MANUAL
)

def run_acceptance_inspection():
    q13 = parse_assigned_questions(WEEK1_13_PROGRAMS_PRESET)
    manual_reqs = parse_instruction_manual(OFFICIAL_INSTRUCTION_MANUAL)
    student_ids = ["N240157", "N240046", "N240081"]

    for sid in student_ids:
        path = os.path.join("output", "sections", "week-01", "SEC1", "students", f"{sid}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ext = data.get("extraction", {})
        ocr_text = data.get("ocr", {}).get("text", "")

        print("\n" + "#" * 60)
        print(f"### EVALUATION INSPECTION: STUDENT {sid}")
        print("#" * 60)

        # For N240046, test on formatted extraction text (which is what rerun evaluations uses)
        # For N240157 and N240081, test on OCR text
        if sid == "N240046":
            eval_input = format_extraction_for_evaluation(ext, fallback_text=ocr_text)
        else:
            eval_input = ocr_text

        eval_res = validate_and_ground_evaluation({}, eval_input, q13, manual_reqs)

        print("\nCRITERIA TOTALS:")
        for crit, stats in eval_res.summary.get("criteria_totals", {}).items():
            print(f"  - {crit}: {stats['obtained']} / {stats['max']}")

        print(f"\nTOTAL MARKS: {eval_res.summary.get('total_obtained')} / {eval_res.summary.get('total_max')}")
        print(f"FINAL SCORE (1-dec): {eval_res.summary.get('final_score_out_of_10')} / 10.0")
        print(f"FINAL SCORE (2-dec): {eval_res.summary.get('final_score')} / 10.0")
        print(f"OVERALL ASSESSMENT: {eval_res.summary.get('overall_assessment')}")

if __name__ == "__main__":
    run_acceptance_inspection()
