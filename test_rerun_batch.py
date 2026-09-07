import os
import json
from batch_processor import BatchPipeline
from verifier import WEEK1_13_PROGRAMS_PRESET, OFFICIAL_INSTRUCTION_MANUAL, parse_evaluation_scores

def test_batch_rerun():
    proc = BatchPipeline(
        week_id="week-01",
        section_id="SEC1",
        assigned_questions=WEEK1_13_PROGRAMS_PRESET,
        instruction_manual=OFFICIAL_INSTRUCTION_MANUAL,
        enable_evaluation=True
    )
    print("Running rerun_evaluations_on_extracted for N240046 and N240081...")
    manifest = proc.rerun_evaluations_on_extracted(selected_student_ids=["N240046", "N240081"])
    print(f"Manifest total students: {manifest.total_students}")

    for sid in ["N240157", "N240046", "N240081"]:
        fpath = os.path.join(proc.students_dir, f"{sid}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            eval_text = d.get("evaluation", "")
            scores = parse_evaluation_scores(eval_text)
            print(f"\nStudent {sid}:")
            print(f"  Score: {scores.get('score_display')}")
            print(f"  Status: {scores.get('status')}")
            print(f"  Variables: {scores.get('variables_table')}")
            print(f"  Observation: {scores.get('what_i_observed')}")

if __name__ == "__main__":
    test_batch_rerun()
