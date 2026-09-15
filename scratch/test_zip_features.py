import sys, os
sys.path.insert(0, os.path.abspath('.'))
from verifier import ingest_student_code_from_json_or_files, match_report_programs_to_questions
import json

coll = ingest_student_code_from_json_or_files('week-4-sec-2.zip')
with open('output/sections/week-04/SEC2/students/N241003.json', 'r', encoding='utf-8') as f:
    s_data = json.load(f)

from verifier import evaluate_holistic_student, render_holistic_evaluation_markdown

res = evaluate_holistic_student(
    report_text=s_data['ocr']['text'],
    extracted_report=s_data['extraction'],
    student_code=coll,
    student_id='N241003',
    week_id='week-04'
)

md = render_holistic_evaluation_markdown(res)
with open('output/sections/week-04/SEC2/students/N241003_evaluation.md', 'w', encoding='utf-8') as f:
    f.write(md)

s_data['holistic_evaluation'] = res.to_canonical_dict()
with open('output/sections/week-04/SEC2/students/N241003.json', 'w', encoding='utf-8') as f:
    json.dump(s_data, f, indent=2)

print("Report saved successfully. Total code problems:", res.total_code_problems)
print("\nReported Programs:")
for r in res.report_entries:
    m = next((m for m in res.matches if m.report_program_id == r.report_program_id), None)
    print(f"  {r.report_program_id} -> {m.matched_problem_id if m else 'None'} ({m.source_file if m else 'None'})")

print("\nConsistency:")
for c in res.code_report_consistency:
    print(f"  {c.report_program_id} ({c.problem}): claim_type={c.claim_type}")
    print(f"    claim: {c.report_claim}")
    print(f"    reality: {c.code_reality}")

