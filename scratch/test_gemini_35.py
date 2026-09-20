import os
import json
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.insert(0, ".")
from verifier import evaluate_with_gemini

sid = "N240035"
json_path = os.path.join(r"output/sections/week-04/SEC2/students", f"{sid}.json")
with open(json_path, "r", encoding="utf-8") as f:
    sdata = json.load(f)

print(f"Testing evaluate_with_gemini using gemini-3.5-flash on {sid}...")
res = evaluate_with_gemini(
    student_id=sid,
    report_text=sdata.get("ocr", {}).get("text", ""),
    extracted_report=sdata.get("extraction"),
    model_name="gemini-3.5-flash",
    week_id="week-04"
)

print(f"Provider: {res.provider}")
print(f"Model: {res.model_name}")
print(f"Score: {res.recommended_score}")
print(f"Grade: {res.grade}")
print(f"Report Length: {len(res.full_report_markdown)} chars")
print("\n=== SAMPLE SECTIONS FROM GEMINI REPORT ===")
for line in res.full_report_markdown.splitlines()[:50]:
    print(line)
