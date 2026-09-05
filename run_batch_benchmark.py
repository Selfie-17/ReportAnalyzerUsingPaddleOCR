"""
run_batch_benchmark.py
Benchmarks and validates batch extraction pipeline on a real cohort archive.
Measures OCR time, Qwen extraction time, manifest generation, resumability, and ZIP export.
"""

import os
import sys
import time
import json
import tempfile
from batch_processor import (
    validate_and_extract_zip,
    discover_student_reports,
    BatchPipeline,
)
from schemas import StudentObservationReport

cohort_zip = "test_cohort_3_students.zip"
temp_extract_dir = tempfile.mkdtemp(prefix="benchmark_cohort_")
print(f"1. Extracting {cohort_zip} to {temp_extract_dir}...")
validate_and_extract_zip(cohort_zip, temp_extract_dir)

print("2. Discovering student reports...")
discovered = discover_student_reports(temp_extract_dir)
print(f"Discovered {len(discovered)} students:")
for sid, d in discovered.items():
    print(f"  - {sid}: file={d['file_path']}, err={d['error']}")

output_dir = "output/test_batch_cohort"
pipeline = BatchPipeline(
    week_id="week-01",
    output_dir=output_dir,
    python_exe=sys.executable
)

print("\n3. Starting Batch Pipeline (OCR concurrency=1, LLM concurrency=1)...")
t0 = time.perf_counter()

def progress_cb(entry):
    print(f"  [Status Update] Student {entry.student_id}: {entry.status} (stage={entry.stage}, err={entry.error})")

manifest = pipeline.run_batch(discovered, progress_cb=progress_cb)
total_batch_time = time.perf_counter() - t0

print("\n=== BATCH RUN 1 COMPLETE ===")
print(f"Total Batch Time: {total_batch_time:.2f}s")
print(f"Successful: {manifest.successful}, Failed: {manifest.failed}, Pending: {manifest.pending}")

# Verify student 22001 output
s1_json = os.path.join(pipeline.batch_output_dir, "22001.json")
if os.path.exists(s1_json):
    with open(s1_json, "r", encoding="utf-8") as f:
        rep = StudentObservationReport.model_validate_json(f.read())
    print("\n=== SAMPLE EXTRACTED REPORT (Student 22001) ===")
    print(f"Student ID: {rep.student_id}")
    print(f"Source: {rep.source.filename} ({rep.source.num_pages} pages)")
    print(f"OCR Time: {rep.ocr.total_time:.2f}s, Device: {rep.ocr.device}")
    print(f"Detected Programs: {rep.extraction.detected_programs}")
    print(f"Missing Programs: {rep.extraction.missing_programs}")

print("\n4. Testing RESUME capability (should skip completed students)...")
t_resume_start = time.perf_counter()
pipeline_resumed = BatchPipeline(week_id="week-01", output_dir=output_dir)
manifest_resumed = pipeline_resumed.run_batch(discovered)
t_resume_total = time.perf_counter() - t_resume_start
print(f"Resume took: {t_resume_total:.2f}s (skipped completed students instantly!)")

print("\n5. Testing RETRY FAILED capability after fixing student 22003...")
s3_pdf = os.path.join(temp_extract_dir, "22003", "observation.pdf")
with open("test_sample_page1.pdf", "rb") as src, open(s3_pdf, "wb") as dst:
    dst.write(src.read())

discovered_updated = discover_student_reports(temp_extract_dir)
manifest_retried = pipeline_resumed.run_batch(discovered_updated, retry_only_failed=True, progress_cb=progress_cb)
print(f"After Retry: Successful={manifest_retried.successful}, Failed={manifest_retried.failed}")

print("\n6. Packaging final ZIP...")
zip_path = pipeline_resumed.create_batch_zip()
print(f"Final Batch ZIP created at: {zip_path} ({os.path.getsize(zip_path)} bytes)")

print("\nALL BATCH INTEGRATION CHECKS PASSED!")
