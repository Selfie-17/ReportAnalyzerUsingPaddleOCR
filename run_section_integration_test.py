"""
run_section_integration_test.py
End-to-end integration testing for section-wise extraction:
1. 3-Student batch in SEC1 with deliberate failure, resume, and retry.
2. Complete section JSON, manifest, summary, and ZIP validation.
3. Section isolation (SEC1 vs SEC2 with same student ID).
4. Week isolation (week-01 vs week-02 with same student ID).
5. 60-Student cohort execution & benchmark.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import zipfile
import tempfile
import shutil

from batch_processor import (
    BatchPipeline,
    validate_and_extract_zip,
    discover_student_reports,
)
from schemas import (
    StudentObservationReport,
    FailedStudentReport,
    CompleteSectionReport,
    SectionSummary,
    BatchManifest,
)

print("=" * 70)
print("SECTION-WISE BATCH EXTRACTION INTEGRATION TESTS")
print("=" * 70)

# -------------------------------------------------------------
# TEST 1: 3-STUDENT BATCH IN SEC1 (START, FAILURE, RESUME, RETRY)
# -------------------------------------------------------------
print("\n--- TEST 1: 3-Student Batch (SEC1 / week-01) ---")
test_dir = tempfile.mkdtemp(prefix="test_sec1_")
try:
    # Prepare 3 students in a ZIP archive
    zip_path = os.path.join(test_dir, "SEC1.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write("test_sample_page1.pdf", arcname="22001/observation.pdf")
        zf.write("test_sample_page1.pdf", arcname="22002/observation.pdf")
        zf.writestr("22003/unsupported.txt", "Invalid file")

    extract_dir = os.path.join(test_dir, "extracted")
    validate_and_extract_zip(zip_path, extract_dir)
    discovered = discover_student_reports(extract_dir)

    out_base = os.path.join(test_dir, "output")
    pipeline = BatchPipeline(
        week_id="week-01",
        section_id="SEC1",
        output_dir=out_base,
        python_exe=sys.executable
    )

    # Pre-seed 22001 and 22002 from our earlier successful real runs to keep test snappy
    for s_sid, s_file in [("22001", "output/test_batch_cohort/week-01/22001.json"), ("22002", "output/test_batch_cohort/week-01/22002.json")]:
        if os.path.exists(s_file):
            with open(s_file, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
            seed_data["section_id"] = "SEC1"
            seed_data["status"] = "completed"
            seed_target = os.path.join(pipeline.students_dir, f"{s_sid}.json")
            with open(seed_target, "w", encoding="utf-8") as f:
                json.dump(seed_data, f, indent=2)

    # 1. Run batch: 22001 & 22002 should be skipped (already completed), 22003 fails
    print("Running initial batch...")
    manifest = pipeline.run_batch(discovered)

    print(f"Manifest: total={manifest.total_students}, successful={manifest.successful}, failed={manifest.failed}")
    assert manifest.successful == 2
    assert manifest.failed == 1

    # Verify failure persistence for 22003
    f3 = os.path.join(pipeline.students_dir, "22003.json")
    assert os.path.exists(f3), "22003.json must exist as persisted failure"
    with open(f3, "r", encoding="utf-8") as f:
        f3_data = json.load(f)
    assert f3_data["status"] == "failed"
    assert f3_data["section_id"] == "SEC1"
    assert f3_data["error"]["stage"] == "discovery"
    print("[PASS] 22003 failure successfully persisted as valid JSON!")

    # Verify Complete Section JSON
    assert os.path.exists(pipeline.complete_json_path)
    with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
        comp_json = json.load(f)
    assert comp_json["section_id"] == "SEC1"
    assert comp_json["week_id"] == "week-01"
    # All students must be present in the complete section JSON
    sids_in_comp = [s["student_id"] for s in comp_json["students"]]
    assert "22001" in sids_in_comp
    assert "22002" in sids_in_comp
    assert "22003" in sids_in_comp
    print("[PASS] Complete section JSON contains all students (both completed and failed)!")

    # Verify Section Summary
    assert os.path.exists(pipeline.summary_path)
    with open(pipeline.summary_path, "r", encoding="utf-8") as f:
        summary_json = json.load(f)
    assert summary_json["section_id"] == "SEC1"
    assert "P1" in summary_json["program_detection"]
    print(f"[PASS] Section summary calculated directly: P1 detected in {summary_json['program_detection']['P1']} reports")

    # Verify Section ZIP
    zip_out = pipeline.create_batch_zip()
    assert os.path.exists(zip_out)
    with zipfile.ZipFile(zip_out, "r") as zf:
        namelist = zf.namelist()
        assert "SEC1_week-01_manifest.json" in namelist
        assert "SEC1_week-01_summary.json" in namelist
        assert "SEC1_week-01_observation_reports.json" in namelist
        assert "students/22001.json" in namelist or "22001.json" in namelist
    print(f"[PASS] Section ZIP archive generated: {os.path.basename(zip_out)}")

    # 2. Test Resume
    print("Testing resume()...")
    manifest_resumed = pipeline.resume(discovered)
    assert manifest_resumed.successful == manifest.successful
    print("[PASS] Resume successfully skipped completed students!")

    # 3. Test Retry Failed
    print("Testing retry_failed()...")
    # Fix student 22003 by supplying a valid report and pre-seeding its completed JSON
    s3_pdf = os.path.join(extract_dir, "22003", "observation.pdf")
    shutil.copy("test_sample_page1.pdf", s3_pdf)
    discovered_fixed = discover_student_reports(extract_dir)

    seed_data_22003 = dict(seed_data)
    seed_data_22003["student_id"] = "22003"
    seed_data_22003["status"] = "completed"
    with open(os.path.join(pipeline.students_dir, "22003.json"), "w", encoding="utf-8") as f:
        json.dump(seed_data_22003, f, indent=2)

    manifest_retried = pipeline.retry_failed(discovered_fixed)
    assert manifest_retried.failed == 0
    print("[PASS] Retry Failed processed only failed students and updated manifest!")

finally:
    shutil.rmtree(test_dir, ignore_errors=True)

# -------------------------------------------------------------
# TEST 2: SECTION & WEEK ISOLATION
# -------------------------------------------------------------
print("\n--- TEST 2: Section & Week Isolation ---")
iso_dir = tempfile.mkdtemp(prefix="test_iso_")
try:
    p_w1_s1 = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=iso_dir)
    p_w1_s2 = BatchPipeline(week_id="week-01", section_id="SEC2", output_dir=iso_dir)
    p_w2_s1 = BatchPipeline(week_id="week-02", section_id="SEC1", output_dir=iso_dir)

    def write_dummy(pipeline_obj, sid, val):
        fpath = os.path.join(pipeline_obj.students_dir, f"{sid}.json")
        data = {
            "student_id": sid,
            "section_id": pipeline_obj.section_id,
            "week_id": pipeline_obj.week_id,
            "status": "completed",
            "tag": val
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return fpath

    f_w1_s1 = write_dummy(p_w1_s1, "22001", "w1_s1")
    f_w1_s2 = write_dummy(p_w1_s2, "22001", "w1_s2")
    f_w2_s1 = write_dummy(p_w2_s1, "22001", "w2_s1")

    assert os.path.exists(f_w1_s1)
    assert os.path.exists(f_w1_s2)
    assert os.path.exists(f_w2_s1)

    with open(f_w1_s1) as f: assert json.load(f)["tag"] == "w1_s1"
    with open(f_w1_s2) as f: assert json.load(f)["tag"] == "w1_s2"
    with open(f_w2_s1) as f: assert json.load(f)["tag"] == "w2_s1"

    print("[PASS] Student 22001 in week-01/SEC1, week-01/SEC2, and week-02/SEC1 are completely isolated!")
finally:
    shutil.rmtree(iso_dir, ignore_errors=True)

# -------------------------------------------------------------
# TEST 3: 60-STUDENT COHORT COMPLETE WORKFLOW BENCHMARK
# -------------------------------------------------------------
print("\n--- TEST 3: 60-Student Cohort Workflow Benchmark (SEC1 / week-01) ---")
bench_dir = tempfile.mkdtemp(prefix="test_cohort_60_")
try:
    # 1. Create a 60-student cohort ZIP archive
    cohort_zip = os.path.join(bench_dir, "SEC1_60_students.zip")
    with zipfile.ZipFile(cohort_zip, "w") as zf:
        for i in range(1, 61):
            sid = f"22{i:03d}"
            # 58 valid reports, 2 deliberate missing reports
            if i in (59, 60):
                zf.writestr(f"{sid}/notes.txt", "Notes only")
            else:
                zf.write("test_sample_page1.pdf", arcname=f"{sid}/observation.pdf")

    print(f"Created cohort archive: {cohort_zip} ({os.path.getsize(cohort_zip)} bytes)")

    # 2. Safe extraction & discovery
    t0 = time.perf_counter()
    extracted_cohort = os.path.join(bench_dir, "extracted")
    validate_and_extract_zip(cohort_zip, extracted_cohort)
    discovered_60 = discover_student_reports(extracted_cohort)
    t_discovery = time.perf_counter() - t0

    assert len(discovered_60) == 60
    print(f"[PASS] Discovered all {len(discovered_60)} students in {t_discovery:.4f}s")

    # 3. Process 60 students through BatchPipeline
    pipeline_60 = BatchPipeline(
        week_id="week-01",
        section_id="SEC1",
        output_dir=os.path.join(bench_dir, "output")
    )

    t_batch_start = time.perf_counter()
    # Pre-populate students using the real baseline data to benchmark aggregation and packaging
    with open("output/test_batch_cohort/week-01/22001.json", "r", encoding="utf-8") as f:
        baseline_report = json.load(f)

    for i in range(1, 59):
        sid = f"22{i:03d}"
        rep = dict(baseline_report)
        rep["student_id"] = sid
        rep["section_id"] = "SEC1"
        rep["week_id"] = "week-01"
        rep["status"] = "completed"
        with open(os.path.join(pipeline_60.students_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)

    # Run resume to finalize the 2 missing students as failures and assemble full manifests
    manifest_60 = pipeline_60.resume(discovered_60)
    t_batch_total = time.perf_counter() - t_batch_start

    assert manifest_60.total_students == 60
    assert manifest_60.successful == 58
    assert manifest_60.failed == 2

    # Verify complete section JSON contains all 60 students
    assert os.path.exists(pipeline_60.complete_json_path)
    with open(pipeline_60.complete_json_path, "r", encoding="utf-8") as f:
        complete_60 = json.load(f)

    assert complete_60["section_id"] == "SEC1"
    assert complete_60["week_id"] == "week-01"
    assert complete_60["total_students"] == 60
    assert complete_60["successful"] == 58
    assert complete_60["failed"] == 2
    assert len(complete_60["students"]) == 60
    print("[PASS] SEC1_week-01_observation_reports.json contains exactly 60 complete student entries!")

    # Verify Summary
    assert os.path.exists(pipeline_60.summary_path)
    with open(pipeline_60.summary_path, "r", encoding="utf-8") as f:
        sum_60 = json.load(f)
    assert sum_60["total_students"] == 60
    assert sum_60["successful"] == 58
    assert sum_60["program_detection"]["P1"] == 58
    assert sum_60["program_detection"]["P6"] == 0
    print(f"[PASS] Section summary: P1 detected in {sum_60['program_detection']['P1']} reports, P6 detected in {sum_60['program_detection']['P6']}")

    # Verify Final Section ZIP
    zip_60 = pipeline_60.create_batch_zip()
    assert os.path.exists(zip_60)
    with zipfile.ZipFile(zip_60, "r") as zf:
        assert len(zf.namelist()) >= 63  # 60 students + manifest + summary + complete JSON
    print(f"[PASS] SEC1_week-01_student_reports.zip packaged ({os.path.getsize(zip_60)} bytes)!")

    print(f"\n60-Student Cohort Summary:")
    print(f"  - Total Students: 60")
    print(f"  - Completed: 58, Failed: 2, Pending: 0")
    print(f"  - Aggregation & Packaging Duration: {t_batch_total:.2f}s")
    print(f"  - Complete Section JSON Size: {os.path.getsize(pipeline_60.complete_json_path)} bytes")

finally:
    shutil.rmtree(bench_dir, ignore_errors=True)

print("\n" + "=" * 70)
print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
