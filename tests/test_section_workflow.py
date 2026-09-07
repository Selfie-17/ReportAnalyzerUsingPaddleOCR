"""
tests/test_section_workflow.py
Comprehensive unit tests for section-wise batch extraction (SEC1-SEC6),
aggregation, summary calculation, section/week isolation, resume, and retry.
All tests use lightweight mocks (no PaddleOCR or Ollama invocation).
"""

import os
import json
import zipfile
import tempfile
import pytest
from unittest.mock import patch

from batch_processor import (
    BatchPipeline,
    discover_student_reports,
    validate_and_extract_zip,
)
from schemas import (
    StudentObservationReport,
    FailedStudentReport,
    CompleteSectionReport,
    SectionSummary,
    SourceMeta,
    OcrResult,
    ExtractionResult,
    ProgramDetails,
    VariableItem,
)


def mock_fast_ocr(file_path, python_exe=None, **kwargs):
    return {
        "success": True,
        "text": "1. Check whether a number is even or odd\nExplanation: Uses mod 2.\n✓ Executed.",
        "num_pages": 1,
        "page_breakdown": [{"page": 1, "text": "1. Check whether a number is even or odd"}],
        "device": "gpu:0",
        "paddle_version": "3.3.0",
        "total_time": 0.05
    }


def mock_fast_extraction(report_text, base_url=None, model=None, temperature=0.1, page_breakdown=None, **kwargs):
    return ExtractionResult(
        status="success",
        objective_of_lab="Lab Objective text",
        programs={
            "P1": ProgramDetails(
                status="detected",
                source_pages=[1],
                problem_understanding="Check even odd",
                logic_approach="Use mod 2",
                important_variables=[VariableItem(variable="n", purpose="Input")],
                what_i_observed="Program executed"
            ),
            "P2": ProgramDetails(
                status="detected",
                source_pages=[1],
                problem_understanding="Check positive negative",
                logic_approach="Comparison ladder",
                important_variables=[],
                what_i_observed="Executed"
            )
        },
        detected_programs=["P1", "P2"],
        missing_programs=[f"P{i}" for i in range(3, 11)],
        errors=[]
    )


def mock_fast_verification(report_text, assigned_questions=None, **kwargs):
    return "# 📊 Observation Report Verification Report\n\n## Overall Evaluation\n- **Total Score:** 9.5 / 10.0\n- **Grade:** A\n- **Status:** Approved\n"


@pytest.fixture(autouse=True)
def mock_verify_for_section_tests():
    with patch("batch_processor.verify_observation_report_sync", side_effect=mock_fast_verification):
        yield


def test_section_id():
    """Verifies section_id field presence and serialization."""
    report = StudentObservationReport(
        student_id="22001",
        section_id="SEC1",
        week_id="week-01",
        status="completed",
        source=SourceMeta(filename="obs.pdf", num_pages=1),
        ocr=OcrResult(text="sample text", num_pages=1),
        extraction=ExtractionResult()
    )
    assert report.section_id == "SEC1"
    data = json.loads(report.model_dump_json())
    assert data["section_id"] == "SEC1"
    assert data["status"] == "completed"


def test_failed_student_persists_error():
    """Verifies that failure persistence creates a valid FailedStudentReport."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        status_entry = pipeline.process_student(
            student_id="22003",
            file_path=None,
            error_precheck="No observation report found"
        )
        assert status_entry.status == "failed"
        assert status_entry.stage == "discovery"

        persisted_file = os.path.join(pipeline.students_dir, "22003.json")
        assert os.path.exists(persisted_file)
        with open(persisted_file, "r", encoding="utf-8") as f:
            failed_data = json.load(f)

        failed_obj = FailedStudentReport.model_validate(failed_data)
        assert failed_obj.student_id == "22003"
        assert failed_obj.section_id == "SEC1"
        assert failed_obj.week_id == "week-01"
        assert failed_obj.status == "failed"
        assert failed_obj.error.stage == "discovery"
        assert "No observation report found" in failed_obj.error.message


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_build_complete_section_json(mock_ext, mock_ocr):
    """Verifies aggregation of multiple students into CompleteSectionReport."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)

        # Create dummy student reports
        s1_file = os.path.join(tmp_dir, "s1.pdf")
        with open(s1_file, "w") as f:
            f.write("test")

        pipeline.process_student("22001", file_path=s1_file)
        pipeline.process_student("22002", file_path=None, error_precheck="File corrupted")

        complete_rep = pipeline.build_complete_section_json()
        assert complete_rep.section_id == "SEC1"
        assert complete_rep.week_id == "week-01"
        assert complete_rep.total_students == 2
        assert complete_rep.successful == 1
        assert complete_rep.failed == 1
        assert len(complete_rep.students) == 2

        # Check that complete section JSON file exists on disk
        assert os.path.exists(pipeline.complete_json_path)
        with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["section_id"] == "SEC1"
        assert saved_data["successful"] == 1
        assert saved_data["failed"] == 1


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_section_with_60_students(mock_ext, mock_ocr):
    """Fast simulation of a full cohort of 60 students in SEC1."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)

        discovered = {}
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("pdf data")

        for i in range(1, 61):
            sid = f"22{i:03d}"
            discovered[sid] = {"file_path": sample_pdf, "error": None}

        manifest = pipeline.run_batch(discovered)
        assert manifest.total_students == 60
        assert manifest.successful == 60
        assert manifest.failed == 0

        # Verify 60 files in students/
        student_files = [f for f in os.listdir(pipeline.students_dir) if f.endswith(".json")]
        assert len(student_files) == 60

        # Verify Complete section JSON
        assert os.path.exists(pipeline.complete_json_path)
        with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        assert comp_data["total_students"] == 60
        assert len(comp_data["students"]) == 60

        # Verify Summary tallies
        assert os.path.exists(pipeline.summary_path)
        with open(pipeline.summary_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
        assert sum_data["program_detection"]["P1"] == 60
        assert sum_data["program_detection"]["P2"] == 60
        assert sum_data["program_detection"]["P3"] == 0


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_same_student_id_different_sections(mock_ext, mock_ocr):
    """Section isolation: student 22001 in SEC1 and SEC2 do not collide or overwrite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        p_sec1 = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        p_sec2 = BatchPipeline(week_id="week-01", section_id="SEC2", output_dir=tmp_dir)

        p_sec1.process_student("22001", file_path=sample_pdf)
        p_sec2.process_student("22001", file_path=sample_pdf)

        s1_file = os.path.join(p_sec1.students_dir, "22001.json")
        s2_file = os.path.join(p_sec2.students_dir, "22001.json")

        assert os.path.exists(s1_file)
        assert os.path.exists(s2_file)
        assert s1_file != s2_file

        with open(s1_file, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        with open(s2_file, "r", encoding="utf-8") as f:
            d2 = json.load(f)

        assert d1["section_id"] == "SEC1"
        assert d2["section_id"] == "SEC2"


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_same_section_different_weeks(mock_ext, mock_ocr):
    """Week isolation: week-01/SEC1 and week-02/SEC1 are completely isolated."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        p_w1 = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        p_w2 = BatchPipeline(week_id="week-02", section_id="SEC1", output_dir=tmp_dir)

        p_w1.process_student("22001", file_path=sample_pdf)
        p_w2.process_student("22001", file_path=sample_pdf)

        f1 = os.path.join(p_w1.students_dir, "22001.json")
        f2 = os.path.join(p_w2.students_dir, "22001.json")

        assert os.path.exists(f1)
        assert os.path.exists(f2)
        assert f1 != f2


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_resume_skips_completed_students(mock_ext, mock_ocr):
    """Verifies that resume skips already completed students."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        discovered = {
            "22001": {"file_path": sample_pdf, "error": None},
            "22002": {"file_path": sample_pdf, "error": None}
        }

        # Process 22001 first
        pipeline.process_student("22001", file_path=sample_pdf)
        assert mock_ocr.call_count == 1

        mock_ocr.reset_mock()
        mock_ext.reset_mock()

        # Resume batch across both 22001 and 22002
        pipeline.resume(discovered)

        # 22001 should be skipped; only 22002 should be processed!
        assert mock_ocr.call_count == 1
        assert pipeline.state["22001"].status == "completed"
        assert pipeline.state["22002"].status == "completed"


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_retry_only_failed_students(mock_ext, mock_ocr):
    """Verifies retry_failed processes ONLY failed students, skipping completed & pending."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        discovered = {
            "22001": {"file_path": sample_pdf, "error": None},
            "22002": {"file_path": None, "error": "Missing report"},
            "22003": {"file_path": sample_pdf, "error": None}
        }

        # Process: 22001 completes, 22002 fails
        pipeline.process_student("22001", file_path=sample_pdf)
        pipeline.process_student("22002", file_path=None, error_precheck="Missing report")
        # 22003 remains pending in state
        pipeline.state["22003"] = pipeline.state.get("22003") or pipeline.process_student(
            "22003", file_path=sample_pdf
        ) if False else None

        from schemas import BatchStudentStatus
        pipeline.state["22003"] = BatchStudentStatus(student_id="22003", section_id="SEC1", status="pending")

        assert pipeline.state["22001"].status == "completed"
        assert pipeline.state["22002"].status == "failed"
        assert pipeline.state["22003"].status == "pending"

        # Now fix 22002
        discovered["22002"] = {"file_path": sample_pdf, "error": None}

        mock_ocr.reset_mock()
        mock_ext.reset_mock()

        # Call retry_failed
        pipeline.retry_failed(discovered)

        # Only 22002 should be processed!
        assert mock_ocr.call_count == 1
        assert pipeline.state["22002"].status == "completed"
        # 22003 must remain pending (not run by retry_failed)
        assert pipeline.state["22003"].status == "pending"


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_section_json_contains_all_students(mock_ext, mock_ocr):
    """Verifies that complete section JSON contains both successful and failed students."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        pipeline.process_student("22001", file_path=sample_pdf)
        pipeline.process_student("22002", file_path=None, error_precheck="Corrupt PDF")

        complete_rep = pipeline.build_complete_section_json()
        sids = [(s.student_id if hasattr(s, "student_id") else s.get("student_id")) for s in complete_rep.students]
        assert "22001" in sids
        assert "22002" in sids

        # Student 22002 must have status failed and error recorded
        s2 = next(s for s in complete_rep.students if (s.student_id if hasattr(s, "student_id") else s.get("student_id")) == "22002")
        status = s2.status if hasattr(s2, "status") else s2["status"]
        assert status == "failed"
        err_msg = s2.error.message if hasattr(s2, "error") and hasattr(s2.error, "message") else s2["error"]["message"]
        assert "Corrupt PDF" in err_msg


def test_duplicate_student_ids_rejected():
    """Duplicate / ambiguous student mappings within the same ZIP must be flagged as ambiguous."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create folder 22001/observation.pdf
        s_dir = os.path.join(tmp_dir, "22001")
        os.makedirs(s_dir)
        with open(os.path.join(s_dir, "observation.pdf"), "w") as f:
            f.write("pdf1")

        # Also create flat file 22001_observation.pdf
        with open(os.path.join(tmp_dir, "22001_observation.pdf"), "w") as f:
            f.write("pdf2")

        discovered = discover_student_reports(tmp_dir)
        assert "22001" in discovered
        assert discovered["22001"]["file_path"] is None
        assert "Ambiguous: multiple observation reports" in discovered["22001"]["error"]


def test_multiple_reports_rejected():
    """Multiple report files inside one student directory must be flagged as ambiguous."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        s_dir = os.path.join(tmp_dir, "22002")
        os.makedirs(s_dir)
        with open(os.path.join(s_dir, "obs1.pdf"), "w") as f:
            f.write("1")
        with open(os.path.join(s_dir, "obs2.pdf"), "w") as f:
            f.write("2")

        discovered = discover_student_reports(tmp_dir)
        assert "22002" in discovered
        assert discovered["22002"]["file_path"] is None
        assert "Ambiguous: found multiple report files" in discovered["22002"]["error"]


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_fast_ocr)
@patch("batch_processor.extract_observation_report", side_effect=mock_fast_extraction)
def test_section_summary_program_detection(mock_ext, mock_ocr):
    """Verifies that SectionSummary computes exact P1..P10 detection counts from JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=tmp_dir)
        sample_pdf = os.path.join(tmp_dir, "sample.pdf")
        with open(sample_pdf, "w") as f:
            f.write("dummy")

        # Process 2 completed students and 1 failed student
        pipeline.process_student("22001", file_path=sample_pdf)
        pipeline.process_student("22002", file_path=sample_pdf)
        pipeline.process_student("22003", file_path=None, error_precheck="Failed file")

        summary = pipeline.build_section_summary()
        assert summary.section_id == "SEC1"
        assert summary.week_id == "week-01"
        assert summary.total_students == 3
        assert summary.successful == 2
        assert summary.failed == 1
        # P1 and P2 were detected in both successful students -> count must be 2
        assert summary.program_detection["P1"] == 2
        assert summary.program_detection["P2"] == 2
        # P3..P10 were not detected -> count must be 0
        assert summary.program_detection["P3"] == 0
