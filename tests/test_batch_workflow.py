"""
tests/test_batch_workflow.py
End-to-end tests for batch pipeline execution, resumability, retry failed, and manifest/ZIP export.
"""

import os
import json
import zipfile
import tempfile
import pytest
from unittest.mock import patch
from batch_processor import BatchPipeline
from schemas import (
    ExtractionResult,
    ProgramDetails,
    VariableItem,
    StudentObservationReport,
    BatchManifest,
)


def mock_ocr_success(file_path, python_exe=None, **kwargs):
    return {
        "success": True,
        "text": "1. Check whether a number is even or odd\nExplanation: Uses mod 2.",
        "num_pages": 1,
        "page_breakdown": [{"page": 1, "text": "1. Check whether a number is even or odd"}],
        "device": "gpu:0",
        "paddle_version": "3.3.0",
        "total_time": 1.2
    }


def mock_extraction_success(report_text, base_url=None, model=None, temperature=0.1, page_breakdown=None, **kwargs):
    return ExtractionResult(
        status="success",
        objective_of_lab="Understand loops",
        programs={
            "P1": ProgramDetails(
                status="detected",
                source_pages=[1],
                problem_understanding="Even odd check",
                logic_approach="Mod 2 check",
                important_variables=[VariableItem(variable="n", purpose="Input")],
                what_i_observed="Ran well"
            )
        },
        detected_programs=["P1"],
        missing_programs=[f"P{i}" for i in range(2, 11)],
        errors=[]
    )


def mock_fast_verification(report_text, assigned_questions=None, **kwargs):
    return "# 📊 Observation Report Verification Report\n\n## Overall Evaluation\n- **Total Score:** 9.5 / 10.0\n- **Grade:** A\n- **Status:** Approved\n"


@pytest.fixture(autouse=True)
def mock_verify_for_batch_tests():
    with patch("batch_processor.verify_observation_report_sync", side_effect=mock_fast_verification):
        yield


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_ocr_success)
@patch("batch_processor.extract_observation_report", side_effect=mock_extraction_success)
def test_batch_pipeline_3_students_with_failure_resume_and_retry(mock_ext, mock_ocr):
    with tempfile.TemporaryDirectory() as base_dir:
        # Create 3 student test files
        s1_dir = os.path.join(base_dir, "22001")
        s2_dir = os.path.join(base_dir, "22002")
        s3_dir = os.path.join(base_dir, "22003")
        os.makedirs(s1_dir)
        os.makedirs(s2_dir)
        os.makedirs(s3_dir)

        s1_file = os.path.join(s1_dir, "obs.pdf")
        s2_file = os.path.join(s2_dir, "obs.pdf")
        with open(s1_file, "w") as f:
            f.write("content 1")
        with open(s2_file, "w") as f:
            f.write("content 2")
        # s3 intentionally has NO report file -> deliberate failure

        discovered = {
            "22001": {"file_path": s1_file, "error": None},
            "22002": {"file_path": s2_file, "error": None},
            "22003": {"file_path": None, "error": "No observation report found"}
        }

        output_dir = os.path.join(base_dir, "output")
        pipeline = BatchPipeline(week_id="week-01", output_dir=output_dir)

        # Run batch
        manifest = pipeline.run_batch(discovered)
        assert manifest.total_students == 3
        assert manifest.successful == 2
        assert manifest.failed == 1
        assert pipeline.state["22001"].status == "completed"
        assert pipeline.state["22002"].status == "completed"
        assert pipeline.state["22003"].status == "failed"

        # Verify output JSON files exist and are valid
        s1_json = os.path.join(pipeline.batch_output_dir, "22001.json")
        assert os.path.exists(s1_json)
        with open(s1_json, "r", encoding="utf-8") as f:
            report_s1 = StudentObservationReport.model_validate_json(f.read())
            assert report_s1.student_id == "22001"
            assert report_s1.extraction.programs["P1"].status == "detected"
            assert report_s1.evaluation is not None
            assert "Total Score" in report_s1.evaluation

        # Verify standalone evaluation markdown file exists
        s1_eval_md = os.path.join(pipeline.students_dir, "22001_evaluation.md")
        assert os.path.exists(s1_eval_md)

        # Test Resumability: Create new pipeline instance and run again
        mock_ocr.reset_mock()
        mock_ext.reset_mock()
        pipeline_resumed = BatchPipeline(week_id="week-01", output_dir=output_dir)
        manifest_resumed = pipeline_resumed.run_batch(discovered)

        # Completed students should be skipped! No OCR calls for 22001 or 22002
        assert mock_ocr.call_count == 0
        assert manifest_resumed.successful == 2
        assert manifest_resumed.failed == 1

        # Test Retry Failed: Now fix student 22003 by adding report file
        s3_file = os.path.join(s3_dir, "obs.pdf")
        with open(s3_file, "w") as f:
            f.write("content 3")
        discovered["22003"] = {"file_path": s3_file, "error": None}

        # Run retry failed
        manifest_retried = pipeline_resumed.run_batch(discovered, retry_only_failed=True)
        assert manifest_retried.successful == 3
        assert manifest_retried.failed == 0
        assert os.path.exists(os.path.join(pipeline.batch_output_dir, "22003.json"))

        # Test Final ZIP generation
        zip_path = pipeline_resumed.create_batch_zip()
        assert os.path.exists(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "batch_manifest.json" in namelist
            assert "22001.json" in namelist
            assert "22002.json" in namelist
            assert "22003.json" in namelist
            assert "22001_evaluation.md" in namelist


@patch("subprocess.run")
def test_run_paddle_worker_sync_default_no_timeout(mock_subproc):
    import subprocess
    from batch_processor import run_paddle_worker_sync
    mock_subproc.return_value.returncode = 0
    mock_subproc.return_value.stdout = json.dumps({"success": True, "text": "Extracted text"})
    mock_subproc.return_value.stderr = ""

    res = run_paddle_worker_sync("dummy.pdf", python_exe="python")
    assert res.get("success") is True
    _, kwargs = mock_subproc.call_args
    assert kwargs.get("timeout") is None


@patch("subprocess.run")
def test_run_paddle_worker_sync_custom_timeout_expiry(mock_subproc):
    import subprocess
    from batch_processor import run_paddle_worker_sync
    mock_subproc.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=10)

    res = run_paddle_worker_sync("dummy.pdf", python_exe="python", timeout=10)
    assert res.get("success") is False
    assert "timed out after 10 seconds" in res.get("error")


@patch("batch_processor.run_paddle_worker_sync", side_effect=mock_ocr_success)
@patch("batch_processor.extract_observation_report", side_effect=mock_extraction_success)
def test_rerun_evaluations_on_extracted(mock_ext, mock_ocr):
    with tempfile.TemporaryDirectory() as base_dir:
        s1_dir = os.path.join(base_dir, "22001")
        s2_dir = os.path.join(base_dir, "22002")
        os.makedirs(s1_dir)
        os.makedirs(s2_dir)

        s1_file = os.path.join(s1_dir, "obs.pdf")
        s2_file = os.path.join(s2_dir, "obs.pdf")
        with open(s1_file, "w") as f:
            f.write("content 1")
        with open(s2_file, "w") as f:
            f.write("content 2")

        discovered = {
            "22001": {"file_path": s1_file, "error": None},
            "22002": {"file_path": s2_file, "error": None},
        }

        output_dir = os.path.join(base_dir, "output")
        pipeline = BatchPipeline(week_id="week-01", section_id="SEC1", output_dir=output_dir)

        # Initial batch run
        manifest = pipeline.run_batch(discovered)
        assert manifest.successful == 2
        assert mock_ocr.call_count == 2
        assert mock_ext.call_count == 2

        # Reset call counts
        mock_ocr.reset_mock()
        mock_ext.reset_mock()

        # Now re-run evaluations on extracted text for 22001 only
        with patch("batch_processor.verify_observation_report_sync", return_value="# Re-evaluated Report: 10/10") as mock_verify:
            manifest_rerun = pipeline.rerun_evaluations_on_extracted(
                discovered_students=discovered,
                selected_student_ids=["22001"]
            )
            assert mock_verify.call_count == 1
            # OCR and Extraction must NOT have been called again!
            assert mock_ocr.call_count == 0
            assert mock_ext.call_count == 0

        # Verify student JSON updated
        s1_json = os.path.join(pipeline.students_dir, "22001.json")
        with open(s1_json, "r", encoding="utf-8") as f:
            s1_data = json.load(f)
        assert s1_data["evaluation"] == "# Re-evaluated Report: 10/10"

        # Verify standalone evaluation md updated
        s1_md = os.path.join(pipeline.students_dir, "22001_evaluation.md")
        with open(s1_md, "r", encoding="utf-8") as f:
            assert f.read() == "# Re-evaluated Report: 10/10"

