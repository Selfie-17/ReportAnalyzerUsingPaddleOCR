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


def mock_ocr_success(file_path, python_exe=None):
    return {
        "success": True,
        "text": "1. Check whether a number is even or odd\nExplanation: Uses mod 2.",
        "num_pages": 1,
        "page_breakdown": [{"page": 1, "text": "1. Check whether a number is even or odd"}],
        "device": "gpu:0",
        "paddle_version": "3.3.0",
        "total_time": 1.2
    }


def mock_extraction_success(report_text, base_url=None, model=None, temperature=0.1, page_breakdown=None):
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
