"""
tests/test_schemas.py
Unit tests for canonical Pydantic schemas.
"""

import pytest
from schemas import (
    VariableItem,
    ProgramDetails,
    ExtractionResult,
    SourceMeta,
    OcrResult,
    StudentObservationReport,
    BatchManifest,
    BatchStudentStatus,
)


def test_variable_item():
    v = VariableItem(variable="count", purpose="Loop counter")
    assert v.variable == "count"
    assert v.purpose == "Loop counter"


def test_program_details_defaults():
    p = ProgramDetails()
    assert p.status == "not_detected"
    assert p.source_pages == []
    assert p.problem_understanding is None
    assert p.logic_approach is None
    assert p.important_variables == []
    assert p.what_i_observed is None


def test_extraction_result_defaults():
    er = ExtractionResult()
    assert er.status == "success"
    assert er.objective_of_lab is None
    assert er.programs == {}
    assert er.detected_programs == []
    assert er.missing_programs == []
    assert er.errors == []


def test_student_observation_report_serialization():
    report = StudentObservationReport(
        student_id="22001",
        week_id="week-01",
        source=SourceMeta(filename="obs.pdf", num_pages=2),
        ocr=OcrResult(text="Sample raw OCR", num_pages=2),
        extraction=ExtractionResult(
            status="success",
            objective_of_lab="Understand conditionals",
            programs={
                "P1": ProgramDetails(
                    status="detected",
                    source_pages=[1],
                    problem_understanding="Even or Odd check",
                    important_variables=[
                        VariableItem(variable="num", purpose="Stores input value")
                    ]
                )
            },
            detected_programs=["P1"],
            missing_programs=[f"P{i}" for i in range(2, 11)]
        )
    )

    data = report.model_dump()
    assert data["student_id"] == "22001"
    assert data["source"]["filename"] == "obs.pdf"
    assert data["ocr"]["text"] == "Sample raw OCR"
    assert data["extraction"]["programs"]["P1"]["status"] == "detected"
    assert data["extraction"]["programs"]["P1"]["source_pages"] == [1]

    # Verify JSON round-trip
    json_str = report.model_dump_json()
    reconstructed = StudentObservationReport.model_validate_json(json_str)
    assert reconstructed.student_id == "22001"
    assert reconstructed.extraction.programs["P1"].important_variables[0].variable == "num"


def test_batch_manifest():
    manifest = BatchManifest(
        week_id="week-01",
        total_students=2,
        successful=1,
        failed=1,
        students=[
            BatchStudentStatus(student_id="22001", status="completed", output="22001.json"),
            BatchStudentStatus(student_id="22002", status="failed", stage="ocr", error="File corrupted")
        ]
    )
    assert manifest.total_students == 2
    assert manifest.successful == 1
    assert manifest.failed == 1
    assert manifest.students[1].error == "File corrupted"
