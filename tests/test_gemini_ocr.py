"""
tests/test_gemini_ocr.py
Unit tests for GeminiKeyManager, rotation, failover, and gemini OCR integration.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from gemini_ocr import GeminiKeyManager, extract_text_with_gemini, _parse_page_breakdown


def test_key_manager_comma_separated():
    km = GeminiKeyManager(api_keys="key_A, key_B , key_C")
    assert km.key_count() == 3
    assert km.get_all_keys() == ["key_A", "key_B", "key_C"]
    assert km.has_keys() is True

    # Test round-robin
    k1 = km.get_next_key()
    k2 = km.get_next_key()
    k3 = km.get_next_key()
    k4 = km.get_next_key()
    assert [k1, k2, k3, k4] == ["key_A", "key_B", "key_C", "key_A"]


def test_key_manager_json_list():
    km = GeminiKeyManager(api_keys='["key_1", "key_2"]')
    assert km.key_count() == 2
    assert km.get_all_keys() == ["key_1", "key_2"]


def test_key_manager_env_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "env_key_1, env_key_2")
    monkeypatch.setenv("GEMINI_API_KEY", "env_key_3")

    km = GeminiKeyManager()
    assert km.key_count() == 3
    assert km.get_all_keys() == ["env_key_1", "env_key_2", "env_key_3"]


def test_key_manager_single_env_only(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "single_key_xyz")

    km = GeminiKeyManager()
    assert km.key_count() == 1
    assert km.get_next_key() == "single_key_xyz"


def test_key_manager_exhaustion_failover():
    km = GeminiKeyManager(api_keys=["key1", "key2", "key3"], exhaustion_cooldown_seconds=10)
    # Mark key1 as exhausted
    km.mark_exhausted("key1")
    # Next key should skip key1 and return key2
    assert km.get_next_key() == "key2"
    assert km.get_next_key() == "key3"


def test_parse_page_breakdown_with_markers():
    text = (
        "<!-- Page 1 -->\n"
        "# Lab Report Page 1\n"
        "Aim: To write a C program\n\n"
        "<!-- Page 2 -->\n"
        "# Lab Report Page 2\n"
        "Variables Table and Code\n"
    )
    breakdown = _parse_page_breakdown(text, 2)
    assert len(breakdown) == 2
    assert breakdown[0]["page"] == 1
    assert "Lab Report Page 1" in breakdown[0]["text"]
    assert breakdown[1]["page"] == 2
    assert "Lab Report Page 2" in breakdown[1]["text"]


def test_extract_text_with_gemini_no_keys():
    km = GeminiKeyManager(api_keys="")
    res = extract_text_with_gemini("dummy.pdf", key_manager=km)
    assert res["success"] is False
    assert "No Gemini API keys found" in res["error"]


def test_batch_pipeline_step2_gemini_ocr(tmp_path):
    from batch_processor import BatchPipeline
    import json

    # Create dummy student report
    dummy_pdf = tmp_path / "N249999_report.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

    pipeline = BatchPipeline(week_id="week-04", section_id="SEC9", output_dir=str(tmp_path / "output"))

    mock_gemini_return = {
        "success": True,
        "text": "# Aim: Test Program\nTo verify Gemini OCR integration.\n<!-- Page 1 -->\nVariables Table\nCode:\n```c\n#include <stdio.h>\nint main() { return 0; }\n```",
        "num_pages": 1,
        "page_breakdown": [{"page": 1, "text": "Variables Table and Code"}],
        "engine": "gemini",
        "model": "gemini-2.5-flash",
        "total_time": 1.25,
        "error": None
    }

    with patch("gemini_ocr.extract_text_with_gemini", return_value=mock_gemini_return):
        res = pipeline.step2_run_ocr(
            discovered_students={"N249999": {"file_path": str(dummy_pdf)}},
            ocr_engine="gemini",
            gemini_model="gemini-2.5-flash"
        )

    assert "N249999" in res
    assert res["N249999"]["success"] is True
    assert res["N249999"]["engine"] == "gemini"

    # Verify student JSON persisted on disk
    student_json_path = os.path.join(pipeline.students_dir, "N249999.json")
    assert os.path.exists(student_json_path)

    with open(student_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["ocr"]["engine"] == "gemini"
    assert data["ocr"]["model"] == "gemini-2.5-flash"
    assert len(data["ocr"]["text"]) > 0
    assert "observation_report" in data
    assert "extraction" in data

