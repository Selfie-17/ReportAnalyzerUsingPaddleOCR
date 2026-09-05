"""
tests/test_batch_security.py
Security tests for ZIP handling and cohort student report discovery.
"""

import os
import zipfile
import tempfile
import pytest
from batch_processor import (
    validate_and_extract_zip,
    discover_student_reports,
    SecurityError,
)


def test_safe_zip_extraction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "safe.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("22001/observation.pdf", b"%PDF-1.4 test")
            zf.writestr("22002/observation.pdf", b"%PDF-1.4 test 2")

        extract_target = os.path.join(tmp_dir, "extracted")
        validate_and_extract_zip(zip_path, extract_target)

        assert os.path.exists(os.path.join(extract_target, "22001", "observation.pdf"))
        assert os.path.exists(os.path.join(extract_target, "22002", "observation.pdf"))


def test_zip_slip_protection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        evil_zip_path = os.path.join(tmp_dir, "evil.zip")
        with zipfile.ZipFile(evil_zip_path, "w") as zf:
            # Attempt to escape extraction directory
            zf.writestr("../evil_file.txt", "malicious payload")

        extract_target = os.path.join(tmp_dir, "extracted")
        os.makedirs(extract_target, exist_ok=True)

        with pytest.raises(SecurityError, match="Zip Slip path traversal attempt detected"):
            validate_and_extract_zip(evil_zip_path, extract_target)

        assert not os.path.exists(os.path.join(tmp_dir, "evil_file.txt"))


def test_student_discovery_directory_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create student 22001 with 1 PDF
        s1 = os.path.join(tmp_dir, "22001")
        os.makedirs(s1)
        with open(os.path.join(s1, "obs.pdf"), "w") as f:
            f.write("content")

        # Create student 22002 with 1 image
        s2 = os.path.join(tmp_dir, "22002")
        os.makedirs(s2)
        with open(os.path.join(s2, "report.png"), "w") as f:
            f.write("image")

        discovered = discover_student_reports(tmp_dir)
        assert "22001" in discovered
        assert discovered["22001"]["error"] is None
        assert discovered["22001"]["file_path"].endswith("obs.pdf")

        assert "22002" in discovered
        assert discovered["22002"]["error"] is None
        assert discovered["22002"]["file_path"].endswith("report.png")


def test_student_discovery_ambiguity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Student with multiple PDFs
        s = os.path.join(tmp_dir, "22003")
        os.makedirs(s)
        with open(os.path.join(s, "p1.pdf"), "w") as f:
            f.write("1")
        with open(os.path.join(s, "p2.pdf"), "w") as f:
            f.write("2")

        discovered = discover_student_reports(tmp_dir)
        assert "22003" in discovered
        assert discovered["22003"]["file_path"] is None
        assert "Ambiguous: found multiple report files" in discovered["22003"]["error"]


def test_student_discovery_missing_report():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Student directory with only .c code and no PDF/image
        s = os.path.join(tmp_dir, "22004")
        os.makedirs(s)
        with open(os.path.join(s, "p1.c"), "w") as f:
            f.write("int main() {}")

        discovered = discover_student_reports(tmp_dir)
        assert "22004" in discovered
        assert discovered["22004"]["file_path"] is None
        assert "No observation report" in discovered["22004"]["error"]
