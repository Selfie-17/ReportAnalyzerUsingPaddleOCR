"""
batch_processor.py
Safe, resumable section-wise batch observation report extraction pipeline.
Supports cohort ZIP inputs, strict Zip Slip defense, sequential GPU OCR (concurrency=1),
sequential LLM extraction (concurrency=1), complete section JSON aggregation, section summaries,
manifest generation, and ZIP packaging for Sections 1-6.
"""

import os
import sys
import json
import time
import zipfile
import tempfile
import subprocess
import re
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime
import pymupdf

from schemas import (
    StudentObservationReport,
    FailedStudentReport,
    FailedStudentError,
    CompleteSectionReport,
    SectionSummary,
    SourceMeta,
    OcrResult,
    ExtractionResult,
    BatchStudentStatus,
    BatchManifest,
)
from extractor import extract_observation_report, DEFAULT_OLLAMA_URL, DEFAULT_MODEL

# Allowed file extensions for student observation reports
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Security limits for ZIP extraction
MAX_ZIP_SIZE_BYTES = 500 * 1024 * 1024       # 500 MB max archive size
MAX_TOTAL_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB max total uncompressed


def get_paddle_python() -> str:
    """Finds the Python interpreter that has paddle installed (e.g. paddle_vl conda env)."""
    try:
        import paddle
        return sys.executable
    except ImportError:
        pass

    base_conda = os.path.dirname(sys.executable)
    candidates = [
        os.path.join(base_conda, "envs", "paddle_vl", "python.exe"),
        r"C:\Users\kampa\anaconda3\envs\paddle_vl\python.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable


class SecurityError(Exception):
    """Raised when a file or ZIP archive violates security policies."""
    pass


def atomic_write_json(file_path: str, data: Any) -> None:
    """
    Writes JSON data atomically using a process-unique temporary file and atomic rename.
    Ensures that process interruptions never leave corrupted or partially written JSON.
    """
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    temp_path = f"{file_path}.tmp_{os.getpid()}_{time.time_ns()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                f.write(data)
            elif hasattr(data, "model_dump_json"):
                f.write(data.model_dump_json(indent=2))
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def validate_and_extract_zip(
    zip_path: str,
    extract_to: str,
    max_zip_size: int = MAX_ZIP_SIZE_BYTES,
    max_uncompressed_size: int = MAX_TOTAL_EXTRACTED_BYTES
) -> None:
    """
    Safely extracts a ZIP archive preventing Zip Slip, path traversal, and zip bombs.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    zip_file_size = os.path.getsize(zip_path)
    if zip_file_size > max_zip_size:
        raise SecurityError(
            f"ZIP file size ({zip_file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit ({max_zip_size / (1024*1024):.1f} MB)"
        )

    resolved_extract_to = os.path.abspath(extract_to)
    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # Check for zip bomb
            total_uncompressed += member.file_size
            if total_uncompressed > max_uncompressed_size:
                raise SecurityError(
                    f"Total uncompressed ZIP size exceeds safety limit of {max_uncompressed_size / (1024*1024):.1f} MB"
                )

            # Check for Zip Slip / path traversal
            target_path = os.path.abspath(os.path.join(resolved_extract_to, member.filename))
            try:
                common = os.path.commonpath([resolved_extract_to, target_path])
            except ValueError:
                raise SecurityError(f"Path traversal detected across drives in ZIP member: {member.filename}")

            if common != resolved_extract_to:
                raise SecurityError(f"Zip Slip path traversal attempt detected in member: {member.filename}")

        # If all members are safe, extract all
        archive.extractall(resolved_extract_to)


def discover_student_reports(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Discovers student reports from an extracted batch directory.
    Supports:
      1. Top-level wrapper directory unwrapping (e.g. E1_SEC1/)
      2. Inner student ZIP archives (e.g. N240360_week1.zip, Week 1.zip, N241095.zip)
      3. Already extracted student folders (e.g. N240360/, 22001/)
      4. Flat files (e.g. N240360_observation.pdf)
      5. Smart multi-candidate resolution (prioritizing report/lab/week/pstc keywords and non-empty text)
      6. Multi-tier Student ID discovery: filename regex -> folder name -> PDF text scan
    """
    students: Dict[str, Dict[str, Any]] = {}
    base_dir = os.path.abspath(base_dir)

    # 1. Detect if base_dir contains a single wrapper directory (e.g. E1_SEC1)
    entries = [e for e in os.listdir(base_dir) if not e.startswith(".") and e != "_extracted_students"]
    batch_root = base_dir
    subdirs = [e for e in entries if os.path.isdir(os.path.join(base_dir, e))]
    non_dirs = [e for e in entries if not os.path.isdir(os.path.join(base_dir, e))]
    if len(subdirs) == 1 and len(non_dirs) == 0:
        batch_root = os.path.join(base_dir, subdirs[0])

    extracted_inner_dir = os.path.join(base_dir, "_extracted_students")
    os.makedirs(extracted_inner_dir, exist_ok=True)

    # 2. Iterate through items in batch_root
    for item in sorted(os.listdir(batch_root)):
        if item == "_extracted_students" or item.startswith(".") or item.startswith("__MACOSX"):
            continue

        item_path = os.path.join(batch_root, item)
        student_dir = None

        # Case A: Inner ZIP file per student
        if item.lower().endswith(".zip"):
            clean_stem = re.sub(r"[^\w\-]", "_", os.path.splitext(item)[0].strip())
            student_dir = os.path.join(extracted_inner_dir, clean_stem)
            os.makedirs(student_dir, exist_ok=True)
            try:
                validate_and_extract_zip(item_path, student_dir)
            except Exception as e:
                sid_match = re.search(r"[Nn]\d{6}", item)
                sid = sid_match.group(0).upper() if sid_match else clean_stem.upper()
                students[sid] = {
                    "file_path": None,
                    "error": f"Failed to extract inner ZIP {item}: {e}"
                }
                continue

        # Case B: Already extracted directory per student
        elif os.path.isdir(item_path):
            student_dir = item_path

        # Case C: Single flat observation file
        elif os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                sid_match = re.search(r"[Nn]\d{6}", item)
                sid = sid_match.group(0).upper() if sid_match else os.path.splitext(item)[0].split("_")[0].split("-")[0].strip().upper()
                if sid in students:
                    students[sid] = {
                        "file_path": None,
                        "error": f"Ambiguous: multiple observation reports found mapping to student ID {sid}"
                    }
                else:
                    students[sid] = {
                        "file_path": item_path,
                        "error": None
                    }
            continue

        if not student_dir or not os.path.exists(student_dir):
            continue

        # Find candidate report files inside the student directory
        candidates = []
        for root, _, files in os.walk(student_dir):
            if "__MACOSX" in root:
                continue
            for f in files:
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    candidates.append(os.path.join(root, f))

        # Detect Student ID
        # Priority 1: Check item name (zip or folder name)
        sid_match = re.search(r"[Nn]\d{6}", item)
        sid = sid_match.group(0).upper() if sid_match else None

        # Priority 2: Check subfolder names
        if not sid:
            for root, dirs, _ in os.walk(student_dir):
                for d in dirs:
                    m = re.search(r"[Nn]\d{6}", d)
                    if m:
                        sid = m.group(0).upper()
                        break
                if sid:
                    break

        # Priority 3: Check first page of candidate PDFs
        if not sid and candidates:
            for cf in candidates:
                if cf.lower().endswith(".pdf"):
                    try:
                        doc = pymupdf.open(cf)
                        if len(doc) > 0:
                            txt = doc[0].get_text()
                            m = re.search(r"[Nn]\d{6}", txt)
                            if m:
                                sid = m.group(0).upper()
                                break
                    except Exception:
                        pass

        # Fallback to sanitized item name
        if not sid:
            sid = re.sub(r"[^\w\-]", "_", os.path.splitext(item)[0].strip()).upper()

        # Handle duplicate student ID in the batch
        if sid in students:
            students[sid] = {
                "file_path": None,
                "error": f"Ambiguous: duplicate student ID {sid} found in batch archives"
            }
            continue

        # Resolve candidate report file
        best_file = None
        error_msg = None

        if not candidates:
            error_msg = "No observation report (PDF/Image) found in student submission"
        elif len(candidates) == 1:
            best_file = candidates[0]
        else:
            # Score candidates
            scored = []
            for c in candidates:
                score = 0
                low = os.path.basename(c).lower()
                if any(k in low for k in ["report", "lab", "week", "pstc", "observation"]):
                    score += 10
                if any(k in low for k in ["docscanner", "camscanner", "scan_", "scanner", "backup"]):
                    score -= 10
                if c.lower().endswith(".pdf"):
                    try:
                        doc = pymupdf.open(c)
                        if len(doc) > 0 and doc[0].get_text().strip():
                            score += 5
                    except Exception:
                        pass
                scored.append((score, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            if len(scored) >= 2 and scored[0][0] == scored[1][0] and scored[0][0] <= 0:
                rel_names = [os.path.basename(c) for c in candidates]
                error_msg = f"Ambiguous: found multiple report files: {', '.join(rel_names)}"
            else:
                best_file = scored[0][1]

        students[sid] = {
            "file_path": best_file,
            "error": error_msg
        }

    return students


def run_paddle_worker_sync(
    file_path: str,
    python_exe: Optional[str] = None,
    pages: Optional[str] = None,
    max_pages: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs paddle_worker.py in a dedicated subprocess to ensure CUDA isolation.
    Accepts optional page selection.
    """
    if python_exe is None:
        python_exe = get_paddle_python()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "paddle_worker.py")

    if not os.path.exists(worker_script):
        return {
            "success": False,
            "error": f"Worker script not found at {worker_script}"
        }

    cmd = [python_exe, worker_script, file_path]
    if pages:
        cmd.extend(["--pages", str(pages)])
    if max_pages and max_pages > 0:
        cmd.extend(["--max-pages", str(max_pages)])

    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300  # 5 minutes max per document
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "PaddleOCR-VL worker timed out after 300 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Subprocess invocation error: {str(e)}"
        }

    total_time = time.perf_counter() - start_time
    stdout = proc.stdout.strip()

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or stdout or f"Exited with code {proc.returncode}"
        return {
            "success": False,
            "error": f"PaddleOCR-VL failed: {err_msg[:400]}",
            "total_time": total_time
        }

    # Locate and parse JSON output
    parsed = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            cand = json.loads(line)
            if isinstance(cand, dict) and "success" in cand:
                parsed = cand
                break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        return {
            "success": False,
            "error": "Could not parse valid JSON output from PaddleOCR worker",
            "total_time": total_time
        }

    parsed["total_time"] = total_time
    return parsed


class BatchPipeline:
    """
    Orchestrates section-wise batch observation report extraction for student cohorts.
    Adheres strictly to OCR concurrency = 1 and LLM concurrency = 1.
    Maintains clean section and week isolation.
    """

    def __init__(
        self,
        week_id: str,
        section_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        python_exe: Optional[str] = None
    ):
        self.week_id = week_id.strip()
        self.section_id = section_id.strip() if section_id else None

        # Sanitize section_id against path traversal
        if self.section_id and (".." in self.section_id or "/" in self.section_id or "\\" in self.section_id):
            raise ValueError(f"Invalid characters in section_id: {self.section_id}")

        if self.section_id:
            base_out = output_dir or "output/sections"
            self.batch_output_dir = os.path.join(base_out, self.week_id, self.section_id)
            self.students_dir = os.path.join(self.batch_output_dir, "students")
            self.manifest_path = os.path.join(self.batch_output_dir, f"{self.section_id}_{self.week_id}_manifest.json")
            self.summary_path = os.path.join(self.batch_output_dir, f"{self.section_id}_{self.week_id}_summary.json")
            self.complete_json_path = os.path.join(self.batch_output_dir, f"{self.section_id}_{self.week_id}_observation_reports.json")
            self.zip_path = os.path.join(self.batch_output_dir, f"{self.section_id}_{self.week_id}_student_reports.zip")
        else:
            # Backwards-compatible layout for earlier tests without section_id
            base_out = output_dir or "output/batches"
            self.batch_output_dir = os.path.join(base_out, self.week_id)
            self.students_dir = self.batch_output_dir
            self.manifest_path = os.path.join(self.batch_output_dir, "batch_manifest.json")
            self.summary_path = os.path.join(self.batch_output_dir, f"{self.week_id}_summary.json")
            self.complete_json_path = os.path.join(self.batch_output_dir, f"{self.week_id}_observation_reports.json")
            self.zip_path = os.path.join(self.batch_output_dir, f"{self.week_id}_observation_reports.zip")

        os.makedirs(self.students_dir, exist_ok=True)
        self.ollama_url = ollama_url
        self.model = model
        self.temperature = temperature
        self.python_exe = python_exe or get_paddle_python()

        self.state: Dict[str, BatchStudentStatus] = {}
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Loads existing manifest state and rebuilds status from persisted student JSON files."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("students", []):
                        status_obj = BatchStudentStatus.model_validate(item)
                        self.state[status_obj.student_id] = status_obj
            except Exception:
                pass

        # Rebuild / verify state from persisted student JSON files in students directory
        if os.path.exists(self.students_dir):
            for fname in os.listdir(self.students_dir):
                if fname.endswith(".json") and not fname.endswith("_manifest.json") and not fname.endswith("_summary.json") and not fname.endswith("_reports.json"):
                    student_id = os.path.splitext(fname)[0]
                    full_path = os.path.join(self.students_dir, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        st = data.get("status")
                        if st == "completed":
                            # Validate completed schema
                            StudentObservationReport.model_validate(data)
                            self.state[student_id] = BatchStudentStatus(
                                student_id=student_id,
                                section_id=self.section_id,
                                status="completed",
                                stage=None,
                                output=fname,
                                error=None,
                                ocr_status="completed",
                                extraction_status="completed"
                            )
                        elif st == "failed":
                            err_info = data.get("error", {})
                            stage = err_info.get("stage") if isinstance(err_info, dict) else "general"
                            msg = err_info.get("message") if isinstance(err_info, dict) else str(err_info)
                            self.state[student_id] = BatchStudentStatus(
                                student_id=student_id,
                                section_id=self.section_id,
                                status="failed",
                                stage=stage,
                                output=fname,
                                error=msg,
                                ocr_status="completed" if stage == "extraction" else "failed",
                                extraction_status="failed" if stage == "extraction" else None
                            )
                    except Exception:
                        pass

    def save_manifest(self, current_section_status: Optional[str] = None) -> BatchManifest:
        """Computes summary statistics and saves manifest atomically."""
        students_list = list(self.state.values())
        successful = sum(1 for s in students_list if s.status == "completed")
        failed = sum(1 for s in students_list if s.status == "failed")
        pending = sum(1 for s in students_list if s.status not in ("completed", "failed"))
        total = len(students_list)

        if current_section_status:
            overall_status = current_section_status
        elif pending > 0:
            overall_status = "running" if (successful + failed) > 0 else "pending"
        elif failed > 0 and successful > 0:
            overall_status = "partial"
        elif failed > 0 and successful == 0 and total > 0:
            overall_status = "failed"
        elif successful == total and total > 0:
            overall_status = "completed"
        else:
            overall_status = "pending"

        manifest = BatchManifest(
            week_id=self.week_id,
            section_id=self.section_id,
            status=overall_status,
            total_students=total,
            successful=successful,
            failed=failed,
            pending=pending,
            students=students_list,
            generated_at=datetime.now().isoformat()
        )

        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def persist_failure(
        self,
        student_id: str,
        stage: str,
        error_message: str,
        status_entry: BatchStudentStatus,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchStudentStatus:
        """Persists a failed student JSON record atomically without fabricating fake observations."""
        output_filename = f"{student_id}.json"
        output_full_path = os.path.join(self.students_dir, output_filename)

        failed_report = FailedStudentReport(
            student_id=student_id,
            section_id=self.section_id or "SEC1",
            week_id=self.week_id,
            status="failed",
            error=FailedStudentError(stage=stage, message=error_message),
            created_at=datetime.now().isoformat()
        )
        atomic_write_json(output_full_path, failed_report)

        status_entry.status = "failed"
        status_entry.stage = stage
        status_entry.error = error_message
        status_entry.output = output_filename
        if stage == "extraction":
            status_entry.ocr_status = "completed"
            status_entry.extraction_status = "failed"
        else:
            status_entry.ocr_status = "failed"
            status_entry.extraction_status = None

        if progress_cb:
            progress_cb(status_entry)

        return status_entry

    def process_student(
        self,
        student_id: str,
        file_path: Optional[str],
        error_precheck: Optional[str] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchStudentStatus:
        """
        Processes a single student sequentially:
        Checks resume -> OCR (gpu:0) -> Qwen extraction -> Save JSON.
        Persists failures as real JSON records.
        """
        status_entry = self.state.get(student_id)
        if status_entry is None:
            status_entry = BatchStudentStatus(
                student_id=student_id,
                section_id=self.section_id,
                status="pending"
            )
            self.state[student_id] = status_entry

        def update(st: str, stage: Optional[str] = None, err: Optional[str] = None, out: Optional[str] = None):
            status_entry.status = st
            status_entry.stage = stage
            status_entry.error = err
            if out:
                status_entry.output = out
            if progress_cb:
                progress_cb(status_entry)

        # Handle pre-check errors (ambiguity, duplicates, or missing report file)
        if error_precheck:
            return self.persist_failure(
                student_id=student_id,
                stage="discovery",
                error_message=error_precheck,
                status_entry=status_entry,
                progress_cb=progress_cb
            )

        if not file_path or not os.path.exists(file_path):
            return self.persist_failure(
                student_id=student_id,
                stage="discovery",
                error_message=f"Report file not found: {file_path}",
                status_entry=status_entry,
                progress_cb=progress_cb
            )

        # Resume check: Check if valid completed JSON already exists
        output_filename = f"{student_id}.json"
        output_full_path = os.path.join(self.students_dir, output_filename)
        if os.path.exists(output_full_path):
            try:
                with open(output_full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "completed":
                    StudentObservationReport.model_validate(data)
                    update("completed", None, None, output_filename)
                    status_entry.ocr_status = "completed"
                    status_entry.extraction_status = "completed"
                    return status_entry
            except Exception:
                pass

        t_start = time.perf_counter()

        # Step 1: Run PaddleOCR-VL (sequential concurrency = 1)
        update("processing_ocr", "ocr")
        status_entry.ocr_status = "running"
        ocr_res = run_paddle_worker_sync(file_path, python_exe=self.python_exe)

        if not ocr_res.get("success", False):
            err_text = ocr_res.get("error", "Unknown OCR failure")
            status_entry.time_taken = time.perf_counter() - t_start
            return self.persist_failure(
                student_id=student_id,
                stage="ocr",
                error_message=err_text,
                status_entry=status_entry,
                progress_cb=progress_cb
            )

        ocr_text = str(ocr_res.get("text", "")).strip()
        if not ocr_text:
            status_entry.time_taken = time.perf_counter() - t_start
            return self.persist_failure(
                student_id=student_id,
                stage="ocr",
                error_message="OCR completed but returned empty text",
                status_entry=status_entry,
                progress_cb=progress_cb
            )

        update("ocr_complete", "ocr")
        status_entry.ocr_status = "completed"

        # Step 2: Run Qwen Structured Extraction (sequential concurrency = 1)
        update("extracting", "extraction")
        status_entry.extraction_status = "running"
        page_breakdown = ocr_res.get("page_breakdown", [])
        ext_res = extract_observation_report(
            report_text=ocr_text,
            base_url=self.ollama_url,
            model=self.model,
            temperature=self.temperature,
            page_breakdown=page_breakdown
        )

        if ext_res.status == "failed" and not ext_res.detected_programs:
            err_msg = "; ".join(ext_res.errors) if ext_res.errors else "Extraction failed"
            status_entry.time_taken = time.perf_counter() - t_start
            return self.persist_failure(
                student_id=student_id,
                stage="extraction",
                error_message=err_msg,
                status_entry=status_entry,
                progress_cb=progress_cb
            )

        # Step 3: Build Canonical Student Report
        source_meta = SourceMeta(
            filename=os.path.basename(file_path),
            num_pages=ocr_res.get("num_pages", 1)
        )
        ocr_obj = OcrResult(
            status="success",
            text=ocr_text,
            num_pages=ocr_res.get("num_pages", 1),
            page_breakdown=page_breakdown,
            device=ocr_res.get("device", "gpu:0"),
            paddle_version=ocr_res.get("paddle_version", "Unknown"),
            total_time=ocr_res.get("total_time", 0.0)
        )

        student_report = StudentObservationReport(
            student_id=student_id,
            section_id=self.section_id or "SEC1",
            week_id=self.week_id,
            status="completed",
            source=source_meta,
            ocr=ocr_obj,
            extraction=ext_res
        )

        # Atomic Save student JSON
        atomic_write_json(output_full_path, student_report)

        status_entry.time_taken = time.perf_counter() - t_start
        status_entry.ocr_status = "completed"
        status_entry.extraction_status = "completed"
        update("completed", None, None, output_filename)
        return status_entry

    def resume(
        self,
        discovered_students: Dict[str, Dict[str, Any]],
        selected_student_ids: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchManifest:
        """
        Resume incomplete work while skipping already successfully completed students.
        Optionally restricts processing to selected_student_ids.
        """
        # Register all discovered students
        for sid in discovered_students:
            if sid not in self.state:
                self.state[sid] = BatchStudentStatus(
                    student_id=sid,
                    section_id=self.section_id,
                    status="pending"
                )

        to_process = []
        for sid, info in discovered_students.items():
            if selected_student_ids is not None and sid not in selected_student_ids:
                continue
            current_st = self.state[sid].status
            if current_st != "completed":
                to_process.append((sid, info))

        for sid, info in to_process:
            self.process_student(
                student_id=sid,
                file_path=info.get("file_path"),
                error_precheck=info.get("error"),
                progress_cb=progress_cb
            )
            self.save_manifest()

        # Build complete JSON and summary after batch run
        self.build_complete_section_json()
        self.build_section_summary()
        return self.save_manifest()

    def retry_failed(
        self,
        discovered_students: Dict[str, Dict[str, Any]],
        selected_student_ids: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchManifest:
        """
        Processes ONLY students whose persisted status is strictly 'failed'.
        Optionally restricts processing to selected_student_ids.
        """
        to_retry = []
        for sid, info in discovered_students.items():
            if selected_student_ids is not None and sid not in selected_student_ids:
                continue
            st_entry = self.state.get(sid)
            if st_entry and st_entry.status == "failed":
                to_retry.append((sid, info))

        for sid, info in to_retry:
            self.process_student(
                student_id=sid,
                file_path=info.get("file_path"),
                error_precheck=info.get("error"),
                progress_cb=progress_cb
            )
            self.save_manifest()

        # Rebuild complete JSON and summary
        self.build_complete_section_json()
        self.build_section_summary()
        return self.save_manifest()

    def run_batch(
        self,
        discovered_students: Dict[str, Dict[str, Any]],
        selected_student_ids: Optional[List[str]] = None,
        retry_only_failed: bool = False,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchManifest:
        """
        Dispatches to resume (standard execution) or retry_failed with optional student filtering.
        """
        if retry_only_failed:
            return self.retry_failed(discovered_students, selected_student_ids=selected_student_ids, progress_cb=progress_cb)
        else:
            return self.resume(discovered_students, selected_student_ids=selected_student_ids, progress_cb=progress_cb)

    def build_complete_section_json(self) -> CompleteSectionReport:
        """
        Compiles all student reports (both completed and failed) from the students directory
        into the complete aggregated section JSON file. Preserves full raw OCR and extraction data.
        """
        students_data: List[Dict[str, Any]] = []
        completed_count = 0
        failed_count = 0
        seen_sids = set()

        if os.path.exists(self.students_dir):
            for fname in sorted(os.listdir(self.students_dir)):
                if fname.endswith(".json") and not fname.endswith("_manifest.json") and not fname.endswith("_summary.json") and not fname.endswith("_reports.json"):
                    full_path = os.path.join(self.students_dir, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            entry_data = json.load(f)
                        sid = entry_data.get("student_id", os.path.splitext(fname)[0])
                        seen_sids.add(sid)
                        st = entry_data.get("status")
                        if st == "completed":
                            completed_count += 1
                        elif st == "failed":
                            failed_count += 1
                        students_data.append(entry_data)
                    except Exception:
                        pass

        # Also account for any registered students that are still pending
        for sid, status_obj in self.state.items():
            if sid not in seen_sids:
                students_data.append({
                    "student_id": sid,
                    "section_id": self.section_id or "SEC1",
                    "week_id": self.week_id,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                })

        # Sort students cleanly by student ID
        students_data.sort(key=lambda s: str(s.get("student_id", "")))

        total = len(students_data)
        pending_count = max(0, total - completed_count - failed_count)

        complete_report = CompleteSectionReport(
            section_id=self.section_id or "SEC1",
            week_id=self.week_id,
            total_students=total,
            successful=completed_count,
            failed=failed_count,
            pending=pending_count,
            students=students_data,
            generated_at=datetime.now().isoformat()
        )

        atomic_write_json(self.complete_json_path, complete_report)
        return complete_report

    def build_section_summary(self) -> SectionSummary:
        """
        Computes section-level summary statistics including exact P1..P10 detection tallies
        derived directly from persisted student JSON files. Never asks LLM for counts.
        """
        program_counts = {f"P{i}": 0 for i in range(1, 11)}
        successful = 0
        failed = 0

        if os.path.exists(self.students_dir):
            for fname in os.listdir(self.students_dir):
                if fname.endswith(".json") and not fname.endswith("_manifest.json") and not fname.endswith("_summary.json") and not fname.endswith("_reports.json"):
                    full_path = os.path.join(self.students_dir, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("status") == "completed":
                            successful += 1
                            ext = data.get("extraction", {})
                            progs = ext.get("programs", {})
                            for pkey, pdet in progs.items():
                                if isinstance(pdet, dict) and pdet.get("status") == "detected":
                                    if pkey in program_counts:
                                        program_counts[pkey] += 1
                        elif data.get("status") == "failed":
                            failed += 1
                    except Exception:
                        pass

        total = len(self.state) if self.state else (successful + failed)
        pending = max(0, total - successful - failed)

        summary = SectionSummary(
            section_id=self.section_id or "SEC1",
            week_id=self.week_id,
            total_students=total,
            successful=successful,
            failed=failed,
            pending=pending,
            program_detection=program_counts,
            generated_at=datetime.now().isoformat()
        )

        atomic_write_json(self.summary_path, summary)
        return summary

    def create_batch_zip(self) -> str:
        """
        Creates the section archive containing all individual student JSON files,
        manifest, summary, and complete section JSON.
        """
        self.save_manifest()
        self.build_complete_section_json()
        self.build_section_summary()

        with zipfile.ZipFile(self.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Include manifest
            if os.path.exists(self.manifest_path):
                zf.write(self.manifest_path, arcname=os.path.basename(self.manifest_path))

            # Include summary
            if os.path.exists(self.summary_path):
                zf.write(self.summary_path, arcname=os.path.basename(self.summary_path))

            # Include complete section JSON
            if os.path.exists(self.complete_json_path):
                zf.write(self.complete_json_path, arcname=os.path.basename(self.complete_json_path))

            # Include individual student JSON files
            if os.path.exists(self.students_dir):
                for fname in sorted(os.listdir(self.students_dir)):
                    if (
                        fname.endswith(".json")
                        and not fname.endswith("_manifest.json")
                        and not fname.endswith("_summary.json")
                        and not fname.endswith("_reports.json")
                        and fname != "batch_manifest.json"
                    ):
                        full_path = os.path.join(self.students_dir, fname)
                        arc_name = fname if self.students_dir == self.batch_output_dir else os.path.join("students", fname)
                        zf.write(full_path, arcname=arc_name)

        return self.zip_path
