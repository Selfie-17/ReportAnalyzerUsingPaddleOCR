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
import io
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
    HolisticEvaluationResult,
    CalculatedScore,
    LLMJudgeEvaluation,
)
from extractor import extract_observation_report, DEFAULT_OLLAMA_URL, DEFAULT_MODEL
from verifier import (
    verify_observation_report_sync,
    format_extraction_for_evaluation,
    parse_evaluation_scores,
    find_student_code,
    calculate_deterministic_score,
    evaluate_with_ollama,
    evaluate_with_gemini,
)

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
        single_subdir = subdirs[0]
        # Only unwrap if the single subdir is not a student directory (e.g., 22003, N240360)
        if not re.match(r"^[a-zA-Z]?\d{4,7}$", single_subdir):
            batch_root = os.path.join(base_dir, single_subdir)

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
                        "student_dir": batch_root,
                        "code_files": [],
                        "code_count": 0,
                        "error": f"Ambiguous: multiple observation reports found mapping to student ID {sid}"
                    }
                else:
                    students[sid] = {
                        "file_path": item_path,
                        "student_dir": batch_root,
                        "code_files": [],
                        "code_count": 0,
                        "error": None
                    }
            continue

        if not student_dir or not os.path.exists(student_dir):
            continue

        # Find candidate report files inside the student directory
        candidates = []
        code_files = []
        for root, _, files in os.walk(student_dir):
            if "__MACOSX" in root:
                continue
            for f in sorted(files):
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    candidates.append(os.path.join(root, f))
                elif ext in (".c", ".cpp", ".py", ".h", ".java"):
                    code_files.append(os.path.join(root, f))

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
                "student_dir": student_dir,
                "code_files": code_files,
                "code_count": len(code_files),
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
            "student_dir": student_dir,
            "code_files": code_files,
            "code_count": len(code_files),
            "error": error_msg
        }

    # Check for flat C files directly inside batch_root
    flat_code_files = []
    for f in sorted(os.listdir(batch_root)):
        if f.startswith(".") or f.startswith("__MACOSX"):
            continue
        fp = os.path.join(batch_root, f)
        if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in (".c", ".cpp", ".py", ".h", ".java"):
            flat_code_files.append(fp)

    if flat_code_files:
        if len(students) == 1:
            only_sid = next(iter(students.keys()))
            students[only_sid]["student_dir"] = students[only_sid].get("student_dir") or batch_root
            existing = students[only_sid].get("code_files", [])
            for f in flat_code_files:
                if f not in existing:
                    existing.append(f)
            students[only_sid]["code_files"] = existing
            students[only_sid]["code_count"] = len(existing)
        else:
            for sid in students:
                matched = [f for f in flat_code_files if sid.lower() in os.path.basename(f).lower()]
                if matched:
                    students[sid]["student_dir"] = students[sid].get("student_dir") or batch_root
                    existing = students[sid].get("code_files", [])
                    for f in matched:
                        if f not in existing:
                            existing.append(f)
                    students[sid]["code_files"] = existing
                    students[sid]["code_count"] = len(existing)

    return students


def run_paddle_worker_sync(
    file_path: str,
    python_exe: Optional[str] = None,
    pages: Optional[str] = None,
    max_pages: Optional[int] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs paddle_worker.py in a dedicated subprocess to ensure CUDA isolation.
    Accepts optional page selection.
    timeout=None runs until completion without timing out (matching normal extraction).
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
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if timeout is not None and timeout > 0:
            run_kwargs["timeout"] = timeout
        proc = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"PaddleOCR-VL worker timed out after {timeout} seconds"
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
        python_exe: Optional[str] = None,
        assigned_questions: Optional[str] = None,
        instruction_manual: Optional[str] = None,
        ocr_timeout: Optional[int] = None,
        enable_evaluation: bool = True,
        **kwargs
    ):
        self.week_id = week_id.strip()
        self.section_id = section_id.strip() if section_id else None
        self.python_exe = python_exe or kwargs.get("python_exe") or get_paddle_python()
        q_val = assigned_questions if assigned_questions is not None else kwargs.get("assigned_questions")
        self.assigned_questions = q_val.strip() if (q_val and isinstance(q_val, str)) else None
        man_val = instruction_manual if instruction_manual is not None else kwargs.get("instruction_manual")
        self.instruction_manual = man_val.strip() if (man_val and isinstance(man_val, str)) else None
        self.ocr_timeout = ocr_timeout if ocr_timeout is not None else kwargs.get("ocr_timeout", None)
        self.enable_evaluation = enable_evaluation if enable_evaluation is not None else kwargs.get("enable_evaluation", True)
        self.evaluate_extracted_text = kwargs.get("evaluate_extracted_text", True)

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
                            has_eval = bool(data.get("evaluation"))
                            self.state[student_id] = BatchStudentStatus(
                                student_id=student_id,
                                section_id=self.section_id,
                                status="completed",
                                stage=None,
                                output=fname,
                                error=None,
                                ocr_status="completed",
                                extraction_status="completed",
                                evaluation_status="completed" if has_eval else None
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
        file_path: Optional[str] = None,
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

        # Resume check: Check if valid completed JSON already exists
        output_filename = f"{student_id}.json"
        output_full_path = os.path.join(self.students_dir, output_filename)
        if os.path.exists(output_full_path):
            try:
                with open(output_full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "completed":
                    StudentObservationReport.model_validate(data)

                    # If evaluation is enabled and missing or incomplete (missing 5 criteria breakdown), run evaluation directly on existing extracted data
                    cur_eval = data.get("evaluation")
                    eval_scores = parse_evaluation_scores(cur_eval) if cur_eval else {}
                    needs_eval = not cur_eval or eval_scores.get("objective") == "—"
                    if self.enable_evaluation and needs_eval and data.get("extraction"):
                        update("evaluating", "evaluation")
                        status_entry.ocr_status = "completed"
                        status_entry.extraction_status = "completed"
                        setattr(status_entry, "evaluation_status", "running")

                        ext_dict = data.get("extraction", {})
                        ocr_txt = data.get("ocr", {}).get("text", "")
                        eval_text = ocr_txt.strip() if ocr_txt and ocr_txt.strip() else format_extraction_for_evaluation(ext_dict, fallback_text=ocr_txt)
                        s_dir = os.path.dirname(file_path) if file_path else None
                        student_code = find_student_code(student_id=student_id, week_id=self.week_id, student_dir=s_dir)
                        try:
                            eval_md = verify_observation_report_sync(
                                report_text=eval_text,
                                base_url=self.ollama_url,
                                model=self.model,
                                temperature=self.temperature,
                                student_code=student_code,
                                extracted_report=ext_dict,
                                student_id=student_id,
                                week_id=self.week_id
                            )
                            setattr(status_entry, "evaluation_status", "completed")
                            data["evaluation"] = eval_md
                            holistic_res = getattr(verify_observation_report_sync, "last_holistic_result", None)
                            if isinstance(holistic_res, (HolisticEvaluationResult, dict)):
                                data["holistic_evaluation"] = holistic_res.model_dump() if hasattr(holistic_res, "model_dump") else (holistic_res.dict() if hasattr(holistic_res, "dict") else holistic_res)
                            with open(output_full_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            eval_md_path = os.path.join(self.students_dir, f"{student_id}_evaluation.md")
                            with open(eval_md_path, "w", encoding="utf-8") as f:
                                f.write(eval_md)
                        except Exception:
                            setattr(status_entry, "evaluation_status", "failed")

                    update("completed", None, None, output_filename)
                    status_entry.ocr_status = "completed"
                    status_entry.extraction_status = "completed"
                    setattr(status_entry, "evaluation_status", "completed" if data.get("evaluation") else None)
                    return status_entry
            except Exception:
                pass

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

        t_start = time.perf_counter()

        # Step 1: Run PaddleOCR-VL (sequential concurrency = 1)
        update("processing_ocr", "ocr")
        status_entry.ocr_status = "running"
        ocr_kwargs = {"python_exe": self.python_exe}
        if self.ocr_timeout is not None:
            ocr_kwargs["timeout"] = self.ocr_timeout
        ocr_res = run_paddle_worker_sync(file_path, **ocr_kwargs)

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
        ext_kwargs = {
            "report_text": ocr_text,
            "base_url": self.ollama_url,
            "model": self.model,
            "temperature": self.temperature,
            "page_breakdown": page_breakdown,
        }
        if self.assigned_questions:
            ext_kwargs["assigned_questions"] = self.assigned_questions
        ext_res = extract_observation_report(**ext_kwargs)

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

        status_entry.extraction_status = "completed"

        # Step 3: Run Qwen Model Evaluation against Lab Manual & Questions (concurrency = 1)
        eval_md: Optional[str] = None
        if self.enable_evaluation:
            update("evaluating", "evaluation")
            setattr(status_entry, "evaluation_status", "running")
            if getattr(self, "evaluate_extracted_text", True):
                eval_text = format_extraction_for_evaluation(ext_res, fallback_text=ocr_text)
            else:
                eval_text = ocr_text.strip() if ocr_text and ocr_text.strip() else format_extraction_for_evaluation(ext_res, fallback_text=ocr_text)
            s_dir = os.path.dirname(file_path) if file_path else None
            student_code = find_student_code(student_id=student_id, week_id=self.week_id, student_dir=s_dir)
            try:
                eval_md = verify_observation_report_sync(
                    report_text=eval_text,
                    base_url=self.ollama_url,
                    model=self.model,
                    temperature=self.temperature,
                    student_code=student_code,
                    extracted_report=ext_res,
                    student_id=student_id,
                    week_id=self.week_id
                )
                setattr(status_entry, "evaluation_status", "completed")
            except Exception as e:
                eval_md = f"⚠️ Evaluation error: {str(e)}"
                setattr(status_entry, "evaluation_status", "failed")

        # Step 4: Build Canonical Student Report
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

        holistic_eval = getattr(verify_observation_report_sync, "last_holistic_result", None)
        if not isinstance(holistic_eval, (HolisticEvaluationResult, dict)):
            holistic_eval = None
        student_report = StudentObservationReport(
            student_id=student_id,
            section_id=self.section_id or "SEC1",
            week_id=self.week_id,
            status="completed",
            source=source_meta,
            ocr=ocr_obj,
            extraction=ext_res,
            evaluation=eval_md,
            holistic_evaluation=holistic_eval
        )

        # Atomic Save student JSON
        atomic_write_json(output_full_path, student_report)

        # Also save standalone student evaluation markdown file
        if eval_md:
            eval_md_path = os.path.join(self.students_dir, f"{student_id}_evaluation.md")
            try:
                with open(eval_md_path, "w", encoding="utf-8") as f:
                    f.write(eval_md)
            except Exception:
                pass

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
            needs_eval = self.enable_evaluation and (getattr(self.state[sid], "evaluation_status", None) != "completed")
            if current_st != "completed" or needs_eval:
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
            is_failed = st_entry and st_entry.status == "failed"
            eval_failed = self.enable_evaluation and st_entry and getattr(st_entry, "evaluation_status", None) == "failed"
            if is_failed or eval_failed:
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

    def rerun_evaluations_on_extracted(
        self,
        discovered_students: Optional[Dict[str, Dict[str, Any]]] = None,
        selected_student_ids: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> BatchManifest:
        """
        Re-runs Qwen rubric evaluation directly on already-extracted student text.
        Skips re-running PaddleOCR and Qwen structured extraction when extraction
        already exists on disk.
        Updates student JSON files, standalone evaluation markdown files,
        section aggregated JSON, section summary, and manifest.
        """
        if discovered_students is None:
            discovered_students = {}
            if os.path.exists(self.students_dir):
                for fname in os.listdir(self.students_dir):
                    if fname.endswith(".json") and not any(fname.endswith(sfx) for sfx in ("_manifest.json", "_summary.json", "_reports.json", "_observation_reports.json")):
                        sid = os.path.splitext(fname)[0]
                        discovered_students[sid] = {"file_path": os.path.join(self.students_dir, fname)}

        for sid, info in discovered_students.items():
            if selected_student_ids is not None and sid not in selected_student_ids:
                continue

            status_entry = self.state.setdefault(
                sid,
                BatchStudentStatus(
                    student_id=sid,
                    section_id=self.section_id,
                    status="pending"
                )
            )

            out_json = os.path.join(self.students_dir, f"{sid}.json")
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        student_data = json.load(f)

                    ext_data = student_data.get("extraction")
                    ocr_data = student_data.get("ocr", {})
                    ocr_text = ocr_data.get("text", "") if isinstance(ocr_data, dict) else ""

                    if ext_data and isinstance(ext_data, dict) and ext_data.get("status") in ("success", "partial"):
                        status_entry.status = "evaluating"
                        status_entry.stage = "evaluation"
                        setattr(status_entry, "evaluation_status", "running")
                        if progress_cb:
                            progress_cb(status_entry)

                        eval_text = format_extraction_for_evaluation(ext_data, fallback_text=ocr_text)
                        s_info = discovered_students.get(sid, {}) if discovered_students else {}
                        s_dir = s_info.get("student_dir") or (os.path.dirname(s_info.get("file_path")) if s_info.get("file_path") else None)
                        student_code = find_student_code(student_id=sid, week_id=self.week_id, student_dir=s_dir)
                        try:
                            eval_md = verify_observation_report_sync(
                                report_text=eval_text,
                                base_url=self.ollama_url,
                                model=self.model,
                                temperature=self.temperature,
                                student_code=student_code,
                                extracted_report=ext_data,
                                student_id=sid,
                                week_id=self.week_id
                            )
                            setattr(status_entry, "evaluation_status", "completed")
                        except Exception as e:
                            eval_md = f"⚠️ Evaluation error: {str(e)}"
                            setattr(status_entry, "evaluation_status", "failed")

                        student_data["evaluation"] = eval_md
                        holistic_res = getattr(verify_observation_report_sync, "last_holistic_result", None)
                        if isinstance(holistic_res, (HolisticEvaluationResult, dict)):
                            student_data["holistic_evaluation"] = holistic_res.model_dump() if hasattr(holistic_res, "model_dump") else (holistic_res.dict() if hasattr(holistic_res, "dict") else holistic_res)
                        atomic_write_json(out_json, student_data)

                        # Write standalone markdown
                        eval_md_path = os.path.join(self.students_dir, f"{sid}_evaluation.md")
                        try:
                            with open(eval_md_path, "w", encoding="utf-8") as f:
                                f.write(eval_md or "")
                        except Exception:
                            pass

                        status_entry.status = "completed"
                        status_entry.stage = None
                        status_entry.error = None
                        if progress_cb:
                            progress_cb(status_entry)
                        self.save_manifest()
                        continue
                except Exception:
                    # Fallback to standard process_student if JSON parsing or update failed
                    pass

            # If student does not have existing extraction, run standard process_student
            self.process_student(
                student_id=sid,
                file_path=info.get("file_path"),
                error_precheck=info.get("error"),
                progress_cb=progress_cb
            )
            self.save_manifest()

        # Rebuild section deliverables
        self.build_complete_section_json()
        self.build_section_summary()
        self.create_batch_zip()
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
                if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json", "_observation_report.json", "_evaluations.json", "_ollama_evaluation.json", "_gemini_evaluation.json", "_brief_evaluation.json")):
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
        # Initialize program detection dynamically from assigned questions or fallback to P1..P10
        from verifier import parse_assigned_questions
        assigned_q_list = parse_assigned_questions(self.assigned_questions) if self.assigned_questions else []
        if assigned_q_list:
            program_counts = {f"P{q.question_number}": 0 for q in assigned_q_list}
        else:
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
                                    if pkey not in program_counts:
                                        program_counts[pkey] = 0
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

            # Include individual student JSON and evaluation MD files
            if os.path.exists(self.students_dir):
                for fname in sorted(os.listdir(self.students_dir)):
                    if (
                        (
                            fname.endswith(".json")
                            and not fname.endswith("_manifest.json")
                            and not fname.endswith("_summary.json")
                            and not fname.endswith("_reports.json")
                            and fname != "batch_manifest.json"
                        )
                        or fname.endswith("_evaluation.md")
                    ):
                        full_path = os.path.join(self.students_dir, fname)
                        arc_name = fname if self.students_dir == self.batch_output_dir else os.path.join("students", fname)
                        zf.write(full_path, arcname=arc_name)

        return self.zip_path

    # ============================================================
    # 4-STEP PIPELINE: UNAMBIGUOUS STEP EXECUTION METHODS
    # ============================================================

    def step1_ingest_zip(
        self,
        zip_source: Any,
        target_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Step 1: Ingest and validate cohort ZIP archive.
        Discovers student observation reports and C source code files.
        """
        unpacked_dir = target_dir or os.path.join(self.batch_output_dir, "unpacked_cohort")
        os.makedirs(unpacked_dir, exist_ok=True)

        # Handle Streamlit UploadedFile, bytes, or file path string
        if hasattr(zip_source, "read"):
            temp_zip = os.path.join(self.batch_output_dir, "input_cohort.zip")
            with open(temp_zip, "wb") as f:
                f.write(zip_source.read())
            validate_and_extract_zip(temp_zip, extract_to=unpacked_dir)
        elif isinstance(zip_source, bytes):
            temp_zip = os.path.join(self.batch_output_dir, "input_cohort.zip")
            with open(temp_zip, "wb") as f:
                f.write(zip_source)
            validate_and_extract_zip(temp_zip, extract_to=unpacked_dir)
        else:
            validate_and_extract_zip(str(zip_source), extract_to=unpacked_dir)

        self.unpacked_dir = unpacked_dir
        discovered = discover_student_reports(unpacked_dir)
        self.discovered_students = discovered

        # Register in manifest state
        for sid, info in discovered.items():
            st_dir = info.get("student_dir")
            c_files = info.get("code_files", [])
            c_count = info.get("code_count", len(c_files))
            if sid not in self.state:
                self.state[sid] = BatchStudentStatus(
                    student_id=sid,
                    section_id=self.section_id,
                    status="pending",
                    ocr_status="pending",
                    calculated_score_status="pending",
                    ollama_evaluation_status="pending",
                    gemini_evaluation_status="pending",
                    student_dir=st_dir,
                    code_files=c_files,
                    code_count=c_count
                )
            else:
                self.state[sid].student_dir = st_dir
                self.state[sid].code_files = c_files
                self.state[sid].code_count = c_count

        self.save_manifest()
        return {
            "unpacked_dir": unpacked_dir,
            "total_students": len(discovered),
            "students": discovered
        }

    def step2_run_ocr(
        self,
        discovered_students: Optional[Dict[str, Dict[str, Any]]] = None,
        selected_student_ids: Optional[List[str]] = None,
        force_rerun: bool = False,
        ocr_engine: str = "paddleocr",
        gemini_model: Optional[str] = None,
        gemini_keys: Optional[Union[List[str], str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> Dict[str, Any]:
        """
        Step 2: Run OCR on discovered student reports.
        Supports dual engines:
          - "paddleocr": PaddleOCR-VL 1.6 on local GPU
          - "gemini": Google Gemini API Multimodal OCR with multi-key rotation and failover
        Extracts raw text and per-page breakdown into {student_id}.json.
        """
        students_map = discovered_students or getattr(self, "discovered_students", None)
        if not students_map:
            # Fallback to scanning unpacked directory or student files
            students_map = {}
            if hasattr(self, "unpacked_dir") and os.path.exists(self.unpacked_dir):
                students_map = discover_student_reports(self.unpacked_dir)

        # Prepare Gemini Key Manager if Gemini engine is selected
        gemini_km = None
        target_gemini_model = gemini_model or os.environ.get("GEMINI_OCR_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if ocr_engine.lower() == "gemini":
            from gemini_ocr import GeminiKeyManager, get_gemini_key_manager, extract_text_with_gemini
            gemini_km = GeminiKeyManager(api_keys=gemini_keys) if gemini_keys else get_gemini_key_manager()

        results = {}
        for sid, info in students_map.items():
            if selected_student_ids and sid not in selected_student_ids:
                continue

            status_entry = self.state.setdefault(
                sid,
                BatchStudentStatus(student_id=sid, section_id=self.section_id, status="pending")
            )

            file_path = info.get("file_path")
            out_json = os.path.join(self.students_dir, f"{sid}.json")

            # Check if valid OCR text already exists unless force_rerun is requested
            if not force_rerun and os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        cur_data = json.load(f)
                    ocr_data = cur_data.get("ocr", {})
                    ocr_txt = ocr_data.get("text", "").strip()
                    if ocr_txt:
                        status_entry.ocr_status = "completed"
                        results[sid] = {
                            "success": True,
                            "cached": True,
                            "text_length": len(ocr_txt),
                            "engine": ocr_data.get("engine", "paddleocr")
                        }
                        if progress_cb:
                            progress_cb(status_entry)
                        continue
                except Exception:
                    pass

            if not file_path or not os.path.exists(file_path):
                status_entry.ocr_status = "failed"
                status_entry.error = "Report document not found"
                results[sid] = {"success": False, "error": status_entry.error}
                if progress_cb:
                    progress_cb(status_entry)
                continue

            # Run Selected OCR engine
            status_entry.status = "processing_ocr"
            status_entry.ocr_status = "running"
            if progress_cb:
                progress_cb(status_entry)

            if ocr_engine.lower() == "gemini":
                from gemini_ocr import extract_text_with_gemini
                ocr_res = extract_text_with_gemini(
                    file_path=file_path,
                    model=target_gemini_model,
                    key_manager=gemini_km
                )
            else:
                ocr_res = run_paddle_worker_sync(
                    file_path=file_path,
                    python_exe=self.python_exe,
                    timeout=self.ocr_timeout
                )

            if not ocr_res.get("success", False):
                status_entry.ocr_status = "failed"
                status_entry.error = ocr_res.get("error", "OCR failed")
                results[sid] = {"success": False, "error": status_entry.error}
            else:
                ocr_text = str(ocr_res.get("text", "")).strip()
                status_entry.ocr_status = "completed"
                status_entry.error = None
                results[sid] = {
                    "success": True,
                    "text_length": len(ocr_text),
                    "engine": ocr_engine.lower()
                }

                st_info = students_map.get(sid, {})
                st_dir = st_info.get("student_dir") or (os.path.dirname(file_path) if file_path else None)
                c_files = st_info.get("code_files", [])
                if not c_files and st_dir and os.path.exists(st_dir):
                    c_files = [os.path.join(root, f) for root, _, files in os.walk(st_dir) for f in sorted(files) if f.endswith((".c", ".cpp", ".py", ".h", ".java"))]

                # Extract Stage 1 canonical ObservationReport from OCR text immediately
                from extractor import extract_observation_report_stage1
                obs_report = extract_observation_report_stage1(ocr_text)
                obs_dict = obs_report.model_dump()
                ext_dict = obs_report.to_extraction_result().model_dump()

                # Save standalone Stage 1 ObservationReport JSON artifact
                obs_out_json = os.path.join(self.students_dir, f"{sid}_observation_report.json")
                atomic_write_json(obs_out_json, obs_dict)

                total_pages_detected = ocr_res.get("num_pages") or ocr_res.get("total_pages", len(ocr_res.get("page_breakdown", [])))

                # Update or initialize student JSON
                student_payload = {
                    "student_id": sid,
                    "section_id": self.section_id or "SEC1",
                    "week_id": self.week_id,
                    "status": "completed",
                    "student_dir": st_dir,
                    "code_files": [os.path.basename(cf) for cf in c_files],
                    "code_count": len(c_files),
                    "source": {
                        "filename": os.path.basename(file_path),
                        "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        "mime_type": "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"
                    },
                    "ocr": {
                        "engine": ocr_engine.lower(),
                        "model": ocr_res.get("model", target_gemini_model) if ocr_engine.lower() == "gemini" else "paddleocr-vl-1.6",
                        "text": ocr_text,
                        "page_breakdown": ocr_res.get("page_breakdown", []),
                        "total_pages": total_pages_detected,
                        "total_time": ocr_res.get("total_time", 0.0)
                    },
                    "observation_report": obs_dict,
                    "extraction": ext_dict
                }

                # If existing JSON has evaluation or scores, preserve them
                if os.path.exists(out_json):
                    try:
                        with open(out_json, "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                        for k in ("calculated_score", "ollama_evaluation", "gemini_evaluation", "evaluation", "evaluation_report"):
                            if k in old_data:
                                student_payload[k] = old_data[k]
                    except Exception:
                        pass

                atomic_write_json(out_json, student_payload)

            if progress_cb:
                progress_cb(status_entry)

        self.save_manifest()
        return results

    def step3_calculate_scores(
        self,
        discovered_students: Optional[Dict[str, Dict[str, Any]]] = None,
        selected_student_ids: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> Dict[str, CalculatedScore]:
        """
        Step 3: Deterministic Rubric Score Calculation.
        Computes 100% reproducible rubric scores (/10) for all students based on OCR evidence,
        question matching, and C code static checks. Zero LLM hallucination.
        """
        students_map = discovered_students or getattr(self, "discovered_students", {})
        scores = {}

        # If discovered_students is empty, gather from students_dir
        if not students_map and os.path.exists(self.students_dir):
            for fname in os.listdir(self.students_dir):
                if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json")):
                    sid = os.path.splitext(fname)[0]
                    students_map[sid] = {"file_path": os.path.join(self.students_dir, fname)}

        for sid, info in students_map.items():
            if selected_student_ids and sid not in selected_student_ids:
                continue

            status_entry = self.state.setdefault(
                sid,
                BatchStudentStatus(student_id=sid, section_id=self.section_id, status="pending")
            )
            status_entry.status = "calculating_score"
            status_entry.stage = "scoring"
            if progress_cb:
                progress_cb(status_entry)

            out_json = os.path.join(self.students_dir, f"{sid}.json")
            ocr_text = ""
            ext_data = None
            student_data = {}

            if os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        student_data = json.load(f)
                    ocr_text = student_data.get("ocr", {}).get("text", "")
                    ext_data = student_data.get("extraction")
                except Exception:
                    pass

            s_dir = info.get("student_dir") or student_data.get("student_dir") or (os.path.dirname(info.get("file_path")) if info.get("file_path") else None)
            student_code = find_student_code(student_id=sid, week_id=self.week_id, student_dir=s_dir)

            # Deterministic Score Calculation
            calc_res = calculate_deterministic_score(
                student_id=sid,
                ocr_text=ocr_text,
                extracted_report=ext_data,
                student_code=student_code,
                week_id=self.week_id,
                assigned_questions=self.assigned_questions
            )

            scores[sid] = calc_res
            status_entry.calculated_score = calc_res.total_score
            status_entry.calculated_score_status = "completed"
            status_entry.stage = None
            status_entry.status = "completed"

            # Save in student JSON
            student_data["calculated_score"] = calc_res.model_dump()
            atomic_write_json(out_json, student_data)

            if progress_cb:
                progress_cb(status_entry)

        self.save_manifest()
        self.build_scores_summary_csv()
        return scores

    def step4_run_llm_judges(
        self,
        providers: List[str] = ["ollama", "gemini"],
        discovered_students: Optional[Dict[str, Dict[str, Any]]] = None,
        selected_student_ids: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None,
        skip_existing: bool = True,
        gemini_model: Optional[str] = None,
        gemini_key: Optional[str] = None,
        auto_build_artifacts: bool = True,
    ) -> Dict[str, Dict[str, LLMJudgeEvaluation]]:
        """
        Step 4: LLM as Judge Evaluation.
        Runs Ollama and/or Gemini independently to generate two separate reports and recommended scores.
        Neither model sees the other's score.
        Supports resume/skip_existing and per-student error isolation.
        """
        students_map = discovered_students or getattr(self, "discovered_students", {})
        if not students_map and os.path.exists(self.students_dir):
            for fname in os.listdir(self.students_dir):
                if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json")):
                    sid = os.path.splitext(fname)[0]
                    students_map[sid] = {"file_path": os.path.join(self.students_dir, fname)}

        results: Dict[str, Dict[str, LLMJudgeEvaluation]] = {}

        for sid, info in students_map.items():
            if selected_student_ids and sid not in selected_student_ids:
                continue

            status_entry = self.state.setdefault(
                sid,
                BatchStudentStatus(student_id=sid, section_id=self.section_id, status="pending")
            )

            out_json = os.path.join(self.students_dir, f"{sid}.json")
            student_data = {}
            ocr_text = ""
            ext_data = None
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        student_data = json.load(f)
                    ocr_text = student_data.get("ocr", {}).get("text", "")
                    ext_data = student_data.get("observation_report") or student_data.get("extraction")
                except Exception:
                    pass

            # Ensure observation report has actual detected programs; extract if missing
            num_entries = 0
            if isinstance(ext_data, dict):
                if ext_data.get("entries"):
                    num_entries = len(ext_data["entries"])
                elif ext_data.get("detected_programs"):
                    num_entries = len(ext_data["detected_programs"])
                elif ext_data.get("programs"):
                    num_entries = len(ext_data["programs"])

            if num_entries == 0 and ocr_text:
                from extractor import extract_observation_report_stage1
                obs_rep = extract_observation_report_stage1(ocr_text)
                student_data["observation_report"] = obs_rep.model_dump()
                student_data["extraction"] = obs_rep.to_extraction_result().model_dump()
                ext_data = student_data["observation_report"]
                atomic_write_json(out_json, student_data)
                obs_file = os.path.join(self.students_dir, f"{sid}_observation_report.json")
                atomic_write_json(obs_file, student_data["observation_report"])

            s_dir = info.get("student_dir") or student_data.get("student_dir") or (os.path.dirname(info.get("file_path")) if info.get("file_path") else None)
            student_code = find_student_code(student_id=sid, week_id=self.week_id, student_dir=s_dir)

            results[sid] = {}

            # 1. Ollama Judge
            if "ollama" in providers:
                ollama_md_path = os.path.join(self.students_dir, f"{sid}_ollama_evaluation.md")
                has_valid_ollama = (
                    skip_existing
                    and os.path.exists(ollama_md_path)
                    and os.path.getsize(ollama_md_path) > 1000
                    and "ollama_evaluation" in student_data
                    and student_data["ollama_evaluation"].get("recommended_score") is not None
                )
                if has_valid_ollama:
                    status_entry.ollama_score = student_data["ollama_evaluation"].get("recommended_score")
                    status_entry.ollama_evaluation_status = "completed"
                else:
                    status_entry.status = "evaluating_ollama"
                    status_entry.ollama_evaluation_status = "running"
                    if progress_cb:
                        progress_cb(status_entry)

                    try:
                        ollama_eval = evaluate_with_ollama(
                            student_id=sid,
                            report_text=ocr_text,
                            extracted_report=ext_data,
                            student_code=student_code,
                            assigned_questions=self.assigned_questions,
                            base_url=self.ollama_url,
                            model=self.model,
                            temperature=self.temperature,
                            week_id=self.week_id
                        )
                        results[sid]["ollama"] = ollama_eval
                        ollama_dump = ollama_eval.model_dump()
                        ollama_dump["ocr_extracted_text"] = ocr_text
                        ollama_dump["ocr_text"] = ocr_text
                        student_data["ollama_evaluation"] = ollama_dump
                        student_data["evaluation"] = ollama_eval.full_report_markdown

                        with open(ollama_md_path, "w", encoding="utf-8") as f:
                            f.write(ollama_eval.full_report_markdown)

                        if getattr(ollama_eval, "evaluation_report", None):
                            ollama_json_path = os.path.join(self.students_dir, f"{sid}_ollama_evaluation.json")
                            atomic_write_json(ollama_json_path, ollama_eval.evaluation_report)

                        status_entry.ollama_score = ollama_eval.recommended_score
                        status_entry.ollama_evaluation_status = "completed"
                    except Exception as e:
                        safe_print(f"Error evaluating student {sid} with Ollama: {e}")
                        status_entry.ollama_evaluation_status = "failed"

            # 2. Gemini Judge
            if "gemini" in providers:
                gemini_md_path = os.path.join(self.students_dir, f"{sid}_gemini_evaluation.md")
                has_full_gemini = False
                if skip_existing and os.path.exists(gemini_md_path) and os.path.getsize(gemini_md_path) > 3000:
                    try:
                        with open(gemini_md_path, "r", encoding="utf-8") as f:
                            c_txt = f.read()
                        if "## 1. Submission Summary" in c_txt and "## 7. Final Score" in c_txt:
                            has_full_gemini = True
                    except Exception:
                        has_full_gemini = False

                if has_full_gemini and "gemini_evaluation" in student_data and student_data["gemini_evaluation"].get("recommended_score") is not None:
                    status_entry.gemini_score = student_data["gemini_evaluation"].get("recommended_score")
                    status_entry.gemini_evaluation_status = "completed"
                else:
                    status_entry.status = "evaluating_gemini"
                    status_entry.gemini_evaluation_status = "running"
                    if progress_cb:
                        progress_cb(status_entry)

                    try:
                        target_g_key = gemini_key or getattr(self, "gemini_key", None) or os.environ.get("GEMINI_API_KEY", "")
                        target_g_model = gemini_model or getattr(self, "gemini_model", None) or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

                        gemini_eval = evaluate_with_gemini(
                            student_id=sid,
                            report_text=ocr_text,
                            extracted_report=ext_data,
                            student_code=student_code,
                            assigned_questions=self.assigned_questions,
                            api_key=target_g_key,
                            model_name=target_g_model,
                            temperature=self.temperature,
                            week_id=self.week_id
                        )
                        results[sid]["gemini"] = gemini_eval
                        gemini_dump = gemini_eval.model_dump()
                        gemini_dump["ocr_extracted_text"] = ocr_text
                        gemini_dump["ocr_text"] = ocr_text
                        student_data["gemini_evaluation"] = gemini_dump

                        with open(gemini_md_path, "w", encoding="utf-8") as f:
                            f.write(gemini_eval.full_report_markdown)

                        if getattr(gemini_eval, "evaluation_report", None):
                            gemini_json_path = os.path.join(self.students_dir, f"{sid}_gemini_evaluation.json")
                            atomic_write_json(gemini_json_path, gemini_eval.evaluation_report)

                        status_entry.gemini_score = gemini_eval.recommended_score
                        status_entry.gemini_evaluation_status = "completed"
                    except Exception as e:
                        safe_print(f"Error evaluating student {sid} with Gemini: {e}")
                        status_entry.gemini_evaluation_status = "failed"

            status_entry.status = "completed"
            atomic_write_json(out_json, student_data)

            if progress_cb:
                progress_cb(status_entry)

        if auto_build_artifacts:
            self.save_manifest()
            self.build_scores_summary_csv()
            self.create_model_reports_zip("ollama")
            self.create_model_reports_zip("gemini")
            self.build_provider_section_json("ollama")
            self.build_provider_section_json("gemini")
        return results

    def build_scores_summary_csv(self) -> str:
        """
        Builds a comprehensive comparison CSV showing Step 3 Calculated Score vs Step 4 Ollama vs Step 4 Gemini.
        """
        rows = []
        if os.path.exists(self.students_dir):
            for fname in sorted(os.listdir(self.students_dir)):
                if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json")):
                    sid = os.path.splitext(fname)[0]
                    j_path = os.path.join(self.students_dir, fname)
                    try:
                        with open(j_path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        calc = d.get("calculated_score", {})
                        o_eval = d.get("ollama_evaluation", {})
                        g_eval = d.get("gemini_evaluation", {})

                        rows.append({
                            "Student ID": sid,
                            "Detected Programs": len(calc.get("detected_programs", [])),
                            "Step 3 Calculated Score": calc.get("total_score", "—"),
                            "Calculated Grade": calc.get("grade", "—"),
                            "Calculated Status": calc.get("status", "—"),
                            "Step 4 Ollama Score": o_eval.get("recommended_score", "—"),
                            "Ollama Grade": o_eval.get("grade", "—"),
                            "Step 4 Gemini Score": g_eval.get("recommended_score", "—"),
                            "Gemini Grade": g_eval.get("grade", "—"),
                            "Consensus Verdict": "Approved" if (calc.get("status") == "Approved" or o_eval.get("status") == "Approved" or g_eval.get("status") == "Approved") else "Needs Revision"
                        })
                    except Exception:
                        pass

        csv_path = os.path.join(self.batch_output_dir, f"{self.section_id or 'SEC1'}_{self.week_id}_scores_comparison.csv")
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path

    def create_model_reports_zip(self, provider: str = "ollama") -> str:
        """
        Packages individual markdown reports for a specific provider into a ZIP.
        """
        zip_fname = f"{self.section_id or 'SEC1'}_{self.week_id}_{provider}_reports.zip"
        target_zip = os.path.join(self.batch_output_dir, zip_fname)
        suffix = f"_{provider}_evaluation.md"

        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if os.path.exists(self.students_dir):
                for fname in sorted(os.listdir(self.students_dir)):
                    if fname.endswith(suffix):
                        full_path = os.path.join(self.students_dir, fname)
                        zf.write(full_path, arcname=fname)

        return target_zip

    def build_provider_section_json(self, provider: str = "ollama") -> str:
        """
        Compiles an overall section aggregated JSON containing evaluations for a specific provider (Ollama or Gemini).
        Saved as {section_id}_{week_id}_{provider}_evaluations.json.
        """
        out_fname = f"{self.section_id or 'SEC1'}_{self.week_id}_{provider}_evaluations.json"
        target_json = os.path.join(self.batch_output_dir, out_fname)

        students_evaluations: Dict[str, Any] = {}
        scores: List[float] = []

        if os.path.exists(self.students_dir):
            for fname in sorted(os.listdir(self.students_dir)):
                if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json", "_observation_report.json", "_evaluations.json", "_ollama_evaluation.json", "_gemini_evaluation.json", "_brief_evaluation.json")):
                    full_path = os.path.join(self.students_dir, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            sdata = json.load(f)
                        sid = sdata.get("student_id", os.path.splitext(fname)[0])
                        key = f"{provider}_evaluation"
                        peval = sdata.get(key)
                        if peval:
                            peval_copy = dict(peval)
                            ocr_obj = sdata.get("ocr")
                            ocr_text = ""
                            if isinstance(ocr_obj, dict):
                                ocr_text = ocr_obj.get("text", "")
                            elif isinstance(ocr_obj, str):
                                ocr_text = ocr_obj
                            if not ocr_text:
                                ocr_text = sdata.get("ocr_text", "") or sdata.get("ocr_extracted_text", "") or peval.get("ocr_extracted_text", "") or peval.get("ocr_text", "")

                            peval_copy["student_id"] = sid
                            peval_copy["ocr_extracted_text"] = ocr_text
                            peval_copy["ocr_text"] = ocr_text
                            if isinstance(ocr_obj, dict) and ocr_obj:
                                peval_copy["ocr"] = ocr_obj

                            if isinstance(peval_copy.get("evaluation_report"), dict):
                                peval_copy["evaluation_report"]["ocr_extracted_text"] = ocr_text
                                peval_copy["evaluation_report"]["ocr_text"] = ocr_text

                            students_evaluations[sid] = peval_copy
                            rec = peval.get("recommended_score")
                            if rec is not None:
                                try:
                                    scores.append(float(rec))
                                except (ValueError, TypeError):
                                    pass
                    except Exception:
                        pass

        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        section_payload = {
            "schema_version": "2.0",
            "section_id": self.section_id or "SEC1",
            "week_id": self.week_id,
            "provider": provider,
            "total_students": len(self.state) or len(students_evaluations),
            "evaluated_students": len(students_evaluations),
            "average_score": avg_score,
            "generated_at": datetime.now().isoformat(),
            "students": students_evaluations
        }

        atomic_write_json(target_json, section_payload)
        return target_json

    def run_all_steps(
        self,
        zip_source: Optional[Union[str, bytes, io.BytesIO]] = None,
        ocr_engine: str = "paddleocr",
        providers: List[str] = ["ollama", "gemini"],
        progress_cb: Optional[Callable[[BatchStudentStatus], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes all 4 steps end-to-end sequentially:
        Step 1: Ingest Zip
        Step 2: Do OCR (PaddleOCR or Gemini)
        Step 3: Calculate Scores
        Step 4: LLM as Judge (Dual Reports: Ollama & Gemini)
        """
        if zip_source:
            self.step1_ingest_zip(zip_source)

        self.step2_run_ocr(ocr_engine=ocr_engine, progress_cb=progress_cb)
        self.step3_calculate_scores(progress_cb=progress_cb)
        self.step4_run_llm_judges(providers=providers, progress_cb=progress_cb)
        self.build_complete_section_json()
        self.build_section_summary()
        self.create_batch_zip()
        self.build_provider_section_json("ollama")
        self.build_provider_section_json("gemini")

        return {
            "status": "completed",
            "section_id": self.section_id,
            "week_id": self.week_id,
            "manifest": self.save_manifest()
        }

