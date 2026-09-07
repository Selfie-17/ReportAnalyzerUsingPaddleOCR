# PaddleOCR-VL & Section-Wise Batch Observation Report Extraction Engine

A high-performance local platform for extracting text from student laboratory observation reports using **PaddleOCR-VL 1.6 on GPU (gpu:0)**, transforming raw OCR into strictly validated canonical JSON with **Ollama Qwen 2.5 Coder 3B**, and performing safe, resumable **Section-Wise Batch Extraction for student cohorts across Sections 1 to 6 (SEC1–SEC6)**.

---

## 🌟 Key Architecture & Section Pipeline

```text
               Section ZIP Archive (e.g. SEC1.zip, SEC2.zip)
                                     │
                                     ▼
                      [Safe ZIP Unpacker & Validator]
                      (Zip Slip & path traversal defense)
                                     │
                                     ▼
                           Student Discovery Engine
                    (Flags ambiguities, duplicate IDs, missing)
                                     │
                                     ▼
                    Sequential Batch Execution Engine
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
          PaddleOCR-VL 1.6                   Qwen 2.5 Coder 3B
      (GPU:0, Concurrency = 1)          (Ollama JSON, Concurrency = 1)
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                          Strict Pydantic Validation
                        (Canonical Schemas & Failures)
                                     │
                                     ▼
        Isolated Section Storage (output/sections/<week_id>/<section_id>/)
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    Individual Students     Complete Section JSON       Section Summary
    students/<sid>.json    <sec>_<week>_reports.json  <sec>_<week>_summary.json
    (Completed & Failed)      (100% full reports)       (P1..P10 exact counts)
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                         Section Reports ZIP Archive
                       <sec>_<week>_student_reports.zip
```

### Core Highlights
- **Section-Aware Pipeline (SEC1–SEC6)**: Explicit section tagging in every report, directory structure, and manifest. Supports cohorts of ~60 students (or any count) per section without hardcoding.
- **Section & Week Isolation**: Dedicated output hierarchy `output/sections/<week_id>/<section_id>/students/<student_id>.json`. The same student ID in `SEC1` vs `SEC2` or `week-01` vs `week-02` are completely isolated.
- **Strict Non-Hallucination Policy**: Converts OCR text into structured format without inventing, assuming, or improving student answers. Missing sections or programs evaluate strictly to `null` or `not_detected`.
- **Ground-Truth Source Page Tracking**: Automatically associates original page numbers (`source_pages: [1, 2]`) to each detected program from OCR segmentation—never guessed by the LLM.
- **Hardware-Aware Concurrency**: Empirically benchmarked on NVIDIA RTX 4050 (6 GB VRAM) where 1 PaddleOCR-VL worker consumes **5.9 GB VRAM**. OCR and LLM concurrency are strictly set to **1** to prevent Out-Of-Memory crashes.
- **Safe Cohort ZIP Handling**: Defense against Zip Slip, directory traversal, and uncompressed bomb limits.
- **Distinct Resume vs. Retry Failed**:
  - `Resume`: Skips valid, completed student JSONs; processes only pending or incomplete students.
  - `Retry Failed`: Processes strictly students whose persisted status is `failed`. Skips completed and pending students!
- **Atomic JSON Writes**: All JSON writes use a temporary file with `os.replace` and `os.fsync` to guarantee zero corruption upon sudden interruption.
- **Complete Accounting in Master Section JSON**: Failed students never disappear. The complete section JSON (`<section_id>_<week_id>_observation_reports.json`) contains 100% of student entries, with failed entries preserving the stage and error details.
- **Direct Section Summaries**: Tallies program detection counts (`P1`–`P10`) directly from student JSON records. Never asks the LLM to invent counts.

---

## 📋 Data Schemas & Deliverables

### 1. Student Observation Report (`<student_id>.json`)
Conforms to `StudentObservationReport`:
```json
{
  "student_id": "22001",
  "section_id": "SEC1",
  "week_id": "week-01",
  "status": "completed",
  "source": {
    "filename": "observation.pdf",
    "num_pages": 1
  },
  "ocr": {
    "status": "success",
    "text": "...",
    "num_pages": 1,
    "page_breakdown": [{"page": 1, "text": "..."}],
    "device": "gpu:0",
    "paddle_version": "3.3.0",
    "total_time": 195.65
  },
  "extraction": {
    "status": "success",
    "objective_of_lab": null,
    "programs": {
      "P1": {
        "status": "detected",
        "source_pages": [1],
        "problem_understanding": "Check whether a given number is even or odd",
        "logic_approach": "Reads an integer, checks if number % 2 == 0",
        "important_variables": [{"variable": "num", "purpose": "Input integer value"}],
        "what_i_observed": "Program successfully executed"
      },
      "P6": {
        "status": "not_detected",
        "source_pages": [],
        "problem_understanding": null,
        "logic_approach": null,
        "important_variables": [],
        "what_i_observed": null
      }
    },
    "detected_programs": ["P1", "P2", "P3", "P4", "P5"],
    "missing_programs": ["P6", "P7", "P8", "P9", "P10"],
    "errors": []
  },
  "created_at": "2026-09-03T21:44:54.052848"
}
```

### 2. Failed Student Report (`<student_id>.json` on failure)
Conforms to `FailedStudentReport`:
```json
{
  "student_id": "22003",
  "section_id": "SEC1",
  "week_id": "week-01",
  "status": "failed",
  "error": {
    "stage": "discovery",
    "message": "No observation report (PDF/Image) found in student directory"
  },
  "created_at": "2026-09-03T21:44:54.052848"
}
```

### 3. Complete Section JSON (`<section_id>_<week_id>_observation_reports.json`)
Aggregates all students in the section while preserving 100% full student report content, raw OCR, and failed entries:
```json
{
  "section_id": "SEC1",
  "week_id": "week-01",
  "total_students": 60,
  "successful": 58,
  "failed": 2,
  "pending": 0,
  "students": [ ... ],
  "generated_at": "2026-09-03T22:34:50.123456"
}
```

### 4. Section Summary (`<section_id>_<week_id>_summary.json`)
Calculates exact detection statistics across the cohort:
```json
{
  "section_id": "SEC1",
  "week_id": "week-01",
  "total_students": 60,
  "successful": 58,
  "failed": 2,
  "pending": 0,
  "program_detection": {
    "P1": 58,
    "P2": 58,
    "P3": 54,
    "P4": 52,
    "P5": 48,
    "P6": 0,
    "P7": 0,
    "P8": 0,
    "P9": 0,
    "P10": 0
  },
  "generated_at": "2026-09-03T22:34:50.123456"
}
```

---

## 🛠️ Requirements & Environment

- **OS**: Windows 10/11 with NVIDIA GPU (CUDA enabled)
- **Python**: 3.11 with conda environment (e.g. `paddle_vl`)
- **GPU Packages**: `paddlepaddle-gpu==3.3.0`, `paddleocr>=2.8`
- **Application Packages**: `streamlit`, `pymupdf`, `requests`, `pydantic>=2.0`, `pytest`
- **Ollama**: Running locally on `http://127.0.0.1:11434` with model `qwen2.5-coder:3b`

---

## 🚀 Usage Guide

### 1. Launch the Streamlit Web Application

Activate your CUDA-enabled Python environment:

```powershell
conda activate paddle_vl
streamlit run app.py
```

### 2. Section-Wise Batch Workflow (SEC1–SEC6)

1. Select the **📦 Section-Wise Batch Extraction (SEC1–SEC6)** tab.
2. Select your **Week Identifier** (`week-01`) and **Section Identifier** (`SEC1` through `SEC6`).
3. Upload the section ZIP archive (e.g. `SEC1.zip`).
   - The engine validates against Zip Slip, unpacks to a safe temp folder, and runs discovery.
   - Pre-flight confirmation is displayed:
     ```text
     Section: SEC1 | Week: week-01 | Students found: 60 (58 valid, 2 flagged) | Ready to process.
     ```
4. Click **🚀 Start Section Batch** to begin sequential GPU OCR and LLM extraction.
5. Monitor live progress:
   - Live stage indicator displays active student and stage (e.g. `Stage: OCR (GPU:0)` vs `Stage: Structured extraction (Qwen)`).
   - Real-time student status table with OCR (`✓`/`✗`) and Extraction (`✓`/`✗`) indicators.
6. If interrupted, click **🔄 Resume Incomplete** (skips already completed students).
7. If any students failed, resolve the report files and click **⚠️ Retry Failed Only** (processes strictly failed students).
8. Download section deliverables:
   - **📥 Download Complete Section JSON** (`SEC1_week-01_observation_reports.json`)
   - **📥 Download Section Manifest** (`SEC1_week-01_manifest.json`)
   - **📊 Download Section Summary** (`SEC1_week-01_summary.json`)
   - **📦 Download Student Reports ZIP** (`SEC1_week-01_student_reports.zip`)
9. Inspect any individual student using the interactive **Student Report Inspector**.

---

## 📋 Observation Report Verification Engine (Instruction Manual & Assigned Questions)

The platform includes an advanced verification engine that evaluates extracted observation reports against the official **"Instruction Manual for Writing the Observation Report"** submission guide and instructor-assigned lab questions:

1. **Instructor Assigned Questions Input**:
   - Instructors can input the specific problems/questions assigned for the lab session before evaluation.
   - Includes a 1-click **Manual Preset** (`Program 1: Factors of a Number`, `Program 2: Factorial of a Number`, `Program 3: Palindrome Number`, `Program 4: Prime Number`, `Program 5: Fibonacci Series`).
2. **Submission Guide Comparison**:
   - Evaluates against the 5 core sections (each scored out of 2.0, total 10.0 points):
     - `Objective of the Lab` (2.0 pts): Session-wide purpose, concepts, problem types, skills.
     - `Problem Understanding` (2.0 pts): Student's own words (input, calculation, output); no C code.
     - `Logic / Approach Used` (2.0 pts): Conceptual step-by-step reasoning (loops, conditions, value changes); no syntax copying.
     - `Important Variables Table` (2.0 pts): Markdown table (`| Variable | Purpose |`).
     - `What I Observed` (2.0 pts): Genuine observations on execution behavior; penalizes generic statements.
3. **Question Coverage & Integrity Checks**:
   - Cross-references the report against the assigned questions to generate a **Coverage Analysis Table**.
   - Checks answers to the **5 Core Questions** and flags academic integrity violations (full C code dumping).
4. **Exportable Deliverables**:
   - Token-streamed live evaluation report with 1-click markdown download (`<student_id>_evaluation.md`).

---

## 🧪 Testing & Verification

The test suite includes 37 comprehensive automated tests:

```powershell
conda activate paddle_vl
python -m pytest tests/ -v
```

### Test Coverage Summary:
- **Security & Extraction**: Zip Slip defense, path traversal prevention, bomb limits, markdown fence stripping, 1-shot self-healing, ground-truth source page assignment.
- **Section Workflow**: Explicit `section_id`, complete section aggregation, failure persistence, section isolation, week isolation, resume skipping completed, retry targeting only failed, duplicate ID rejection, section summary calculations.
- **Verification Engine**: Official Instruction Manual prompt formatting, assigned questions message construction, token streaming, synchronous evaluation, and Ollama connection health checks.
- **End-to-End Integration**: 3-student cohort integration run and 60-student full cohort benchmark via `python run_section_integration_test.py`.
