# 🔬 PaddleOCR-VL 1.6 & Section-Wise Observation Report Extraction Engine

A high-performance, privacy-first local platform for extracting, validating, and grading student laboratory observation reports using **PaddleOCR-VL 1.6 on GPU (gpu:0)**, transforming raw OCR into strictly validated canonical JSON with **Ollama Qwen 2.5 Coder 3B**, and performing safe, resumable **Section-Wise Batch Extraction & Dynamic Rubric Evaluation across Sections 1 to 6 (SEC1–SEC6)**.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PaddlePaddle GPU 3.3.0](https://img.shields.io/badge/PaddlePaddle--GPU-3.3.0-brightgreen.svg)](https://www.paddlepaddle.org.cn/)
[![PaddleOCR 3.7.0](https://img.shields.io/badge/PaddleOCR-3.7.0-green.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![Ollama Qwen 2.5 Coder 3B](https://img.shields.io/badge/Ollama-Qwen%202.5%20Coder%203B-purple.svg)](https://ollama.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-114%20Passing-success.svg)](file:///c:/Users/kampa/OneDrive/Desktop/ReportTest/tests)

---

## 📖 Table of Contents
- [🌟 Key Architecture & Section Pipeline](#-key-architecture--section-pipeline)
- [✨ Core Capabilities](#-core-capabilities)
- [📋 Data Schemas & Deliverables](#-data-schemas--deliverables)
- [🛠️ Requirements & Quick Setup](#️-requirements--quick-setup)
- [🚀 Usage Guide (Streamlit Web UI)](#-usage-guide-streamlit-web-ui)
- [📊 Evaluation Engine & Rubric Scoring](#-evaluation-engine--rubric-scoring)
- [🧪 Automated Test Suite (114 Tests)](#-automated-test-suite-114-tests)
- [📁 Project Structure](#-project-structure)

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
                       Dynamic Rubric Evaluation Layer
               (Verbatim OCR Grounding & Deterministic Scoring)
                                     │
                                     ▼
        Isolated Section Storage (output/sections/<week_id>/<section_id>/)
                                     │
     ┌───────────────────────┬───────┴───────────────┬───────────────────────┐
     ▼                       ▼                       ▼                       ▼
Individual Students   Evaluation Markdown   Complete Section JSON       Section Summary
students/<sid>.json   students/<sid>_eval.md <sec>_<week>_reports.json  <sec>_<week>_summary.json
(Completed & Failed)   (Full Rubric Review)   (100% full reports)       (P1..P13 exact counts)
     │                       │                       │                       │
     └───────────────────────┼───────────────────────┼───────────────────────┘
                             │                       │
                             ▼                       ▼
                 Section Reports ZIP Archive   Cohort Scores Table
               <sec>_<week>_student_reports.zip <sec>_<week>_scores.csv
               (All JSONs & Evaluation MDs)    (Grades, Scores, Breakdown)
```

---

## ✨ Core Capabilities

- **Section-Aware Pipeline (SEC1–SEC6)**: Explicit section tagging in every report, directory hierarchy, and manifest. Accommodates standard cohorts of ~60 students (or any count) per section without hardcoded limits.
- **Section & Week Isolation**: Dedicated output hierarchy `output/sections/<week_id>/<section_id>/students/<student_id>.json`. The same student ID across `SEC1` vs `SEC2` or `week-01` vs `week-02` is completely isolated.
- **Subprocess CUDA Isolation (`paddle_worker.py`)**: Runs PaddleOCR-VL 1.6 in a dedicated isolated worker process to ensure clean GPU VRAM allocation, automatic memory cleanup, and bypass Windows `bfloat16` capability issues.
- **Strict Non-Hallucination Policy**: Structured extraction never invents, assumes, or improves student answers. Absent sections or programs evaluate strictly to `null` or `not_detected`.
- **Dynamic Assigned Questions & Manuals**: Runtime parsing supports any number of lab questions (e.g. 5-program manual preset, 13-program Week 1 preset from `questions_13.json`, or arbitrary user questions).
- **Verbatim Evidence Grounding (`verifier.py`)**: All claims evaluated by the LLM are checked against raw OCR text via Python post-validation (`is_evidence_grounded`). Unverified claims are automatically flagged or discounted.
- **Deterministic Python-Side Scoring**: LLM identifies criteria status (`PRESENT`, `PARTIAL`, `MISSING`), while final grades, category totals, and percentages are computed deterministically in Python to prevent mathematical hallucinations.
- **Token & Context Optimizations**:
  - Automatically converts verbose HTML tables in OCR output into compact Markdown tables (`_clean_ocr_text_for_llm`), saving substantial LLM prompt tokens.
  - Expanded context window (`num_ctx: 16384`) in Ollama to seamlessly process long multi-page reports.
  - Flexible schema parsing handling lists, dictionaries, aliases, and nested variable objects.
- **Hardware-Aware Concurrency**: Empirically benchmarked on NVIDIA RTX 4050 (6 GB VRAM) where 1 PaddleOCR-VL worker consumes **~5.9 GB VRAM**. OCR and LLM concurrency are strictly set to **1** to prevent Out-Of-Memory crashes.
- **Safe Cohort ZIP Handling**: Complete defense against Zip Slip, absolute path traversal, and uncompressed bomb limits.
- **Distinct Resume vs. Retry Failed vs. Re-evaluate**:
  - `Resume Incomplete`: Skips valid, completed student JSONs; processes only pending or interrupted students.
  - `Retry Failed Only`: Processes strictly students whose persisted status is `failed`. Skips completed and pending students.
  - `Rerun Evaluation Only`: Re-evaluates rubric grading directly on existing extracted data without re-doing GPU OCR (`rerun_evaluations_on_extracted`).
- **Atomic JSON Writes**: All file persistence uses unique temporary files with `os.fsync` and `os.replace` to guarantee zero corruption upon sudden interruption.
- **Complete Accounting in Master Section JSON**: Failed students never disappear. The complete section JSON (`<section_id>_<week_id>_observation_reports.json`) contains 100% of student entries, with failed entries preserving the stage and error details.
- **Cohort Scores Table & CSV Export**: Streamlit provides a live aggregated cohort grading table displaying Student ID, Total Score, Grade, Verdict, Questions Covered, 5-criterion breakdown, Cohort Metrics (Average, Highest, Approved count), and a 1-click CSV download.

---

## 📋 Data Schemas & Deliverables

### 1. Student Observation Report (`students/<student_id>.json`)
Conforms to `StudentObservationReport`:
```json
{
  "student_id": "N240046",
  "section_id": "SEC1",
  "week_id": "week-01",
  "status": "completed",
  "source": {
    "filename": "observation.pdf",
    "num_pages": 4
  },
  "ocr": {
    "status": "success",
    "text": "...",
    "num_pages": 4,
    "page_breakdown": [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}],
    "device": "gpu:0",
    "paddle_version": "3.3.0",
    "total_time": 42.15
  },
  "extraction": {
    "status": "success",
    "objective_of_lab": "Understand and implement fundamental C programming constructs",
    "programs": {
      "P1": {
        "status": "detected",
        "source_pages": [1],
        "problem_understanding": "Check whether a given number is even or odd",
        "logic_approach": "Reads an integer, checks if number % 2 == 0",
        "important_variables": [{"variable": "num", "purpose": "Input integer value"}],
        "what_i_observed": "Program successfully evaluated test cases"
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
    "missing_programs": ["P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13"],
    "errors": []
  },
  "evaluation": "# Observation Report Evaluation\n...",
  "dynamic_evaluation": { ... },
  "created_at": "2026-09-08T19:40:12.102938"
}
```

### 2. Standalone Evaluation Markdown (`students/<student_id>_evaluation.md`)
Generated for every student and bundled in the section ZIP deliverable:
- Submission status badges (Total Marks, Grade, Coverage, Integrity Verdict).
- Section-by-section rubric evaluation table (Objective, Problem Understanding, Logic, Variables Table, Observations).
- Assigned Question Coverage Analysis.
- Verbatim grounded evidence quotes from student OCR text.
- Actionable recommendations for student improvement.

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
  "generated_at": "2026-09-08T20:15:30.451230"
}
```

### 4. Section Summary (`<section_id>_<week_id>_summary.json`)
Deterministic program detection counts tallied directly in Python:
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
    "P6": 40,
    "P7": 38,
    "P8": 35,
    "P9": 32,
    "P10": 28,
    "P11": 25,
    "P12": 22,
    "P13": 20
  },
  "generated_at": "2026-09-08T20:15:30.451230"
}
```

### 5. Section Manifest (`<section_id>_<week_id>_manifest.json`)
Tracks operational status, timings, errors, and granular stages (`ocr_status`, `extraction_status`, `evaluation_status`) for every student.

### 6. Section Deliverables ZIP (`<section_id>_<week_id>_student_reports.zip`)
Contains all individual student JSON files and evaluation Markdown files ready for instructor archiving or LMS distribution.

---

## 🛠️ Requirements & Quick Setup

> [!TIP]
> For detailed instructions, prerequisites, and troubleshooting, see the [Complete Setup Guide (SETUP.md)](file:///c:/Users/kampa/OneDrive/Desktop/ReportTest/SETUP.md) or [Quick Reference (HOW_TO_SETUP.md)](file:///c:/Users/kampa/OneDrive/Desktop/ReportTest/HOW_TO_SETUP.md).

### 1. Environment & Hardware
- **OS**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA (tested on RTX 4050 / RTX 3060 / 4060, minimum 6GB VRAM)
- **Python**: Python 3.11 in Conda environment (`paddle_vl`)
- **Ollama**: Running locally on `http://127.0.0.1:11434` with model `qwen2.5-coder:3b`

### 2. Quick Setup Commands

```powershell
# 1. Pull Ollama LLM
ollama pull qwen2.5-coder:3b

# 2. Create Conda environment
conda create -n paddle_vl python=3.11 -y
conda activate paddle_vl

# 3. Install PaddlePaddle GPU (for CUDA 11.8 or 12.x)
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 4. Install Project Requirements
python -m pip install -r requirements.txt

# 5. Run Automated Tests
python -m pytest tests/ -v
```

---

## 🚀 Usage Guide (Streamlit Web UI)

### 1. Launch the Application

```powershell
conda activate paddle_vl
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Single Student Workflow (Tab 1)
1. **Upload Document**: Upload a single PDF or image. View page thumbnails with the interactive slider.
2. **Step 2 (OCR)**: Click **🚀 Run PaddleOCR-VL 1.6** to execute GPU text extraction.
3. **Step 3 (Review)**: Review or edit OCR text directly in the UI.
4. **Step 4 (Structured Extraction)**: Click **🤖 Extract Structured Report with Qwen**.
5. **Step 5 (Validation)**: Inspect the canonical Pydantic JSON structure.
6. **Step 6 (Evaluation)**: Run the dynamic verification rubric with live token streaming and download the evaluation report (`<student_id>_evaluation.md`).

### 3. Section-Wise Batch Workflow (Tab 2: SEC1–SEC6)
1. **Configure Scope**: Select the **Week** (`week-01` to `week-16`) and **Section** (`SEC1` to `SEC6`).
2. **Upload Cohort ZIP**: Upload `SEC1.zip`. Safe unpacking discovers valid students and flags anomalies.
3. **Set Runtime Questions / Rubric**: Choose presets (Week 1 13 programs or Manual 5 programs) or enter custom lab questions.
4. **Run Batch Pipeline**:
   - **🚀 Start Section Batch**: Runs sequential GPU OCR, Structured Extraction, and Rubric Evaluation.
   - **🔄 Resume Incomplete**: Safely picks up where it left off.
   - **⚠️ Retry Failed Only**: Retries only students that previously failed.
   - **🔄 Rerun Evaluation Only**: Re-grades already extracted reports without repeating OCR.
5. **Download Deliverables**:
   - Complete Section JSON (`<sec>_<week>_observation_reports.json`)
   - Section Manifest (`<sec>_<week>_manifest.json`)
   - Section Summary (`<sec>_<week>_summary.json`)
   - Student Reports ZIP (`<sec>_<week>_student_reports.zip`)
6. **Inspect Cohort Scores Table**:
   - View student-by-student scores, grades, verdicts, question coverage, and criteria marks.
   - Click **📥 Download Cohort Scores Summary (CSV)** for spreadsheet analysis.

---

## 📊 Evaluation Engine & Rubric Scoring

Reports are evaluated against the official 5-criterion rubric (each scored out of 2.0, total 10.0 points):

| Section | Max Score | Key Evaluation Criteria |
| :--- | :---: | :--- |
| **Objective of the Lab** | 2.0 | Session-wide purpose, concepts, and skills in student's own words. |
| **Problem Understanding** | 2.0 | Explains inputs, calculations, and expected outputs. Direct C code copying is penalized. |
| **Logic / Approach Used** | 2.0 | Conceptual step-by-step reasoning (loops, branches, state changes). |
| **Important Variables Table** | 2.0 | Structured markdown table format (`\| Variable \| Purpose \|`). |
| **What I Observed** | 2.0 | Genuine observations on runtime behavior; generic filler ("Code executed successfully") is flagged. |

### Anti-Hallucination & Integrity Features:
- **Verbatim Evidence Verification**: Every piece of extracted evidence is cross-referenced with raw OCR text.
- **Python-Calculated Marks**: Math and totals are calculated deterministically in Python.
- **Code Leak Defense**: Evaluator prompt strictly prevents leaking Python test code, evaluator source functions, or dummy implementations into evaluation reports.

---

## 🧪 Automated Test Suite (114 Tests)

The test suite includes **114 comprehensive automated tests** passing in ~6.3 seconds:

```powershell
conda activate paddle_vl
python -m pytest tests/ -v
```

### Test Suite Breakdown:
- **`test_batch_security.py`**: Zip Slip defense, path traversal prevention, uncompressed size / bomb limits.
- **`test_batch_workflow.py`**: Sequential processing, resumability, retry isolation, atomic persistence.
- **`test_section_workflow.py`**: Section isolation (SEC1–SEC6), week isolation, complete aggregation, direct section summaries.
- **`test_extractor.py`**: Schema parsing, HTML table to Markdown conversion, list/dict alias handling, cross-program deduplication, context window options.
- **`test_verifier.py`**: Runtime question parsing, rubric parsing, Ollama health checks, token streaming, synchronous evaluation.
- **`test_dynamic_evidence_evaluation.py`**: Block segmentation, question-to-block fuzzy matching, verbatim grounding verification.
- **`test_final_scoring_layer.py`**: Objective scoring, criterion scoring, deterministic totals, grade classifications, report rendering.
- **`test_schemas.py`**: Pydantic models integrity, extra field handling, serialization tests.
- **`test_source_separation.py`**: Multi-page PDF page tracking, OCR device verification, metadata preservation.

---

## 📁 Project Structure

```text
ReportTest/
├── app.py                                  # Streamlit interactive web interface
├── batch_processor.py                      # Section batch engine, safe unzipping, state management
├── paddle_worker.py                        # Dedicated GPU PaddleOCR-VL subprocess worker
├── extractor.py                            # Structured canonical JSON extraction with Qwen
├── verifier.py                             # Dynamic rubric evaluation & verbatim grounding engine
├── schemas.py                              # Canonical Pydantic schemas (Extraction, Evaluation, Cohort)
├── questions_13.json                       # 13 standard C programming lab questions
├── requirements.txt                        # Python dependencies
├── SETUP.md                                # Complete setup & installation guide
├── HOW_TO_SETUP.md                         # Quick reference setup guide
├── README.md                               # Project documentation
├── output/
│   └── sections/                           # Isolated section output directory
│       └── <week_id>/<section_id>/
│           ├── <sec>_<week>_manifest.json
│           ├── <sec>_<week>_observation_reports.json
│           ├── <sec>_<week>_summary.json
│           ├── <sec>_<week>_student_reports.zip
│           └── students/
│               ├── <student_id>.json
│               └── <student_id>_evaluation.md
├── tests/                                  # 114 automated unit & integration tests
│   ├── test_batch_security.py
│   ├── test_batch_workflow.py
│   ├── test_dynamic_evidence_evaluation.py
│   ├── test_extractor.py
│   ├── test_final_scoring_layer.py
│   ├── test_schemas.py
│   ├── test_section_workflow.py
│   ├── test_source_separation.py
│   └── test_verifier.py
└── run_section_integration_test.py         # End-to-end cohort integration benchmark
```

---

## 📜 License & Acknowledgments

Built for university laboratory assessment workflows using:
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by Baidu PaddlePaddle
- [Qwen 2.5 Coder](https://github.com/QwenLM/Qwen2.5-Coder) by Alibaba Cloud Qwen Team
- [Ollama](https://ollama.com/) for local LLM inference
- [Streamlit](https://streamlit.io/) for web application delivery
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing and rendering
