# 🛠️ Complete Setup Guide: PaddleOCR-VL & Observation Report Extraction Engine

This document provides a step-by-step walkthrough for configuring and running the **PaddleOCR-VL 1.6 & Section-Wise Batch Observation Report Extraction Engine** on Windows 10/11 with NVIDIA GPU acceleration.

---

## 📋 System & Hardware Requirements

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11 (64-bit) | Windows 11 (64-bit) |
| **GPU** | NVIDIA GPU with CUDA support | NVIDIA RTX 3060 / 4050 / 4060 or better |
| **VRAM** | 6 GB VRAM *(strictly Concurrency = 1)* | 8 GB+ VRAM |
| **RAM** | 16 GB System Memory | 32 GB System Memory |
| **Python** | Python 3.11 (via Anaconda / Miniconda) | Python 3.11 in isolated Conda environment |
| **LLM Server** | Ollama for Windows | Ollama running locally (`http://127.0.0.1:11434`) |
| **LLM Model** | `qwen2.5-coder:3b` | `qwen2.5-coder:3b` (with 16K context support) |

> [!IMPORTANT]
> **VRAM & Concurrency Notice:** Empirically benchmarked on an NVIDIA RTX 4050 (6 GB VRAM), 1 active PaddleOCR-VL worker consumes **~5.9 GB VRAM**. For this reason, OCR and LLM concurrency are intentionally set to **1** to prevent Out-Of-Memory (OOM) crashes. Do not increase worker concurrency unless using a GPU with 16 GB+ VRAM.

---

## 🚀 Step-by-Step Installation

### Step 1: Install and Configure Ollama

The extraction and dynamic evaluation engines require Ollama running locally to serve `qwen2.5-coder:3b`.

1. **Download and install Ollama:**
   - Download the Windows installer from [ollama.com/download](https://ollama.com/download).
   - Alternatively, install via Windows Package Manager:
     ```powershell
     winget install Ollama.Ollama
     ```

2. **Pull the Qwen 2.5 Coder 3B model:**
   Open a terminal and run:
   ```powershell
   ollama pull qwen2.5-coder:3b
   ```

3. **Verify Ollama is running:**
   Visit `http://127.0.0.1:11434` in your browser or test with `curl`:
   ```powershell
   curl http://127.0.0.1:11434/api/tags
   ```
   You should see `qwen2.5-coder:3b` in the list of available models.

---

### Step 2: Create an Isolated Conda Environment

We recommend using Anaconda or Miniconda with **Python 3.11**.

1. **Open Anaconda Prompt or PowerShell** and create the environment:
   ```powershell
   conda create -n paddle_vl python=3.11 -y
   ```

2. **Activate the environment:**
   ```powershell
   conda activate paddle_vl
   ```

3. **Ensure pip is updated:**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```

---

### Step 3: Install PaddlePaddle with CUDA GPU Support

Install the CUDA-enabled build of `paddlepaddle-gpu==3.3.0`.

#### For CUDA 11.8:
```powershell
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

#### For CUDA 12.x:
```powershell
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```

#### Verify GPU Acceleration in Paddle:
Run the following verification snippet:
```powershell
python -c "import paddle; print('Paddle Version:', paddle.__version__); print('CUDA Available:', paddle.is_compiled_with_cuda()); print('Device:', paddle.device.get_device())"
```
**Expected output:**
```text
Paddle Version: 3.3.0
CUDA Available: True
Device: gpu:0
```

> [!TIP]
> If you encounter an error regarding `paddle.amp.is_bfloat16_supported`, note that `paddle_worker.py` already includes an automatic monkey-patch (`_safe_bfloat16_supported`) to ensure compatibility with Windows CUDA drivers.

---

### Step 4: Install Project Dependencies

Install all remaining Python dependencies using `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

This installs:
- `streamlit` (interactive web UI)
- `paddleocr` & `paddlex` (vision-language OCR engine)
- `pymupdf` (PDF rendering & page breakdown)
- `pydantic>=2.0` (strict schema validation & serialization)
- `pandas` (scores table aggregation & CSV generation)
- `requests` (Ollama HTTP streaming & API client)
- `pytest` (automated testing framework)

---

### Step 5: Verify Installation with the Test Suite

Run the full automated test suite to ensure all security policies, schema validators, batch processors, and dynamic evaluation engines are functioning properly:

```powershell
python -m pytest tests/ -v
```

**Expected output:**
```text
============================= 114 passed in ~6.3s =============================
```

All **114 tests** across all 9 test suites should pass cleanly:
- `test_batch_security.py` (Zip Slip, directory traversal, bomb defense)
- `test_batch_workflow.py` (Resumable batch processing & retry pipelines)
- `test_section_workflow.py` (Section isolation SEC1–SEC6, week isolation)
- `test_extractor.py` (Canonical JSON schema, HTML table conversion, deduplication)
- `test_verifier.py` (Dynamic evaluation, rubrics, anti-hallucination)
- `test_dynamic_evidence_evaluation.py` (Grounded evidence verification)
- `test_final_scoring_layer.py` (Deterministic scoring calculations)
- `test_schemas.py` (Pydantic model integrity)
- `test_source_separation.py` (Multi-page PDF OCR page tracking)

---

## 🖥️ Running the Application

### 1. Start the Streamlit Web Application

Activate the conda environment and run:

```powershell
conda activate paddle_vl
streamlit run app.py
```

The application will launch and open in your default browser at:
`http://localhost:8501`

---

## 📖 Application Workflows

### Tab 1: 📄 Single Student Report Extraction & Verification

1. **Step 1: Upload Document:** Upload any student lab report (PDF, PNG, JPG, WEBP). Multi-page PDFs provide an interactive page preview slider.
2. **Step 2: Run OCR:** Click **🚀 Run PaddleOCR-VL 1.6** to extract text via GPU.
3. **Step 3: Text Inspection & Edit:** Review the extracted OCR text. You can manually edit or clean up text before extraction if necessary.
4. **Step 4: Structured Extraction:** Click **🤖 Extract Structured Report with Qwen** to generate canonical Pydantic JSON.
5. **Step 5: Review JSON:** Inspect detected vs. missing programs, problem understanding, logic approaches, variables tables, and observations.
6. **Step 6: Evaluate Report:** Run the dynamic rubric evaluation against the official Instruction Manual and assigned questions. Watch token-streamed evaluation results and download `<student_id>_evaluation.md`.

---

### Tab 2: 📦 Section-Wise Batch Extraction (SEC1–SEC6)

1. **Configure Cohort Scope:**
   - Select the **Week Identifier** (`week-01` through `week-16`).
   - Select the **Section Identifier** (`SEC1` through `SEC6`).
2. **Upload Section ZIP:**
   - Upload the cohort archive (e.g. `SEC1.zip`).
   - The engine validates against Zip Slip and directory traversal, extracts safely, and discovers all student directories.
   - Pre-flight discovery displays valid students, flagged directories, and duplicate ID alerts.
3. **Configure Runtime Questions & Rubric (Optional):**
   - Click the top expander to customize the assigned questions (1-click preset for Week 1's 13 programs or Manual's 5 programs) and evaluation rubric.
4. **Execute Batch Pipeline:**
   - **🚀 Start Section Batch:** Sequential processing (OCR on GPU:0, Qwen Structured Extraction, Qwen Rubric Evaluation).
   - **🔄 Resume Incomplete:** Skips already completed students; processes pending/interrupted students.
   - **⚠️ Retry Failed Only:** Selectively re-runs only students whose status is `failed`.
   - **🔄 Rerun Evaluation Only:** Re-runs rubric grading directly on existing extracted text without re-doing OCR.
5. **Export Section Deliverables:**
   - **📥 Download Complete Section JSON:** `<section>_<week>_observation_reports.json` (contains 100% of student reports, including failed diagnostic logs).
   - **📥 Download Section Manifest:** `<section>_<week>_manifest.json` (student status, stages, timings).
   - **📊 Download Section Summary:** `<section>_<week>_summary.json` (program detection statistics P1..P10/P13).
   - **📦 Download Student Reports ZIP:** `<section>_<week>_student_reports.zip` (packages all individual `<student_id>.json` and `<student_id>_evaluation.md` files).
6. **Cohort Scores & Evaluation Summary Table:**
   - View aggregated metrics: Cohort Average Score, Highest Score, Evaluated Count, Approved Count.
   - Interactive table detailing Student ID, Total Score, Grade, Verdict, Questions Covered, and criterion breakdown (/2 each).
   - **📥 Download Cohort Scores Summary (CSV):** 1-click export to `<section>_<week>_scores_summary.csv`.

---

## 🛠️ CLI Testing & Utility Scripts

The repository includes several standalone command-line scripts:

### 1. Test PaddleOCR-VL Subprocess Directly
```powershell
python paddle_worker.py test_sample_page1.pdf
```
Extracts text from a sample PDF and outputs page-by-page JSON.

### 2. Run Cohort Integration Test
```powershell
python run_section_integration_test.py
```
Runs an end-to-end integration test using a 3-student cohort (`test_cohort_3_students.zip`).

### 3. Run Direct Evaluation on Extracted Text
```powershell
python run_evaluation_on_extracted.py
```
Demonstrates running dynamic rubric evaluation across all 13 questions without needing to re-run OCR.

---

## ❓ Troubleshooting & FAQs

### Q1: Paddle throws `UserWarning: You are using GPU version Paddle, but your CUDA device is not set properly.`
- **Cause:** Paddle is running in an environment without CUDA toolkit / cuDNN binaries visible in PATH.
- **Fix:** Ensure you activated the `paddle_vl` conda environment (`conda activate paddle_vl`). Verify your NVIDIA drivers are up to date.

### Q2: Streamlit shows `🔴 Ollama Offline`
- **Cause:** Ollama service is not running on `http://127.0.0.1:11434`.
- **Fix:** Start Ollama from the Windows Start menu or run `ollama serve` in a separate command prompt. Confirm with `curl http://127.0.0.1:11434/api/tags`.

### Q3: `qwen2.5-coder:3b` model not found
- **Cause:** Ollama is running but the model has not been pulled yet.
- **Fix:** Run `ollama pull qwen2.5-coder:3b` in your terminal.

### Q4: Out of Memory (OOM) error on GPU
- **Cause:** Multiple GPU processes attempted to allocate VRAM simultaneously.
- **Fix:** Concurrency is enforced at 1. Ensure no other GPU-intensive applications (e.g., PyTorch, local Stable Diffusion, or games) are consuming VRAM concurrently.

### Q5: Camera scan / PDF rendering is slow or looks oversampled
- **Cause:** Default renderers sometimes apply a 2x render scale.
- **Fix:** The engine sets `PADDLE_PDX_PDF_RENDER_SCALE="1.0"` by default in `paddle_worker.py` to maintain optimal 1.0x native resolution.

---

## 📁 Output Directory Structure Reference

```text
output/
└── sections/
    └── <week_id>/                     (e.g., week-01)
        └── <section_id>/              (e.g., SEC1, SEC2)
            ├── <section>_<week>_manifest.json
            ├── <section>_<week>_observation_reports.json
            ├── <section>_<week>_summary.json
            ├── <section>_<week>_student_reports.zip
            └── students/
                ├── <student_id>.json
                ├── <student_id>_evaluation.md
                └── ...
```
