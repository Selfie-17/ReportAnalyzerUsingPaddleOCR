import streamlit as st
import subprocess
import sys
import tempfile
import os
import json
import time
import zipfile
import pymupdf as fitz  # Modern PyMuPDF import for PDF preview and page rendering

from verifier import (
    check_ollama_status,
    stream_observation_verification,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MODEL,
)
from extractor import extract_observation_report
from schemas import (
    StudentObservationReport,
    SourceMeta,
    OcrResult,
    CompleteSectionReport,
    SectionSummary,
)
from batch_processor import (
    validate_and_extract_zip,
    discover_student_reports,
    BatchPipeline,
    SecurityError,
    get_paddle_python,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaddleOCR-VL 1.6 & Section-Wise Report Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "uploaded_file_id" not in st.session_state:
    st.session_state["uploaded_file_id"] = None

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

if "edited_text" not in st.session_state:
    st.session_state["edited_text"] = ""

if "ocr_meta" not in st.session_state:
    st.session_state["ocr_meta"] = {}

if "structured_extraction" not in st.session_state:
    st.session_state["structured_extraction"] = None

if "single_student_report" not in st.session_state:
    st.session_state["single_student_report"] = None

if "verification_result" not in st.session_state:
    st.session_state["verification_result"] = ""

if "debug_stdout" not in st.session_state:
    st.session_state["debug_stdout"] = ""

# Batch session states
if "batch_zip_id" not in st.session_state:
    st.session_state["batch_zip_id"] = None

if "batch_extracted_dir" not in st.session_state:
    st.session_state["batch_extracted_dir"] = None

if "batch_discovered" not in st.session_state:
    st.session_state["batch_discovered"] = {}

if "batch_manifest" not in st.session_state:
    st.session_state["batch_manifest"] = None

if "batch_zip_path" not in st.session_state:
    st.session_state["batch_zip_path"] = None


# ============================================================
# SIDEBAR: ENVIRONMENT & OLLAMA SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings & System")

    st.subheader("🖥️ OCR Environment")
    st.write("**Worker:** `paddle_worker.py`")
    st.write("**Model:** `PaddleOCR-VL 1.6 (GPU:0)`")
    st.info("⚡ Concurrency strictly set to **1** for RTX 4050 6GB VRAM safety.")
    with st.expander("Paddle Python Executable"):
        st.code(get_paddle_python(), language="text")

    st.divider()

    st.subheader("🧠 Qwen 2.5 Coder 3B")
    ollama_url = st.text_input(
        "Ollama URL",
        value=DEFAULT_OLLAMA_URL,
        help="Base URL for local Ollama server"
    )

    ollama_status = check_ollama_status(base_url=ollama_url)

    if ollama_status["ok"]:
        if ollama_status["model_found"]:
            st.success("🟢 Ollama Connected • Qwen 2.5 Coder 3B Ready")
        else:
            st.warning(f"🟡 Ollama online, but '{DEFAULT_MODEL}' not found.")
            st.caption(f"Available models: {', '.join(ollama_status['models']) or 'None'}")
    else:
        st.error("🔴 Ollama Offline")
        st.caption(ollama_status.get("error", "Cannot connect to Ollama server"))

    # Model selector with fallback
    avail_models = ollama_status.get("models", [])
    if DEFAULT_MODEL not in avail_models and avail_models:
        default_idx = 0
    elif DEFAULT_MODEL in avail_models:
        default_idx = avail_models.index(DEFAULT_MODEL)
    else:
        avail_models = [DEFAULT_MODEL]
        default_idx = 0

    selected_model = st.selectbox(
        "Model",
        options=avail_models,
        index=default_idx,
        help="Local LLM for structured extraction and evaluation"
    )

    temperature = st.slider(
        "Sampling Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Low temperature (0.1) produces deterministic structured JSON without hallucinations"
    )

    st.divider()

    with st.expander("📖 5-Section Rubric Guidelines"):
        st.markdown(
            """
            **5 Required Report Sections:**
            1. **Objective of Lab:** Concepts practiced & skills intended (written once).
            2. **Problem Understanding:** Inputs, outputs, calculation (own words, NO C code).
            3. **Logic / Approach Used:** Step-by-step reasoning, loops, branches (NO code dumping).
            4. **Important Variables:** `| Variable | Purpose |` format.
            5. **What I Observed:** Actual behavior, condition handling (reject generic fluff).
            """
        )


# ============================================================
# MAIN HEADER & NAVIGATION TABS
# ============================================================

st.title("🔬 PaddleOCR-VL & Section-Wise Report Extractor")
st.caption(
    "GPU-Accelerated Document OCR + Qwen 2.5 Coder 3B Conservative Structured JSON Extraction (Sections 1–6)"
)

tab_single, tab_batch = st.tabs([
    "📄 Single Student Report",
    "📦 Section-Wise Batch Extraction (SEC1–SEC6)"
])


# ============================================================
# TAB 1: SINGLE STUDENT WORKFLOW
# ============================================================

with tab_single:
    st.subheader("📤 Step 1: Upload Document (Image or Multi-Page PDF)")

    uploaded_file = st.file_uploader(
        "Upload laboratory report",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
        key="single_file_uploader"
    )

    if uploaded_file is None:
        st.session_state["uploaded_file_id"] = None
        st.session_state["ocr_text"] = ""
        st.session_state["edited_text"] = ""
        st.session_state["ocr_meta"] = {}
        st.session_state["structured_extraction"] = None
        st.session_state["single_student_report"] = None
        st.session_state["verification_result"] = ""
        st.session_state["debug_stdout"] = ""
        st.info("👆 Upload an image or PDF report to begin OCR extraction.")
    else:
        # Detect new file upload and reset session state accordingly
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["uploaded_file_id"] != current_file_id:
            st.session_state["uploaded_file_id"] = current_file_id
            st.session_state["ocr_text"] = ""
            st.session_state["edited_text"] = ""
            st.session_state["ocr_meta"] = {}
            st.session_state["structured_extraction"] = None
            st.session_state["single_student_report"] = None
            st.session_state["verification_result"] = ""
            st.session_state["debug_stdout"] = ""

        # Preview Section
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        is_pdf = file_ext == ".pdf"

        prev_col1, prev_col2 = st.columns([3, 1])

        with prev_col1:
            if is_pdf:
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    total_pdf_pages = len(doc)

                    if total_pdf_pages > 1:
                        page_sel = st.slider(
                            "Select page to preview:",
                            min_value=1,
                            max_value=total_pdf_pages,
                            value=1,
                            step=1,
                            key="single_pdf_slider"
                        )
                    else:
                        page_sel = 1

                    page = doc[page_sel - 1]
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    st.image(
                        img_bytes,
                        caption=f"{uploaded_file.name} — Page {page_sel} of {total_pdf_pages}",
                        use_container_width=True
                    )
                    doc.close()
                except Exception as e:
                    st.warning(f"Could not render PDF preview: {e}")
            else:
                st.image(
                    uploaded_file,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

        with prev_col2:
            st.write("**Filename:**")
            st.info(uploaded_file.name)
            st.write("**Size:**")
            st.write(f"{uploaded_file.size / 1024:.2f} KB")
            st.write("**Format:**")
            st.write("📄 PDF Document" if is_pdf else f"🖼️ Image ({uploaded_file.type or file_ext})")
            if is_pdf:
                st.write("**Pages:**")
                st.write(f"{total_pdf_pages} page(s)")

        # STEP 2: OCR EXTRACTION
        st.divider()
        st.subheader("🚀 Step 2: Extract Text with PaddleOCR-VL 1.6")

        pages_to_extract_arg = None
        if is_pdf and total_pdf_pages > 1:
            st.write("**Page Extraction Selection:**")
            p_sel_col1, p_sel_col2 = st.columns([1, 2])
            with p_sel_col1:
                page_mode = st.radio(
                    "Pages to extract:",
                    [f"All Pages (1 to {total_pdf_pages})", "Select Specific Pages"],
                    key="single_page_mode"
                )
            with p_sel_col2:
                if page_mode == "Select Specific Pages":
                    sel_pages = st.multiselect(
                        "Choose pages to extract:",
                        options=list(range(1, total_pdf_pages + 1)),
                        default=list(range(1, total_pdf_pages + 1)),
                        key="single_page_multiselect"
                    )
                    if sel_pages:
                        pages_to_extract_arg = ",".join(map(str, sorted(sel_pages)))
                        st.caption(f"Will extract {len(sel_pages)} selected page(s): {pages_to_extract_arg}")
                    else:
                        st.warning("No pages selected. Defaulting to all pages.")
                else:
                    st.caption(f"All {total_pdf_pages} pages will be rendered and extracted page-by-page sequentially.")

        extract_col, _ = st.columns([1, 2])
        with extract_col:
            extract_button = st.button(
                "⚡ Run PaddleOCR-VL Extraction",
                type="primary",
                use_container_width=True,
                key="btn_run_single_ocr"
            )

        if extract_button:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                worker_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "paddle_worker.py"
                )

                status_box = st.empty()
                page_desc = f"{pages_to_extract_arg or f'all {total_pdf_pages}'} pages" if is_pdf else "1 image"
                status_box.info(f"⏳ Running PaddleOCR-VL 1.6 worker process on GPU:0 (Extracting {page_desc} one-by-one)...")

                worker_python = get_paddle_python()
                cmd = [worker_python, worker_path, temp_path]
                if pages_to_extract_arg:
                    cmd.extend(["--pages", pages_to_extract_arg])

                start_time = time.perf_counter()
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                total_time = time.perf_counter() - start_time
                stdout = process.stdout.strip()
                st.session_state["debug_stdout"] = stdout

                if process.returncode != 0:
                    status_box.error("❌ PaddleOCR-VL worker encountered an error.")
                    with st.expander("🔴 Worker Error Output (stderr)", expanded=True):
                        st.code(process.stderr)
                    st.stop()

                worker_result = None
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        candidate = json.loads(line)
                        if isinstance(candidate, dict) and "success" in candidate:
                            worker_result = candidate
                            break
                    except json.JSONDecodeError:
                        continue

                if worker_result is None or not worker_result.get("success", False):
                    status_box.error("❌ Could not parse valid result from worker.")
                    st.stop()

                status_box.success(f"✅ OCR Extraction complete in {total_time:.2f}s!")
                extracted_text = str(worker_result.get("text", "")).strip()
                st.session_state["ocr_text"] = extracted_text
                st.session_state["edited_text"] = extracted_text
                st.session_state["ocr_meta"] = {
                    "total_time": total_time,
                    "num_pages": worker_result.get("num_pages", 1),
                    "page_breakdown": worker_result.get("page_breakdown", []),
                    "device": worker_result.get("device", "Unknown"),
                    "paddle_version": worker_result.get("paddle_version", "Unknown"),
                }
                # Reset downstream outputs
                st.session_state["structured_extraction"] = None
                st.session_state["single_student_report"] = None
                st.session_state["verification_result"] = ""

            except Exception as e:
                st.error(f"Failed to execute OCR process: {e}")
                st.exception(e)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # STEP 3: REVIEW & EDIT EXTRACTED TEXT
        current_ocr_text = st.session_state.get("ocr_text", "")
        if current_ocr_text:
            st.divider()
            st.subheader("📝 Step 3: Extracted Text & User Review")

            meta = st.session_state.get("ocr_meta", {})
            words_count = len(st.session_state["edited_text"].split())
            chars_count = len(st.session_state["edited_text"])

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("⚡ OCR Time", f"{meta.get('total_time', 0):.2f}s")
            with m2:
                st.metric("📑 Total Pages", meta.get("num_pages", 1))
            with m3:
                st.metric("📝 Words", words_count)
            with m4:
                st.metric("🔤 Characters", chars_count)
            with m5:
                st.metric("🎮 Device", meta.get("device", "gpu:0"))

            st.caption("💡 Review or correct any OCR misspellings before structured extraction.")

            tab_edit, tab_preview, tab_pages = st.tabs([
                "✏️ Edit Extracted Text",
                "👁️ Markdown Preview",
                "📄 Page Breakdown"
            ])

            with tab_edit:
                edited_text = st.text_area(
                    "Extracted Markdown (Editable):",
                    value=st.session_state["edited_text"],
                    height=300,
                    key="text_editor_area"
                )
                st.session_state["edited_text"] = edited_text

            with tab_preview:
                st.markdown(st.session_state["edited_text"])

            with tab_pages:
                page_breakdown = meta.get("page_breakdown", [])
                if page_breakdown:
                    for p in page_breakdown:
                        with st.expander(f"📄 Page {p.get('page', 1)} Text", expanded=(len(page_breakdown) == 1)):
                            st.text_area(
                                f"Page {p.get('page', 1)} Content",
                                value=p.get("text", ""),
                                height=180,
                                disabled=True,
                                key=f"single_page_view_{p.get('page')}"
                            )
                else:
                    st.info("Single-page content available in the main editor tab.")

            d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
            with d_col1:
                st.download_button(
                    label="📥 Download Markdown (.md)",
                    data=st.session_state["edited_text"],
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with d_col2:
                st.download_button(
                    label="📄 Download Plain Text (.txt)",
                    data=st.session_state["edited_text"],
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with d_col3:
                if st.button("🔄 Reset Edits to Original OCR", use_container_width=True):
                    st.session_state["edited_text"] = st.session_state["ocr_text"]
                    st.rerun()

            # STEP 4: STRUCTURED EXTRACTION
            st.divider()
            st.subheader("🤖 Step 4: Extract Structured Report with Qwen 2.5 Coder 3B")
            st.caption(
                "Converts OCR text into strict canonical JSON (Objective + P1..P10 sections). "
                "Never invents or hallucinates missing sections."
            )

            c_s1, c_s2, c_s3 = st.columns([1, 1, 1])
            with c_s1:
                student_id_input = st.text_input(
                    "Student ID",
                    value=os.path.splitext(uploaded_file.name)[0].split("_")[0],
                    help="Identifier for the student (used for output JSON filename)"
                )
            with c_s2:
                section_id_input = st.selectbox(
                    "Section ID",
                    options=["SEC1", "SEC2", "SEC3", "SEC4", "SEC5", "SEC6"],
                    index=0,
                    help="Section identifier (e.g. SEC1..SEC6)"
                )
            with c_s3:
                week_id_input = st.text_input(
                    "Week ID",
                    value="week-01",
                    help="Lab session week identifier"
                )

            extract_struct_btn = st.button(
                "⚡ Extract Structured JSON",
                type="primary",
                use_container_width=True,
                key="btn_run_structured_ext"
            )

            if extract_struct_btn:
                if not ollama_status["ok"]:
                    st.error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is running.")
                    st.stop()

                text_to_extract = st.session_state["edited_text"].strip()
                if not text_to_extract:
                    st.warning("Cannot extract from empty text.")
                    st.stop()

                with st.spinner(f"Extracting structured observation data with {selected_model}..."):
                    t_ext_start = time.perf_counter()
                    ext_result = extract_observation_report(
                        report_text=text_to_extract,
                        base_url=ollama_url,
                        model=selected_model,
                        temperature=temperature,
                        page_breakdown=st.session_state["ocr_meta"].get("page_breakdown", [])
                    )
                    t_ext_total = time.perf_counter() - t_ext_start

                st.session_state["structured_extraction"] = ext_result

                # Assemble complete canonical student report
                report = StudentObservationReport(
                    student_id=student_id_input.strip() or "student_01",
                    section_id=section_id_input.strip() or "SEC1",
                    week_id=week_id_input.strip() or "week-01",
                    status="completed",
                    source=SourceMeta(
                        filename=uploaded_file.name,
                        num_pages=st.session_state["ocr_meta"].get("num_pages", 1)
                    ),
                    ocr=OcrResult(
                        status="success",
                        text=st.session_state["ocr_text"],
                        num_pages=st.session_state["ocr_meta"].get("num_pages", 1),
                        page_breakdown=st.session_state["ocr_meta"].get("page_breakdown", []),
                        device=st.session_state["ocr_meta"].get("device", "gpu:0"),
                        paddle_version=st.session_state["ocr_meta"].get("paddle_version", "Unknown"),
                        total_time=st.session_state["ocr_meta"].get("total_time", 0.0)
                    ),
                    extraction=ext_result
                )
                st.session_state["single_student_report"] = report
                st.success(f"✅ Structured extraction complete in {t_ext_total:.2f}s!")

            # STEP 5: REVIEW STRUCTURED JSON
            if st.session_state.get("structured_extraction") and st.session_state.get("single_student_report"):
                st.divider()
                st.subheader("📊 Step 5: Review Structured Report & Pydantic Validation")

                report = st.session_state["single_student_report"]
                ext = report.extraction

                # Summary Metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Extraction Status", ext.status.upper())
                with c2:
                    st.metric("Detected Programs", f"{len(ext.detected_programs)} / 10")
                with c3:
                    st.metric("Missing Programs", f"{len(ext.missing_programs)} / 10")
                with c4:
                    has_obj = "Present" if ext.objective_of_lab else "Not Detected"
                    st.metric("Lab Objective", has_obj)

                # Program status badges
                badge_cols = st.columns(10)
                for i, b_col in enumerate(badge_cols, start=1):
                    pkey = f"P{i}"
                    pdet = ext.programs.get(pkey)
                    is_det = pdet and pdet.status == "detected"
                    with b_col:
                        if is_det:
                            st.success(f"**{pkey}**\n\n✓ Detected")
                        else:
                            st.caption(f"**{pkey}**\n\n— Absent")

                if ext.objective_of_lab:
                    with st.expander("🎯 Extracted Objective of Lab", expanded=True):
                        st.write(ext.objective_of_lab)

                # Detailed Program Cards
                with st.expander("🔍 Detailed Program Breakdown (P1..P10)", expanded=True):
                    for pkey in [f"P{i}" for i in range(1, 11)]:
                        pdet = ext.programs.get(pkey)
                        if not pdet:
                            continue

                        pages_info = f"(Pages: {', '.join(map(str, pdet.source_pages))})" if pdet.source_pages else ""
                        title = f"**{pkey}** — {'🟢 Detected ' + pages_info if pdet.status == 'detected' else '⚪ Not Detected'}"

                        with st.expander(title, expanded=(pdet.status == "detected")):
                            if pdet.status == "detected":
                                st.write("**Problem Understanding:**")
                                st.write(pdet.problem_understanding or "_Not provided (null)_")

                                st.write("**Logic / Approach Used:**")
                                st.write(pdet.logic_approach or "_Not provided (null)_")

                                st.write("**Important Variables:**")
                                if pdet.important_variables:
                                    var_data = [{"Variable": v.variable, "Purpose": v.purpose} for v in pdet.important_variables]
                                    st.table(var_data)
                                else:
                                    st.write("_None listed (empty)_")

                                st.write("**What I Observed:**")
                                st.write(pdet.what_i_observed or "_Not provided (null)_")
                            else:
                                st.info(f"Program {pkey} was not detected in the report OCR text. All fields set to null.")

                # JSON Viewer and Download
                st.subheader("📥 Export Canonical JSON")
                json_str = report.model_dump_json(indent=2)
                st.download_button(
                    label=f"📥 Download {report.student_id}.json",
                    data=json_str,
                    file_name=f"{report.student_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

                with st.expander("📄 View Canonical JSON Payload"):
                    st.code(json_str, language="json")

            # STEP 6: OPTIONAL MARKDOWN RUBRIC VERIFICATION (PRESERVED STANDALONE)
            st.divider()
            with st.expander("🧪 Step 6: Standalone Markdown Rubric Verification (Instruction Manual Verifier)"):
                st.caption(
                    "Optional prototype verification engine: Evaluates the report text against the "
                    "10.0-point Instruction Manual rubric using Ollama streaming."
                )

                verify_clicked = st.button(
                    "Evaluate Report with Instruction Manual Rubric",
                    key="btn_run_standalone_verifier"
                )

                if verify_clicked:
                    if not ollama_status["ok"]:
                        st.error(f"Cannot connect to Ollama at {ollama_url}.")
                    else:
                        eval_container = st.empty()
                        with st.spinner("Generating rubric evaluation..."):
                            stream_gen = stream_observation_verification(
                                report_text=st.session_state["edited_text"].strip(),
                                base_url=ollama_url,
                                model=selected_model,
                                temperature=temperature
                            )
                            final_eval = eval_container.write_stream(stream_gen)
                            st.session_state["verification_result"] = final_eval

                if st.session_state.get("verification_result") and not verify_clicked:
                    st.markdown(st.session_state["verification_result"])


# ============================================================
# TAB 2: SECTION-WISE BATCH EXTRACTION WORKFLOW (SEC1–SEC6)
# ============================================================

with tab_batch:
    st.subheader("📦 Section-Wise Batch Observation Report Extraction")
    st.markdown(
        """
        Upload a section cohort ZIP archive (e.g. `SEC1.zip`, `SEC2.zip`) containing student observation reports:
        ```text
        SEC1.zip
        ├── 22001/observation.pdf
        ├── 22002/observation.pdf
        └── ...
        ```
        - **Section & Week Isolation:** Output stored cleanly under `output/sections/<week_id>/<section_id>/`.
        - **Sequential Execution:** Strictly sequential OCR (concurrency = 1) and LLM extraction (concurrency = 1).
        - **Distinct Resume vs Retry:** Resume skips completed students; Retry Failed processes only failed students.
        - **Complete Section JSON:** Download one master JSON file containing all students in the section.
        """
    )

    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    with b_col1:
        batch_week_id = st.text_input(
            "Week Identifier",
            value="week-01",
            help="Lab session week identifier (e.g. week-01, week-02)"
        )
    with b_col2:
        batch_section_id = st.selectbox(
            "Section Identifier",
            options=["SEC1", "SEC2", "SEC3", "SEC4", "SEC5", "SEC6"],
            index=0,
            help="Select student section (SEC1 through SEC6)"
        )
    with b_col3:
        batch_out_dir = st.text_input(
            "Base Output Directory",
            value="output/sections",
            help="Base directory where section folders are created"
        )

    batch_file = st.file_uploader(
        f"Upload {batch_section_id} ZIP Archive (Max 500 MB)",
        type=["zip"],
        key="batch_zip_uploader"
    )

    if batch_file is not None:
        batch_id = f"{batch_file.name}_{batch_file.size}_{batch_section_id}_{batch_week_id}"
        # If new zip uploaded or section/week changed, unpack into clean temp folder
        if st.session_state["batch_zip_id"] != batch_id:
            st.session_state["batch_zip_id"] = batch_id
            temp_extract_dir = tempfile.mkdtemp(prefix=f"batch_{batch_section_id}_")
            st.session_state["batch_extracted_dir"] = temp_extract_dir

            temp_zip_path = os.path.join(temp_extract_dir, "archive.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(batch_file.getbuffer())

            try:
                validate_and_extract_zip(temp_zip_path, temp_extract_dir)
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)

                discovered = discover_student_reports(temp_extract_dir)
                st.session_state["batch_discovered"] = discovered
                st.success(f"✅ Archive validated & extracted safely. Discovered {len(discovered)} student entries.")
            except SecurityError as se:
                st.error(f"❌ Security Error during ZIP extraction: {se}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Failed to extract ZIP: {e}")
                st.stop()

        discovered = st.session_state.get("batch_discovered", {})
        if discovered:
            # Pre-flight Validation Summary
            ambiguities = [sid for sid, d in discovered.items() if d.get("error")]
            ready_students = [sid for sid, d in discovered.items() if not d.get("error")]

            st.info(
                f"📋 **Pre-Flight Summary:** Section: **{batch_section_id}** | Week: **{batch_week_id}** | "
                f"Students found: **{len(discovered)}** ({len(ready_students)} valid, {len(ambiguities)} flagged) | **Ready to process.**"
            )

            if ambiguities:
                with st.expander(f"⚠️ Flagged Student Entries ({len(ambiguities)})", expanded=True):
                    for sid in ambiguities:
                        st.warning(f"**Student {sid}:** {discovered[sid].get('error')}")

            # Initialize Section Batch Pipeline
            pipeline = BatchPipeline(
                week_id=batch_week_id.strip() or "week-01",
                section_id=batch_section_id.strip() or "SEC1",
                output_dir=batch_out_dir.strip() or "output/sections",
                ollama_url=ollama_url,
                model=selected_model,
                temperature=temperature,
                python_exe=get_paddle_python()
            )

            # Student Extraction Scope Selector
            st.divider()
            st.subheader("🎯 Student Extraction Scope")
            st.caption("Select how many or which students to extract from the batch archive (ideal for testing / checking).")

            scope_col1, scope_col2 = st.columns([1, 2])
            all_sids = sorted(list(discovered.keys()))

            with scope_col1:
                scope_mode = st.radio(
                    "Extraction Scope:",
                    options=[
                        f"All Students ({len(discovered)})",
                        "Quick Check (First N Students)",
                        "Custom Student Selection"
                    ],
                    index=0,
                    key="batch_scope_mode"
                )

            with scope_col2:
                if scope_mode == "Quick Check (First N Students)":
                    default_n = min(3, len(discovered))
                    subset_count = st.slider(
                        "Number of students to extract:",
                        min_value=1,
                        max_value=len(discovered),
                        value=default_n,
                        step=1,
                        key="batch_subset_slider"
                    )
                    selected_student_ids = all_sids[:subset_count]
                    st.info(f"Targeting first **{len(selected_student_ids)}** students: `{', '.join(selected_student_ids)}`")
                elif scope_mode == "Custom Student Selection":
                    selected_student_ids = st.multiselect(
                        "Select specific students to process:",
                        options=all_sids,
                        default=all_sids[:min(3, len(all_sids))],
                        key="batch_custom_multiselect"
                    )
                    st.info(f"Targeting **{len(selected_student_ids)}** custom selected student(s).")
                else:
                    selected_student_ids = all_sids
                    st.success(f"Targeting **all {len(selected_student_ids)}** discovered students.")

            # Execution controls: Distinct Start, Resume, and Retry Failed
            st.divider()
            c_run1, c_run2, c_run3 = st.columns(3)
            with c_run1:
                btn_start_batch = st.button(f"🚀 Start Batch ({len(selected_student_ids)} students)", type="primary", use_container_width=True)
            with c_run2:
                btn_resume_batch = st.button(f"🔄 Resume Incomplete ({len(selected_student_ids)} target)", use_container_width=True)
            with c_run3:
                btn_retry_failed = st.button("⚠️ Retry Failed Only", use_container_width=True)

            # Status and live stage display containers
            live_stage_box = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            table_placeholder = st.empty()

            def update_dashboard():
                students_list = list(pipeline.state.values())
                total = len(discovered)
                target_count = len(selected_student_ids)
                completed = sum(1 for s in students_list if s.status == "completed")
                failed = sum(1 for s in students_list if s.status == "failed")
                in_prog = sum(1 for s in students_list if s.status in ("processing_ocr", "ocr_complete", "extracting"))
                pending = max(0, total - completed - failed - in_prog)

                pct = (completed + failed) / total if total > 0 else 0.0
                progress_bar.progress(min(pct, 1.0))
                status_text.write(f"**Progress for {batch_section_id} / {batch_week_id}:** {completed + failed} / {total} processed ({pct*100:.1f}%) | Target: {target_count} students")

                with metrics_placeholder.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Found", total)
                    m2.metric("🎯 Target Run", target_count)
                    m3.metric("✅ Completed", completed)
                    m4.metric("❌ Failed", failed)
                    m5.metric("⏳ Pending", pending)

                # Student Status Table
                table_rows = []
                for sid, info in sorted(discovered.items()):
                    st_entry = pipeline.state.get(sid)
                    cur_st = st_entry.status if st_entry else "pending"
                    err = st_entry.error if st_entry and st_entry.error else "-"
                    in_target = sid in selected_student_ids
                    scope_display = "🎯 Target" if in_target else "⚪ Outside"

                    ocr_mark = "✓" if st_entry and st_entry.ocr_status == "completed" else ("✗" if st_entry and st_entry.ocr_status == "failed" else "—")
                    ext_mark = "✓" if st_entry and st_entry.extraction_status == "completed" else ("✗" if st_entry and st_entry.extraction_status == "failed" else "—")

                    status_display = {
                        "completed": "✅ Completed",
                        "processing_ocr": "⚡ Running OCR",
                        "ocr_complete": "📑 OCR Done",
                        "extracting": "🧠 Extracting",
                        "failed": "❌ Failed",
                        "pending": "⏳ Pending"
                    }.get(cur_st, cur_st)

                    table_rows.append({
                        "Student ID": sid,
                        "Scope": scope_display,
                        "Status": status_display,
                        "OCR": ocr_mark,
                        "Extraction": ext_mark,
                        "Error / Details": err[:65] if err else "-"
                    })

                table_placeholder.dataframe(
                    table_rows,
                    use_container_width=True,
                    height=320
                )

            # Update initial dashboard
            update_dashboard()

            # Execute Batch actions
            if btn_start_batch or btn_resume_batch or btn_retry_failed:
                if not ollama_status["ok"]:
                    st.error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is online.")
                    st.stop()

                def on_progress(entry):
                    stage_name = "OCR (GPU:0)" if entry.stage == "ocr" else ("Structured extraction (Qwen)" if entry.stage == "extraction" else (entry.stage or "Processing"))
                    live_stage_box.info(f"📍 **Currently processing:** Student `{entry.student_id}` | **Stage:** `{stage_name}`")
                    update_dashboard()

                if btn_retry_failed:
                    status_text.info(f"⏳ Running Retry Failed for {batch_section_id}... (Only processing failed students)")
                    manifest = pipeline.retry_failed(discovered_students=discovered, selected_student_ids=selected_student_ids, progress_cb=on_progress)
                else:
                    status_text.info(f"⏳ Running Section Batch for {batch_section_id} ({len(selected_student_ids)} students targeted)...")
                    manifest = pipeline.resume(discovered_students=discovered, selected_student_ids=selected_student_ids, progress_cb=on_progress)

                live_stage_box.empty()
                st.session_state["batch_manifest"] = manifest
                update_dashboard()
                st.success(f"🎉 Processing finished for {batch_section_id}! Completed: {manifest.successful}, Failed: {manifest.failed}")

            # DOWNLOADS & EXPORTS SECTION
            st.divider()
            st.subheader(f"📥 Section Deliverables ({batch_section_id} / {batch_week_id})")

            # Always ensure complete json and summary are generated if any students exist
            if os.path.exists(pipeline.students_dir) and os.listdir(pipeline.students_dir):
                pipeline.build_complete_section_json()
                pipeline.build_section_summary()
                pipeline.create_batch_zip()

            d1, d2 = st.columns(2)
            with d1:
                if os.path.exists(pipeline.complete_json_path):
                    with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
                        complete_json_bytes = f.read()
                    st.download_button(
                        label=f"📥 Download Complete Section JSON ({os.path.basename(pipeline.complete_json_path)})",
                        data=complete_json_bytes,
                        file_name=os.path.basename(pipeline.complete_json_path),
                        mime="application/json",
                        type="primary",
                        use_container_width=True
                    )
                if os.path.exists(pipeline.summary_path):
                    with open(pipeline.summary_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label=f"📊 Download Section Summary ({os.path.basename(pipeline.summary_path)})",
                            data=f.read(),
                            file_name=os.path.basename(pipeline.summary_path),
                            mime="application/json",
                            use_container_width=True
                        )

            with d2:
                if os.path.exists(pipeline.manifest_path):
                    with open(pipeline.manifest_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label=f"📥 Download Section Manifest ({os.path.basename(pipeline.manifest_path)})",
                            data=f.read(),
                            file_name=os.path.basename(pipeline.manifest_path),
                            mime="application/json",
                            use_container_width=True
                        )
                if os.path.exists(pipeline.zip_path):
                    with open(pipeline.zip_path, "rb") as f:
                        st.download_button(
                            label=f"📦 Download Student Reports ZIP ({os.path.basename(pipeline.zip_path)})",
                            data=f.read(),
                            file_name=os.path.basename(pipeline.zip_path),
                            mime="application/zip",
                            use_container_width=True
                        )

            # Interactive Student JSON Inspector
            student_json_files = []
            if os.path.exists(pipeline.students_dir):
                student_json_files = sorted([os.path.splitext(f)[0] for f in os.listdir(pipeline.students_dir) if f.endswith(".json")])

            if student_json_files:
                st.divider()
                st.subheader("🔍 Inspect Student Report")
                inspected_sid = st.selectbox("Select student to inspect:", student_json_files, key="select_inspect_student")
                if inspected_sid:
                    json_path = os.path.join(pipeline.students_dir, f"{inspected_sid}.json")
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            st_data = json.load(f)

                        st.write(f"**Student ID:** `{inspected_sid}` | **Section:** `{st_data.get('section_id', batch_section_id)}` | **Week:** `{st_data.get('week_id', batch_week_id)}` | **Status:** `{st_data.get('status', 'unknown').upper()}`")

                        if st_data.get("status") == "failed":
                            err_info = st_data.get("error", {})
                            st.error(f"❌ **Stage:** {err_info.get('stage')} | **Error:** {err_info.get('message')}")
                        else:
                            ext_data = st_data.get("extraction", {})
                            det_progs = ext_data.get("detected_programs", [])
                            miss_progs = ext_data.get("missing_programs", [])
                            st.success(f"Detected {len(det_progs)} programs: {', '.join(det_progs) or 'None'}")
                            if miss_progs:
                                st.caption(f"Missing programs: {', '.join(miss_progs)}")

                        st.download_button(
                            label=f"📥 Download {inspected_sid}.json",
                            data=json.dumps(st_data, indent=2, ensure_ascii=False),
                            file_name=f"{inspected_sid}.json",
                            mime="application/json"
                        )

                        with st.expander("📄 View Full JSON Payload"):
                            st.json(st_data)


# ============================================================
# DIAGNOSTICS FOOTER
# ============================================================

if st.session_state.get("debug_stdout"):
    with st.expander("🔧 Diagnostics & Raw Worker Logs"):
        st.code(st.session_state["debug_stdout"], language="text")

st.divider()
st.caption(
    "PaddleOCR-VL 1.6 • PyMuPDF • Ollama Qwen 2.5 Coder 3B • NVIDIA RTX 4050 6GB • Concurrency: OCR=1, LLM=1 • Sections 1–6"
)