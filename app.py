import streamlit as st
import subprocess
import sys
import tempfile
import os
import json
import time
import zipfile
import re
import pandas as pd
import pymupdf as fitz  # Modern PyMuPDF import for PDF preview and page rendering

from verifier import (
    check_ollama_status,
    stream_observation_verification,
    evaluate_observation_report,
    render_verification_markdown,
    sanitize_student_ocr_text,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MODEL,
    DEFAULT_MANUAL_QUESTIONS_PRESET,
    WEEK1_13_PROGRAMS_PRESET,
    OFFICIAL_INSTRUCTION_MANUAL,
    format_extraction_for_evaluation,
    parse_evaluation_scores,
    parse_assigned_questions,
    parse_instruction_manual,
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

if "assigned_questions" not in st.session_state:
    st.session_state["assigned_questions"] = WEEK1_13_PROGRAMS_PRESET

if "instruction_manual" not in st.session_state:
    st.session_state["instruction_manual"] = OFFICIAL_INSTRUCTION_MANUAL

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

# ============================================================
# TOP CONFIGURATION: RUNTIME ASSIGNED QUESTIONS & INSTRUCTION MANUAL
# ============================================================
with st.expander("📝 Runtime Configuration: Assigned Questions & Instruction Manual", expanded=True):
    st.caption(
        "Both the assigned questions and instruction manual are dynamic runtime inputs. "
        "The system evaluates whatever questions and criteria are supplied here without hardcoded assumptions."
    )
    col_q, col_m = st.columns(2)
    with col_q:
        st.markdown("**1. Assigned Questions (Runtime Data)**")
        cp1, cp2, cp3 = st.columns([1.2, 1, 0.8])
        with cp1:
            if st.button("📋 Week 1 (13 Progs)", use_container_width=True, help="Load 13 standard C programs"):
                st.session_state["assigned_questions"] = WEEK1_13_PROGRAMS_PRESET
                st.rerun()
        with cp2:
            if st.button("📋 Manual (5 Progs)", use_container_width=True, help="Load 5 standard programs"):
                st.session_state["assigned_questions"] = DEFAULT_MANUAL_QUESTIONS_PRESET
                st.rerun()
        with cp3:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear assigned questions", key="btn_clear_questions"):
                st.session_state["assigned_questions"] = ""
                st.rerun()

        parsed_q = parse_assigned_questions(st.session_state.get("assigned_questions", ""))
        st.caption(f"✓ **{len(parsed_q)} questions parsed dynamically**")

        assigned_questions_top = st.text_area(
            "Assigned Questions:",
            value=st.session_state.get("assigned_questions", WEEK1_13_PROGRAMS_PRESET),
            height=140,
            placeholder="Enter the lab questions (any language, any topic, any count)...",
            key="top_assigned_questions"
        )
        st.session_state["assigned_questions"] = assigned_questions_top

    with col_m:
        st.markdown("**2. Instruction Manual / Criteria (Runtime Data)**")
        mp1, mp2 = st.columns(2)
        with mp1:
            if st.button("📖 Default Manual", use_container_width=True, help="Load official 5-section manual"):
                st.session_state["instruction_manual"] = OFFICIAL_INSTRUCTION_MANUAL
                st.rerun()
        with mp2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear manual", key="btn_clear_manual"):
                st.session_state["instruction_manual"] = ""
                st.rerun()

        parsed_man = parse_instruction_manual(st.session_state.get("instruction_manual", ""))
        sec_names = parsed_man.get("per_question_requirements", [])
        st.caption(f"✓ **{len(sec_names)} required section(s) detected:** `{', '.join(sec_names)}`")

        manual_text_top = st.text_area(
            "Instruction Manual / Evaluation Requirements:",
            value=st.session_state.get("instruction_manual", OFFICIAL_INSTRUCTION_MANUAL),
            height=140,
            placeholder="Enter the instruction manual defining what sections and rules are required...",
            key="top_instruction_manual"
        )
        st.session_state["instruction_manual"] = manual_text_top

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
                        page_breakdown=st.session_state["ocr_meta"].get("page_breakdown", []),
                        assigned_questions=st.session_state.get("assigned_questions", "")
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

                # Sort programs canonically by program number
                all_pkeys = sorted(
                    list(ext.programs.keys()),
                    key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x) else 999
                )
                if not all_pkeys:
                    all_pkeys = [f"P{i}" for i in range(1, 11)]

                # Summary Metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Extraction Status", ext.status.upper())
                with c2:
                    st.metric("Detected Programs", f"{len(ext.detected_programs)} / {len(all_pkeys)}")
                with c3:
                    st.metric("Missing Programs", f"{len(ext.missing_programs)} / {len(all_pkeys)}")
                with c4:
                    has_obj = "Present" if ext.objective_of_lab else "Not Detected"
                    st.metric("Lab Objective", has_obj)

                # Program status badges (in rows of up to 10 columns)
                for row_start in range(0, len(all_pkeys), 10):
                    row_keys = all_pkeys[row_start:row_start + 10]
                    badge_cols = st.columns(len(row_keys))
                    for pkey, b_col in zip(row_keys, badge_cols):
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
                range_label = f"{all_pkeys[0]}..{all_pkeys[-1]}" if len(all_pkeys) > 1 else all_pkeys[0]
                with st.expander(f"🔍 Detailed Program Breakdown ({range_label})", expanded=True):
                    for pkey in all_pkeys:
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

            # STEP 6: EVALUATION & INSTRUCTION MANUAL RUBRIC VERIFICATION
            st.divider()
            st.subheader("🧪 Step 6: Evaluate Report with Instruction Manual & Rubric")
            st.caption(
                "Strictly verifies the student's report against the official Instruction Manual submission guide "
                "and compares coverage against the assigned lab questions configured at the top."
            )

            assigned_q_val = st.session_state.get("assigned_questions", "").strip()
            if assigned_q_val:
                st.info("📋 **Assigned Questions Active:** Qwen will verify report coverage against the assigned questions configured at the top.")
            else:
                st.warning("⚠️ No assigned questions specified at the top. Qwen will evaluate using general rubric criteria only.")

            with st.expander("📝 Review / Quick-Edit Assigned Questions for this Evaluation", expanded=False):
                step6_assigned_val = st.text_area(
                    "Assigned Lab Questions / Problem Statements:",
                    value=st.session_state.get("assigned_questions", DEFAULT_MANUAL_QUESTIONS_PRESET),
                    height=120,
                    help="Modify here if you wish to override the questions configured at the top for this evaluation.",
                    key="step6_assigned_questions"
                )
                if step6_assigned_val != st.session_state.get("assigned_questions"):
                    st.session_state["assigned_questions"] = step6_assigned_val

            # STRICT SOURCE SEPARATION: Evaluation source is strictly Student OCR Text extracted from uploaded report
            raw_ocr = st.session_state.get("ocr_text", "").strip()
            student_ocr_source = st.session_state.get("edited_text", "").strip() or raw_ocr

            parsed_assigned = parse_assigned_questions(st.session_state.get("assigned_questions", ""))
            manual_text_val = st.session_state.get("instruction_manual", "")

            # Sanitize and isolate student OCR
            evaluation_source = sanitize_student_ocr_text(
                raw_text=student_ocr_source,
                filename=uploaded_file.name,
                assigned_count=len(parsed_assigned),
                manual_char_count=len(manual_text_val),
                debug=True
            )

            st.success("📄 **Evaluation Source:** Strictly isolated Student OCR Text extracted from uploaded report.")

            with st.expander("🔍 [INPUT DEBUG] Source Separation Verification", expanded=False):
                st.write(f"**PDF Filename:** `{uploaded_file.name}`")
                st.write(f"**OCR Character Count:** `{len(raw_ocr)}`")
                st.write(f"**Evaluation Source Character Count:** `{len(evaluation_source)}`")
                st.write(f"**Assigned Questions Count:** `{len(parsed_assigned)}`")
                st.write(f"**Instruction Manual Character Count:** `{len(manual_text_val)}`")
                st.text_area("OCR First 500 Characters", value=raw_ocr[:500], height=90, disabled=True)
                st.text_area("OCR Last 500 Characters", value=raw_ocr[-500:], height=90, disabled=True)

            c_btn1, c_btn2 = st.columns([2, 1])
            with c_btn1:
                verify_clicked = st.button(
                    "🚀 Evaluate Report Against Assigned Questions & Rubric",
                    type="primary",
                    use_container_width=True,
                    key="btn_run_standalone_verifier"
                )
            with c_btn2:
                if st.session_state.get("verification_result"):
                    if st.button("🔄 Clear Evaluation Result", use_container_width=True):
                        st.session_state["verification_result"] = ""
                        st.rerun()

            if verify_clicked:
                if not ollama_status["ok"]:
                    st.error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is running.")
                elif not evaluation_source:
                    st.warning("Cannot evaluate empty report text.")
                else:
                    # Strict source separation assertions
                    assert "## 📊 Final Score" not in evaluation_source, "Generated final score markdown must not be in evaluation source"
                    assert "# 📊 Laboratory Observation Report" not in evaluation_source, "Generated report markdown must not be in evaluation source"
                    assert "def add(" not in evaluation_source, "Evaluator python test code must not be in evaluation source"
                    assert "def subtract(" not in evaluation_source, "Evaluator python test code must not be in evaluation source"
                    assert "def multiply(" not in evaluation_source, "Evaluator python test code must not be in evaluation source"
                    assert "test_cases =" not in evaluation_source, "Test cases must not be in evaluation source"
                    assert "Expected qualitative behavior" not in evaluation_source, "Expected qualitative behavior must not be in evaluation source"

                    eval_container = st.empty()
                    with st.spinner("Auditing report evidence against assigned questions and Instruction Manual rubric..."):
                        eval_result = evaluate_observation_report(
                            report_text=evaluation_source,
                            assigned_questions=st.session_state.get("assigned_questions", "").strip(),
                            instruction_manual=st.session_state.get("instruction_manual", "").strip(),
                            base_url=ollama_url,
                            model=selected_model,
                            temperature=temperature,
                            filename=uploaded_file.name
                        )
                        final_eval = render_verification_markdown(eval_result)
                        eval_container.markdown(final_eval)
                        st.session_state["verification_result"] = final_eval

            if st.session_state.get("verification_result") and not verify_clicked:
                st.markdown(st.session_state["verification_result"])

            if st.session_state.get("verification_result"):
                eval_filename = f"{os.path.splitext(uploaded_file.name)[0]}_evaluation.md"
                st.download_button(
                    label="📥 Download Evaluation Report (.md)",
                    data=st.session_state["verification_result"],
                    file_name=eval_filename,
                    mime="text/markdown",
                    use_container_width=True,
                    key="btn_download_eval_report"
                )


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
                python_exe=get_paddle_python(),
                assigned_questions=st.session_state.get("assigned_questions", ""),
                instruction_manual=st.session_state.get("instruction_manual", ""),
                enable_evaluation=True
            )
            # Safely set assigned questions & manual for batch pipeline
            pipeline.assigned_questions = st.session_state.get("assigned_questions", "")
            pipeline.instruction_manual = st.session_state.get("instruction_manual", "")

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

            eval_extracted_toggle = st.checkbox(
                "Evaluate extracted text instead of raw OCR (recommended)",
                value=True,
                help="When enabled, Qwen evaluates against structured student report extractions rather than unsegmented OCR text."
            )
            pipeline.evaluate_extracted_text = eval_extracted_toggle

            # Execution controls: Distinct Start, Resume, Retry Failed, and Rerun Evaluation
            st.divider()
            c_run1, c_run2, c_run3, c_run4 = st.columns([1, 1, 1, 1.2])
            with c_run1:
                btn_start_batch = st.button(f"🚀 Start Batch ({len(selected_student_ids)} students)", type="primary", use_container_width=True)
            with c_run2:
                btn_resume_batch = st.button(f"🔄 Resume Incomplete ({len(selected_student_ids)} target)", use_container_width=True)
            with c_run3:
                btn_retry_failed = st.button("⚠️ Retry Failed Only", use_container_width=True)
            with c_run4:
                btn_rerun_eval = st.button("🧠 Re-run Qwen Evaluation (Extracted)", help="Re-runs Qwen rubric evaluation directly on already-extracted student text without repeating OCR or extraction.", use_container_width=True)

            # Status and live stage display containers
            live_stage_box = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            table_placeholder = st.empty()

            def update_dashboard():
                # Filter pipeline state to strictly the students belonging to the current discovered batch
                students_list = [pipeline.state[sid] for sid in discovered if sid in pipeline.state]
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

                    ocr_st = getattr(st_entry, "ocr_status", None) if st_entry else None
                    ext_st = getattr(st_entry, "extraction_status", None) if st_entry else None
                    eval_st = getattr(st_entry, "evaluation_status", None) if st_entry else None

                    ocr_mark = "✓" if ocr_st == "completed" else ("✗" if ocr_st == "failed" else ("⏳" if ocr_st == "running" else "—"))
                    ext_mark = "✓" if ext_st == "completed" else ("✗" if ext_st == "failed" else ("⏳" if ext_st == "running" else "—"))
                    eval_mark = "✓" if eval_st == "completed" else ("✗" if eval_st == "failed" else ("⏳" if eval_st == "running" else "—"))

                    status_display = {
                        "completed": "✅ Completed",
                        "processing_ocr": "⚡ Running OCR",
                        "ocr_complete": "📑 OCR Done",
                        "extracting": "🧠 Extracting",
                        "evaluating": "📊 Evaluating",
                        "failed": "❌ Failed",
                        "pending": "⏳ Pending"
                    }.get(cur_st, cur_st)

                    table_rows.append({
                        "Student ID": sid,
                        "Scope": scope_display,
                        "Status": status_display,
                        "OCR": ocr_mark,
                        "Extraction": ext_mark,
                        "Evaluation": eval_mark,
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
            if btn_start_batch or btn_resume_batch or btn_retry_failed or btn_rerun_eval:
                if not ollama_status["ok"]:
                    st.error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is online.")
                    st.stop()

                def on_progress(entry):
                    stage_name = (
                        "OCR (GPU:0)" if entry.stage == "ocr"
                        else ("Structured extraction (Qwen)" if entry.stage == "extraction"
                        else ("Rubric Evaluation (Qwen)" if entry.stage == "evaluation"
                        else (entry.stage or "Processing")))
                    )
                    live_stage_box.info(f"📍 **Currently processing:** Student `{entry.student_id}` | **Stage:** `{stage_name}`")
                    update_dashboard()

                if btn_rerun_eval:
                    status_text.info(f"⏳ Re-running Qwen evaluation on extracted text for {batch_section_id} ({len(selected_student_ids)} students targeted)...")
                    manifest = pipeline.rerun_evaluations_on_extracted(discovered_students=discovered, selected_student_ids=selected_student_ids, progress_cb=on_progress)
                elif btn_retry_failed:
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

            # Interactive Student & Section Report Explorer
            student_json_files = []
            if os.path.exists(pipeline.students_dir):
                student_json_files = sorted([os.path.splitext(f)[0] for f in os.listdir(pipeline.students_dir) if f.endswith(".json")])

            if student_json_files:
                st.divider()
                st.subheader("🔍 Cohort Deliverables & Report Inspector")
                
                inspect_mode = st.radio(
                    "Select View:",
                    options=["👤 Single Student Report & Evaluation", "🌐 Entire Section Aggregated JSON (All Students)"],
                    horizontal=True,
                    key="cohort_inspect_mode"
                )

                if inspect_mode == "👤 Single Student Report & Evaluation":
                    inspected_sid = st.selectbox("Select student to inspect:", student_json_files, key="select_inspect_student")
                    if inspected_sid:
                        json_path = os.path.join(pipeline.students_dir, f"{inspected_sid}.json")
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                st_data = json.load(f)

                            eval_text = st_data.get("evaluation")
                            eval_scores = parse_evaluation_scores(eval_text) if eval_text else {}

                            # Summary Banner with Scores & Status
                            st.write(f"**Student ID:** `{inspected_sid}` | **Section:** `{st_data.get('section_id', batch_section_id)}` | **Week:** `{st_data.get('week_id', batch_week_id)}`")
                            
                            if st_data.get("status") == "failed":
                                err_info = st_data.get("error", {})
                                st.error(f"❌ **Stage:** {err_info.get('stage')} | **Error:** {err_info.get('message')}")
                            else:
                                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                                with score_col1:
                                    score_col1.metric("Total Score", eval_scores.get("score_display", "—"))
                                with score_col2:
                                    score_col2.metric("Grade", eval_scores.get("grade", "—"))
                                with score_col3:
                                    score_col3.metric("Status / Verdict", eval_scores.get("status", "—"))
                                with score_col4:
                                    score_col4.metric("Questions Addressed", eval_scores.get("questions_covered", "—"))

                            # Student Details Tabs
                            t_eval, t_extract, t_ocr, t_json = st.tabs([
                                "📊 Rubric Evaluation Report",
                                "🧠 Structured Extraction (5 Sections)",
                                "📑 Raw OCR Scanned Text",
                                "📄 Student JSON Payload"
                            ])

                            with t_eval:
                                if eval_text:
                                    st.markdown(eval_text)
                                    st.download_button(
                                        label=f"📥 Download {inspected_sid}_evaluation.md",
                                        data=eval_text,
                                        file_name=f"{inspected_sid}_evaluation.md",
                                        mime="text/markdown",
                                        type="primary",
                                        key=f"dl_eval_inspect_{inspected_sid}"
                                    )
                                else:
                                    st.info("ℹ️ Evaluation has not been generated for this student yet. Click 'Resume Incomplete Work' above to run evaluation.")

                            with t_extract:
                                ext_data = st_data.get("extraction", {})
                                det_progs = ext_data.get("detected_programs", [])
                                miss_progs = ext_data.get("missing_programs", [])
                                st.success(f"**Detected Programs ({len(det_progs)}):** {', '.join(det_progs) or 'None'}")
                                if miss_progs:
                                    st.caption(f"**Missing Programs:** {', '.join(miss_progs)}")
                                
                                progs_dict = ext_data.get("programs", {})
                                for p_code, p_info in progs_dict.items():
                                    with st.expander(f"📌 Program {p_code} Details", expanded=False):
                                        st.markdown(f"**Problem Understanding:** {p_info.get('problem_understanding') or 'Not provided'}")
                                        st.markdown(f"**Logic / Approach:** {p_info.get('logic_approach') or 'Not provided'}")
                                        v_list = p_info.get("important_variables", [])
                                        if v_list:
                                            st.markdown("**Important Variables:**")
                                            var_df = []
                                            for v in v_list:
                                                vname = v.get("variable", "") if isinstance(v, dict) else getattr(v, "variable", str(v))
                                                vpurp = v.get("purpose", "") if isinstance(v, dict) else getattr(v, "purpose", "")
                                                var_df.append({"Variable": vname, "Purpose": vpurp})
                                            st.dataframe(var_df, use_container_width=True)
                                        st.markdown(f"**What I Observed:** {p_info.get('what_i_observed') or 'Not provided'}")

                            with t_ocr:
                                ocr_data = st_data.get("ocr", {})
                                st.write(f"**Device:** `{ocr_data.get('device', 'gpu:0')}` | **Pages:** `{ocr_data.get('num_pages', 1)}` | **Total Time:** `{ocr_data.get('total_time', 0.0):.1f}s`")
                                page_breakdown = ocr_data.get("page_breakdown", [])
                                if page_breakdown:
                                    for pb in page_breakdown:
                                        with st.expander(f"📄 Page {pb.get('page', 1)} Raw Text", expanded=False):
                                            st.markdown(pb.get("text", ""))
                                else:
                                    st.markdown(ocr_data.get("text", "No OCR text available."))

                            with t_json:
                                st.download_button(
                                    label=f"📥 Download {inspected_sid}.json",
                                    data=json.dumps(st_data, indent=2, ensure_ascii=False),
                                    file_name=f"{inspected_sid}.json",
                                    mime="application/json",
                                    key=f"dl_json_inspect_{inspected_sid}"
                                )
                                st.json(st_data)

                else:
                    # Entire Section Aggregated JSON Viewer
                    st.write(f"#### 🌐 Aggregated Section Observation Reports (`{os.path.basename(pipeline.complete_json_path)}`)")
                    if os.path.exists(pipeline.complete_json_path):
                        with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
                            complete_data = json.load(f)

                        st.write(f"**Section ID:** `{complete_data.get('section_id', batch_section_id)}` | **Week:** `{complete_data.get('week_id', batch_week_id)}` | **Total Students:** `{complete_data.get('total_students', len(student_json_files))}` | **Successful:** `{complete_data.get('successful', 0)}` | **Failed:** `{complete_data.get('failed', 0)}`")
                        
                        st.download_button(
                            label=f"📥 Download Complete Section JSON ({os.path.basename(pipeline.complete_json_path)})",
                            data=json.dumps(complete_data, indent=2, ensure_ascii=False),
                            file_name=os.path.basename(pipeline.complete_json_path),
                            mime="application/json",
                            type="primary",
                            key="dl_entire_section_json"
                        )
                        st.json(complete_data)
                    else:
                        st.info("Aggregated section JSON file not found on disk yet.")

                # ============================================================
                # COHORT SCORES & EVALUATION SUMMARY TABLE
                # ============================================================
                st.divider()
                st.subheader("📊 Cohort Scores & Evaluation Summary Table")
                st.caption("Consolidated grading table and criteria breakdown for all students in this section.")

                score_table_rows = []
                total_scores_list = []

                for sid in student_json_files:
                    j_path = os.path.join(pipeline.students_dir, f"{sid}.json")
                    if os.path.exists(j_path):
                        try:
                            with open(j_path, "r", encoding="utf-8") as f:
                                s_data = json.load(f)
                            e_text = s_data.get("evaluation", "")
                            e_sc = parse_evaluation_scores(e_text) if e_text else {}
                            if e_sc.get("total_score") is not None:
                                total_scores_list.append(e_sc["total_score"])
                            
                            det_p = s_data.get("extraction", {}).get("detected_programs", [])
                            det_p_str = f"{len(det_p)} ({', '.join(det_p[:6])}{'...' if len(det_p) > 6 else ''})" if det_p else "—"

                            score_table_rows.append({
                                "Student ID": sid,
                                "Total Score": e_sc.get("score_display", "—"),
                                "Grade": e_sc.get("grade", "—"),
                                "Verdict / Status": e_sc.get("status", "—"),
                                "Questions Addressed": e_sc.get("questions_covered", "—"),
                                "Objective (/2)": e_sc.get("objective", "—"),
                                "Problem Understanding (/2)": e_sc.get("problem_understanding", "—"),
                                "Logic / Approach (/2)": e_sc.get("logic_approach", "—"),
                                "Variables Table (/2)": e_sc.get("variables_table", "—"),
                                "What I Observed (/2)": e_sc.get("what_i_observed", "—"),
                                "Detected Programs": det_p_str
                            })
                        except Exception:
                            pass

                if score_table_rows:
                    # Cohort Metrics Row
                    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                    evaluated_count = len(total_scores_list)
                    avg_score = sum(total_scores_list) / evaluated_count if evaluated_count > 0 else 0.0
                    approved_count = sum(1 for r in score_table_rows if "Approved" in r.get("Verdict / Status", ""))
                    top_score = max(total_scores_list) if total_scores_list else 0.0

                    c_m1.metric("Evaluated Students", f"{evaluated_count} / {len(student_json_files)}")
                    c_m2.metric("Cohort Average Score", f"{avg_score:.1f} / 10.0" if evaluated_count > 0 else "—")
                    c_m3.metric("Approved Students", f"{approved_count} / {len(student_json_files)}")
                    c_m4.metric("Highest Score", f"{top_score:.1f} / 10.0" if evaluated_count > 0 else "—")

                    score_df = pd.DataFrame(score_table_rows)
                    st.dataframe(score_df, use_container_width=True)

                    csv_data = score_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Cohort Scores Summary (CSV)",
                        data=csv_data,
                        file_name=f"{batch_section_id}_{batch_week_id}_scores_summary.csv",
                        mime="text/csv",
                        type="primary",
                        key="dl_cohort_scores_csv"
                    )


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