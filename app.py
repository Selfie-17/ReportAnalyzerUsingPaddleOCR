"""
app.py
Academic Laboratory Evaluation Studio - 4-Step Pipeline.
Step 1: Upload Zip
Step 2: Do OCR (PaddleOCR-VL 1.6)
Step 3: Calculate Score (Deterministic Rubric Heuristics)
Step 4: LLM as Judge Recommended Score (Dual Reports: Ollama & Gemini)
"""

import os
import sys
import json
import time
import zipfile
import io
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import pymupdf as fitz  # PyMuPDF for PDF preview rendering

load_dotenv()

from schemas import (
    StudentObservationReport,
    CalculatedScore,
    LLMJudgeEvaluation,
    BatchStudentStatus,
    BatchManifest,
)
from verifier import (
    check_ollama_status,
    calculate_deterministic_score,
    evaluate_with_ollama,
    evaluate_with_gemini,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MODEL,
    find_student_code,
)
from batch_processor import (
    BatchPipeline,
    validate_and_extract_zip,
    discover_student_reports,
    get_paddle_python,
    run_paddle_worker_sync,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Academic Laboratory Evaluation Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean step indicators
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .step-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None

if "section_id" not in st.session_state:
    st.session_state["section_id"] = "SEC2"

if "week_id" not in st.session_state:
    st.session_state["week_id"] = "week-04"

if "active_step" not in st.session_state:
    st.session_state["active_step"] = 0

if "uploaded_zip_name" not in st.session_state:
    st.session_state["uploaded_zip_name"] = None

if "discovered_students" not in st.session_state:
    st.session_state["discovered_students"] = {}


# ============================================================
# SIDEBAR: SYSTEM DIAGNOSTICS & ENGINE SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Evaluation Engines")

    # 1. OCR Engine
    st.subheader("🖥️ Step 2: OCR Engine")
    st.caption("PaddleOCR-VL 1.6 (GPU:0 • Concurrency = 1)")
    with st.expander("Paddle Python Executable"):
        st.code(get_paddle_python(), language="text")

    st.divider()

    # 2. Local Ollama Engine
    st.subheader("🦙 Step 4: Ollama Judge")
    ollama_url = st.text_input("Ollama URL", value=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
    ollama_status = check_ollama_status(base_url=ollama_url)

    if ollama_status["ok"]:
        if ollama_status["model_found"]:
            st.success("🟢 Ollama Connected • Qwen Ready")
        else:
            st.warning(f"🟡 Ollama online • '{DEFAULT_MODEL}' not active")
    else:
        st.error("🔴 Ollama Offline")
        st.caption(ollama_status.get("error", "Run 'ollama serve'"))

    ollama_model = st.text_input("Ollama Model", value=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))

    st.divider()

    # 3. Google Gemini Engine
    st.subheader("♊ Step 4: Gemini Judge")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = st.text_input("Gemini Model", value=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))

    if gemini_key:
        st.success(f"🟢 Gemini API Key Active (`{gemini_key[:8]}...`)")
    else:
        st.warning("🟡 Gemini API Key Missing in .env")

    st.divider()
    temperature = st.slider("Evaluation Temperature", min_value=0.0, max_value=0.7, value=0.1, step=0.05)


# ============================================================
# INITIALIZE / GET BATCH PIPELINE
# ============================================================

def get_or_create_pipeline(section_id: str, week_id: str) -> BatchPipeline:
    pipeline = BatchPipeline(
        week_id=week_id,
        section_id=section_id,
        ollama_url=ollama_url,
        model=ollama_model,
        temperature=temperature
    )
    st.session_state["pipeline"] = pipeline
    return pipeline

pipeline = get_or_create_pipeline(st.session_state["section_id"], st.session_state["week_id"])


# ============================================================
# MAIN APPLICATION HEADER
# ============================================================

st.markdown('<div class="main-title">🔬 Academic Laboratory Evaluation Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Structured 4-Step Pipeline • Deterministic Rubric Scoring • Dual Independent LLM Judges (Ollama & Gemini)</div>',
    unsafe_allow_html=True
)

# 4-Step Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Step 1: Upload Zip",
    "🔍 Step 2: Do OCR",
    "📐 Step 3: Calculate Score",
    "⚖️ Step 4: LLM as Judge (Dual Reports)"
])


# ============================================================
# STEP 1: UPLOAD ZIP
# ============================================================

with tab1:
    st.subheader("📦 Step 1: Ingest Cohort Archive")
    st.caption("Upload a ZIP archive containing student observation reports (PDF/images) and C source code files.")

    c1, c2 = st.columns(2)
    with c1:
        sec_input = st.text_input("Section ID", value=st.session_state["section_id"], key="sec_in")
        if sec_input != st.session_state["section_id"]:
            st.session_state["section_id"] = sec_input
            pipeline = get_or_create_pipeline(sec_input, st.session_state["week_id"])
    with c2:
        wk_input = st.text_input("Week ID", value=st.session_state["week_id"], key="wk_in")
        if wk_input != st.session_state["week_id"]:
            st.session_state["week_id"] = wk_input
            pipeline = get_or_create_pipeline(st.session_state["section_id"], wk_input)

    uploaded_zip = st.file_uploader(
        "Upload Section ZIP File",
        type=["zip"],
        help="Upload cohort ZIP file (e.g. week-4-sec-2.zip)"
    )

    # Also detect if local week-4-sec-2.zip exists in workspace for instant loading
    local_default_zip = f"{st.session_state['week_id']}-{st.session_state['section_id'].lower()}.zip"
    alt_local_zip = "week-4-sec-2.zip"

    use_local_zip = None
    if os.path.exists(local_default_zip):
        use_local_zip = local_default_zip
    elif os.path.exists(alt_local_zip):
        use_local_zip = alt_local_zip

    if use_local_zip and not uploaded_zip:
        st.info(f"💡 Detected local workspace archive: `{use_local_zip}` ({os.path.getsize(use_local_zip) / (1024*1024):.2f} MB)")
        if st.button(f"📥 Load Local Archive ({use_local_zip})", type="secondary"):
            with st.spinner("Extracting and discovering students from local archive..."):
                res = pipeline.step1_ingest_zip(use_local_zip)
                st.session_state["discovered_students"] = res["students"]
                st.session_state["uploaded_zip_name"] = use_local_zip
                st.success(f"Discovered {res['total_students']} student(s) successfully!")
                st.rerun()

    if uploaded_zip is not None:
        if st.session_state.get("uploaded_zip_name") != uploaded_zip.name:
            with st.spinner(f"Extracting {uploaded_zip.name} and scanning submissions..."):
                res = pipeline.step1_ingest_zip(uploaded_zip)
                st.session_state["discovered_students"] = res["students"]
                st.session_state["uploaded_zip_name"] = uploaded_zip.name
                st.success(f"Discovered {res['total_students']} student(s) from {uploaded_zip.name}!")
                st.rerun()

    # Load existing state if already available on disk
    if not st.session_state["discovered_students"] and os.path.exists(pipeline.students_dir):
        discovered = {}
        for fname in os.listdir(pipeline.students_dir):
            if fname.endswith(".json") and not any(fname.endswith(s) for s in ("_manifest.json", "_summary.json", "_reports.json")):
                sid = os.path.splitext(fname)[0]
                discovered[sid] = {"file_path": os.path.join(pipeline.students_dir, fname)}
        if discovered:
            st.session_state["discovered_students"] = discovered

    # Display Discovered Students Table
    discovered_map = st.session_state["discovered_students"]
    if discovered_map:
        st.divider()
        st.write(f"### 📋 Discovered Submissions ({len(discovered_map)} Students)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Students Discovered", len(discovered_map))
        docs_count = sum(1 for s in discovered_map.values() if s.get("file_path"))
        m2.metric("Report Documents Found", docs_count)
        code_count = 0
        for sid, info in discovered_map.items():
            s_dir = info.get("student_dir") or (os.path.dirname(info.get("file_path")) if info.get("file_path") else None)
            coll = find_student_code(student_id=sid, week_id=st.session_state["week_id"], student_dir=s_dir)
            if coll and coll.problems:
                code_count += len(coll.problems)
            elif info.get("code_count"):
                code_count += info.get("code_count", 0)
        m3.metric("C Source Files Ingested", code_count)

        roster_rows = []
        for sid, info in discovered_map.items():
            f_path = info.get("file_path")
            f_name = os.path.basename(f_path) if f_path else "Not Found"
            s_dir = info.get("student_dir") or (os.path.dirname(info.get("file_path")) if info.get("file_path") else None)
            st_code = find_student_code(student_id=sid, week_id=st.session_state["week_id"], student_dir=s_dir)
            c_filenames = [p.source_file for p in st_code.problems] if st_code and st_code.problems else [os.path.basename(cf) for cf in info.get("code_files", [])]
            c_count = len(c_filenames)
            c_display = f"✅ {c_count} file(s) ({', '.join(c_filenames[:3])}{'...' if c_count > 3 else ''})" if c_count > 0 else "⚠️ 0 files"
            roster_rows.append({
                "Student ID": sid,
                "Report Document": f_name,
                "Submitted C Files": c_display,
                "Discovered Status": "Ready for OCR" if f_path else "Missing Document"
            })

        st.dataframe(pd.DataFrame(roster_rows), use_container_width=True)

        # Interactive C Code Inspector
        with st.expander("📂 Inspect Submitted C Source Code Files per Student"):
            code_students = list(discovered_map.keys())
            selected_student_for_code = st.selectbox(
                "Select Student to Inspect Code",
                options=code_students,
                key="code_viewer_student"
            )
            if selected_student_for_code:
                s_info = discovered_map[selected_student_for_code]
                s_dir = s_info.get("student_dir") or (os.path.dirname(s_info.get("file_path")) if s_info.get("file_path") else None)
                st_code = find_student_code(student_id=selected_student_for_code, week_id=st.session_state["week_id"], student_dir=s_dir)
                if st_code and st_code.problems:
                    st.write(f"**Found {len(st_code.problems)} C Program(s) for `{selected_student_for_code}`:**")
                    code_tabs = st.tabs([p.source_file or p.problem_id for p in st_code.problems[:15]])
                    for i, c_tab in enumerate(code_tabs):
                        prob = st_code.problems[i]
                        with c_tab:
                            st.caption(f"Filename: `{prob.source_file}` • Problem ID: `{prob.problem_id}`")
                            st.code(prob.source_code, language="c")
                else:
                    st.info("No C source code files found for this student.")
    else:
        st.info("Please upload a ZIP file or load an existing archive to proceed.")


# ============================================================
# STEP 2: DO OCR
# ============================================================

with tab2:
    st.subheader("🔍 Step 2: Extract Text with PaddleOCR-VL 1.6")
    st.caption("Runs PaddleOCR sequentially on each student's observation report. Extracts verbatim text and page breakdowns.")

    discovered_map = st.session_state["discovered_students"]
    if not discovered_map:
        st.warning("⚠️ No students discovered yet. Please complete Step 1 (Upload Zip) first.")
    else:
        # Status calculation
        ocr_completed = 0
        ocr_failed = 0
        ocr_pending = 0

        for sid in discovered_map:
            st_entry = pipeline.state.get(sid)
            st_ocr = getattr(st_entry, "ocr_status", None)
            if st_ocr == "completed":
                ocr_completed += 1
            elif st_ocr == "failed":
                ocr_failed += 1
            else:
                # Check if JSON on disk has OCR text
                j_path = os.path.join(pipeline.students_dir, f"{sid}.json")
                if os.path.exists(j_path):
                    try:
                        with open(j_path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        if d.get("ocr", {}).get("text"):
                            ocr_completed += 1
                            continue
                    except Exception:
                        pass
                ocr_pending += 1

        c_o1, c_o2, c_o3 = st.columns(3)
        c_o1.metric("OCR Completed", f"{ocr_completed} / {len(discovered_map)}")
        c_o2.metric("OCR Pending", ocr_pending)
        c_o3.metric("OCR Failures", ocr_failed)

        force_ocr = st.checkbox("🔄 Force Re-extract OCR (bypass cached results)", value=False)

        col_ocr_btn1, col_ocr_btn2 = st.columns(2)
        with col_ocr_btn1:
            run_all_ocr = st.button("🚀 Extract OCR for All Students", type="primary")
        with col_ocr_btn2:
            inspected_single = st.selectbox("Or select single student:", options=list(discovered_map.keys()), key="ocr_single_sel")
            run_single_ocr = st.button(f"🔍 Extract OCR for {inspected_single} Only", type="secondary")

        if run_all_ocr or run_single_ocr:
            target_sids = [inspected_single] if run_single_ocr else list(discovered_map.keys())
            p_bar = st.progress(0.0)
            status_text = st.empty()
            errors = []

            for idx, sid in enumerate(target_sids):
                status_text.write(f"Extracting OCR for **{sid}** ({idx+1}/{len(target_sids)})...")
                p_bar.progress(idx / len(target_sids))

                res = pipeline.step2_run_ocr(
                    discovered_students={sid: discovered_map[sid]},
                    force_rerun=force_ocr
                )
                p_bar.progress((idx + 1) / len(target_sids))

                if not res.get(sid, {}).get("success", False):
                    errors.append((sid, res.get(sid, {}).get("error", "Unknown error")))

            status_text.empty()
            p_bar.empty()

            if errors:
                for s_err, msg in errors:
                    st.error(f"❌ OCR extraction failed for **{s_err}**: {msg}")
            else:
                st.success("✅ OCR extraction completed successfully!")
            st.rerun()

        st.divider()
        st.write("### 📑 OCR Document & Text Inspector")

        inspected_sid = inspected_single
        if inspected_sid:
            out_json = os.path.join(pipeline.students_dir, f"{inspected_sid}.json")
            ocr_text = ""
            doc_file = discovered_map.get(inspected_sid, {}).get("file_path")

            if os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    ocr_text = d.get("ocr", {}).get("text", "")
                except Exception:
                    pass

            v_col1, v_col2 = st.columns([1, 1])
            with v_col1:
                st.write(f"**Document File:** `{os.path.basename(doc_file) if doc_file else 'N/A'}`")
                if doc_file and doc_file.lower().endswith(".pdf") and os.path.exists(doc_file):
                    try:
                        doc = fitz.open(doc_file)
                        page_num = st.number_input("Page Preview", min_value=1, max_value=len(doc), value=1)
                        page = doc[page_num - 1]
                        pix = page.get_pixmap(dpi=150)
                        st.image(pix.tobytes("png"), caption=f"{inspected_sid} - Page {page_num}/{len(doc)}", use_container_width=True)
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
                elif doc_file and os.path.exists(doc_file) and doc_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
                    st.image(doc_file, caption=f"{inspected_sid} Scan", use_container_width=True)
                else:
                    st.info("Document file preview not found on disk.")

            with v_col2:
                st.write(f"**Extracted OCR Text:** ({len(ocr_text)} characters)")
                if ocr_text:
                    st.text_area("Extracted Markdown / Text", value=ocr_text, height=450, key=f"ocr_ta_{inspected_sid}")
                    st.download_button(
                        label=f"📥 Download {inspected_sid}_ocr.txt",
                        data=ocr_text,
                        file_name=f"{inspected_sid}_ocr.txt",
                        mime="text/plain"
                    )
                else:
                    st.info(f"No OCR text extracted yet for {inspected_sid}. Click 'Run OCR Extraction' above.")

            # Show C files attached to this student
            s_info = discovered_map.get(inspected_sid, {})
            s_dir = s_info.get("student_dir") or (os.path.dirname(doc_file) if doc_file else None)
            st_code = find_student_code(student_id=inspected_sid, week_id=st.session_state["week_id"], student_dir=s_dir)
            if st_code and st_code.problems:
                st.write(f"##### 💻 Attached C Code Submissions ({len(st_code.problems)} files)")
                with st.expander(f"View {len(st_code.problems)} C files for {inspected_sid}"):
                    code_tabs = st.tabs([p.source_file or p.problem_id for p in st_code.problems[:15]])
                    for i, c_tab in enumerate(code_tabs):
                        prob = st_code.problems[i]
                        with c_tab:
                            st.caption(f"Filename: `{prob.source_file}` • Problem ID: `{prob.problem_id}`")
                            st.code(prob.source_code, language="c")


# ============================================================
# STEP 3: CALCULATE SCORE
# ============================================================

with tab3:
    st.subheader("📐 Step 3: Calculate Deterministic Rubric Score")
    st.caption("100% reproducible algorithmic scoring (0 to 10 marks). Evaluates attempted questions, logic, variables table completeness, observations, and static C code validation. Zero LLM hallucination.")

    discovered_map = st.session_state["discovered_students"]
    if not discovered_map:
        st.warning("⚠️ Please complete Step 1 (Upload Zip) first.")
    else:
        if st.button("⚙️ Calculate Algorithmic Scores for All Students", type="primary"):
            with st.spinner("Calculating deterministic rubric scores across cohort..."):
                scores_res = pipeline.step3_calculate_scores(discovered_students=discovered_map)
                st.success(f"✅ Calculated scores for {len(scores_res)} student(s) successfully!")
                st.rerun()

        # Build and render calculated score table
        calc_rows = []
        total_scores = []
        for sid in discovered_map:
            j_path = os.path.join(pipeline.students_dir, f"{sid}.json")
            if os.path.exists(j_path):
                try:
                    with open(j_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    c_data = data.get("calculated_score")
                    if c_data:
                        tot = c_data.get("total_score", 0.0)
                        total_scores.append(tot)
                        det_p = c_data.get("detected_programs", [])
                        det_p_str = f"{len(det_p)} ({', '.join(det_p[:4])}{'...' if len(det_p)>4 else ''})" if det_p else "—"

                        calc_rows.append({
                            "Student ID": sid,
                            "Calculated Score": c_data.get("score_display", f"{tot:.1f} / 10.0"),
                            "Grade": c_data.get("grade", "—"),
                            "Verdict": c_data.get("status", "—"),
                            "Attempted Programs": det_p_str,
                            "Objective (/2)": c_data.get("objective", 0.0),
                            "Understanding (/2)": c_data.get("problem_understanding", 0.0),
                            "Logic (/2)": c_data.get("logic_approach", 0.0),
                            "Variables Table (/2)": c_data.get("variables_table", 0.0),
                            "Observations (/2)": c_data.get("what_i_observed", 0.0),
                        })
                except Exception:
                    pass

        if calc_rows:
            st.divider()
            m_s1, m_s2, m_s3, m_s4 = st.columns(4)
            avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
            app_count = sum(1 for r in calc_rows if r.get("Verdict") == "Approved")
            m_s1.metric("Calculated Students", len(calc_rows))
            m_s2.metric("Cohort Average", f"{avg_score:.2f} / 10.0")
            m_s3.metric("Approved Count", f"{app_count} / {len(calc_rows)}")
            m_s4.metric("Highest Calculated Score", f"{max(total_scores):.1f} / 10.0" if total_scores else "—")

            df_calc = pd.DataFrame(calc_rows)
            st.dataframe(df_calc, use_container_width=True)

            csv_calc_data = df_calc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Step 3 Calculated Scores (CSV)",
                data=csv_calc_data,
                file_name=f"{st.session_state['section_id']}_{st.session_state['week_id']}_calculated_scores.csv",
                mime="text/csv",
                type="secondary"
            )
        else:
            st.info("No scores calculated yet. Click 'Calculate Algorithmic Scores' above to evaluate cohort.")


# ============================================================
# STEP 4: LLM AS JUDGE (DUAL REPORTS: OLLAMA & GEMINI)
# ============================================================

with tab4:
    st.subheader("⚖️ Step 4: LLM as Judge (Independent Dual Reports)")
    st.caption("Executes qualitative LLM evaluation through Ollama and Google Gemini independently. Generates TWO distinct reports with no ambiguity.")

    discovered_map = st.session_state["discovered_students"]
    if not discovered_map:
        st.warning("⚠️ Please complete Step 1 (Upload Zip) first.")
    else:
        # Engine selection
        # Engine selection
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            run_ollama_chk = st.checkbox("🦙 Run Ollama Judge (Qwen 2.5 Coder 3B)", value=True)
        with c_p2:
            run_gemini_chk = st.checkbox("♊ Run Gemini Judge (Google Gemini API)", value=bool(gemini_key))
        with c_p3:
            skip_existing_chk = st.checkbox("⏩ Resume / Skip Completed", value=True, help="Skip students who already have completed evaluations.")

        providers_to_run = []
        if run_ollama_chk:
            providers_to_run.append("ollama")
        if run_gemini_chk:
            providers_to_run.append("gemini")

        if st.button("⚖️ Run LLM as Judge Evaluation", type="primary", disabled=not providers_to_run):
            p_bar = st.progress(0.0)
            status_text = st.empty()

            total_sids = list(discovered_map.keys())
            for idx, sid in enumerate(total_sids):
                status_text.write(f"Running LLM Judge ({', '.join(providers_to_run)}) for **{sid}** ({idx+1}/{len(total_sids)})...")
                p_bar.progress(idx / len(total_sids))

                try:
                    pipeline.step4_run_llm_judges(
                        providers=providers_to_run,
                        discovered_students={sid: discovered_map[sid]},
                        skip_existing=skip_existing_chk,
                        gemini_model=gemini_model,
                        gemini_key=gemini_key,
                        auto_build_artifacts=False
                    )
                except Exception as e:
                    st.warning(f"Warning during evaluation of {sid}: {e}")

                p_bar.progress((idx + 1) / len(total_sids))

            # Build summary artifacts and zips ONCE at the end of the batch
            status_text.write("Packaging results and generating download archives...")
            pipeline.save_manifest()
            pipeline.build_scores_summary_csv()
            if "ollama" in providers_to_run:
                pipeline.create_model_reports_zip("ollama")
                pipeline.build_provider_section_json("ollama")
            if "gemini" in providers_to_run:
                pipeline.create_model_reports_zip("gemini")
                pipeline.build_provider_section_json("gemini")

            status_text.empty()
            p_bar.empty()
            st.success("✅ LLM as Judge evaluation completed for all students!")
            st.rerun()

        # ============================================================
        # TRI-SCORE COMPARISON TABLE (Step 3 vs Ollama vs Gemini)
        # ============================================================
        st.divider()
        st.write("### 📊 Tri-Score Comparison Table (Calculated vs Ollama vs Gemini)")

        comparison_rows = []
        for sid in discovered_map:
            j_path = os.path.join(pipeline.students_dir, f"{sid}.json")
            if os.path.exists(j_path):
                try:
                    with open(j_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    c_data = data.get("calculated_score", {})
                    o_data = data.get("ollama_evaluation", {})
                    g_data = data.get("gemini_evaluation", {})

                    c_score = c_data.get("total_score")
                    o_score = o_data.get("recommended_score")
                    g_score = g_data.get("recommended_score")

                    # Delta between judges
                    delta_str = "—"
                    if o_score is not None and g_score is not None:
                        d_val = abs(o_score - g_score)
                        delta_str = f"Δ {d_val:.1f}"

                    comparison_rows.append({
                        "Student ID": sid,
                        "Step 3: Calculated Score": f"{c_score:.1f} / 10.0" if c_score is not None else "—",
                        "Calculated Grade": c_data.get("grade", "—"),
                        "Step 4: Ollama Score": f"{o_score:.1f} / 10.0" if o_score is not None else "—",
                        "Ollama Grade": o_data.get("grade", "—"),
                        "Step 4: Gemini Score": f"{g_score:.1f} / 10.0" if g_score is not None else "—",
                        "Gemini Grade": g_data.get("grade", "—"),
                        "Judge Delta": delta_str,
                        "Consensus Status": "Approved" if (c_data.get("status") == "Approved" or o_data.get("status") == "Approved" or g_data.get("status") == "Approved") else "Needs Revision"
                    })
                except Exception:
                    pass

        if comparison_rows:
            df_comp = pd.DataFrame(comparison_rows)
            st.dataframe(df_comp, use_container_width=True)

            csv_comp_data = df_comp.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Tri-Score Comparison Table (CSV)",
                data=csv_comp_data,
                file_name=f"{st.session_state['section_id']}_{st.session_state['week_id']}_tri_score_comparison.csv",
                mime="text/csv",
                type="secondary"
            )

        # ============================================================
        # STUDENT DUAL REPORT INSPECTOR
        # ============================================================
        st.divider()
        st.write("### 🔍 Student Dual Report Inspector")

        eval_sid = st.selectbox("Select Student for In-Depth Report Inspection", options=list(discovered_map.keys()), key="eval_sel")
        if eval_sid:
            j_path = os.path.join(pipeline.students_dir, f"{eval_sid}.json")
            st_data = {}
            if os.path.exists(j_path):
                with open(j_path, "r", encoding="utf-8") as f:
                    st_data = json.load(f)

            ollama_eval = st_data.get("ollama_evaluation", {})
            gemini_eval = st_data.get("gemini_evaluation", {})
            calc_data = st_data.get("calculated_score", {})
            ocr_data = st_data.get("ocr", {})

            r_tab_ollama, r_tab_gemini, r_tab_side, r_tab_calc, r_tab_ocr = st.tabs([
                "🦙 Ollama Report",
                "♊ Gemini Report",
                "⚖️ Side-by-Side Comparison",
                "📐 Calculated Scorecard",
                "📑 Raw OCR Evidence"
            ])

            with r_tab_ollama:
                if ollama_eval and ollama_eval.get("full_report_markdown"):
                    st.write(f"### 🦙 Ollama Evaluation Report (`{ollama_eval.get('model_name', 'qwen2.5-coder:3b')}`)")
                    st.metric("Ollama Recommended Score", f"{ollama_eval.get('recommended_score', 0.0):.1f} / 10.0", f"Grade: {ollama_eval.get('grade', '—')}")
                    st.markdown(ollama_eval.get("full_report_markdown", ""))

                    col_dl1, col_dl2 = st.columns(2)
                    col_dl1.download_button(
                        label=f"📥 Download {eval_sid}_ollama_evaluation.md",
                        data=ollama_eval.get("full_report_markdown", ""),
                        file_name=f"{eval_sid}_ollama_evaluation.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    col_dl2.download_button(
                        label=f"📄 Download {eval_sid}_ollama_report.json",
                        data=json.dumps(ollama_eval, indent=2),
                        file_name=f"{eval_sid}_ollama_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No Ollama evaluation generated yet for this student.")

            with r_tab_gemini:
                if gemini_eval and gemini_eval.get("full_report_markdown"):
                    st.write(f"### ♊ Google Gemini Evaluation Report (`{gemini_eval.get('model_name', 'gemini-3.6-flash')}`)")
                    st.metric("Gemini Recommended Score", f"{gemini_eval.get('recommended_score', 0.0):.1f} / 10.0", f"Grade: {gemini_eval.get('grade', '—')}")
                    st.markdown(gemini_eval.get("full_report_markdown", ""))

                    col_g1, col_g2 = st.columns(2)
                    col_g1.download_button(
                        label=f"📥 Download {eval_sid}_gemini_evaluation.md",
                        data=gemini_eval.get("full_report_markdown", ""),
                        file_name=f"{eval_sid}_gemini_evaluation.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    col_g2.download_button(
                        label=f"📄 Download {eval_sid}_gemini_report.json",
                        data=json.dumps(gemini_eval, indent=2),
                        file_name=f"{eval_sid}_gemini_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No Gemini evaluation generated yet for this student.")

            with r_tab_side:
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.write("#### 🦙 Ollama Judge")
                    if ollama_eval:
                        st.metric("Score", f"{ollama_eval.get('recommended_score', 0.0):.1f} / 10.0")
                        st.write(f"**Grade:** `{ollama_eval.get('grade', '—')}` | **Status:** `{ollama_eval.get('status', '—')}`")
                        st.write("**Key Strengths:**")
                        for s in ollama_eval.get("strengths", []):
                            st.write(f"- {s}")
                        st.write("**Recommendations:**")
                        for r in ollama_eval.get("recommendations", []):
                            st.write(f"- {r}")
                    else:
                        st.info("Ollama report not generated.")

                with s_col2:
                    st.write("#### ♊ Gemini Judge")
                    if gemini_eval:
                        st.metric("Score", f"{gemini_eval.get('recommended_score', 0.0):.1f} / 10.0")
                        st.write(f"**Grade:** `{gemini_eval.get('grade', '—')}` | **Status:** `{gemini_eval.get('status', '—')}`")
                        st.write("**Key Strengths:**")
                        for s in gemini_eval.get("strengths", []):
                            st.write(f"- {s}")
                        st.write("**Recommendations:**")
                        for r in gemini_eval.get("recommendations", []):
                            st.write(f"- {r}")
                    else:
                        st.info("Gemini report not generated.")

            with r_tab_calc:
                st.write("#### 📐 Step 3 Calculated Rubric Scorecard")
                if calc_data:
                    st.json(calc_data)
                else:
                    st.info("Calculated scorecard not available.")

            with r_tab_ocr:
                st.write("#### 📑 Step 2 Raw OCR Text")
                if ocr_data and ocr_data.get("text"):
                    st.text_area("OCR Text", value=ocr_data.get("text", ""), height=350, key=f"ocr_show_{eval_sid}")
                else:
                    st.info("No OCR text available.")

        # ============================================================
        # SECTION DOWNLOAD CENTER
        # ============================================================
        st.divider()
        st.write("### 📦 Section Batch Download Center")

        # 1. Dedicated Overall Section JSON Downloads (Separate Ollama & Gemini)
        st.write("#### 📄 Overall Section JSON Downloads (Model-Specific)")
        j_col1, j_col2 = st.columns(2)

        with j_col1:
            ollama_sec_json = os.path.join(pipeline.batch_output_dir, f"{pipeline.section_id or 'SEC1'}_{pipeline.week_id}_ollama_evaluations.json")
            if not os.path.exists(ollama_sec_json) and hasattr(pipeline, "build_provider_section_json"):
                try:
                    pipeline.build_provider_section_json("ollama")
                except Exception:
                    pass
            if os.path.exists(ollama_sec_json):
                with open(ollama_sec_json, "r", encoding="utf-8") as f:
                    o_sec_data = json.load(f)
                num_evals = o_sec_data.get("evaluated_students", len(o_sec_data.get("students", {})))
                st.download_button(
                    label=f"🦙 Download Overall Ollama Section JSON ({num_evals} students)",
                    data=json.dumps(o_sec_data, indent=2),
                    file_name=os.path.basename(ollama_sec_json),
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.caption("Ollama section JSON not generated yet.")

        with j_col2:
            gemini_sec_json = os.path.join(pipeline.batch_output_dir, f"{pipeline.section_id or 'SEC1'}_{pipeline.week_id}_gemini_evaluations.json")
            if not os.path.exists(gemini_sec_json) and hasattr(pipeline, "build_provider_section_json"):
                try:
                    pipeline.build_provider_section_json("gemini")
                except Exception:
                    pass
            if os.path.exists(gemini_sec_json):
                with open(gemini_sec_json, "r", encoding="utf-8") as f:
                    g_sec_data = json.load(f)
                num_evals = g_sec_data.get("evaluated_students", len(g_sec_data.get("students", {})))
                st.download_button(
                    label=f"♊ Download Overall Gemini Section JSON ({num_evals} students)",
                    data=json.dumps(g_sec_data, indent=2),
                    file_name=os.path.basename(gemini_sec_json),
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.caption("Gemini section JSON not generated yet.")

        # 2. ZIP Archives and Aggregated Reports
        st.write("#### 📦 Report ZIP Archives & Complete Aggregates")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            ollama_zip_path = os.path.join(pipeline.batch_output_dir, f"{pipeline.section_id or 'SEC1'}_{pipeline.week_id}_ollama_reports.zip")
            if os.path.exists(ollama_zip_path):
                with open(ollama_zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download All Ollama Reports (.zip)",
                        data=f.read(),
                        file_name=os.path.basename(ollama_zip_path),
                        mime="application/zip",
                        use_container_width=True
                    )
            else:
                st.caption("Ollama reports zip not generated yet.")

        with dl_col2:
            gemini_zip_path = os.path.join(pipeline.batch_output_dir, f"{pipeline.section_id or 'SEC1'}_{pipeline.week_id}_gemini_reports.zip")
            if os.path.exists(gemini_zip_path):
                with open(gemini_zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download All Gemini Reports (.zip)",
                        data=f.read(),
                        file_name=os.path.basename(gemini_zip_path),
                        mime="application/zip",
                        use_container_width=True
                    )
            else:
                st.caption("Gemini reports zip not generated yet.")

        with dl_col3:
            if os.path.exists(pipeline.complete_json_path):
                with open(pipeline.complete_json_path, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                st.download_button(
                    label="📄 Download Section Aggregated JSON",
                    data=json.dumps(comp_data, indent=2),
                    file_name=os.path.basename(pipeline.complete_json_path),
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.caption("Complete aggregated JSON not built yet.")