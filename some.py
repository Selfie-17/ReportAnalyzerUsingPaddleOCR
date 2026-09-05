
import streamlit as st
import subprocess
import sys
import tempfile
import os
import json
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PaddleOCR-VL 1.6",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔍 PaddleOCR-VL 1.6")

st.caption(
    "GPU-accelerated OCR using a dedicated PaddleOCR worker"
)


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

st.subheader("🖥️ Environment")

c1, c2 = st.columns(2)

with c1:
    st.write("**Python:**")
    st.code(sys.executable)

with c2:
    st.write("**Worker:**")
    st.code("paddle_worker.py")


# ============================================================
# FILE UPLOAD
# ============================================================

st.divider()

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tif",
        "tiff"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload an image to start OCR."
    )

    st.stop()


# ============================================================
# PREVIEW
# ============================================================

st.subheader("🖼️ Input Image")

left, right = st.columns(
    [3, 1]
)


with left:

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        width="stretch"
    )


with right:

    st.write(
        "**Filename**"
    )

    st.write(
        uploaded_file.name
    )

    st.write(
        "**Size**"
    )

    st.write(
        f"{uploaded_file.size / 1024:.2f} KB"
    )

    st.write(
        "**Type**"
    )

    st.write(
        uploaded_file.type
    )


# ============================================================
# EXTRACT BUTTON
# ============================================================

st.divider()

extract_button = st.button(
    "🚀 Extract Text",
    type="primary",
    width="stretch"
)


# ============================================================
# OCR
# ============================================================

if extract_button:

    temp_path = None

    try:

        # ====================================================
        # SAVE UPLOADED IMAGE
        # ====================================================

        extension = os.path.splitext(
            uploaded_file.name
        )[1]


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        # ====================================================
        # WORKER PATH
        # ====================================================

        worker_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "paddle_worker.py"
        )


        if not os.path.exists(worker_path):

            st.error(
                f"Worker not found:\n{worker_path}"
            )

            st.stop()


        # ====================================================
        # RUN WORKER
        # ====================================================

        st.subheader(
            "⚙️ PaddleOCR-VL"
        )


        status = st.empty()


        status.info(
            "Starting PaddleOCR-VL worker..."
        )


        start_time = time.perf_counter()


        process = subprocess.run(
            [
                sys.executable,
                worker_path,
                temp_path
            ],

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace"
        )


        total_time = (
            time.perf_counter()
            - start_time
        )


        # ====================================================
        # PROCESS FAILED
        # ====================================================

        if process.returncode != 0:

            status.error(
                "PaddleOCR-VL worker failed."
            )


            st.error(
                "OCR process returned an error."
            )


            with st.expander(
                "🔴 Worker Error"
            ):

                st.code(
                    process.stderr
                )


            st.stop()


        # ====================================================
        # PARSE WORKER OUTPUT
        # ====================================================

        stdout = process.stdout.strip()


        if not stdout:

            st.error(
                "Worker returned no output."
            )

            with st.expander(
                "Worker stdout"
            ):

                st.code(
                    process.stdout
                )

            st.stop()


        # ====================================================
        # FIND JSON
        #
        # Paddle/PaddleX prints logs before our JSON.
        # Therefore we search for the final JSON line.
        # ====================================================

        worker_result = None


        lines = stdout.splitlines()


        for line in reversed(lines):

            line = line.strip()


            if not line:
                continue


            try:

                candidate = json.loads(
                    line
                )


                if isinstance(
                    candidate,
                    dict
                ):

                    worker_result = candidate

                    break


            except json.JSONDecodeError:

                continue


        # ====================================================
        # JSON NOT FOUND
        # ====================================================

        if worker_result is None:

            st.error(
                "Could not parse worker output."
            )


            with st.expander(
                "🔧 Worker stdout"
            ):

                st.code(
                    stdout
                )


            with st.expander(
                "🔴 Worker stderr"
            ):

                st.code(
                    process.stderr
                )


            st.stop()


        # ====================================================
        # WORKER SUCCESS
        # ====================================================

        if not worker_result.get(
            "success",
            False
        ):

            st.error(
                "PaddleOCR-VL reported failure."
            )


            st.json(
                worker_result
            )


            st.stop()


        status.success(
            "PaddleOCR-VL completed successfully."
        )


        # ====================================================
        # TEXT
        # ====================================================

        final_text = str(
            worker_result.get(
                "text",
                ""
            )
        ).strip()


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "📝 Extracted Text"
        )


        if final_text:

            st.text_area(
                "OCR Result",
                value=final_text,
                height=300
            )


            # =================================================
            # METRICS
            # =================================================

            word_count = len(
                final_text.split()
            )


            character_count = len(
                final_text
            )


            m1, m2, m3, m4 = st.columns(
                4
            )


            with m1:

                st.metric(
                    "⚡ Total Time",
                    f"{total_time:.2f}s"
                )


            with m2:

                st.metric(
                    "📝 Words",
                    word_count
                )


            with m3:

                st.metric(
                    "🔤 Characters",
                    character_count
                )


            with m4:

                st.metric(
                    "🎮 Device",
                    worker_result.get(
                        "device",
                        "Unknown"
                    )
                )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.download_button(
                label="📄 Download TXT",
                data=final_text,
                file_name="ocr_result.txt",
                mime="text/plain",
                width="stretch"
            )


        else:

            st.warning(
                "PaddleOCR-VL completed but "
                "returned empty text."
            )


        # ====================================================
        # DEBUG
        # ====================================================

        with st.expander(
            "🔧 Debug Information"
        ):

            st.write(
                "Worker Python:",
                sys.executable
            )

            st.write(
                "Paddle version:",
                worker_result.get(
                    "paddle_version",
                    "Unknown"
                )
            )

            st.write(
                "Device:",
                worker_result.get(
                    "device",
                    "Unknown"
                )
            )

            st.write(
                "Total execution time:",
                f"{total_time:.2f}s"
            )

            st.write(
                "Worker return code:",
                process.returncode
            )


        # ====================================================
        # OPTIONAL RAW LOG
        # ====================================================

        with st.expander(
            "📋 Paddle Worker Log"
        ):

            st.code(
                stdout
            )


    # ========================================================
    # PYTHON EXCEPTION
    # ========================================================

    except Exception as e:

        st.error(
            "Failed to run PaddleOCR worker."
        )

        st.exception(e)


    finally:

        # ====================================================
        # DELETE TEMP FILE
        # ====================================================

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:
                pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PaddleOCR-VL 1.6 • PaddlePaddle 3.3.0 • RTX 4050 6GB"
)