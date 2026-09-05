import sys
import os
import json
import tempfile
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pymupdf  # Modern PyMuPDF import (replaces deprecated fitz)
import paddle


def _safe_bfloat16_supported(device=None):
    """Handle undefined Paddle device objects on some CUDA builds."""
    try:
        target = paddle.device.get_device() if device is None else device
        if target is None:
            return False

        target_text = str(target).lower()
        if not any(token in target_text for token in ("gpu", "xpu", "npu", "mlu")):
            return False

        try:
            return bool(paddle.amp.is_bfloat16_supported(target))
        except TypeError:
            try:
                if "gpu" in target_text:
                    return bool(paddle.base.core.is_bfloat16_supported(paddle.CUDAPlace(0)))
                return False
            except Exception:
                return False
    except Exception:
        return False


# Make the capability check safe before any model is created.
paddle.amp.is_bfloat16_supported = _safe_bfloat16_supported

# ============================================================
# GPU INITIALIZATION & VALIDATION
# ============================================================

if not paddle.is_compiled_with_cuda():
    raise RuntimeError("Paddle is NOT compiled with CUDA")

try:
    paddle.set_device("gpu:0")
except Exception as e:
    raise RuntimeError(f"Failed to set GPU device: {e}")

if "gpu" not in paddle.device.get_device().lower():
    raise RuntimeError(
        f"Paddle is NOT using GPU: {paddle.device.get_device()}"
    )

try:
    paddle.amp.is_bfloat16_supported = lambda device=None: (
        bool(paddle.base.core.is_bfloat16_supported(paddle.CUDAPlace(0)))
        if (device is None or "gpu" in str(device).lower())
        else False
    )
except Exception:
    pass

from paddleocr import PaddleOCRVL


# ============================================================
# CLI ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(description="PaddleOCR-VL 1.6 Page-by-Page OCR Worker")
parser.add_argument("input_path", help="Path to input PDF or image")
parser.add_argument(
    "--pages",
    type=str,
    default=None,
    help="Comma-separated 1-indexed page numbers to extract (e.g. '1,2' or '1-3'). Default: all pages."
)
parser.add_argument(
    "--max-pages",
    type=int,
    default=None,
    help="Maximum number of pages to extract from the document."
)

args = parser.parse_args()
input_path = os.path.abspath(args.input_path)

if not os.path.exists(input_path):
    print(json.dumps({"success": False, "error": f"File not found: {input_path}"}))
    sys.exit(1)


def parse_page_selection(pages_arg: str, total_pages: int) -> list:
    """Parses page strings like '1,2,3' or '1-3' into a list of valid 1-indexed page numbers."""
    if not pages_arg:
        return list(range(1, total_pages + 1))

    selected = set()
    parts = pages_arg.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            subparts = part.split("-")
            if len(subparts) == 2 and subparts[0].isdigit() and subparts[1].isdigit():
                start = max(1, int(subparts[0]))
                end = min(total_pages, int(subparts[1]))
                for p in range(start, end + 1):
                    selected.add(p)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= total_pages:
                selected.add(p)

    res = sorted(list(selected))
    return res if res else list(range(1, total_pages + 1))


def extract_text_from_result(result) -> str:
    """Extracts markdown text from a PaddleOCRVL result item."""
    page_text = ""
    if hasattr(result, "markdown_texts"):
        value = result.markdown_texts
        if isinstance(value, dict):
            page_text = value.get("markdown_texts", "")
        else:
            page_text = str(value)
    elif hasattr(result, "markdown"):
        value = result.markdown
        if isinstance(value, dict):
            page_text = value.get("markdown_texts", "")
        else:
            page_text = str(value)

    if not page_text and hasattr(result, "parsing_res_list"):
        parsed = []
        for item in getattr(result, "parsing_res_list", []):
            if isinstance(item, dict) and "content" in item:
                parsed.append(item["content"])
        if parsed:
            page_text = "\n\n".join(parsed)

    return str(page_text or "").strip()


# ============================================================
# CREATE MODEL
# ============================================================

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    device="gpu:0",
)


# ============================================================
# EXECUTE EXTRACTION PAGE BY PAGE
# ============================================================

page_breakdown = []
page_texts = []
file_ext = os.path.splitext(input_path)[1].lower()

if file_ext == ".pdf":
    doc = pymupdf.open(input_path)
    total_doc_pages = len(doc)

    pages_to_process = parse_page_selection(args.pages, total_doc_pages)
    if args.max_pages and args.max_pages > 0:
        pages_to_process = pages_to_process[:args.max_pages]

    tmp_sub_path = None
    if len(pages_to_process) < total_doc_pages:
        # Create a lightweight sub-PDF containing only the requested pages
        sub_doc = pymupdf.open()
        for p in pages_to_process:
            sub_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
        tmp_fd, tmp_sub_path = tempfile.mkstemp(suffix="_subset.pdf")
        os.close(tmp_fd)
        sub_doc.save(tmp_sub_path)
        sub_doc.close()
        pdf_target = tmp_sub_path
    else:
        pdf_target = input_path
        pages_to_process = list(range(1, total_doc_pages + 1))
    doc.close()

    sys.stderr.write(f"[PaddleOCR-VL] Running native OCR on {len(pages_to_process)}/{total_doc_pages} pages from PDF...\n")

    try:
        results = pipeline.predict(pdf_target)
        for idx, res in enumerate(results):
            p_num = pages_to_process[idx] if idx < len(pages_to_process) else (idx + 1)
            p_text = extract_text_from_result(res)
            page_breakdown.append({
                "page": p_num,
                "text": p_text
            })
            if p_text:
                page_texts.append(p_text)
    finally:
        if tmp_sub_path and os.path.exists(tmp_sub_path):
            try:
                os.remove(tmp_sub_path)
            except Exception:
                pass

else:
    # Single image file
    sys.stderr.write(f"[PaddleOCR-VL] Running OCR on single image: {os.path.basename(input_path)}...\n")
    results = pipeline.predict(input_path)
    img_text = ""
    for res in results:
        t = extract_text_from_result(res)
        if t:
            img_text = f"{img_text}\n\n{t}".strip() if img_text else t

    page_breakdown.append({
        "page": 1,
        "text": img_text
    })
    if img_text:
        page_texts.append(img_text)

total_pages = len(page_breakdown)

if total_pages > 1:
    combined_parts = []
    for item in page_breakdown:
        combined_parts.append(f"<!-- Page {item['page']} -->\n\n{item['text']}")
    final_text = "\n\n---\n\n".join(combined_parts).strip()
else:
    final_text = "\n\n".join(page_texts).strip()


# ============================================================
# RETURN JSON RESULT
# ============================================================

output = {
    "success": True,
    "text": final_text,
    "num_pages": total_pages,
    "page_breakdown": page_breakdown,
    "device": paddle.device.get_device(),
    "paddle_version": paddle.__version__,
}

print(
    json.dumps(
        output,
        ensure_ascii=False
    )
)