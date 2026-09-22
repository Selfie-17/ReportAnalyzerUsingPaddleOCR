"""
gemini_ocr.py
Text extraction engine using Google Gemini API (Multimodal Document Transcription).
Supports multi-key rotation, automatic quota failover (429 / RESOURCE_EXHAUSTED),
PDF and image ingestion, and page-by-page breakdown compatible with downstream scoring.
"""

import os
import re
import json
import time
import threading
from typing import Any, Dict, List, Optional, Union
import pymupdf as fitz


class GeminiKeyManager:
    """
    Manages a pool of Google Gemini API keys.
    Provides thread-safe round-robin rotation, usage tracking, and
    automatic failover when a key encounters quota limits (HTTP 429).
    """

    def __init__(
        self,
        api_keys: Optional[Union[List[str], str]] = None,
        exhaustion_cooldown_seconds: int = 60
    ):
        self._lock = threading.Lock()
        self.cooldown_seconds = exhaustion_cooldown_seconds
        self._index = 0
        self._keys: List[str] = []
        self._exhausted_until: Dict[str, float] = {}

        self.reload_keys(api_keys)

    def reload_keys(self, api_keys: Optional[Union[List[str], str]] = None) -> None:
        """Parses API keys from parameters or environment variables."""
        with self._lock:
            raw_keys: List[str] = []

            if api_keys:
                if isinstance(api_keys, str):
                    raw_keys = self._parse_keys_string(api_keys)
                elif isinstance(api_keys, list):
                    raw_keys = [str(k).strip() for k in api_keys if str(k).strip()]
            else:
                # 1. Check GEMINI_API_KEYS (list or comma-separated)
                env_multi = os.environ.get("GEMINI_API_KEYS", "").strip()
                if env_multi:
                    raw_keys = self._parse_keys_string(env_multi)

                # 2. Check GEMINI_API_KEY (single key)
                env_single = os.environ.get("GEMINI_API_KEY", "").strip()
                if env_single and env_single not in raw_keys:
                    raw_keys.append(env_single)

            # Deduplicate while preserving order
            seen = set()
            cleaned = []
            for k in raw_keys:
                k_clean = k.strip().strip("'\"")
                if k_clean and k_clean not in seen and not k_clean.startswith("your_"):
                    seen.add(k_clean)
                    cleaned.append(k_clean)

            self._keys = cleaned
            self._index = 0

    @staticmethod
    def _parse_keys_string(raw: str) -> List[str]:
        """Parses comma-separated, newline-separated, or JSON list string into a list of keys."""
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(k).strip() for k in parsed if str(k).strip()]
            except Exception:
                pass
        # Split on commas or newlines
        tokens = re.split(r"[\r\n,]+", raw)
        return [t.strip().strip("'\"") for t in tokens if t.strip()]

    def has_keys(self) -> bool:
        """Returns True if at least one API key is registered."""
        with self._lock:
            return len(self._keys) > 0

    def key_count(self) -> int:
        """Returns total number of configured keys."""
        with self._lock:
            return len(self._keys)

    def get_all_keys(self) -> List[str]:
        """Returns a copy of all loaded keys."""
        with self._lock:
            return list(self._keys)

    def get_next_key(self) -> Optional[str]:
        """
        Retrieves the next available key in round-robin order.
        Skips temporarily exhausted keys unless all keys are exhausted.
        """
        with self._lock:
            if not self._keys:
                return None

            now = time.time()
            total = len(self._keys)

            # Try to find a non-exhausted key starting from current index
            for i in range(total):
                idx = (self._index + i) % total
                candidate = self._keys[idx]
                cooldown = self._exhausted_until.get(candidate, 0)
                if now >= cooldown:
                    self._index = (idx + 1) % total
                    return candidate

            # If all are in cooldown, pick the one that expires soonest
            soonest_key = min(self._keys, key=lambda k: self._exhausted_until.get(k, 0))
            self._index = (self._keys.index(soonest_key) + 1) % total
            return soonest_key

    def mark_exhausted(self, key: str, cooldown_seconds: Optional[int] = None) -> None:
        """Marks a key as temporarily exhausted due to rate limits (429)."""
        duration = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        with self._lock:
            if key in self._keys:
                self._exhausted_until[key] = time.time() + duration


# Global shared manager instance
_GLOBAL_KEY_MANAGER: Optional[GeminiKeyManager] = None


def get_gemini_key_manager() -> GeminiKeyManager:
    """Returns the singleton instance of GeminiKeyManager."""
    global _GLOBAL_KEY_MANAGER
    if _GLOBAL_KEY_MANAGER is None:
        _GLOBAL_KEY_MANAGER = GeminiKeyManager()
    return _GLOBAL_KEY_MANAGER


def _parse_page_breakdown(full_text: str, total_pages: int) -> List[Dict[str, Any]]:
    """
    Parses <!-- Page X --> markers into structured page breakdowns.
    Falls back to single or proportional segmentation if markers are absent.
    """
    page_marker_pat = re.compile(r"<!--\s*Page\s*(\d+)\s*-->", re.IGNORECASE)
    matches = list(page_marker_pat.finditer(full_text))

    if matches:
        breakdown: List[Dict[str, Any]] = []
        for i, m in enumerate(matches):
            p_num = int(m.group(1))
            start_pos = m.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            p_txt = full_text[start_pos:end_pos].strip()
            # Clean possible markdown hr dividers
            p_txt = re.sub(r"^---+\s*", "", p_txt).strip()
            breakdown.append({
                "page": p_num,
                "text": p_txt
            })
        return breakdown

    # Fallback when no page markers returned
    if total_pages <= 1:
        return [{"page": 1, "text": full_text.strip()}]

    # Return full text assigned to page 1
    return [{"page": 1, "text": full_text.strip()}]


def extract_text_with_gemini(
    file_path: str,
    model: str = "gemini-3.6-flash",
    key_manager: Optional[GeminiKeyManager] = None,
    temperature: float = 0.0,
    max_key_attempts: int = 5
) -> Dict[str, Any]:
    """
    Extracts text from a student laboratory observation report (PDF or image) using Google Gemini.
    Rotates keys automatically on 429 quota exhaustion.

    Returns dict matching PaddleOCR worker format:
    {
        "success": bool,
        "text": str,
        "num_pages": int,
        "page_breakdown": List[Dict[str, Any]],
        "engine": "gemini",
        "model": str,
        "total_time": float,
        "error": Optional[str]
    }
    """
    mgr = key_manager or get_gemini_key_manager()
    if not mgr.has_keys():
        return {
            "success": False,
            "error": "No Gemini API keys found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env",
            "text": "",
            "num_pages": 0,
            "page_breakdown": [],
            "engine": "gemini",
            "total_time": 0.0
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "text": "",
            "num_pages": 0,
            "page_breakdown": [],
            "engine": "gemini",
            "total_time": 0.0
        }

    start_time = time.perf_counter()
    ext = os.path.splitext(file_path)[1].lower()

    # Determine total pages and MIME type
    total_pages = 1
    mime_type = "application/pdf"
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
        except Exception:
            total_pages = 1
        mime_type = "application/pdf"
    elif ext in (".png", ".webp"):
        mime_type = f"image/{ext[1:]}"
    elif ext in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif ext in (".bmp", ".tiff"):
        mime_type = "image/png"
    else:
        mime_type = "application/octet-stream"

    # Read binary content
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    system_instruction = (
        "You are an expert OCR transcription engine specializing in academic laboratory observation reports and lab manuals. "
        "Perform a complete, faithful, verbatim transcription of the provided document.\n\n"
        "Rules:\n"
        "1. Transcribe all text verbatim. Preserve all section headers:\n"
        "   - Aim / Objective\n"
        "   - Problem Description / Understanding\n"
        "   - Algorithm / Flowchart / Logic / Approach\n"
        "   - Variables Table (format strictly as a Markdown table with columns: Variable Name, Data Type, Purpose/Description)\n"
        "   - Program / Source Code (format strictly as Markdown C code block: ```c ... ```)\n"
        "   - Sample Input / Output / Expected Results\n"
        "   - What I Observed / Results / Conclusion\n"
        "2. For every page in the document, mark its beginning with an HTML comment marker:\n"
        "   <!-- Page 1 -->\n"
        "   ... content of page 1 ...\n"
        "   <!-- Page 2 -->\n"
        "   ... content of page 2 ...\n"
        "3. Do NOT summarize. Do NOT omit any handwritten or printed text. Do NOT add conversational commentary.\n"
        "4. Output verbatim transcribed markdown only."
    )

    prompt = (
        f"Please transcribe this academic laboratory observation report document ({total_pages} page(s)). "
        "Preserve all headings, tables, C code blocks, and separate each page with '<!-- Page <number> -->'."
    )

    from google import genai
    from google.genai import types

    last_error: Optional[str] = None
    successful_model = model

    candidate_models = [model]
    for alt_m in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
        if alt_m not in candidate_models:
            candidate_models.append(alt_m)

    attempts = 0
    while attempts < max_key_attempts:
        current_key = mgr.get_next_key()
        if not current_key:
            break

        attempts += 1
        client = genai.Client(api_key=current_key)

        for target_m in candidate_models:
            try:
                part = types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type
                )
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature
                )

                resp = client.models.generate_content(
                    model=target_m,
                    contents=[part, prompt],
                    config=config
                )

                extracted_text = (resp.text or "").strip()
                if extracted_text:
                    elapsed = time.perf_counter() - start_time
                    page_breakdown = _parse_page_breakdown(extracted_text, total_pages)

                    # Build combined text with page dividers if multiple pages
                    if len(page_breakdown) > 1:
                        combined_parts = []
                        for item in page_breakdown:
                            combined_parts.append(f"<!-- Page {item['page']} -->\n\n{item['text']}")
                        final_text = "\n\n---\n\n".join(combined_parts).strip()
                    else:
                        final_text = extracted_text

                    return {
                        "success": True,
                        "text": final_text,
                        "num_pages": total_pages,
                        "page_breakdown": page_breakdown,
                        "engine": "gemini",
                        "model": target_m,
                        "total_time": round(elapsed, 2),
                        "error": None
                    }
            except Exception as e:
                err_msg = str(e)
                last_error = err_msg

                # Check if error is quota exhaustion / rate limit (429 / RESOURCE_EXHAUSTED)
                is_rate_limit = any(token in err_msg.lower() for token in [
                    "429", "resource_exhausted", "quota", "rate limit"
                ])

                if is_rate_limit:
                    # Mark key as exhausted and break to try the next key
                    mgr.mark_exhausted(current_key, cooldown_seconds=60)
                    break
                else:
                    # Model not found or other non-quota error, try alternative model
                    continue

    elapsed = time.perf_counter() - start_time
    return {
        "success": False,
        "error": f"Gemini OCR failed: {last_error or 'All keys and retries exhausted'}",
        "text": "",
        "num_pages": total_pages,
        "page_breakdown": [],
        "engine": "gemini",
        "model": successful_model,
        "total_time": round(elapsed, 2)
    }
