import os
import sys
from dotenv import load_dotenv
load_dotenv()
from batch_processor import BatchPipeline

print("=== STARTING FULL BATCH LLM EVALUATION ===")
pipeline = BatchPipeline(
    week_id="week-04",
    section_id="SEC2"
)

def progress_callback(status):
    print(f"[{status.student_id}] status={status.status}, ollama={status.ollama_evaluation_status} ({status.ollama_score}), gemini={status.gemini_evaluation_status} ({status.gemini_score})", flush=True)

results = pipeline.step4_run_llm_judges(
    providers=["ollama", "gemini"],
    skip_existing=True,
    progress_cb=progress_callback
)

print("=== BATCH EVALUATION FINISHED SUCCESSFULLY ===")
