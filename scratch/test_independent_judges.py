import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from unittest.mock import MagicMock, patch
from verifier import evaluate_with_ollama, evaluate_with_gemini

# Test 1: Mock Ollama giving a score of 7.2 (grade B)
mock_ollama_resp = json.dumps({
    "dimensions": [
        {"dimension": "D1", "score": 1.4, "justification": "Ollama D1: Minor syntax warnings."},
        {"dimension": "D2", "score": 1.5, "justification": "Ollama D2: Good logic."},
        {"dimension": "D3", "score": 1.3, "justification": "Ollama D3: Basic report."},
        {"dimension": "D4", "score": 1.5, "justification": "Ollama D4: Good consistency."},
        {"dimension": "D5", "score": 1.5, "justification": "Ollama D5: Average novelty."}
    ],
    "strengths": ["Solid basic code."],
    "improvement_areas": ["Add more details."],
    "overall_summary": "Ollama summary: Completed with basic competency."
})

# Test 2: Mock Gemini giving a score of 9.2 (grade O)
mock_gemini_resp = json.dumps({
    "dimensions": [
        {"dimension": "D1", "score": 1.9, "justification": "Gemini D1: Highly compliant code."},
        {"dimension": "D2", "score": 1.9, "justification": "Gemini D2: Advanced algorithmic structures."},
        {"dimension": "D3", "score": 1.8, "justification": "Gemini D3: Well-structured observation."},
        {"dimension": "D4", "score": 1.8, "justification": "Gemini D4: High code-report alignment."},
        {"dimension": "D5", "score": 1.8, "justification": "Gemini D5: Outstanding recursion logic."}
    ],
    "strengths": ["Exemplary modularity", "Recursive depth"],
    "improvement_areas": ["Minor style polish"],
    "overall_summary": "Gemini summary: Outstanding work across all dimensions."
})

with patch("verifier._call_ollama", return_value=(True, mock_ollama_resp, None)):
    ollama_res = evaluate_with_ollama(student_id="N241003", report_text="Sample OCR", week_id="week-04")

mock_client = MagicMock()
mock_gemini_obj = MagicMock()
mock_gemini_obj.text = mock_gemini_resp
mock_client.models.generate_content.return_value = mock_gemini_obj

with patch("google.genai.Client", return_value=mock_client):
    gemini_res = evaluate_with_gemini(student_id="N241003", report_text="Sample OCR", api_key="dummy", week_id="week-04")

print(f"Ollama Score: {ollama_res.recommended_score} (Grade: {ollama_res.grade})")
print(f"Gemini Score: {gemini_res.recommended_score} (Grade: {gemini_res.grade})")
print(f"Ollama Report Header: {ollama_res.full_report_markdown.splitlines()[0]}")
print(f"Gemini Report Header: {gemini_res.full_report_markdown.splitlines()[0]}")
assert ollama_res.recommended_score == 7.2, f"Expected 7.2, got {ollama_res.recommended_score}"
assert gemini_res.recommended_score == 9.2, f"Expected 9.2, got {gemini_res.recommended_score}"
assert ollama_res.recommended_score != gemini_res.recommended_score, "Scores should be independent!"
print("SUCCESS: Both LLM judges evaluated and scored independently!")
