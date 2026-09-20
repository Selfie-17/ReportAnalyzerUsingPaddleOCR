import re
import json

def parse_robust_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    # Attempt 1: direct parse with strict=False
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Attempt 2: sanitize invalid backslash escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    sanitized = re.sub(r'\\(?![/"\\bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
    try:
        return json.loads(sanitized, strict=False)
    except Exception:
        pass

    # Attempt 3: replace raw newlines in string literals
    try:
        # replace unescaped control chars
        sanitized2 = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', sanitized)
        return json.loads(sanitized2, strict=False)
    except Exception as e:
        print(f"JSON Parse failed: {e}")
        return None

# Test with invalid escape
test_str = '{"evidence": "Checks \\0 and \\c and \\%d in code", "lines": "Line 1\nLine 2"}'
res = parse_robust_json(test_str)
print("Result:", res)
