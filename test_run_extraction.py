"""
test_run_extraction.py
Runs structured extraction on real OCR text using Qwen 2.5 Coder 3B and validates against schemas.
"""

import json
from extractor import extract_observation_report

ocr_text = """## Level 1: Compulsory Programs

### 1. Check whether a given number is even or odd

Functions / Concepts Used:
Integer variable, scanf(), printf(), modulus operator (%), equality operator (==), if-else

Explanation:
• Reads an integer using scanf().
• Checks number % 2 == 0 using the modulus operator.
• if-else prints "Even" or "Odd" accordingly.

✓ Program successfully executed.

### 2. Check whether a given number is positive, negative, or zero

Functions / Concepts Used:
Variables, relational operators (>, <, ==), if-else-if ladder

Explanation:
• Reads a number into a variable.
• Compares it with 0 using >, <, == in an if-else-if ladder.
• Prints Positive, Negative, or Zero based on the match.

✓ Program successfully executed.

### 3. Check whether a character is uppercase, lowercase, digit, or special character

Functions / Concepts Used:
char data type, character input, relational operators, logical AND (&&), else-if ladder

Explanation:
• Reads a character using the char data type.
• Uses relational operators with && to check A-Z, a-z, 0-9 ranges.
• else-if ladder prints the matching category.

✓ Program successfully executed.

### 4. Check whether a given year is a leap year

Functions / Concepts Used:
Integer variable, modulus operator (%), relational and logical operators, nested conditions

Explanation:
• Reads the year as an integer.
• Checks divisibility by 4, 100, 400 using % and nested conditions.
• Prints Leap Year or Not a Leap Year based on the rule.

✓ Program successfully executed.

### 5. Print the memory allocation for all data types in C

Functions / Concepts Used:
Data types, printf(), format specifiers, sizeof operator

Explanation:
• Declares variables of int, char, float, double, etc.
• Uses sizeof() to get bytes occupied by each type.
• printf() with format specifiers displays each size.

✓ Program successfully executed.
"""

page_breakdown = [
    {"page": 1, "text": ocr_text}
]

print("Running extract_observation_report with Qwen 2.5 Coder 3B...")
result = extract_observation_report(
    report_text=ocr_text,
    page_breakdown=page_breakdown
)

print("\n=== STATUS ===")
print("Overall status:", result.status)
print("Objective of lab:", result.objective_of_lab)
print("Detected programs:", result.detected_programs)
print("Missing programs:", result.missing_programs)
print("Errors:", result.errors)

print("\n=== P1 DETAILS ===")
print(result.programs["P1"].model_dump_json(indent=2))

print("\n=== P5 DETAILS ===")
print(result.programs["P5"].model_dump_json(indent=2))

print("\n=== P6 (NOT PRESENT) DETAILS ===")
print(result.programs["P6"].model_dump_json(indent=2))

# Save output for inspection
with open("test_extraction_output.json", "w", encoding="utf-8") as f:
    json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
print("\nSaved output to test_extraction_output.json")
