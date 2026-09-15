# Unified Student Evaluation

**Student ID:** `N241003` | **Week:** `week-04`

## 1. Submission Summary

| Metric | Value |
|---|---:|
| Code Programs | 17 |
| Report Entries | 4 |
| Matched | 4 |
| Undocumented | 13 |

## 2. Reported Programs

| Report | Code | File | Program | Status |
|---|---|---|---|---|
| R1 | P3 | `p3.c` | Find Length of a String Without using strlen | Matched |
| R2 | P5 | `p5.c` | Find Factorial Using Recursion. | Matched |
| R3 | P7 | `p7.c` | Print Armstrong number within a range using a Function: | Matched |
| R4 | P12 | `p12.c` | Pind Frequency of each character In a string | Matched |

## 3. Evaluation

### D1 — Syntax & Code Validity

**Score:** 1.80 / 2

**Assessment**

Static inspection confirms 15 of 17 submitted source files exhibit clean function declarations, standard library header inclusions, and balanced block scopes without obvious declaration anomalies. Potential declaration/return-statement warnings were identified in 2 file(s).

**Evidence**

- Static inspection indicates 15/17 source files have complete function declarations, standard header inclusions, and balanced block scopes.
- Declaration/return-statement warnings: `p6_1.c`: Function 'swap' declared with return type 'int' lacks a return statement.; `p6_1.c`: Function 'main' declared with return type 'int' lacks a return statement.; `p6_2.c`: Function 'swap' declared with return type 'int' lacks a return statement.; `p6_2.c`: Function 'main' declared with return type 'int' lacks a return statement.
- Modular helper function separation statically verified across 11 source file(s).

### D2 — Algorithmic Logic & Functional Correctness

**Score:** 1.80 / 2

**Assessment**

Static reasoning indicates concrete algorithmic workflows across loops, pointer manipulations, and recursive decompositions.

**Evidence**

- Parameter Passing: `p6_1.c` uses pass-by-value, swapping local parameters without modifying caller variables in main(); `p6_2.c` correctly uses pointer dereferencing (`*x`, `*y`) to modify caller variables across stack frames.
- Arithmetic & Precision: `p7.c` uses `pow()` from `<math.h>` for integer Armstrong calculation, which operates on `double` floating-point values and introduces potential type conversions in integer comparisons.
- Recurrence & Recursion Analysis: `p9.c` implements genuine branching tree recursion with multiple distinct recursive branches per frame (`fibonacci(n-1) + fibonacci(n-2)`); `p13.c` implements a doubling recurrence (`f(n-1) + f(n-1)`) with base case `n<=1` returning `1`, evaluating identical subproblems repeatedly rather than forming a Fibonacci recurrence; `p14.c` recursively decomposes integers via `f(n/2)` and `n%2` to calculate binary set bits (Hamming weight); `p15.c` uses twin recursive calls followed by output statements executed during stack unwinding; `p16.c` uses a piecewise modular recursive transformation conditionally branching on `length % 3`.
- Control Flow & Bounds: `p1.c` bounds prime testing up to `n/2` with early return; `p2.c` tests string palindromes up to `len/2`; `p10.c` implements 2D string sorting using nested loops with `strcmp()` and `strcpy()`.

### D3 — Observation Report Quality & Completeness

**Score:** 1.70 / 2

**Assessment**

The student documented 4 report entries with clear problem understanding, logic, and observations. Quality levels distinguish complete entries from algorithmic divergence in R4. No penalty applied for undocumented programs.

**Evidence**

- Evaluated strictly on the 4 documented report entries (R1, R2, R3, R4). The 13 undocumented code problems are not penalized under D3.
- Entries R1, R2, R3 demonstrate thorough problem understanding, step-by-step logic, defined variable tables with specific purposes, and experimental observations.
- Entry R4 provides clear descriptive structure, but its algorithmic logic description (nested comparison loops) diverges from the submitted source code (direct frequency array).

### D4 — Conceptual Understanding & Code-Report Consistency

**Score:** 1.40 / 2

**Assessment**

Demonstrates solid conceptual grasp across functions and data structures. Identified 1 technical discrepancy between report claims and implementation (R4 / P12 frequency array vs nested loops).

**Evidence**

- Conducted 4 code ↔ report consistency checks between claimed algorithms and source implementations.
- Identified 1 technical inconsistency: R4 claims nested comparison loops, whereas p12.c implements a 256-element frequency array.

### D5 — Novelty, Innovation & Presentation Readiness

**Score:** 1.70 / 2

**Assessment**

Active static code search identified 9 interesting algorithmic approach(es), including modular recursive implementations.

**Evidence**

- Static inspection across all 17 submitted source files identified 9 notable algorithmic implementation(s).
- 9 candidates identified; top 4 selected for presentation relevance.

## 4. Code–Report Consistency

| Report | Code | Claim | Code Reality | Status |
|---|---|---|---|---|
| R1 | P3 | Calculate string length by traversing characters until '\0' without using strlen. | `p3.c` manually iterates over string until '\0' and increments counter. | ✓ Supported |
| R2 | P5 | Calculate factorial using recursion until base condition. | `p5.c` implements recursive factorial with base condition. | ✓ Supported |
| R3 | P7 | Check Armstrong numbers across a range using a modular function. | `p7.c` iterates from start to end and calls isarmstrong() for every value. | ✓ Supported |
| R4 | P12 | Report describes character frequency calculation using comparison/nested-loop logic. | Submitted p12.c actually uses a 256-element frequency array, incrementing freq[(unsigned char)str[i]] rather than nested comparison loops. | ⚠️ Inconsistent |

## 5. Notable Implementations

*9 candidates identified; top 4 selected for presentation relevance.*

### P4
**File:** `p4.c`

**Interesting Logic:** Implementation utilizes linear recursion via sum(n-1) + n with a single recursive call per stack frame.

**Why it is interesting:** Single-branch linear recursion demonstrates clean stack progression and base-case termination without tree branching.

**Presentation Potential:** MEDIUM

### P5
**File:** `p5.c`

**Interesting Logic:** Implementation utilizes single-branch linear recursion via factorial(n-1) * n.

**Why it is interesting:** Standard linear recursive state progression with guard against negative inputs.

**Presentation Potential:** MEDIUM

### P6_2
**File:** `p6_2.c`

**Interesting Logic:** Utilizes pointer dereferencing (*x, *y) for direct memory address swapping across stack frames.

**Why it is interesting:** Demonstrates call-by-reference in C, contrasting directly with pass-by-value parameter copying in local scopes.

**Presentation Potential:** HIGH

### P9
**File:** `p9.c`

**Interesting Logic:** Implementation utilizes genuine branching binary tree recursion via twin recursive calls fibonacci(n-1) + fibonacci(n-2).

**Why it is interesting:** Demonstrates multi-branch recursion with an exponential call tree, contrasting with linear single-call recursion.

**Presentation Potential:** HIGH

## 6. Presentation Preparation

### Recommended Topics

- Natural Number Summation via Linear Recursion
- Factorial Decomposition via Linear Recursion
- Pass-by-Reference Pointer Swapping
- p9.c via Branching Tree Recursion

### Likely Faculty Questions

1. Why does `p6_1.c` fail to modify caller variables in main(), while `p6_2.c` succeeds using pointer dereferencing?
2. How does the branching tree recursion in `p9.c` (fibonacci) differ structurally from the doubling recurrence in `p13.c` (f)?
3. What precision or type conversion issues can arise when using `pow()` from `<math.h>` for integer Armstrong-number calculations in `p7.c`?
4. Why does `p12.c` use a 256-element frequency array rather than the nested comparison loops described in observation report R4?
5. In `p14.c`, how does the recursive decomposition `f(n/2) + n%2` compute the number of set bits (binary Hamming weight)?

## 7. Final Score

| Dimension | Score | Weight |
|---|---:|---:|
| D1 | 1.80 / 2 | 20 |
| D2 | 1.80 / 2 | 20 |
| D3 | 1.70 / 2 | 20 |
| D4 | 1.40 / 2 | 20 |
| D5 | 1.70 / 2 | 20 |
| **Total** | **8.40 / 10** | **100** |

**Final Score:** 84 / 100

## 8. Feedback

### Strengths

- Good separation of algorithmic logic across modular functions.
- Handwritten observation report provides clear problem understanding, step-by-step logic, and observed behaviors.

### Improvements

- Ensure non-void functions include explicit return statements across all code paths.
- Align reported algorithmic logic with submitted code implementations (e.g. character frequency calculation).

### Overall Assessment

Unified evaluation of 17 code files and 4 documented report programs. Static analysis confirms sound algorithmic foundations and presentation readiness.
