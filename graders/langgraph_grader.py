"""
Deterministic Grader for LangGraphBench.
Extracts structured JSON from <final_answer>...</final_answer> tags
and compares fields against reference/solutions.json with zero tolerance for hallucinations.
"""

from typing import Dict, Any, Optional, Tuple
import json
import re


def extract_final_answer_json(text: Any) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON strictly from within <final_answer>...</final_answer> tags.
    Strips markdown code blocks if present.
    Returns None if tags are missing or JSON is invalid.
    """
    if not isinstance(text, str):
        if isinstance(text, dict):
            return text
        return None

    pattern = r"<final_answer>(.*?)</final_answer>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    raw_content = match.group(1).strip()

    # Strip markdown code fences if model wrapped inside tags
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)
        raw_content = re.sub(r"\s*```$", "", raw_content)
        raw_content = raw_content.strip()

    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def normalize_value(v: Any) -> Any:
    """Normalizes whitespace on strings and nested collections for robust comparison."""
    if isinstance(v, str):
        return v.strip()
    elif isinstance(v, list):
        return [normalize_value(x) for x in v]
    elif isinstance(v, dict):
        return {k.strip(): normalize_value(val) for k, val in v.items()}
    return v


def grade_candidate_response(
    expected: Dict[str, Any],
    candidate_raw_output: Any
) -> Dict[str, Any]:
    """
    Compares candidate output against expected ground truth.
    Returns:
      {
        "passed": 1 or 0,
        "partial_score": float (0.0 to 1.0),
        "total_fields": int,
        "matched_fields": int,
        "diagnostics": { field: {"expected": ..., "actual": ..., "matched": bool} },
        "error": Optional[str]
      }
    """
    candidate_json = extract_final_answer_json(candidate_raw_output)
    if candidate_json is None:
        return {
            "passed": 0,
            "partial_score": 0.0,
            "total_fields": len(expected),
            "matched_fields": 0,
            "diagnostics": {},
            "error": "Missing or malformed <final_answer> tags or invalid JSON payload."
        }

    diagnostics = {}
    matched_count = 0
    total_fields = len(expected)

    for key, exp_val in expected.items():
        norm_exp = normalize_value(exp_val)
        actual_val = candidate_json.get(key)
        norm_actual = normalize_value(actual_val)

        is_match = (norm_actual == norm_exp)
        if is_match:
            matched_count += 1

        diagnostics[key] = {
            "expected": exp_val,
            "actual": actual_val,
            "matched": is_match
        }

    partial_score = round(matched_count / total_fields, 4) if total_fields > 0 else 0.0
    passed = 1 if (matched_count == total_fields) else 0

    return {
        "passed": passed,
        "partial_score": partial_score,
        "total_fields": total_fields,
        "matched_fields": matched_count,
        "diagnostics": diagnostics,
        "error": None
    }


def run_self_test():
    """Runs built-in self-tests verifying extraction, grading, and false-pass rejection."""
    print("=" * 65)
    print(" LangGraphBench Deterministic Grader Self-Test")
    print("=" * 65)

    # 1. Extraction test
    sample_raw = "Analysis:\n<final_answer>\n```json\n{\"total_length\": 7, \"status\": \"complete\"}\n```\n</final_answer>"
    extracted = extract_final_answer_json(sample_raw)
    assert extracted == {"total_length": 7, "status": "complete"}, "Extraction failed!"
    print("[OK] JSON Extraction Test (tags, markdown code fences stripped)")

    # 2. Perfect match test
    ref = {"total_length": 7, "status": "complete"}
    res = grade_candidate_response(ref, sample_raw)
    assert res["passed"] == 1 and res["partial_score"] == 1.0, "Perfect match failed!"
    print("[OK] Strict Equivalence Match Test (100% precision)")

    # 3. Corrupted / hallucinated rejection test
    corrupted_raw = "<final_answer>{\"total_length\": 3, \"status\": \"complete\"}</final_answer>"
    res_corrupt = grade_candidate_response(ref, corrupted_raw)
    assert res_corrupt["passed"] == 0 and res_corrupt["partial_score"] == 0.5, "Corrupted rejection failed!"
    print("[OK] Corrupted Input Rejection Test (Zero False Passes)")

    print("=" * 65)
    print(">>> ALL GRADER TESTS PASSED: 100% Deterministic (Zero False Passes) <<<\n")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv or len(sys.argv) == 1:
        run_self_test()
    else:
        print("Usage: python graders/langgraph_grader.py --self-test")

