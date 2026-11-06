"""
Unit Tests for LangGraphBench Grader.
Verifies:
1. All reference solutions achieve passed == 1 and partial_score == 1.0.
2. Mutating any single field results in passed == 0 and 0.0 < partial_score < 1.0.
3. Missing tags or broken JSON score passed == 0 and partial_score == 0.0.
"""

import unittest
import json
from pathlib import Path
import copy

ROOT_DIR = Path(__file__).resolve().parent.parent
from graders.langgraph_grader import grade_candidate_response, extract_final_answer_json


class TestLangGraphGrader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sol_path = ROOT_DIR / "reference" / "solutions.json"
        with open(sol_path, "r", encoding="utf-8") as f:
            cls.solutions = json.load(f)

    def test_all_reference_solutions_pass(self):
        """All 6 ground-truth reference solutions must achieve passed == 1 and partial_score == 1.0."""
        for task_id, ref in self.solutions.items():
            wrapped_text = f"<final_answer>\n{json.dumps(ref, indent=2)}\n</final_answer>"
            result = grade_candidate_response(ref, wrapped_text)
            self.assertEqual(result["passed"], 1, f"Task {task_id} failed to pass")
            self.assertEqual(result["partial_score"], 1.0, f"Task {task_id} score != 1.0")
            self.assertIsNone(result["error"])
            self.assertEqual(result["matched_fields"], result["total_fields"])

    def test_mutating_single_field_causes_failure(self):
        """Mutating any single field must produce passed == 0 and 0.0 < partial_score < 1.0."""
        for task_id, ref in self.solutions.items():
            for key in ref.keys():
                mutated = copy.deepcopy(ref)
                val = mutated[key]

                # Alter value
                if isinstance(val, int):
                    mutated[key] = val + 999
                elif isinstance(val, bool):
                    mutated[key] = not val
                elif isinstance(val, str):
                    mutated[key] = val + "_corrupted"
                elif isinstance(val, list):
                    mutated[key] = val + ["corrupted_item"]
                elif isinstance(val, dict):
                    mutated[key] = {"corrupted": True}

                wrapped_text = f"<final_answer>\n{json.dumps(mutated)}\n</final_answer>"
                result = grade_candidate_response(ref, wrapped_text)
                self.assertEqual(result["passed"], 0, f"Mutating {key} in {task_id} still passed!")
                self.assertGreater(result["partial_score"], 0.0, f"Mutating {key} in {task_id} gave 0 score")
                self.assertLess(result["partial_score"], 1.0, f"Mutating {key} in {task_id} gave 1.0 score")
                self.assertFalse(result["diagnostics"][key]["matched"])

    def test_missing_tags_or_malformed_json(self):
        """Missing <final_answer> tags or invalid JSON must score passed == 0 and partial_score == 0.0."""
        ref = self.solutions["task_01_reducer_duplication"]

        # Raw JSON without tags
        res_no_tags = grade_candidate_response(ref, json.dumps(ref))
        self.assertEqual(res_no_tags["passed"], 0)
        self.assertEqual(res_no_tags["partial_score"], 0.0)
        self.assertIsNotNone(res_no_tags["error"])

        # Broken JSON inside tags
        res_broken = grade_candidate_response(ref, "<final_answer>{broken_json: True,}</final_answer>")
        self.assertEqual(res_broken["passed"], 0)
        self.assertEqual(res_broken["partial_score"], 0.0)
        self.assertIsNotNone(res_broken["error"])

        # Arbitrary conversational text
        res_text = grade_candidate_response(ref, "I think the answer is 7 items.")
        self.assertEqual(res_text["passed"], 0)
        self.assertEqual(res_text["partial_score"], 0.0)

    def test_markdown_code_fences_inside_tags(self):
        """Grader should cleanly strip markdown ```json ... ``` code fences inside <final_answer>."""
        ref = self.solutions["task_01_reducer_duplication"]
        fenced_text = f"<final_answer>\n```json\n{json.dumps(ref)}\n```\n</final_answer>"
        result = grade_candidate_response(ref, fenced_text)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["partial_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

