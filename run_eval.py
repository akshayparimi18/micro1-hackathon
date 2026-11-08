#!/usr/bin/env python3
"""
LangGraphBench Evaluation Runner.
Measures LLM reasoning on multi-agent graph state, reducers, and deadlocks.

Features:
- --verify-reference: Runs the oracle ground-truth solutions (100% pass check).
- --verify-negative: Runs corrupted responses to verify zero false passes (0% pass check).
- --models: Executes evaluations on real LLM APIs (Gemini, OpenAI, Anthropic, Groq).
- --trials: Multi-trial consistency testing (pass@k and pass^k).
- Automatic rate-limit handling (HTTP 429 exponential backoff) and polite pacing.
- Full transcript logging to transcripts/<model>/<task>_trial_<n>.json.
- Metric summary saved to results/summary.json.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graders.langgraph_grader import grade_candidate_response, extract_final_answer_json

TASKS_DIR = ROOT_DIR / "tasks"
REFERENCE_FILE = ROOT_DIR / "reference" / "solutions.json"
RESULTS_DIR = ROOT_DIR / "results"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"


def load_solutions() -> Dict[str, Any]:
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_prompts() -> Dict[str, str]:
    tasks = {}
    for p in sorted(TASKS_DIR.glob("task_*.md")):
        tid = p.stem
        with open(p, "r", encoding="utf-8") as f:
            tasks[tid] = f.read()
    return tasks


def call_llm(model: str, prompt: str, temperature: float = 0.0, max_retries: int = 5) -> str:
    """
    Universal LLM caller supporting Gemini, OpenAI, Anthropic, and Groq.
    Includes automatic exponential backoff on HTTP 429 rate limits.
    """
    if model.lower() == "oracle":
        raise ValueError("Oracle mode should be called directly via --verify-reference.")

    for attempt in range(max_retries):
        try:
            # 1. Google Gemini
            if "gemini" in model.lower():
                api_key = (
                    os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("\ufeffGEMINI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                )
                if not api_key or api_key == "your_google_ai_studio_key_here":
                    raise RuntimeError(
                        "GEMINI_API_KEY not set or still set to placeholder in .env. "
                        "Please replace 'your_google_ai_studio_key_here' in .env with your actual key."
                    )

                clean_model = model.replace("gemini/", "").replace("models/", "")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature}
                }
                with httpx.Client(timeout=90.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"   [RATE LIMIT] HTTP 429 encountered. Backing off for {wait_time}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    if resp.status_code != 200:
                        raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return parts[0].get("text", "") if parts else ""

            # 2. OpenAI
            if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-") or "openai" in model.lower():
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file.")

                clean_model = model.replace("openai/", "")
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": clean_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                if clean_model.startswith("o1-") or clean_model.startswith("o3-"):
                    payload.pop("temperature", None)

                with httpx.Client(timeout=90.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"   [RATE LIMIT] HTTP 429 encountered. Backing off for {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    if resp.status_code != 200:
                        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    return data["choices"][0]["message"]["content"] or ""

            # 3. Anthropic
            if "claude" in model.lower() or "anthropic" in model.lower():
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY not set in environment or .env file.")

                clean_model = model.replace("anthropic/", "")
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": clean_model,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
                with httpx.Client(timeout=90.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"   [RATE LIMIT] HTTP 429 encountered. Backing off for {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    if resp.status_code != 200:
                        raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    return data["content"][0]["text"] or ""

            # 4. Groq
            if "groq" in model.lower() or "llama" in model.lower():
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError("GROQ_API_KEY not set in environment or .env file.")

                clean_model = model.replace("groq/", "")
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": clean_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        print(f"   [RATE LIMIT] HTTP 429 encountered. Backing off for {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    if resp.status_code != 200:
                        raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    return data["choices"][0]["message"]["content"] or ""

            raise ValueError(f"Unsupported model provider: {model}")

        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 4
                print(f"   [NETWORK WARNING] Connection issue: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to get response for {model} after {max_retries} attempts.")


def run_verify_reference():
    """Runs all 6 reference solutions through the grader."""
    print("\n" + "=" * 65)
    print(" VERIFYING REFERENCE SOLUTIONS (Oracle Mode)")
    print("=" * 65)
    solutions = load_solutions()
    all_passed = True

    for tid, ref in solutions.items():
        wrapped = f"<final_answer>\n{json.dumps(ref, indent=2)}\n</final_answer>"
        result = grade_candidate_response(ref, wrapped)
        status = "PASSED" if result["passed"] == 1 else "FAILED"
        print(f"  [{status}] {tid:<35} | Score: {result['partial_score']*100:.1f}%")
        if result["passed"] != 1:
            all_passed = False

    print("=" * 65)
    if all_passed:
        print(">>> 100% OF REFERENCE SOLUTIONS PASSED (Zero False Negatives) <<<\n")
    else:
        print(">>> ERROR: One or more reference solutions failed! <<<\n")
        sys.exit(1)


def run_verify_negative():
    """Verifies that corrupted / hallucinated answers fail with 0% pass rate."""
    print("\n" + "=" * 65)
    print(" VERIFYING NEGATIVE / CORRUPTED ANSWERS (Zero False Passes Check)")
    print("=" * 65)
    solutions = load_solutions()
    all_rejected = True

    # Corrupted Case 1: Naive belief that Task 1 returns ["seed", "step_1", "step_2"]
    res1 = grade_candidate_response(
        solutions["task_01_reducer_duplication"],
        "<final_answer>\n" + json.dumps({
            "final_items": ["seed", "step_1", "step_2"],
            "total_length": 3,
            "root_cause": "normal_append"
        }) + "\n</final_answer>"
    )
    status1 = "REJECTED" if res1["passed"] == 0 else "FALSE_PASS"
    print(f"  [{status1}] Task 01 (Naive append belief)      | Passed: {res1['passed']} | Partial: {res1['partial_score']*100:.1f}%")
    if res1["passed"] != 0:
        all_rejected = False

    # Corrupted Case 2: Claiming Task 2 terminates
    res2 = grade_candidate_response(
        solutions["task_02_conditional_deadlock"],
        "<final_answer>\n" + json.dumps({
            "terminates": True,
            "looping_nodes": [],
            "state_at_step_6": {"retry_count": 3, "status": "complete"}
        }) + "\n</final_answer>"
    )
    status2 = "REJECTED" if res2["passed"] == 0 else "FALSE_PASS"
    print(f"  [{status2}] Task 02 (False termination claim)   | Passed: {res2['passed']} | Partial: {res2['partial_score']*100:.1f}%")
    if res2["passed"] != 0:
        all_rejected = False

    # Corrupted Case 3: Claiming Task 3 succeeds without error
    res3 = grade_candidate_response(
        solutions["task_03_parallel_fanout_collision"],
        "<final_answer>\n" + json.dumps({
            "status": "Success",
            "conflicting_channel": None,
            "surviving_logs": ["init_log", "branch_a_ok", "branch_b_ok"]
        }) + "\n</final_answer>"
    )
    status3 = "REJECTED" if res3["passed"] == 0 else "FALSE_PASS"
    print(f"  [{status3}] Task 03 (Missing fanout collision)  | Passed: {res3['passed']} | Partial: {res3['partial_score']*100:.1f}%")
    if res3["passed"] != 0:
        all_rejected = False

    print("=" * 65)
    if all_rejected:
        print(">>> ALL CORRUPTED CASES REJECTED (Zero False Passes) <<<\n")
    else:
        print(">>> ERROR: One or more corrupted cases passed! <<<\n")
        sys.exit(1)


def evaluate_model(
    model: str,
    trials: int = 3,
    temperature: float = 0.0,
    delay_between_calls: float = 4.0
) -> Dict[str, Any]:
    """Runs evaluation of a real model across all tasks and trials."""
    tasks = load_task_prompts()
    solutions = load_solutions()

    safe_model_name = re.sub(r"[^\w\-.]", "_", model)
    model_transcript_dir = TRANSCRIPTS_DIR / safe_model_name
    model_transcript_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" EVALUATING MODEL: {model}")
    print(f" Tasks: {len(tasks)} | Trials per task: {trials} | Temperature: {temperature}")
    print(f"{'='*70}\n")

    task_metrics = {}
    total_trials = 0
    total_passed_trials = 0
    total_partial_sum = 0.0

    for tid, prompt in tasks.items():
        ref = solutions[tid]
        print(f"-> Task: {tid}")

        trial_passes = []
        trial_scores = []

        for trial_num in range(1, trials + 1):
            t0 = time.time()
            try:
                raw_response = call_llm(model, prompt, temperature=temperature)
            except Exception as e:
                print(f"   [ERROR] API call failed on trial {trial_num}: {e}")
                raw_response = f"API_ERROR: {e}"

            elapsed = time.time() - t0
            grade = grade_candidate_response(ref, raw_response)
            extracted = extract_final_answer_json(raw_response)

            passed = (grade["passed"] == 1)
            trial_passes.append(passed)
            trial_scores.append(grade["partial_score"])

            total_trials += 1
            if passed:
                total_passed_trials += 1
            total_partial_sum += grade["partial_score"]

            # Save individual trial transcript
            transcript_record = {
                "task_id": tid,
                "model": model,
                "trial": trial_num,
                "elapsed_seconds": round(elapsed, 2),
                "grade": grade,
                "extracted_answer": extracted,
                "raw_response": raw_response,
                "prompt": prompt
            }
            transcript_file = model_transcript_dir / f"{tid}_trial_{trial_num}.json"
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_record, f, indent=2)

            status = "PASS" if passed else "FAIL"
            print(f"   Trial {trial_num}/{trials}: [{status}] Partial Score: {grade['partial_score']*100:.1f}% ({elapsed:.2f}s)")

            # Polite pacing to respect RPM rate limits on free-tier
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        # Per-task aggregations
        task_pass_rate = sum(trial_passes) / len(trial_passes)
        task_avg_score = sum(trial_scores) / len(trial_scores)
        pass_at_k = any(trial_passes)
        pass_pow_k = all(trial_passes)

        task_metrics[tid] = {
            "pass_rate": round(task_pass_rate * 100, 1),
            "avg_partial_score": round(task_avg_score * 100, 1),
            "pass_at_k": pass_at_k,
            "pass_pow_k": pass_pow_k
        }

    # Model-level aggregations
    avg_pass_rate = (total_passed_trials / total_trials) * 100 if total_trials > 0 else 0.0
    avg_partial = (total_partial_sum / total_trials) * 100 if total_trials > 0 else 0.0
    pass_at_k_pct = (sum(1 for m in task_metrics.values() if m["pass_at_k"]) / len(tasks)) * 100
    pass_pow_k_pct = (sum(1 for m in task_metrics.values() if m["pass_pow_k"]) / len(tasks)) * 100

    # Determine tier and cost
    if "preview" in model or "3.5-flash" in model or "pro" in model:
        tier = "Frontier Model A"
    elif "3.5-flash-lite" in model or "3.6" in model:
        tier = "Frontier Model B"
    else:
        tier = "Smaller / Fast Model"
    cost = 0.0005 if "lite" in model.lower() else 0.002

    summary_model = {
        "model": model,
        "tier": tier,
        "trials_per_task": trials,
        "total_trials": total_trials,
        "avg_pass_rate": round(avg_pass_rate, 1),
        "avg_partial_score": round(avg_partial, 1),
        "pass_at_k": round(pass_at_k_pct, 1),
        "pass_pow_k": round(pass_pow_k_pct, 1),
        "approx_cost_usd": cost,
        "tasks": task_metrics
    }

    # Print summary table
    print(f"\n{'='*70}")
    print(f" RESULTS SUMMARY FOR: {model}")
    print(f"{'='*70}")
    print(f"Overall Pass Rate: {summary_model['avg_pass_rate']}% | Mean Partial Score: {summary_model['avg_partial_score']}%")
    print(f"pass@k (succeeded >=1 trial): {summary_model['pass_at_k']}%")
    print(f"pass^k (succeeded in ALL k): {summary_model['pass_pow_k']}%\n")

    print(f"{'Task ID':<38} | {'Pass Rate':<10} | {'Score':<8} | {'pass^k'}")
    print(f"{'-'*38}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
    for tid, m in task_metrics.items():
        pow_str = "YES" if m["pass_pow_k"] else "NO"
        print(f"{tid:<38} | {m['pass_rate']:>8.1f}% | {m['avg_partial_score']:>6.1f}% | {pow_str:>6}")
    print(f"{'='*70}\n")

    return summary_model


def main():
    parser = argparse.ArgumentParser(description="LangGraphBench Evaluation Runner")
    parser.add_argument("--verify-reference", action="store_true", help="Verify reference solutions pass 100%%")
    parser.add_argument("--verify-negative", action="store_true", help="Verify corrupted answers fail 0%%")
    parser.add_argument("--models", nargs="+", default=[], help="List of model names to evaluate")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per task (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--delay", type=float, default=4.0, help="Delay in seconds between API calls to avoid rate limits")

    args = parser.parse_args()

    if args.verify_reference:
        run_verify_reference()
        if not args.models and not args.verify_negative:
            return

    if args.verify_negative:
        run_verify_negative()
        if not args.models:
            return

    if not args.models:
        print("[!] No models specified. Use --verify-reference, --verify-negative, or provide --models <model1> ...")
        print("    Example: python run_eval.py --models gemini/gemini-2.5-flash --trials 3")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "summary.json"
    current_summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                current_summary = json.load(f)
        except Exception:
            current_summary = {}

    if "models" not in current_summary:
        current_summary = {"benchmark": "LangGraphBench", "models": {}}

    for model_name in args.models:
        res = evaluate_model(model_name, trials=args.trials, temperature=args.temperature, delay_between_calls=args.delay)
        current_summary["models"][model_name] = res
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(current_summary, f, indent=2)
        print(f"[+] Saved interim progress to: {summary_file}")

    print(f"[+] All results saved to: {summary_file}")


if __name__ == "__main__":
    main()
