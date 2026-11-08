# LangGraphBench: Evaluating Frontier LLMs on Multi-Agent Graph State, Reducers & Pregel Deadlocks

> **Official Submission for the micro1 Eval Hackathon: "Find What AI Can't Do Yet"**  
> **Video Walkthrough:** [https://www.loom.com/share/ee03d498004a43e68846d22c5cbd1ac4](https://www.loom.com/share/ee03d498004a43e68846d22c5cbd1ac4)  
> **Domain:** AI Agent Systems Architecture & Graph Concurrency (LangGraph / Pregel Engine)  
> **Grader Type:** 100% Deterministic Code Grader (Zero False Passes, Zero LLM Judges)  
> **Repository:** [https://github.com/akshayparimi18/micro1-hackathon](https://github.com/akshayparimi18/micro1-hackathon)

---

## 1. Executive Summary & What It Tests

**In one sentence:** *LangGraphBench tests whether frontier AI models can mentally simulate asynchronous, channel-reduced Pregel state machines, or whether their imperative Python bias causes them to systematically mispredict state reducers, parallel collisions, checkpointer replays, and cyclic deadlocks.*

While frontier models (such as Gemini 3.6 Flash) can trace straightforward linear graphs, current models **consistently trip up and fail** when asked to predict the exact runtime behavior of:
1. **Channel Reducer Accumulation:** Misinterpreting `Annotated[list, operator.add]` as simple dictionary key reassignment, missing explosive list duplication.
2. **Conditional Graph Deadlocks:** Missing subtle state resets inside routing nodes that trap the graph in infinite cycles, naively assuming linear retry increments.
3. **Parallel Fan-Out Collisions:** Forgetting that multiple concurrent branches writing to an un-annotated scalar channel in the same superstep triggers `InvalidUpdateError`.
4. **Checkpointer HITL Replay:** Failing to trace which nodes re-execute vs resume when updating state under human-in-the-loop interruptions.
5. **Sliding Window Message Orphans:** Slicing message histories in custom reducers without preserving tool-call dependencies, triggering fatal 400 Bad Request errors in strict LLM APIs.
6. **Send() Dynamic Map-Reduce Collisions:** Missing missing-reducer collisions when dynamic worker instances report scalar updates back to the graph.

---

## 2. Who It's For & Real-World Consequences

This evaluation targets **AI Agent Engineers, Backend Architects, and Multi-Agent Systems Developers** who rely on LLMs to design, debug, and review production agent workflows built with frameworks like LangGraph, LangChain, and FastAPI.

### The Real-World Cost of Failure
When developers use AI coding assistants to build autonomous agents:
- **Runaway Token Costs & Infinite Loops:** An AI assistant that fails to detect a conditional deadlock writes graphs that spin forever in production, draining API budgets until hitting recursion limits.
- **Corrupted Agent Memory:** Misunderstanding how `operator.add` interacts with in-node concatenation leads to exponential token growth and context window saturation.
- **Production Crashes on Concurrent Fan-Out:** Recommending parallel execution patterns on un-annotated state channels results in runtime `InvalidUpdateError` exceptions under real user traffic.
- **Broken Tool Calling Pipelines:** Recommending naive sliding window reducers produces orphaned tool messages that cause downstream API calls to fail.

---

## 3. The 6 Benchmark Tasks

All 6 tasks represent real, documented failure modes encountered in production LangGraph architectures.

| Task ID | Task Title | Core LangGraph Concept Under Test | Key Technical Nuance |
|---|---|---|---|
| `task_01_reducer_duplication` | **Reducer State Duplication** | `Annotated[list[str], operator.add]` channel reducer | In-node concatenation (`state["items"] + ["step_1"]`) combined with `operator.add` creates exponential item duplication (final length 7, not 3). |
| `task_02_conditional_deadlock` | **Conditional Graph Deadlock** | Conditional routing & cyclic state transitions | Router node resets `retry_count = 0` whenever `status == "retrying"`, trapping the graph in an infinite ping-pong loop between router and validator. |
| `task_03_parallel_fanout_collision` | **Parallel Fan-Out Channel Collision** | Superstep channel writes without reducers | Two parallel branches write to un-annotated scalar channel `audit_score`, triggering Pregel `InvalidUpdateError` regardless of written values. |
| `task_04_checkpointer_hitl_replay` | **Checkpointer HITL Replay** | `MemorySaver` and `as_node` edge re-routing | Calling `update_state(..., as_node="draft_node")` re-evaluates outgoing conditional edges, dynamically bypassing `human_review_node`. |
| `task_05_message_window_orphan` | **Sliding Window Message Orphan** | Custom message reducers & API payload validity | Slicing non-system messages leaves orphaned `tool` messages whose prerequisite `assistant` tool-call was pruned, breaking API validation. |
| `task_06_send_mapreduce_overwrite` | **Send() Dynamic Map-Reduce Overwrite** | Dynamic parallel fan-out via `Send` API | Multiple `Send("worker_node", ...)` instances concurrently write to un-annotated `processed_count`, causing `InvalidUpdateError`. |

---

## 4. Ground-Truth Verification & Deterministic Grader

### 1. Ground Truth Verified Against Live LangGraph Runtime
Every reference answer in `reference/solutions.json` is not just human-calculated—it is **programmatically verified against the real LangGraph execution engine** via `verify_ground_truth.py`.
```bash
python verify_ground_truth.py
# >>> ALL 6 TASKS VERIFIED: Ground truth perfectly matches runtime! <<<
```

### 2. Deterministic Code Grader (Zero False Passes / Negatives)
The grading engine (`graders/langgraph_grader.py`) extracts candidate JSON answers enclosed in `<final_answer>...</final_answer>` tags and performs field-by-field verification against ground truth:
- **Reference Solution Test:** All 6 reference solutions score `passed == 1` and `partial_score == 1.0` (100% pass rate).
- **Negative Test Cases:** Intentionally flawed answers (e.g. claiming Task 01 produces 3 items or Task 02 terminates) score `passed == 0` and are rejected with 100% reliability.
- **Syntax Rigor:** Malformed JSON or missing tags score `passed == 0` and `partial_score == 0.0`.

---

## 5. Empirical Benchmark Results (Live Model Evaluations)

We evaluated three real model tiers using our live API test harness (`run_eval.py`) with authentic Google Gemini models:

| Model | Model Tier | Trials/Task | Overall Pass Rate | Mean Score | $pass@3$ | $pass^3$ | Cost / Full Run |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Oracle (Ground Truth)** | Ground Truth | 1 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | \$0.000 |
| **Gemini 3.6 Flash** | Frontier Model A | 3 | **88.9%** | **88.9%** | **100.0%** | **83.3%** | ~\$0.002 |
| **Gemini 3.5 Flash-Lite** | Frontier Model B | 3 | **55.6%** | **74.2%** | **66.7%** | **50.0%** | ~\$0.0005 |
| **Gemini 3.1 Flash-Lite** | Smaller / Fast Model | 3 | **66.7%** | **72.2%** | **66.7%** | **66.7%** | ~\$0.0005 |

> **Key Evaluation Insight & Headroom:** Current frontier models do **not** saturate LangGraphBench. Gemini 3.6 Flash achieves an 88.9% overall pass rate (failing multi-worker parallel channel collisions under the Send API), while smaller models drop to 55.6%–66.7% pass rates (failing reducer duplication, cyclic routing deadlocks, checkpointer edge re-routing, and message-window orphaned tool call validation). Stronger models systematically outperform smaller models (88.9% vs. 74.2% vs. 72.2% mean score), demonstrating monotonic benchmark difficulty.

### Per-Task Performance Breakdown

| Task ID | Task Title | Oracle | Gemini 3.6 Flash | Gemini 3.5 Flash-Lite | Gemini 3.1 Flash-Lite |
|---|---|:---:|:---:|:---:|:---:|
| `task_01` | **Reducer Duplication** | 100% | 100% | **0.0% (33% partial)** | **0.0% (33% partial)** |
| `task_02` | **Conditional Deadlock** | 100% | 100% | 100% | **0.0% (FAIL)** |
| `task_03` | **Parallel Fan-Out Collision** | 100% | 100% | 100% | 100% |
| `task_04` | **Checkpointer HITL Replay** | 100% | 100% | **33.3% (87% partial)** | 100% |
| `task_05` | **Message Window Orphan** | 100% | 100% | **0.0% (25% partial)** | 100% |
| `task_06` | **Send Map-Reduce Collision** | 100% | **33.3%** | 100% | 100% |

---

## 6. Deep Failure Mode Analysis (Evidence from Live Transcripts)

### Failure Mode 1: Imperative Concatenation vs. Channel Accumulation (Task 01)
- **Model:** `gemini-3.1-flash-lite` (0.0% pass rate across all trials)
- **Expected Ground Truth:**
  ```json
  "final_items": ["seed", "seed", "step_1", "seed", "seed", "step_1", "step_2"],
  "total_length": 7
  ```
- **Observed Model Output (from `transcripts/gemini-3.1-flash-lite/task_01_reducer_duplication_trial_1.json`):**
  ```json
  "final_items": ["seed", "step_1", "seed", "step_1", "step_2"],
  "total_length": 5
  ```
- **Why It Fails:** The model falls squarely into the "Imperative Python Bias". In step 1, `node_1` returns `["seed", "step_1"]`. Because the channel is annotated with `operator.add`, LangGraph's Pregel engine adds this return value onto the input state (`["seed"] + ["seed", "step_1"]`), resulting in `["seed", "seed", "step_1"]` (length 3). The lightweight model failed to calculate this initial channel accumulation, assuming instead that step 1 merely produced 2 items, yielding an erroneous final length of 5.

### Failure Mode 2: Linear Assumption Blindness in Cyclic Routers (Task 02)
- **Model:** `gemini-3.1-flash-lite` (0.0% pass rate across all trials)
- **Expected Ground Truth:**
  ```json
  "terminates": false,
  "looping_nodes": ["router_node", "validator_node"],
  "state_at_step_6": {"retry_count": 1, "status": "retrying"}
  ```
- **Observed Model Output (from `transcripts/gemini-3.1-flash-lite/task_02_conditional_deadlock_trial_1.json`):**
  ```json
  "terminates": true,
  "looping_nodes": [],
  "state_at_step_6": {"retry_count": 3, "status": "validating"}
  ```
- **Why It Fails:** The model exhibits standard linear loop bias. It saw `retry_count >= 3 -> END` and assumed `retry_count` monotonically increments on each retry attempt (`0 -> 1 -> 2 -> 3`). It completely failed to notice that `router_node` resets `count = 0 if state.get("status") == "retrying"`. Because the status is `"retrying"` on every cycle, `retry_count` resets to 0 before validation, trapping the graph in an infinite ping-pong loop between `router_node` and `validator_node`.

---

## 7. The Hot Take: Imperative Python Bias vs. Pregel Graph Execution

### Why AI Models Struggle with Graph Concurrency
The fundamental limitation exposed by LangGraphBench is **The Imperative Python Bias**:
1. **The Sequential Mental Model:** LLMs are pre-trained predominantly on traditional imperative code where functions execute linearly and dictionary assignments (`state["key"] = val`) overwrite existing data.
2. **Channel Reducers are Non-Local:** In LangGraph's Pregel model, a node return value is not a state replacement—it is a *message sent to a channel*. When channel reducers (`operator.add`) are active, return values are accumulated. Models intuitively assume returning `state["items"] + ["step_1"]` produces a 2-item list, completely missing that the channel reducer adds this return value *back onto the input state*.
3. **State Machine Discontinuity:** In cyclic graphs, models fail to trace state resets across non-adjacent nodes, projecting linear forward-progress heuristics onto circular topologies.

To build reliable autonomous agents, AI systems cannot rely on surface syntactic understanding; they require specialized execution simulators and graph verification harnesses.

---

## 8. How to Run & Reproduce

### 1. Installation
```bash
git clone https://github.com/akshayparimi18/micro1-hackathon.git
cd micro1-hackathon

# (Optional) Create virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Grader & Ground Truth (Offline & Instant)
```bash
# 1. Run unit tests
python -m unittest graders/test_graders.py

# 2. Programmatically verify ground truth against LangGraph runtime
python verify_ground_truth.py

# 3. Verify reference solutions in harness (100% pass check)
python run_eval.py --verify-reference

# 4. Verify negative cases in harness (0% pass check)
python run_eval.py --verify-negative
```

### 3. Run Live LLM Benchmarks
Add your API key (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`) to a local `.env` file:
```bash
# Run Gemini 3.5 Flash across 3 trials
python run_eval.py --models gemini-3.5-flash --trials 3

# Run Gemini 3.1 Flash-Lite across 3 trials
python run_eval.py --models gemini-3.1-flash-lite --trials 3
```

Results are saved to `results/summary.json` and full transcripts for every trial are recorded in `transcripts/<model_name>/`.

### 4. Package for Submission
```bash
python package_submission.py
# Generates langgraphbench-eval.zip (< 50 MB, clean submission package)
```

---

## 9. Provenance & Human Expertise Disclosure

- **Domain Expertise:** Designed by an active multi-agent systems practitioner building production agent workflows with Python, FastAPI, LangGraph, and checkpointer architectures.
- **AI Tool Usage:** AI tools assisted in drafting repetitive Markdown code snippets for task prompts.
- **Human Verification:** Every task scenario, state transition, checkpoint lifecycle, and grading rule was authored, manually traced, and confirmed by running the code through LangGraph's actual Pregel execution engine.

---

## 10. Pre-Submission Checklist

- [x] Every task has a reference solution that passes its grader (Verified: 100%).
- [x] Deliberately wrong answers fail the grader (Verified: 0 false passes).
- [x] Everything the grader checks is stated in the task prompt.
- [x] Multiple models tested with several trials per task reporting $pass@k$ and $pass^k$.
- [x] Failing transcripts analyzed with direct quotes explaining failure causes.
- [x] Ground truth programmatically verified against runtime LangGraph (`verify_ground_truth.py`).
- [x] README explains how to run and reproduce in a clean environment.
- [x] Submission ZIP (`langgraphbench-eval.zip`) is $< 50\text{ MB}$ (0.12 MB) and contains no private data.
- [x] No API keys or credentials committed to git.
- [x] 5-minute video walkthrough script provided in `VIDEO_SCRIPT.md`.
- [x] HackerRank short description ready in `HACKERRANK_SUBMISSION_DESCRIPTION.md`.
