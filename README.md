# LangGraphBench: Evaluating Frontier LLMs on Multi-Agent Graph State, Reducers & Pregel Deadlocks

> **Official Submission for the micro1 Eval Hackathon: "Find What AI Can't Do Yet"**  
> **Domain:** AI Agent Systems Architecture & Graph Concurrency (LangGraph / Pregel Engine)  
> **Grader Type:** 100% Deterministic Code Grader (Zero False Passes, Zero LLM Judges)  
> **Repository:** [https://github.com/akshayparimi18/micro1-hackathon](https://github.com/akshayparimi18/micro1-hackathon)

---

## 1. Executive Summary & What It Tests

**In one sentence:** *LangGraphBench tests whether frontier AI models can mentally simulate asynchronous, channel-reduced Pregel state machines, or whether their imperative Python bias causes them to systematically mispredict state reducers, parallel collisions, checkpointer replays, and cyclic deadlocks.*

While frontier models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash/Pro) effortlessly generate basic boilerplate for LangGraph, they **consistently hallucinate and fail** when asked to predict the exact runtime behavior of:
1. **Channel Reducer Accumulation:** Misinterpreting `Annotated[list, operator.add]` as simple dictionary key reassignment, missing explosive list duplication.
2. **Parallel Fan-Out Collisions:** Forgetting that multiple concurrent branches writing to an un-annotated scalar channel in the same superstep triggers `InvalidUpdateError`.
3. **Conditional Graph Deadlocks:** Missing subtle state resets inside routing nodes that trap the graph in infinite cycles.
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
| `task_02_conditional_deadlock` | **Conditional Graph Deadlock** | Conditional routing & cyclic state transitions | Router node resets `retry_count = 0` whenever `status == "retrying"`, causing an infinite ping-pong loop between router and validator. |
| `task_03_parallel_fanout_collision` | **Parallel Fan-Out Channel Collision** | Superstep channel writes without reducers | Two parallel branches write distinct integers to un-annotated scalar channel `audit_score`, triggering runtime `InvalidUpdateError`. |
| `task_04_checkpointer_hitl_replay` | **Checkpointer HITL Replay** | `MemorySaver` and `interrupt_before` semantics | Updating state during human-in-the-loop pauses resumes execution at the interrupted node without re-executing preceding nodes. |
| `task_05_message_window_orphan` | **Sliding Window Message Orphan** | Custom message reducers & API payload validity | Slicing message history to the last 5 items leaves an orphaned `tool` message at index 0, violating OpenAI/Anthropic API constraints. |
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

## 5. The Hot Take: Imperative Python Bias vs. Pregel Graph Execution

### Why Frontier AI Models Fail at Graph Concurrency
The fundamental limitation exposed by LangGraphBench is **The Imperative Python Bias**:
1. **The Sequential Mental Model:** Frontier LLMs are pre-trained predominantly on traditional imperative code where functions execute linearly and dictionary assignments (`state["key"] = val`) overwrite existing data.
2. **Channel Reducers are Non-Local:** In LangGraph's Pregel model, a node return value is not a state replacement—it is a *message sent to a channel*. When channel reducers (`operator.add`) are active, return values are accumulated. Models intuitively assume returning `state["items"] + ["step_1"]` produces a 2-item list, completely missing that the channel reducer adds this return value *back onto the input state*.
3. **Superstep Blindness:** In parallel fan-out and dynamic map-reduce (`Send`), models struggle to reason about discrete superstep synchronization barriers, failing to anticipate concurrent write collisions on shared scalar state.

To build reliable autonomous agents, AI systems cannot rely on surface syntactic understanding; they require specialized execution simulators and graph verification harnesses.

---

## 6. How to Run & Reproduce

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
# Run Gemini 2.5 Flash across 3 trials
python run_eval.py --models gemini/gemini-2.5-flash --trials 3

# Run OpenAI GPT-4o
python run_eval.py --models gpt-4o --trials 3

# Run Claude 3.5 Sonnet
python run_eval.py --models claude-3-5-sonnet-20241022 --trials 3
```

Results are saved to `results/summary.json` and full transcripts for every trial are recorded in `transcripts/<model_name>/`.

### 4. Package for Submission
```bash
python package_submission.py
# Generates langgraphbench-eval.zip (< 50 MB, clean submission package)
```

---

## 7. Provenance & Human Expertise Disclosure

- **Domain Expertise:** Designed by an active multi-agent systems practitioner building production agent workflows with Python, FastAPI, LangGraph, and checkpointer architectures.
- **AI Tool Usage:** AI tools assisted in drafting repetitive Markdown code snippets for task prompts.
- **Human Verification:** Every task scenario, state transition, checkpoint lifecycle, and grading rule was authored, manually traced, and confirmed by running the code through LangGraph's actual Pregel execution engine.

---

## 8. Pre-Submission Checklist

- [x] Every task has a reference solution that passes its grader (Verified: 100%).
- [x] Deliberately wrong answers fail the grader (Verified: 0 false passes).
- [x] Everything the grader checks is stated in the task prompt.
- [x] Harness supports multi-trial evaluations reporting $pass@k$ and $pass^k$.
- [x] Ground truth programmatically verified against runtime LangGraph (`verify_ground_truth.py`).
- [x] README explains how to run and reproduce in a clean environment.
- [x] Submission ZIP (`langgraphbench-eval.zip`) is $< 50\text{ MB}$ and contains no private data.
- [x] No API keys or credentials committed to git.
- [x] 5-minute video walkthrough script provided in `VIDEO_SCRIPT.md`.
- [x] HackerRank short description ready in `HACKERRANK_SUBMISSION_DESCRIPTION.md`.
