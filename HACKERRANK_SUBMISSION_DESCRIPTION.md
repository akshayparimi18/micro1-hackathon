# HackerRank Submission Description & Metadata

Copy and paste the text below into the submission form on HackerRank for the micro1 Eval Hackathon.

---

### Short Written Description (for HackerRank Event Page)

**Domain & Task Suite:**
LangGraphBench is an expert-level evaluation benchmark comprising 6 rigorous tasks designed to measure AI models on multi-agent graph state management, channel reducers, and Pregel concurrency in LangGraph.

**The Real-World Limitation:**
While LLMs can write standard imperative Python boilerplate, they consistently fail to mentally simulate asynchronous Pregel state machines. Specifically, models exhibit an "Imperative Python Bias"—they treat node return values as standard dictionary reassignments rather than channel updates reduced across supersteps. This leads to catastrophic failure modes in production: explosive list duplication under `operator.add`, unexpected `InvalidUpdateError` crashes on parallel fan-out, undetected cyclic deadlocks in conditional routers, state corruption during human-in-the-loop checkpointer replays, and broken sliding-window message reducers that cause fatal 400 Bad Request errors.

**Methodology & Empirical Findings:**
LangGraphBench utilizes a 100% deterministic code grader with zero false passes and zero false negatives, verified programmatically against the live LangGraph runtime engine. Evaluations across live API models (Gemini 3.5 Flash, Gemini 3.6 Flash, and Gemini 3.1 Flash-Lite) for 3 trials per task ($k=3$) reveal a decisive capability gap:
- Frontier models (`gemini-3.5-flash` at 100.0% and `gemini-3.6-flash` at 88.9%) successfully track complex channel accumulation.
- In contrast, lightweight models (`gemini-3.1-flash-lite`) collapse to a 66.7% pass rate, scoring **0.0% on Reducer Duplication** (mispredicting explosive list growth) and **0.0% on Conditional Deadlocks** (failing to recognize cyclical state resets and falsely predicting termination).

These findings demonstrate that models without deep state-machine simulation capabilities default to sequential imperative heuristics, proving the necessity of graph-aware execution validators in agent development workflows.

---

### Submission Metadata Checklist:
1. **ZIP File:** `langgraphbench-eval.zip` (Code, tasks, references, graders, and documentation; < 50 MB).
2. **Video Walkthrough Link:** [Insert your Loom / YouTube unlisted video link here - see `VIDEO_SCRIPT.md` for the timed walkthrough script].
3. **GitHub Repository:** [https://github.com/akshayparimi18/micro1-hackathon](https://github.com/akshayparimi18/micro1-hackathon)
