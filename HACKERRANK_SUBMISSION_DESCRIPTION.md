# HackerRank Submission Description & Metadata

Copy and paste the text below into the submission form on HackerRank for the micro1 Eval Hackathon.

---

### Short Written Description (for HackerRank Event Page)

**Domain & Task Suite:**
LangGraphBench is an expert-level evaluation benchmark comprising 6 rigorous tasks designed to measure AI models on multi-agent graph state management, channel reducers, and Pregel concurrency in LangGraph.

**The Real-World Limitation:**
While LLMs can write standard imperative Python boilerplate, they consistently fail to mentally simulate asynchronous Pregel state machines. Specifically, models exhibit an "Imperative Python Bias"—they treat node return values as standard dictionary reassignments rather than channel updates reduced across supersteps. This leads to catastrophic failure modes in production: explosive list duplication under `operator.add`, unexpected `InvalidUpdateError` crashes on parallel fan-out, undetected cyclic deadlocks in conditional routers, state corruption during human-in-the-loop checkpointer replays, and broken sliding-window message reducers that cause fatal 400 Bad Request errors.

**Methodology & Empirical Findings:**
LangGraphBench utilizes a 100% deterministic code grader with zero false passes and zero false negatives, verified programmatically against the live LangGraph runtime engine. Evaluations across live API models (Gemini 3.6 Flash, Gemini 3.5 Flash-Lite, and Gemini 3.1 Flash-Lite) reveal clear headroom and decisive capability gaps:
- Frontier models (`gemini-3.6-flash` at 88.9% pass rate) successfully trace compound checkpointing and fan-outs, but fail dynamic multi-worker parallel channel collisions under the Send API.
- In contrast, smaller/faster models (`gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`) drop to 55.6%–66.7% pass rates across 3 trials, scoring **0.0% on Reducer Duplication** (mispredicting explosive list growth under `operator.add`), failing **Conditional Deadlocks** (predicting linear progress through cyclic state resets), failing **Checkpointer State Replay Re-routing**, and failing **Message Window Trimming** (orphaning tool call IDs).

These findings demonstrate that models without deep state-machine simulation capabilities default to sequential imperative heuristics, proving the necessity of graph-aware execution validators in agent development workflows.

---

### Submission Metadata Checklist:
1. **ZIP File:** `langgraphbench-eval.zip` (Code, tasks, references, graders, and documentation; < 50 MB).
2. **Video Walkthrough Link:** [Insert your Loom / YouTube unlisted video link here - see `VIDEO_SCRIPT.md` for the timed walkthrough script].
3. **GitHub Repository:** [https://github.com/akshayparimi18/micro1-hackathon](https://github.com/akshayparimi18/micro1-hackathon)
