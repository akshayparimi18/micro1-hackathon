# HackerRank Submission Description & Metadata

Copy and paste the text below into the submission form on HackerRank for the micro1 Eval Hackathon.

---

### Short Written Description (for HackerRank Event Page)

**Domain & Task Suite:**
LangGraphBench is an expert-level evaluation benchmark comprising 6 rigorous tasks designed to measure frontier AI models on multi-agent graph state management, channel reducers, and Pregel concurrency in LangGraph.

**The Real-World Limitation:**
While LLMs can write standard imperative Python boilerplate, they consistently fail to mentally simulate asynchronous Pregel state machines. Specifically, models exhibit an "Imperative Python Bias"—they treat node return values as standard dictionary reassignments rather than channel updates reduced across supersteps. This leads to catastrophic failure modes in production: explosive list duplication under `operator.add`, unexpected `InvalidUpdateError` crashes on parallel fan-out, undetected cyclic deadlocks in conditional routers, state corruption during human-in-the-loop checkpointer replays, and broken sliding-window message reducers that cause fatal 400 Bad Request errors.

**Methodology & Findings:**
LangGraphBench utilizes a 100% deterministic code grader with zero false passes and zero false negatives, verified programmatically against the live LangGraph runtime engine. Evaluations across multi-trial runs ($k=3$) reveal that models repeatedly mispredict state reducer accumulation, fail to anticipate concurrent write collisions on un-annotated channels, and miss subtle state resets that trap agent graphs in infinite loops. This benchmark demonstrates that AI models lack true state-machine simulation capabilities, highlighting the urgent need for graph-aware execution validators in agent development workflows.

---

### Submission Metadata Checklist:
1. **ZIP File:** `langgraphbench-eval.zip` (Code, tasks, references, graders, and documentation; < 50 MB).
2. **Video Walkthrough Link:** [Insert your Loom / YouTube unlisted video link here - see `VIDEO_SCRIPT.md` for the timed walkthrough script].
3. **GitHub Repository:** [https://github.com/akshayparimi18/micro1-hackathon](https://github.com/akshayparimi18/micro1-hackathon)
