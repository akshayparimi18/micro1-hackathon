# LangGraphBench: 5-Minute Video Walkthrough Script

Use this structured script to record your video submission for the micro1 Eval Hackathon (HackerRank). You can record your screen showing your terminal running `verify_ground_truth.py` and `run_eval.py`, viewing the task markdown files in VS Code, and inspecting the deterministic grader.

---

### [00:00 - 01:00] 1. The Problem & Real-World Impact (Why It Matters)
* **Visual:** Open `README.md` or a diagram showing a multi-agent LangGraph workflow with parallel fan-outs and conditional edges.
* **Speaker Script:**
> "Hi everyone, welcome to LangGraphBench. As developers building autonomous multi-agent systems with Python, FastAPI, and LangGraph, we increasingly rely on AI models to design and review our workflows.
>
> But there is a massive hidden failure mode: frontier AI models suffer from an **'Imperative Python Bias'**. They look at code as sequential line-by-line execution. But LangGraph is fundamentally different—it is built on Google's **Pregel state-machine architecture**, where state is managed through discrete channels, asynchronous supersteps, and reducer functions.
>
> When an AI model misunderstands this execution model, the consequences in production are catastrophic: runaway API bills from infinite conditional loops, memory leaks from duplicate reducer accumulation, and sudden `InvalidUpdateError` crashes under parallel user traffic. LangGraphBench measures exactly where frontier models fail on these graph dynamics."

---

### [01:00 - 02:30] 2. Task Design & The Deterministic Grader Architecture
* **Visual:** Open `tasks/task_01_reducer_duplication.md`, `tasks/task_03_parallel_fanout_collision.md`, and `graders/langgraph_grader.py`.
* **Speaker Script:**
> "LangGraphBench consists of 6 production-grade tasks covering the core architectural pillars of LangGraph:
>
> In **Task 01**, a developer writes `return {"items": state["items"] + ["step_1"]}` on a channel annotated with `operator.add`. An imperative thinker assumes the list has 3 items. But in Pregel, the return value is reduced back onto the state, exploding into 7 duplicated items.
>
> In **Task 03**, a graph fans out in parallel to two branches. Both branches return updates to `logs` and `audit_score`. Because `audit_score` is an un-annotated scalar channel, LangGraph's engine raises a runtime `InvalidUpdateError`.
>
> In **Task 04**, we test human-in-the-loop checkpointer resumption with `MemorySaver`. When state is updated during an interrupt, models must accurately predict which nodes re-execute versus resume.
>
> Our grading system is 100% deterministic Python code. No LLM judges, no subjective rubrics. It parses the model's `<final_answer>` JSON and validates exact field matches against ground truth."

---

### [02:30 - 04:15] 3. Results & Ground-Truth Verification
* **Visual:** Switch to the terminal. Run:
  1. `python -m unittest graders/test_graders.py`
  2. `python verify_ground_truth.py`
  3. `python run_eval.py --verify-reference`
  4. `python run_eval.py --verify-negative`
* **Speaker Script:**
> "Let's verify the benchmark right here in the terminal.
>
> First, our unit test suite verifies that modifying any single field causes an immediate failure, proving zero false passes.
>
> Second, we run `verify_ground_truth.py`. This script doesn't just check static files—it programmatically executes the actual LangGraph runtime engine and asserts that our ground-truth solutions match the runtime 100%.
>
> Next, running `python run_eval.py --verify-reference` confirms that the oracle reference passes with 100%, and `--verify-negative` confirms that typical LLM hallucinations—such as believing Task 1 only has 3 items—are rejected with a 0% pass rate.
>
> When we benchmark frontier models across multiple trials, models repeatedly trip up. They consistently fail Task 01 because their imperative prior overrides graph reducers. They fail Task 03 because they assume parallel writes overwrite cleanly like Python variables. And in Task 05, they recommend slicing message windows that orphan tool messages, triggering 400 Bad Request errors from LLM APIs."

---

### [04:15 - 05:00] 4. The Hot Take & Takeaways
* **Visual:** Highlight Section 5 in `README.md` ("The Hot Take: Imperative Python Bias vs. Pregel Graph Execution").
* **Speaker Script:**
> "Here is our core insight and hot take:
>
> Frontier LLMs do not simulate state machines. They perform token association based on sequential Python paradigms. In imperative Python, returning an updated dict replaces the state. In graph architectures, returning a dict sends a message into a channel reducer.
>
> Because models cannot distinguish between sequential variable assignment and channel aggregation, they cannot be trusted to architect multi-agent systems without automated runtime verification.
>
> LangGraphBench is fully reproducible, packaged with pinned dependencies, and available open-source on GitHub. Thank you!"
