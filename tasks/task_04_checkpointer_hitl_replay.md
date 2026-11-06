# Task 04: Checkpointer Human-In-The-Loop (HITL) Resume & State Replay

You are an expert engineer evaluating state persistence and interruption replay in LangGraph with `MemorySaver`.

### Code
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DocumentState(TypedDict):
    version: int
    approved_by: str

def draft_node(state: DocumentState) -> dict:
    return {"version": state["version"] + 1, "approved_by": "pending"}

def human_review_node(state: DocumentState) -> dict:
    return {"version": state["version"] + 1}

def publish_node(state: DocumentState) -> dict:
    tag = "supervisor_override" if (state["approved_by"] == "supervisor" and state["version"] == 2) else "rejected"
    return {"approved_by": tag}

checkpointer = MemorySaver()
builder = StateGraph(DocumentState)
builder.add_node("draft_node", draft_node)
builder.add_node("human_review_node", human_review_node)
builder.add_node("publish_node", publish_node)

builder.add_edge(START, "draft_node")
builder.add_edge("draft_node", "human_review_node")
builder.add_edge("human_review_node", "publish_node")
builder.add_edge("publish_node", END)

# Interrupted right before human review node
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review_node"])
config = {"configurable": {"thread_id": "thread-101"}}

# PHASE 1: Initial invocation (pauses before human_review_node)
graph.invoke({"version": 0, "approved_by": "none"}, config=config)

# PHASE 2: State update & resume
graph.update_state(config, {"approved_by": "supervisor"})
final_state = graph.invoke(None, config=config)
```

### Problem Description
Trace the lifecycle of checkpoints and node execution:
1. What happened in Phase 1 before the interrupt took effect?
2. When `graph.update_state(config, {"approved_by": "supervisor"})` is called, how does it affect the thread's checkpoint?
3. When `graph.invoke(None, config=config)` is called to resume:
   - Does `draft_node` re-execute?
   - Which nodes execute upon resume?
   - What are the final values of `version` and `approved_by`?

### Output Instructions
You MUST return your response as a valid JSON object enclosed strictly inside `<final_answer>...</final_answer>` tags. Do not put markdown code fences inside the tags.

The JSON object MUST contain exactly these keys:
- `"executed_nodes_on_resume"`: (list of str) The ordered list of node names executed during Phase 2 resume (e.g. `["node_a", "node_b"]`).
- `"re_executed_nodes"`: (list of str) Any nodes that were already executed in Phase 1 but re-executed in Phase 2, or `[]` if none.
- `"final_approved_by"`: (str) The exact value of `final_state["approved_by"]`.
- `"final_version"`: (int) The exact value of `final_state["version"]`.

Example response format:
<final_answer>
{
  "executed_nodes_on_resume": ["node_1", "node_2"],
  "re_executed_nodes": [],
  "final_approved_by": "supervisor_override",
  "final_version": 2
}
</final_answer>

