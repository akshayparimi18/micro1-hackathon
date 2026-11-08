# Task 04: Checkpointer HITL Replay & as_node Edge Re-routing

You are an expert engineer evaluating state persistence and interruption replay in LangGraph with `MemorySaver`.

### Code
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ReviewState(TypedDict):
    version: int
    approved_by: str
    history: Annotated[list[str], operator.add]

def draft_node(state: ReviewState) -> dict:
    return {"version": state["version"] + 1, "approved_by": "pending", "history": ["drafted"]}

def route_draft(state: ReviewState) -> str:
    if state["approved_by"] == "fast_track":
        return "publish_node"
    return "human_review_node"

def human_review_node(state: ReviewState) -> dict:
    return {"version": state["version"] + 1, "history": ["reviewed"]}

def publish_node(state: ReviewState) -> dict:
    app = state["approved_by"]
    v = state["version"]
    tag = "supervisor_override" if (app == "fast_track" and v == 1) else "standard_publish"
    return {"version": v + 10, "approved_by": tag, "history": [f"published_by_{app}_v{v}"]}

checkpointer = MemorySaver()
builder = StateGraph(ReviewState)
builder.add_node("draft_node", draft_node)
builder.add_node("human_review_node", human_review_node)
builder.add_node("publish_node", publish_node)

builder.add_edge(START, "draft_node")
builder.add_conditional_edges("draft_node", route_draft, ["human_review_node", "publish_node"])
builder.add_edge("human_review_node", "publish_node")
builder.add_edge("publish_node", END)

graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review_node"])
config = {"configurable": {"thread_id": "thread-101"}}

# PHASE 1
graph.invoke({"version": 0, "approved_by": "none", "history": ["init"]}, config=config)

# PHASE 2
graph.update_state(config, {"approved_by": "fast_track", "history": ["override"]}, as_node="draft_node")
final_state = graph.invoke(None, config=config)
```

### Problem Description
Trace the lifecycle across Phase 1 and Phase 2:
1. What state is checkpointed during Phase 1 before the interrupt takes effect?
2. When `graph.update_state` is invoked with `as_node="draft_node"`, how does LangGraph's checkpointer process the reducer for `history`, the state update for `approved_by`, and the outgoing conditional edges from `draft_node`?
3. When `graph.invoke(None, config=config)` is called to resume:
   - Which nodes execute upon resume?
   - Which nodes are bypassed/skipped?
   - What are the final values of `approved_by`, `version`, and `history`?

### Output Instructions
You MUST return your response as a valid JSON object enclosed strictly inside `<final_answer>...</final_answer>` tags. Do not put markdown code fences inside the tags.

The JSON object MUST contain exactly these keys:
- `"executed_nodes_on_resume"`: (list of str) The ordered list of node names that executed during Phase 2 resume (e.g. `["node_a"]`).
- `"skipped_nodes_due_to_as_node"`: (list of str) The list of node names that were skipped because of the `as_node` update, or `[]` if none.
- `"final_approved_by"`: (str) The exact value of `final_state["approved_by"]`.
- `"final_version"`: (int) The exact integer value of `final_state["version"]`.
- `"final_history"`: (list of str) The exact final list of strings in `final_state["history"]`.

Example response format:
<final_answer>
{
  "executed_nodes_on_resume": ["publish_node"],
  "skipped_nodes_due_to_as_node": ["human_review_node"],
  "final_approved_by": "supervisor_override",
  "final_version": 11,
  "final_history": ["init", "drafted", "override", "published_by_fast_track_v1"]
}
</final_answer>
