# Task 03: Conditional Parallel Fan-Out Channel Collision in LangGraph

You are an expert LangGraph architect analyzing a parallel fan-out workflow.

### Code
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    mode: str
    logs: Annotated[list[str], operator.add]
    audit_score: int

def route_branches(state: ParallelState) -> list[str]:
    if state["mode"] == "parallel_audit":
        return ["branch_a", "branch_b"]
    return ["branch_c"]

def branch_a(state: ParallelState) -> dict:
    return {
        "logs": ["a_done"],
        "audit_score": 10
    }

def branch_b(state: ParallelState) -> dict:
    return {
        "logs": ["b_done"],
        "audit_score": state["audit_score"] + 10
    }

def branch_c(state: ParallelState) -> dict:
    return {
        "logs": ["c_done"],
        "audit_score": 5
    }

builder = StateGraph(ParallelState)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)
builder.add_node("branch_c", branch_c)

builder.add_conditional_edges(START, route_branches, ["branch_a", "branch_b", "branch_c"])
builder.add_edge("branch_a", END)
builder.add_edge("branch_b", END)
builder.add_edge("branch_c", END)

graph = builder.compile()

try:
    result = graph.invoke({
        "mode": "parallel_audit",
        "logs": ["init_log"],
        "audit_score": 0
    })
except Exception as e:
    pass
```

### Problem Description
Trace the parallel execution of the graph when invoked with `mode: "parallel_audit"` and initial `audit_score: 0`. Both `branch_a` and `branch_b` execute concurrently in the same superstep. Determine whether the execution succeeds or raises an exception, which channel causes any conflict, the exact integer values written to that channel by the branches, and the resulting list of collected logs.

### Output Instructions
You MUST return your response as a valid JSON object enclosed strictly inside `<final_answer>...</final_answer>` tags. Do not put markdown code fences inside the tags.

The JSON object MUST contain exactly these keys:
- `"status"`: (str) One of the following exact options:
  - `"Success"`
  - `"InvalidUpdateError"`
  - `"GraphRecursionError"`
  - `"KeyError"`
  - `"ValueError"`
  - `"TypeError"`
  - `"AttributeError"`
  - `"RuntimeError"`
- `"conflicting_channel"`: (str or null) The name of the channel causing the error, or `null` if no error occurs.
- `"values_written_to_conflicting_channel"`: (list of int) The list of integer values emitted for that channel by the parallel branches in that superstep (e.g. `[10, 10]`), or `[]` if no conflict.
- `"surviving_logs"`: (list of str) The logs accumulated during the run, starting with `"init_log"` followed by branch logs in alphabetical order (`["init_log", "a_done", "b_done"]`).

Example response format:
<final_answer>
{
  "status": "InvalidUpdateError",
  "conflicting_channel": "my_channel",
  "values_written_to_conflicting_channel": [10, 10],
  "surviving_logs": ["init_log", "a_done", "b_done"]
}
</final_answer>
