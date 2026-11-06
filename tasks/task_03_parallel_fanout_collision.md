# Task 03: Parallel Fan-Out Channel Collision in LangGraph

You are an expert LangGraph architect analyzing a parallel fan-out workflow.

### Code
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    logs: Annotated[list[str], operator.add]
    audit_score: int  # Note: un-annotated scalar channel

def branch_a(state: ParallelState) -> dict:
    return {
        "logs": ["branch_a_ok"],
        "audit_score": 10
    }

def branch_b(state: ParallelState) -> dict:
    return {
        "logs": ["branch_b_ok"],
        "audit_score": 20
    }

builder = StateGraph(ParallelState)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)

# Parallel fan-out from START to both branches
builder.add_edge(START, "branch_a")
builder.add_edge(START, "branch_b")
builder.add_edge("branch_a", END)
builder.add_edge("branch_b", END)

graph = builder.compile()

try:
    result = graph.invoke({"logs": ["init_log"], "audit_score": 0})
except Exception as e:
    # Error raised during parallel channel resolution
    pass
```

### Problem Description
In LangGraph's Pregel execution model, `branch_a` and `branch_b` execute in the same superstep. When both nodes finish, the engine resolves updates for each channel.
- Channel `logs` has reducer `Annotated[list[str], operator.add]`.
- Channel `audit_score` has NO reducer annotation (`audit_score: int`).

Determine the runtime outcome when multiple parallel nodes write conflicting values to an un-annotated channel, identify the exact exception raised, the conflicting channel, and the contents of the `logs` channel.

### Output Instructions
You MUST return your response as a valid JSON object enclosed strictly inside `<final_answer>...</final_answer>` tags. Do not put markdown code fences inside the tags.

The JSON object MUST contain exactly these keys:
- `"status"`: (str) One of the exact enum values:
  - `"Success"`
  - `"InvalidUpdateError"`
  - `"GraphRecursionError"`
  - `"KeyError"`
- `"conflicting_channel"`: (str or null) The name of the channel causing the conflict (e.g. `"channel_name"`), or `null` if no conflict.
- `"surviving_logs"`: (list of str) The logs collected during the execution, starting with `"init_log"` followed by branch logs in alphabetical order (`["init_log", "branch_a_ok", "branch_b_ok"]`).

Example response format:
<final_answer>
{
  "status": "InvalidUpdateError",
  "conflicting_channel": "my_channel",
  "surviving_logs": ["init_log", "branch_a_ok", "branch_b_ok"]
}
</final_answer>
