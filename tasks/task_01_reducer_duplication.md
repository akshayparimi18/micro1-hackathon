# Task 01: Reducer State Duplication in LangGraph

You are an expert Python engineer debugging a LangGraph workflow. Analyze the following code and determine the exact state after the workflow completes.

### Code
```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    items: Annotated[list[str], operator.add]

def node_1(state: WorkflowState) -> dict:
    # Developer intends to add "step_1" to the list
    return {"items": state["items"] + ["step_1"]}

def node_2(state: WorkflowState) -> dict:
    # Developer intends to add "step_2" to the list
    return {"items": state["items"] + ["step_2"]}

builder = StateGraph(WorkflowState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()
result = graph.invoke({"items": ["seed"]})
```

### Problem Description
The developer intended the final list of `items` to be `["seed", "step_1", "step_2"]`. However, LangGraph's Pregel execution engine applies channel reducers to node return values. Trace the exact sequence of state updates and reducer applications from `START` to `END`.

### Output Instructions
You MUST return your response as a valid JSON object enclosed strictly inside `<final_answer>...</final_answer>` tags. Do not put markdown code fences inside the tags.

The JSON object MUST contain exactly these keys:
- `"final_items"`: (list of str) The exact final list of strings in `result["items"]`.
- `"total_length"`: (int) The exact length of `result["items"]`.
- `"root_cause"`: (str) One of the following exact enum values:
  - `"variable_reassignment_overwrite"`
  - `"in_node_concatenation_with_operator_add"`
  - `"mutable_default_argument"`
  - `"normal_append"`

Example response format:
<final_answer>
{
  "final_items": ["item1", "item2"],
  "total_length": 2,
  "root_cause": "in_node_concatenation_with_operator_add"
}
</final_answer>
