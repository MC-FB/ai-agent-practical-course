# DAG QA System Documentation

This project implements an inspectable DAG-based question answering agent. Instead of sending a complex multi-hop question directly to one model call, the system first asks a planner model to decompose the question into a directed acyclic graph. Each graph node is a smaller task with a prompt, declared dependencies, an input contract, and a JSON output schema. The executor then runs independent nodes in parallel, passes child outputs into parent nodes, validates every node output, and returns the final node output as the answer.

## Main Components

The backend package is `dagqa`.

- `dagqa/client.py` is the main application-facing client. It connects planning and execution.
- `dagqa/planning/planner.py` asks the language model to create a DAG plan.
- `dagqa/planning/prompts.py` defines the planning instructions. These instructions require nodes to declare dependencies, input maps, prompt templates, child output policies, and output schemas.
- `dagqa/planning/parser.py` parses the planner response into a `DagPlan`.
- `dagqa/planning/validator.py` checks that the graph is valid before execution.
- `dagqa/graph/scheduler.py` groups nodes into execution waves so independent nodes can run at the same time.
- `dagqa/graph/executor.py` executes the validated graph wave by wave.
- `dagqa/graph/substitution.py` resolves placeholders such as `{q1.answer}` and `input_map` references such as `q1.birthdate`.
- `dagqa/nodes/runner.py` executes one node: it resolves child values, renders the prompt, calls the LLM, parses the response, validates the output, and records a trace.
- `dagqa/nodes/output_validation.py` parses JSON/YAML-like model output and validates it with JSON Schema.
- `dagqa/nodes/prompts.py` renders the actual prompt sent to the model and appends the node's required return schema.
- `app/api.py` exposes FastAPI endpoints for planning, execution, live runs, and benchmark runs.
- `app/web/src/App.tsx` is the React UI for asking questions and inspecting graph traces.

## Data Model

The core schema is in `dagqa/schemas.py`.

A `DagPlan` contains:

- `question`: the original user question.
- `nodes`: the list of graph nodes.
- `final_node`: the node whose output becomes the final answer.

Each `DagNode` contains:

- `id`: stable node identifier, for example `q1`.
- `label`: human-readable node name.
- `task_type`: task category such as `fact_lookup`, `date_lookup`, `comparison`, or `synthesis`.
- `operation`: high-level action such as `answer`, `compare`, or `synthesize`.
- `question`: the node-specific question.
- `depends_on`: child node IDs that must run before this node.
- `prompt.system`: system prompt for the node call.
- `prompt.user_template`: user prompt template for the node call.
- `input_map`: named values this node expects from child nodes.
- `child_output_policy`: optional explanation of how child outputs should be used.
- `output_schema`: JSON Schema describing the exact object this node must return.

Each executed node produces a `NodeTrace` with:

- `dependency_values`: the actual resolved child values used by this node.
- `resolved_question`: the node question after direct placeholders are substituted.
- `rendered_prompt`: the full prompt sent to the LLM.
- `raw_response`: the raw model response.
- `parsed_output`: parsed model output.
- `returned_value`: the validated object passed to parent nodes.
- `validation`: whether the output matched the node schema.
- `duration_ms`, `status`, and possible `error`.

## Graph Structure And Parent-Child Contracts

The graph is a DAG. Edges point from child nodes to parent nodes. A parent can only consume values from nodes listed in its `depends_on`.

The important contract is:

1. A child node declares what it returns through `output_schema`.
2. A parent node declares what it expects from children through `input_map`.
3. During execution, the runner resolves the parent `input_map` from already completed child `returned_value` objects.
4. The resolved values become `dependency_values`.
5. The parent prompt is rendered with those values.
6. The parent model call returns its own JSON object, which is validated against the parent's `output_schema`.
7. That validated object becomes the value available to the parent's own parents.

Example:

```yaml
id: person_facts
output_schema:
  type: object
  properties:
    birthday:
      type: string
    location:
      type: string
  required: [birthday, location]
```

A parent can request these child values:

```yaml
id: compare_birthplace
depends_on: [person_facts]
input_map:
  birthday: person_facts.birthday
  location: person_facts.location
```

Before the parent prompt is sent, the system resolves the map into saturated values:

```json
{
  "birthday": "28.02.2002",
  "location": "Munich"
}
```

The rendered parent prompt can include these values through `{dependencies}`, `{birthday}`, `{location}`, or direct child placeholders such as `{person_facts.birthday}`. This is implemented in `dagqa/nodes/prompts.py`.

This behavior is already implemented. The key code path is:

- `DagExecutor.execute()` stores successful child outputs in an `outputs` dictionary.
- `NodeRunner.run()` calls `resolve_input_map(node.input_map, outputs)`.
- `resolve_input_map()` reads references like `person_facts.birthday` from previous node outputs.
- `render_node_prompt()` substitutes the resolved values into the parent prompt.
- `validate_node_output()` checks the parent return value against `output_schema`.

The planner and validator also enforce the contract:

- The planner prompt tells the model that every dependent node must include child values in `input_map` and `prompt.user_template`.
- The validator rejects references to nodes that are not listed in `depends_on`.
- The validator rejects references to fields that do not exist in the child node's `output_schema`.
- The validator rejects dependent nodes with an empty `input_map`.
- The validator rejects dependent nodes whose prompt does not include child values.

## Execution Flow

1. The user asks a question in the web UI.
2. `app/api.py` receives the request at `/api/ask/live`.
3. The API creates a live run record with phase `planning`.
4. The planner creates a YAML DAG plan.
5. The YAML is parsed into a `DagPlan`.
6. The plan is normalized and validated.
7. The scheduler builds execution waves. Nodes with no unresolved dependencies are placed in the same wave and may run in parallel.
8. For each wave, `NodeRunner` executes all nodes subject to the configured concurrency limit.
9. Successful node `returned_value` objects are stored by node ID.
10. Parent nodes use those stored child objects through `input_map`.
11. When all waves complete, the executor returns the `returned_value` of `final_node`.

## Prompt Rendering

Each node has a `prompt.user_template`. The renderer builds a context containing:

- `node.id`
- `node.label`
- `node.question`
- `resolved_question`
- `dependencies`, which is the complete resolved `input_map` as formatted JSON
- direct dependency names from `input_map`, such as `birthday`
- namespaced child output fields, such as `person_facts.birthday`

The rendered prompt also appends the required JSON Schema and tells the model to return JSON only. That is why each node output can be parsed and validated consistently.

## Output Validation And Repair

Model output is not trusted directly. `parse_node_output()` attempts to parse the raw response as JSON or YAML and can extract JSON-like content from a response. `validate_node_output()` validates the parsed object with `jsonschema`.

If validation fails, the node runner can ask the model for a repair response using the configured number of repair rounds. The repair prompt includes the schema, validation errors, and previous response. If the repaired object validates, the node succeeds. Otherwise, the node fails and the graph run is marked failed.

## API And Live UI

The live UI uses:

- `POST /api/ask/live` to start a run.
- `GET /api/ask/live/{run_id}` to poll progress.

The API stores intermediate live state in memory. During execution it updates:

- current phase
- plan
- scheduler waves
- node traces
- Mermaid graph text
- final answer
- errors

The React UI shows:

- the original question input
- final answer
- run metrics
- Mermaid graph
- trace list
- node inspector

The node inspector now makes the parent-child contract explicit:

- **Expected From Children** shows each value from `input_map`, the source child field, and the expected schema type.
- **Received Saturated Values** shows the actual resolved values used by this node.
- **Rendered Prompt Sent To Model** shows the final prompt after substitution.
- **Return Format Expected By Parent** shows this node's JSON Schema.
- **Returned To Parent** shows the validated object that parent nodes can consume.

## Benchmark Support

The project also includes HotpotQA benchmark support:

- `dagqa/eval/hotpot_loader.py` loads examples.
- `dagqa/eval/benchmark.py` runs the configured system.
- `dagqa/eval/metrics.py` computes exact match, F1, latency, graph depth, node count, and structural metrics.
- `app/api.py` exposes benchmark endpoints.
- The UI has dataset and results tabs for running and reviewing benchmark samples.

## Current System Behavior

The requested parent-child value passing already happens. Parent nodes do not need to manually parse child responses from text. They receive structured child output fields through `input_map`, and those values are inserted into the parent prompt before the parent model call.

The UI has been improved so the contract is visible for each selected node: what it expects, what it received, what prompt was sent, what schema it must return, and what object it actually returned.
