# Initial Project Plan: Inspectable DAG QA Agent

## Objective

Build a single-machine, containerized AI question-answering system that can answer multi-hop questions by planning a dependency DAG, executing node-level LLM calls, and exposing the whole execution flow in a frontend.

HotpotQA is not part of the runtime knowledge source for the first iteration. It is an external benchmark used after the system exists, so we do not tune or validate the system on the same examples we later report as evaluation.

## First Iteration Scope

Included:

- LLM planner that decomposes a user question into a YAML DAG.
- DAG validator that checks structure only.
- Scheduler that supports both dependency chains and independent parallel branches.
- Node executor that calls the configured LLM for each runnable node.
- Config-file driven runtime behavior: model, retries, concurrency, timeouts, planner limits, benchmark limits, and output repair.
- Execution trace for every node call: prompt/input, dependency values, raw model response, parsed return value, status, duration, and errors.
- Frontend for asking questions, inspecting the DAG, inspecting node calls, and running a HotpotQA benchmark.
- Programmatic API and CLI so benchmarks can be launched from another program or script.
- Containerized local app.
- Unit tests, integration tests, and a small end-to-end smoke test.

Excluded from the first iteration:

- No HotpotQA corpus loading for answering user questions.
- No document retrieval graph.
- No BM25/dense retrieval over external documents.
- No internet retrieval.
- No factual verification against evidence.
- No training or fine-tuning.

## System Architecture

```text
User Question
  -> Planner LLM Call
  -> YAML DAG
  -> Structural DAG Validation
  -> Scheduler
  -> Parallel/Dependent Node Execution
  -> Structural Output Validation
  -> Final Node Result
  -> Trace + Graph + Benchmark Artifacts
```

All computation runs in one process/container for the MVP. The DAG can express parallelism, and the scheduler can execute ready nodes concurrently with `asyncio`, but no distributed infrastructure is needed.

## LLM Abstraction

Use DSPy behind a narrow adapter so the rest of the system does not depend on a specific provider.

```python
class LanguageModel:
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

The initial backend is Gemini through DSPy. Later backends can be OpenAI-compatible APIs, vLLM, or a university-cluster-hosted model.

Model switching should remain a config change:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
```

Runtime behavior is also config-driven:

```yaml
planner:
  max_nodes: 10
  max_depth: 3
  repair_rounds: 1
execution:
  max_parallel_nodes: 4
  node_timeout_seconds: 45
  node_repair_rounds: 1
  fail_fast: false
benchmark:
  default_limit: 100
  output_dir: runs/benchmarks
```

## DAG Contract

The planner returns YAML in this shape:

```yaml
question: "Which of the two people was born earlier?"
nodes:
  - id: q1
    label: "Identify author"
    task_type: fact_lookup
    question: "Who is the author of Book A?"
    operation: answer
    depends_on: []
    prompt:
      system: "Answer the subquestion concisely. Return only the requested JSON object."
      user_template: |
        Subquestion: {node.question}
    input_map: {}
    output_schema:
      type: object
      required: [answer]
      properties:
        answer:
          type: string

  - id: q2
    label: "Resolve birth date"
    task_type: fact_lookup
    question: "When was {q1.answer} born?"
    operation: answer
    depends_on: [q1]
    prompt:
      system: "Use the dependency values to answer the subquestion. Return only JSON."
      user_template: |
        Dependency values:
        {dependencies}

        Subquestion:
        {resolved_question}
    input_map:
      subject: q1.answer
    output_schema:
      type: object
      required: [answer]
      properties:
        answer:
          type: string

  - id: q3
    label: "Compare dates"
    task_type: comparison
    question: "Who was born earlier: {q2.answer} or {q5.answer}?"
    operation: compare
    depends_on: [q2, q5]
    prompt:
      system: "Compare the child outputs and return the final JSON object."
      user_template: |
        Child outputs:
        {dependencies}

        Comparison question:
        {resolved_question}
    input_map:
      left_date: q2.answer
      right_date: q5.answer
    child_output_policy: "Use both child answers. If one is missing, return answer=null and explain the missing input."
    output_schema:
      type: object
      required: [answer, reasoning]
      properties:
        answer:
          type: string
        reasoning:
          type: string
final_node: q3
```

Node fields:

- `id`: stable node identifier used by dependencies and traces.
- `label`: short UI label.
- `task_type`: semantic kind of question, for example `fact_lookup`, `entity_resolution`, `date_lookup`, `comparison`, `calculation`, `classification`, or `synthesis`.
- `operation`: executor behavior, for example `answer`, `transform`, `compare`, or `synthesize`.
- `question`: node-level question. It may contain placeholders from dependency outputs.
- `depends_on`: parent outputs required before this node can run.
- `prompt`: instructions and template the node uses for its LLM call.
- `input_map`: named values pulled from child outputs, such as `left_date: q2.answer`.
- `child_output_policy`: how the node should use child values and what to do if a required child value is missing.
- `output_schema`: JSON Schema subset defining the format the node must return.

Supported operations:

- `answer`: answer a subquestion using the LLM.
- `transform`: rewrite, normalize, or extract a structured value from dependency outputs.
- `compare`: compare multiple dependency outputs.
- `synthesize`: produce the final answer from prior node outputs.

## Structural Validation

The validator does not judge truthfulness. It only checks whether the graph and node outputs conform to the expected structure.

DAG validation:

- all node IDs are unique,
- all dependencies reference existing nodes,
- graph is acyclic,
- there is exactly one final node,
- final node exists,
- max depth and max node count are respected,
- placeholders such as `{q1.answer}` only reference declared dependencies,
- `input_map` references only declared dependencies,
- operations are from the supported operation set,
- task types are from the supported task type set,
- each node has prompt instructions or inherits a default prompt for its operation/task type,
- output schemas are valid JSON Schema subsets.

Node output validation:

- raw LLM response can be parsed as JSON or YAML,
- parsed object satisfies the node output schema,
- required fields exist,
- field types match,
- invalid output triggers one repair prompt that includes the schema and parser error.

## Scheduler

The scheduler maintains node state:

- `pending`
- `ready`
- `running`
- `succeeded`
- `failed`
- `skipped`

Execution rules:

- A node is `ready` when all dependencies have `succeeded`.
- Ready nodes with no dependency relation can run concurrently.
- Dependent chains run in order.
- The execution trace records scheduling waves so the frontend can show parallel steps.

Example:

```text
Wave 1: q1, q4
Wave 2: q2, q5
Wave 3: q3, q6
Wave 4: q7
```

The implementation can expose both:

- dependency-aware DFS trace for easy debugging,
- wave-based async execution for actual runtime.

## Frontend

Build a practical inspection UI, not a landing page.

Use ShadCN UI on top of React + TypeScript + Tailwind. The visual direction should be clean defense/operations analytics: dark neutral base, restrained contrast, compact panels, status strips, map-room style graph inspection, sharp typography, and sparse accent colors for state. It should feel closer to Helsing/Palantir operational software than a marketing dashboard.

Main views:

- Ask: question input, run button, model/config selector, final answer.
- Graph: visual DAG with node status colors.
- Trace: chronological list of scheduler waves and LLM calls.
- Node Inspector: selected node details:
  - original node question,
  - resolved question after dependency substitution,
  - dependency values,
  - prompt sent to the model,
  - raw model response,
  - parsed node return value,
  - structural validation result,
  - retry/repair attempt if any,
  - duration and token metadata when available.
- Benchmarks: run HotpotQA benchmark with limit/split controls and display metrics/results.

Graph rendering:

- Backend emits graph JSON and Mermaid text.
- Frontend renders Mermaid initially.
- Use React Flow later only if Mermaid becomes too limiting.

The frontend should look polished from the first iteration: dense but readable layout, status colors, tabs, split-pane graph/inspector view, and clear call trace tables.

Suggested UI components:

- ShadCN `Tabs` for Ask, Trace, Graph, Benchmarks, and Config.
- ShadCN `ResizablePanelGroup` for graph plus node inspector.
- ShadCN `Table` for call traces and benchmark results.
- ShadCN `Sheet` or `Dialog` for raw prompt/response inspection.
- ShadCN `Badge` for node status, task type, and validation result.
- ShadCN `Select`, `Input`, `Switch`, and `Slider` for config overrides.

The app should expose a config view that shows the active config and allows safe runtime overrides for non-secret values such as retry count, max parallel nodes, planner depth, benchmark limit, and timeout.

## Programmatic Interface

The system must be easy to call from another program.

HTTP API:

```text
POST /api/ask
POST /api/plan
POST /api/execute
POST /api/benchmarks/hotpotqa
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/trace
GET  /api/config
POST /api/config/validate
```

CLI:

```bash
dagqa ask "Which person was born earlier?"
dagqa plan "Which person was born earlier?" --output plan.yaml
dagqa execute plan.yaml --output run.json
dagqa benchmark hotpotqa --limit 100 --system dag_agent --output runs/hotpot.jsonl
dagqa config validate configs/local.yaml
```

Python API:

```python
from dagqa import DagQaClient

client = DagQaClient.from_config("configs/local.yaml")
run = await client.ask("Which person was born earlier?")
benchmark = await client.benchmark_hotpotqa(limit=100, system="dag_agent")
```

All interfaces should return stable machine-readable JSON with run IDs, metrics, graph data, and traces.

## HotpotQA Benchmarking

HotpotQA is used only for evaluation.

Flow:

```text
HotpotQA question
  -> same planner/scheduler/node executor as normal user questions
  -> final answer
  -> compare final answer to HotpotQA ground truth
```

No HotpotQA context documents are loaded into the answer pipeline in the first iteration.

Benchmark modes:

- `direct_llm`: ask the configured LLM the question directly.
- `dag_agent`: use the planned DAG execution system.

Metrics:

- Exact Match.
- Answer F1.
- Latency.
- Planning latency.
- Execution latency.
- Per-node latency.
- Total runtime.
- LLM call count.
- Token counts when the provider exposes them.
- Average node count.
- Average graph depth.
- Structural failure rate.
- Repair rate.

Supporting fact metrics are deferred until retrieval/evidence is added.

## Repository Structure

```text
dagqa/
  __init__.py
  cli.py
  config.py
  schemas.py
  llm/
    base.py
    dspy_adapter.py
    gemini.py
    fake.py
  planning/
    planner.py
    prompts.py
    parser.py
    validator.py
  graph/
    scheduler.py
    executor.py
    substitution.py
    render.py
    trace.py
  nodes/
    runner.py
    prompts.py
    output_validation.py
    repair.py
  eval/
    hotpot_loader.py
    metrics.py
    benchmark.py
app/
  main.py
  api.py
  web/
    package.json
    components.json
    src/
      App.tsx
      api.ts
      lib/
        utils.ts
      components/
        GraphView.tsx
        NodeInspector.tsx
        TraceTable.tsx
        BenchmarkPanel.tsx
        ConfigPanel.tsx
tests/
  unit/
  integration/
  e2e/
configs/
  local.yaml
docker/
  Dockerfile
docker-compose.yml
```

## Tooling

Python:

- `uv` for dependency and environment management.
- `ruff` for linting and formatting.
- `mypy` or `pyright` for static typing.
- `pytest` for tests.
- `pytest-asyncio` for scheduler/executor tests.
- `coverage.py` for coverage reporting.
- `pre-commit` for local quality checks.

Frontend:

- Vite + React + TypeScript.
- ShadCN UI.
- Tailwind CSS.
- ESLint.
- Prettier.
- Vitest for component/unit tests.
- Playwright for one browser smoke test.

## Testing Pyramid

Unit tests, largest share:

- YAML parser.
- DAG validator.
- placeholder substitution.
- output schema validation.
- task type and input map validation.
- scheduler readiness and wave generation.
- HotpotQA EM/F1 metrics.
- fake LLM adapter.

Integration tests:

- planner with fake LLM output,
- executor with fake LLM responses,
- repair path for malformed node output,
- API `/ask` using fake LLM.

End-to-end smoke tests:

- start app with fake LLM,
- ask a fixed question,
- assert graph renders,
- inspect a node,
- assert final answer appears,
- run tiny benchmark fixture.

External LLM tests should be opt-in and skipped by default in CI.

## Implementation Milestones

### Milestone 1: Core Schemas And Validation

- Define Pydantic models for DAGs, nodes, traces, and LLM calls.
- Implement YAML parsing.
- Implement DAG structural validation.
- Implement node output schema validation.
- Implement task type, prompt template, input map, and child output policy validation.
- Add unit tests.

### Milestone 2: LLM Adapter And Planner

- Implement DSPy adapter.
- Implement fake adapter for deterministic tests.
- Implement planner prompt and repair-on-invalid-DAG flow.
- Add parser/validator integration tests.

### Milestone 3: Scheduler And Executor

- Implement dependency-aware scheduler with parallel waves.
- Implement node prompt construction and dependency substitution.
- Record full execution traces.
- Add async executor tests.

### Milestone 4: Frontend Demo

- Implement FastAPI `/ask` endpoint.
- Implement ShadCN React UI with graph, trace, node inspector, benchmark panel, and config panel.
- Add polished Helsing/Palantir-style operational layout and status styling.
- Add Docker Compose for backend/frontend.

### Milestone 5: Hotpot Benchmark

- Add HotpotQA loader for evaluation only.
- Implement direct LLM and DAG agent benchmark modes.
- Implement EM/F1 and runtime diagnostics.
- Add benchmark UI and CLI.

## First Demo Definition Of Done

- Docker Compose starts the app.
- User can ask a question in the browser.
- The generated DAG is visible.
- Parallel and dependent nodes are represented correctly.
- Each node clearly shows its task type, operation, prompt behavior, input mapping, and return format.
- The UI shows each LLM call, raw response, parsed value, and returned node value.
- Node output format validation is visible.
- Programmatic CLI, HTTP, and Python entry points can run an ask flow and a benchmark flow.
- A small HotpotQA benchmark can be launched from CLI and frontend.
- Model provider/model can be changed in config without touching planner/executor code.

## Initial Defaults

```yaml
planner:
  max_nodes: 10
  max_depth: 3
  repair_rounds: 1
execution:
  max_parallel_nodes: 4
  node_timeout_seconds: 45
  node_repair_rounds: 1
  fail_fast: false
validation:
  structural_only: true
benchmark:
  dataset: hotpotqa
  split: validation
  limit: 100
```
