# Design Decisions

## 1. HotpotQA Is Evaluation, Not Runtime Knowledge

Decision: The first system does not load HotpotQA contexts into the answer pipeline. HotpotQA is used after implementation as an external benchmark.

Reasoning: If the first version is built around HotpotQA-specific context ingestion, it becomes too easy to accidentally optimize the system around the benchmark distribution instead of building a general DAG-based question-answering agent. The first iteration should answer through model calls and graph orchestration only, then measure how well that performs on HotpotQA.

Implication: Initial HotpotQA metrics are a test of decomposition and orchestration, not retrieval quality. Supporting-fact metrics are deferred until retrieval is part of the system.

## 2. No Retrieval Graph In The First Iteration

Decision: The MVP does not build BM25, dense retrieval, document graphs, atom-entity graphs, or internet search.

Reasoning: The course goal for the first supervisor demo is to show the reasoning system: planner, DAG, scheduler, node execution, and inspectable traces. Retrieval introduces a second large problem: corpus ingestion, indexing, passage ranking, evidence attribution, and retrieval evaluation. That should be added only after the agent loop is inspectable and testable.

Paper reference: AtomicRAG argues for atom-level evidence and graph retrieval, but also shows that graph construction, entity extraction, and propagation are substantial system components. That is valuable later, but too much for the first iteration.

## 3. Structural Validation Only

Decision: The system validates graph shape and node output format, not factual correctness.

Reasoning: A factual verifier without evidence is unreliable and can hide the real behavior of the system behind another LLM judgment. The first version should expose what each node asked, what the model returned, and whether the return value matched the expected schema.

Paper reference: The factuality paper separates evaluation concerns: retrieval relevance, generation quality, and faithfulness are different questions. Since this MVP has no evidence retrieval, it should not claim evidence-based verification.

## 4. DAG Rather Than Chain Or Tree

Decision: The planner emits a DAG.

Reasoning: A chain is too restrictive for questions with independent branches. A tree is useful, but a DAG also allows reuse of an intermediate answer by multiple downstream nodes. The scheduler can still run tree-shaped plans, chain-shaped plans, and parallel branches.

Paper reference: RT-RAG motivates explicit hierarchical decomposition and bottom-up synthesis. The project generalizes this into a DAG so independent subquestions can run in parallel and shared intermediate values do not need to be recomputed.

## 5. Parallel-Ready Scheduler

Decision: The scheduler should execute all ready nodes concurrently up to a configurable limit.

Reasoning: The DAG explicitly represents dependencies. If two nodes do not depend on each other, they should be able to run in the same scheduling wave. This is useful for comparison questions such as "Which person was born earlier?" where both branches can be resolved independently before a final compare node.

Implementation note: The first implementation can use `asyncio` inside one process. No distributed scheduler is needed.

## 6. Inspectability Is A First-Class Feature

Decision: Every node execution records a trace containing prompt inputs, dependency values, raw model output, parsed value, validation result, retry attempts, timing, and final returned value.

Reasoning: The project is research-oriented. When the system fails, the important question is whether the failure came from planning, dependency substitution, a model call, malformed output, or final synthesis. A frontend node inspector makes these failures visible to the supervisor and to future development.

Paper reference: Plan-and-Solve and Tree of Thoughts both emphasize explicit intermediate reasoning. The UI should make those intermediate steps observable rather than hiding them in a final answer.

## 7. DSPy Behind A Local Adapter

Decision: Use DSPy for LLM calls, but isolate it behind a small internal adapter.

Reasoning: DSPy is useful for declarative LLM modules and later optimization, but the project should not couple planner, scheduler, or benchmark code to one provider. The model should be swappable by config.

Implication: Gemini is the first backend. Later, a university-cluster model or OpenAI-compatible endpoint can implement the same adapter.

## 8. YAML DAG From The Planner

Decision: The planner produces YAML, then the backend parses it into typed Pydantic models.

Reasoning: YAML is readable for supervisor demos and easy to show in the frontend. Pydantic gives strict validation and clear repair prompts when the model emits invalid structure.

Implementation note: Internally, the system should work with typed Python objects, not unvalidated dictionaries.

## 9. Repair Invalid Structure Once

Decision: Invalid DAGs or malformed node outputs get one repair attempt.

Reasoning: One repair round handles common LLM formatting mistakes while keeping failures visible. More repair rounds can make traces hard to interpret and hide systematic prompt/schema problems.

Paper reference: RT-RAG uses iterative refinement and query rewriting to recover from failures. For this MVP, the analogous mechanism is format repair, not evidence repair.

## 10. Testing Pyramid

Decision: Most tests should be deterministic unit tests with fake LLM responses.

Reasoning: The core correctness risks are graph validation, dependency scheduling, substitution, output parsing, and trace recording. These can be tested without paying for model calls or depending on provider availability.

Test layers:

- Unit: schemas, validators, scheduler, parser, substitution, metrics.
- Integration: planner/executor/API with fake LLM.
- End-to-end: browser smoke test with fake LLM.
- External model tests: opt-in only.

## 11. Benchmark Metrics For The First Iteration

Decision: Use HotpotQA answer EM/F1, latency, node count, graph depth, LLM call count, structural failure rate, and repair rate.

Reasoning: Since retrieval is excluded, retrieval recall and supporting fact F1 would be misleading. Those metrics become relevant once the system has evidence retrieval.

## 12. Frontend Before Retrieval

Decision: Build the inspection frontend before adding retrieval.

Reasoning: A graph agent is difficult to debug from logs alone. The first demo should show the generated graph, parallel waves, calls, raw responses, parsed values, and final result. That creates a strong foundation for later retrieval experiments because every future evidence call can fit into the same trace model.

## 13. ShadCN Operational Interface

Decision: Build the frontend with ShadCN UI, React, TypeScript, and Tailwind.

Reasoning: The UI needs to feel like an operational analysis tool, not a toy demo or marketing page. ShadCN gives high-quality accessible primitives while still allowing a custom look. The target visual language is clean defense/operations software: compact panels, dark neutral surfaces, precise typography, status badges, trace tables, and a graph-first inspection layout.

Implementation note: Use ShadCN components such as `ResizablePanelGroup`, `Tabs`, `Table`, `Badge`, `Sheet`, `Dialog`, `Select`, `Input`, `Switch`, and `Slider`.

## 14. Programmatic Benchmark Surface

Decision: Benchmarking must be available through CLI, HTTP API, and Python API.

Reasoning: A benchmark that only runs from the frontend is not enough for research. The system should be callable from scripts, CI, notebooks, and later evaluation pipelines. Every run should return stable JSON with metrics, run IDs, traces, graph metadata, latency, and structural failure diagnostics.

Implementation note: The minimum API surface is `/api/ask`, `/api/plan`, `/api/execute`, `/api/benchmarks/hotpotqa`, `/api/runs/{run_id}`, and `/api/runs/{run_id}/trace`.

## 15. Config-Driven System Behavior

Decision: Runtime behavior is controlled by config files with optional safe UI/API overrides.

Reasoning: Research iteration requires changing retry counts, timeouts, planner depth, concurrency, model selection, and benchmark limits without touching source code. Config files also make benchmark runs reproducible.

Implementation note: Store the resolved config with each run artifact so benchmark results can be traced back to the exact settings used.

## 16. Nodes Declare Task Type, Prompt Contract, Input Mapping, And Return Format

Decision: Every DAG node declares what kind of question it asks, how it builds its prompt, what child outputs it consumes, how to handle missing child values, and what schema it returns.

Reasoning: A plain node question plus dependencies is underspecified. The executor needs to know whether a node is doing fact lookup, comparison, transformation, calculation, classification, or synthesis. It also needs explicit rules for mapping child outputs into prompt variables and validating the returned value.

Implementation note: The YAML node contract includes `task_type`, `operation`, `prompt`, `input_map`, `child_output_policy`, and `output_schema`. This makes nodes inspectable in the UI and testable without depending on natural language assumptions.
