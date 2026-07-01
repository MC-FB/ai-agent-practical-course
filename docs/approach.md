# DAG Multi-Hop QA: Approach

## Overview

This system answers complex multi-hop questions by decomposing them into
sub-questions and answering them against provided evidence documents. It
combines three techniques from recent research to outperform a single-prompt
baseline on the MuSiQue benchmark.

## Architecture

```
Question ──► Planner ──► Sub-questions ──┬──► Least-to-Most path ──┐
                                         │                         ├──► Best answer
                                         └──► DAG node-by-node ───┘
```

### 1. Planning: Question Decomposition

**File:** `dagqa/planning/planner.py`

The planner asks the LLM to decompose a complex question into simpler
sub-questions. In "simple" mode (default), the LLM returns a JSON array:

```json
{"steps": [
  {"id": "q1", "question": "Where are Spielberg's grandparents from?", "depends_on": []},
  {"id": "q2", "question": "Who visited {q1.answer} on November 22?", "depends_on": ["q1"]}
]}
```

The system then deterministically enriches this into a full `DagPlan` with
inferred task types, input maps, output schemas, and prompt templates
(`_simple_steps_to_plan`). This keeps the LLM's job simple (just decompose)
while the system handles the structural boilerplate.

**Config:** `planner_mode: "simple"` (default) or `"structured"` (legacy full-DAG mode)

### 2. Execution: Least-to-Most + Self-Consistency

**File:** `dagqa/graph/executor.py`

When evidence documents are available, the executor runs two paths in parallel:

#### Path A: Least-to-Most (Zhou et al., 2022)

Compiles all sub-questions and all evidence into a single LLM prompt:

```
Question: <original question>

Solve step by step:
  Step 1: <sub-question 1>
  Step 2: <sub-question 2>
  Final step: Answer the original question.

<all evidence documents>

Return JSON: {"steps": "...", "answer": "short final answer"}
```

This gives the LLM full visibility over all evidence while providing structured
thinking via the decomposition. It avoids error propagation between isolated
LLM calls — the main failure mode of node-by-node execution.

#### Path B: Node-by-node DAG execution

Standard wave-based execution where each node runs independently with its own
evidence and dependency values. Each intermediate node uses a chain-of-answers
prompt format showing the original question and answers so far.

#### Selection: Evidence grounding (Wang et al., 2022)

The system picks the answer with stronger evidence grounding:

- Answers that appear as exact substrings in the evidence score highest
- 2-6 word answers (specific entities, dates) are preferred over single-word
  matches (too vague) or very long responses (likely sentences, not extractions)
- Verbose answers (>15 words) are penalized

This is a lightweight form of Self-Consistency: two diverse reasoning paths
produce candidate answers, and a grounding heuristic selects the better one.

### 3. Simplified Inter-Node Contract

**File:** `dagqa/planning/normalizer.py`

Intermediate nodes use a minimal 2-field contract:

```json
{"answer": "Ukraine", "reasoning": "Evidence states grandparents emigrated from Ukraine."}
```

The old system required 5+ fields (`bridge_answer`, `bridge_reasoning`,
`bridge_source_span`, `constraint_status`, `_evidence_citations`), which caused:
- Planning failures from over-strict validation
- Prompt bloat burying the actual question
- LLM errors from complex output schemas

Final nodes still include `answer_type` and `answer_source_span` for answer
quality, and `_evidence_citations` for traceability.

## Key Files

| File | Purpose |
|---|---|
| `dagqa/client.py` | Entry point; routes to LtM+SC or standard execution |
| `dagqa/planning/planner.py` | Simple planner (sub-questions) + structured fallback |
| `dagqa/graph/executor.py` | LtM path, DAG path, self-consistency selection |
| `dagqa/nodes/prompts.py` | Chain-of-answers format, evidence rendering |
| `dagqa/nodes/runner.py` | Individual node execution, citation validation (final only) |
| `dagqa/planning/normalizer.py` | Ensures answer/reasoning on intermediates |
| `dagqa/planning/validator.py` | Structural plan validation (cycles, depth, refs) |
| `dagqa/config.py` | `PlannerConfig.planner_mode` setting |

## Benchmark Results

MuSiQue `validation_3hop_plus`, 50 items, seed=42, Mistral-Medium-3.5-128B:

| System | Cosine | EM | F1 |
|---|---|---|---|
| **DAG (this system)** | **0.820–0.834** | **64–66%** | **0.724–0.729** |
| Single-prompt baseline | 0.771 | 58% | 0.630 |

## Research References

- **Least-to-Most Prompting** (Zhou et al., 2022): Decompose then solve sub-problems sequentially in context
- **Self-Consistency** (Wang et al., 2022): Sample multiple reasoning paths, select most consistent
- **Plan-and-Solve** (Wang et al., 2023): Use decomposition as thinking scaffold in single pass
- **Self-Ask** (Press et al., 2022): Follow-up questions as minimal inter-step interface
- **IRCoT** (Trivedi et al., 2022): Interleave retrieval with chain-of-thought
- **DecomP** (Khot et al., 2022): Decomposed prompting for multi-step reasoning
