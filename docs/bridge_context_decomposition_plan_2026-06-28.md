# Bridge Context Decomposition Plan

Date: 2026-06-28

## Trigger Example

Marked MuSiQue row:

- ID: `3hop1__31995_24918_24939`
- Question: "The leader visiting where Steven Spielberg's grandparents are from met with whom on November 22?"
- Gold: `Pope John Paul II`
- DAG prediction: `Amy Irving`

The graph decomposed the question too finely and lost the bridge role:

1. q1 asked where Spielberg's grandparents are from and returned both `Cincinnati` and `Ukraine`.
2. q2 asked "Which leader visited Cincinnati, Ukraine?", which is not a valid subquestion.
3. q2 drifted to Steven Spielberg, then q3 answered a Spielberg biographical fact.

The MuSiQue decomposition for this row makes the intended bridge clearer:

1. Where were Steven Spielberg's grandparents from? -> `Ukraine`
2. Who visited Ukraine while the protests were taking place? -> `Gorbachev`
3. Who met with Gorbachev on November 22? -> `Pope John Paul II`

We should not use gold decompositions at runtime, but this failure is useful for diagnosing the
class of error: dependent nodes are being generated as under-specified local questions instead of
context-complete bridge questions.

## New Marked Subset

Created a path-backed MuSiQue subset for fast iteration:

- File: `data/musique/marked_failures_2026_06_28.json`
- Subset ID: `marked_failures_2026_06_28`
- Label: `Marked MuSiQue failures, 2026-06-28`
- Size: 10 examples

The subset is selectable in the Benchmark tab under MuSiQue. The metadata endpoint reports
`total_examples = 10` and `default_limit = 10`, so a paired quick test can run the whole subset by
default.

## Research Basis

- IRCoT, Trivedi et al. 2022: one-shot retrieve-and-read is insufficient for multi-step QA because
  what to retrieve depends on what has already been derived. Their interleaved retrieval/reasoning
  approach improved retrieval and QA on HotpotQA, 2WikiMultiHopQA, MuSiQue, and IIRC.
  https://arxiv.org/abs/2212.10509
- Decomposed Prompting, Khot et al. 2022: decomposition helps only when subtasks are cleanly
  delegated to specialized prompts. This supports fixing the planner/node contract rather than
  adding another final verifier. https://arxiv.org/abs/2210.02406
- GenDec, Wu et al. 2024: robust decomposition should generate independent and complete
  subquestions using extracted evidence. This directly matches our failure: q2 was not independent
  or complete because "which leader visited Ukraine" omitted the protest/event constraint.
  https://arxiv.org/abs/2402.11166
- GenSco, Fazili et al. 2024: passage alignment based on predicted decompositions improves
  multi-hop QA, including MuSiQue. This suggests each bridge node should carry an aligned evidence
  target forward, not only a scalar answer. https://arxiv.org/abs/2407.10245
- QDAMR, Deng et al. 2022: AMR-style decomposition preserves semantic roles for interpretable
  multi-hop QA. We do not need to add an AMR parser, but the design lesson is relevant: preserve
  roles such as source place, visited place, visiting leader, event/time, and meeting participant.
  https://arxiv.org/abs/2206.08486
- Tang et al. 2020: multi-hop systems often fail the underlying single-hop subquestions even when
  final answers look plausible. This supports inspecting and validating intermediate bridge nodes
  as first-class outputs. https://arxiv.org/abs/2002.09919

## Proposed Implementation

### 1. Add Lightweight Bridge Question Rewriting

Use question rewriting rather than a large new output schema. The planner should write dependent
node questions as context-complete rewrites of the original question with the previous
`bridge_answer` inserted.

Example:

- Original question: "The leader visiting where Steven Spielberg's grandparents are from met with
  whom on November 22?"
- After q1 returns `Ukraine`: "The leader visiting Ukraine met with whom on November 22?"

This preserves the bigger answer target while grounding the bridge value. It should prevent vague
local questions like:

- "Which leader visited Ukraine?"
- "Which leader visited Cincinnati, Ukraine?"

because those questions drop either the final target (`met with whom on November 22`) or the correct
bridge surface (`Ukraine`).

Implementation details:

- Planner prompt: every dependent lookup node must be phrased as a rewritten version of the
  original question with resolved child placeholders inserted. The node may narrow the answer target
  to the next missing bridge value, but it must keep unresolved constraints from the original
  question.
- Runtime prompt: dependent nodes get an automatic "Original question / rewritten question /
  resolved bridge answers" block before the local prompt.
- Trace: store the rendered rewritten question so graph inspection shows whether the bridge
  context was preserved.

### 2. Enforce Single Bridge Answer for Fact Lookup Nodes

Non-final fact lookup nodes should return one most likely bridge value in the context of the
original question. They should not expose multiple peer answer fields unless the original question
explicitly asks for a list or comparison.

For the trigger row, q1 should return:

- `place`: `Ukraine`
- `bridge_answer`: `Ukraine`

It should not return both:

- `city`: `Cincinnati`
- `country`: `Ukraine`

because `Cincinnati` is part of the evidence sentence but not the requested bridge target. If the
LLM sees multiple possible values, it should use `bridge_reasoning` to justify the selected one and
leave incidental values out of the public output.

### 3. Add an Automatic Plan Audit Before Execution

After parsing the planner output and normalizing dependencies, run a deterministic audit over every
dependent node:

- Reject or repair non-final fact lookup nodes that have multiple non-contract scalar answer
  fields, unless the question explicitly asks for multiple values.
- Reject or repair dependent nodes that use multiple fields from the same child in the natural
  language question. This catches `Cincinnati, Ukraine`.
- Require dependent nodes to use `{qN.bridge_answer}` or an input map value derived from
  `qN.bridge_answer` when continuing a bridge chain.
- Require the dependent node question or prompt to keep unresolved original constraints, such as
  `November 22`, `met with whom`, `population`, `capital`, or `largest state`.

If the audit fails, run a planner repair prompt that includes:

- original question,
- current invalid node,
- dependency output fields,
- concrete audit errors,
- instruction to rewrite the node as an independent, context-complete subquestion.

This is a scheduler/planner fix, not a final-answer patch.

### 4. Enrich Child Prompts With the Bridge Chain

For each node, render a short "bridge chain so far" before the local question:

- original question,
- rewritten current question,
- selected dependency `bridge_answer` values,
- dependency `bridge_source_span` and `constraint_status`.

This is narrower than the failed broad candidate-ledger experiment. It should not add new required
LLM output fields. The goal is to make the existing `bridge_answer` and rewritten question hard to
ignore.

### 5. Evidence Selection Should Use Rewritten Question Terms

When ranking or selecting evidence for a bridge node, combine:

- selected dependency value,
- rewritten node question,
- unresolved constraints from the original question,
- entity/title terms from the current node question.

For the trigger row, retrieval/evidence ordering should boost passages containing `Ukraine`,
`visit`, `Gorbachev`, `protests`, and later `November 22`, rather than passages about Spielberg.

## Test Plan

1. Baseline on the new subset:
   - Run paired Mistral DAG and direct prompt on `marked_failures_2026_06_28`.
   - Save the result so it is visible in Results.
2. Implement bridge context contract and plan audit.
3. Re-run paired Mistral on the same subset.
4. Acceptance:
   - Primary: DAG cosine improves on the 10 marked examples.
   - Manual: the trigger row q2 asks about a leader visiting Ukraine in the protest context and
     returns `Gorbachev`, not Spielberg.
   - No structural failure increase.
5. If the subset improves, run a 30-item MuSiQue `validation_3hop_plus` paired benchmark with a new
   seed to check for overfitting.

## Expected Risk

The previous candidate-ledger experiment failed because it added too much schema/output burden to
every bridge node. This proposal should stay smaller: no large candidate tables, no many-field
relation object, and no final verifier. It relies on better question rewriting, single selected
bridge answers, and deterministic plan audit/repair before node execution.
