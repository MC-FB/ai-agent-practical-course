# Info Gathering Improvement Plan

## Context

The 1000-example DAG benchmark showed 593 exact-match failures. The largest
exclusive bucket was information gathering / evidence selection with 307 cases.

This bucket was assigned only when the DAG completed without node execution
failure or final-combination structural failure, but the trace evidence looked
weak:

- `gold_supporting_fact_recall < 0.67`
- `wrong_supporting_text_rate >= 0.35`
- `evidence_citation_count == 0`

Within those 307 cases:

- 288 had low gold-supporting-fact recall.
- 122 had a high wrong-supporting-text rate.
- 106 had zero evidence citations.
- Only 6 were obvious wh-question answers incorrectly returned as `yes` or `no`.

So the main issue is not answer normalization. The main issue is that lookup and
intermediate nodes do not reliably anchor returned fields to the actual
supporting facts.

## Investigation Findings

The large run used the `cluster` provider, which uses the structured planner path
in `dagqa/planning/planner.py`.

The YAML planner prompt already discourages broad candidate-list or exhaustive
enumeration nodes. The structured planner prompt used by the benchmark does not
carry the same strong instruction. That matches sampled failures where the DAG
created broad lookup nodes such as "list all James Bond actors" instead of
directly finding which cast member in the target film was a James Bond actor.

The node evidence prompt requires `_evidence_citations`, but it does not strongly
bind each returned output field to a citation. This allows outputs that cite
some inspected evidence while returning broad lists or inferred fields that are
not directly supported by the cited sentences.

## First Change

Make a prompt-only improvement first because it is cheap, reversible, and
directly targets the observed failure mode.

Planned changes:

- Add the anti-broad-enumeration guidance to the structured planner prompt.
- Tell the structured planner to prefer bridge-specific lookup nodes that return
  the final needed entity/fact instead of broad candidate lists.
- Strengthen the evidence prompt so each returned field must be directly
  supported by citations.
- Tell evidence-backed nodes not to fill broad lists from partial evidence.
- Tell evidence-backed nodes to preserve the requested answer type and avoid
  `yes` / `no` unless the question is yes/no.

## Why This Should Help

The biggest measured signal is low supporting-fact recall. Broad lookup nodes
increase the chance that the model cites distractor or generic facts and misses
the exact bridge sentence. More targeted lookup nodes should reduce the evidence
surface and improve recall.

Field-level citation grounding should reduce cases where a node returns a
plausible but unsupported value, or returns a broad array from one partial
sentence.

This will not solve rate limits, dependency contract failures, or disconnected
final prompts. Those need separate scheduler and planner validation work.

## Tests

Add focused unit tests for:

- The structured planner prompt includes the anti-broad-enumeration instruction.
- The evidence prompt requires direct support for each returned field.
- The evidence prompt warns against unsupported broad lists.
- The evidence prompt preserves answer type and rejects `yes` / `no` for
  non-yes/no questions.

After unit tests, run a 20-item HotpotQA E2E benchmark to check whether the
change runs end to end and inspect the resulting metrics.
