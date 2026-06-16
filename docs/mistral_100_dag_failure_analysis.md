# Mistral 100-Item DAG Failure Analysis

Analyzed DAG run: `runs/benchmarks/hotpotqa-20260608T130936Z-621c9986-59d2-4c7d-933d-08e014d71406.json`
Compared direct run: `runs/benchmarks/hotpotqa-20260608T130936Z-5332b32e-ef64-4b4b-88b7-8521e259fda9.json`

## Summary

- Examples: 100
- DAG exact match: 0.490; average F1: 0.684; median F1: 0.899
- DAG exact-match failures classified: 51 (51.0%)
- Structural failures: 2 (2.0%)
- Record-level errors: 2 (2.0%)
- Failed node traces: 0
- Average latency: 21.1s; p50 latency: 20.2s

## Ranked Failure Modes

| Rank | Failure mode | Severity | Cases | Rate of all examples | Rate of DAG EM failures |
|---:|---|---|---:|---:|---:|
| 1 | Information gathering / evidence selection failure | High | 24 | 24.0% | 47.1% |
| 2 | Reasoning or comparison over gathered facts failed | Medium | 15 | 15.0% | 29.4% |
| 3 | Answer normalization / alias mismatch | Low | 12 | 12.0% | 23.5% |

## Failure Details

### Information gathering / evidence selection failure

- Severity: High
- Frequency: 24/100 examples (24.0%); 47.1% of DAG exact-match failures
- What failed: The DAG completed, but cited too little gold evidence or used wrong supporting evidence before producing a wrong answer.
- Improvement target: Improve retrieval/evidence atomization and force each lookup node to ground claims in the relevant supporting facts.
- Sample cases: 5a80799b5542992bc0c4a72e: predicted `XIX Corps` vs gold `Panzer`; 5ae762835542997b22f6a711: predicted `northwestern Mexico` vs gold `tip of the Baja California`; 5a776fc15542997042120a3a: predicted `Mercer Bears` vs gold `The Bears`

### Reasoning or comparison over gathered facts failed

- Severity: Medium
- Frequency: 15/100 examples (15.0%); 29.4% of DAG exact-match failures
- What failed: The DAG appears to have enough evidence, but the final answer is still wrong.
- Improvement target: Add stricter final-node verification, typed intermediate outputs, and answer consistency checks against cited facts.
- Sample cases: 5a7a9a2255429941d65f26eb: predicted `119` vs gold `119 minutes`; 5ae713af554299572ea546bb: predicted `Lake Placid, United States` vs gold `Soviet Union`; 5ae09b7055429906c02daae4: predicted `through YouTube` vs gold `YouTube`

### Answer normalization / alias mismatch

- Severity: Low
- Frequency: 12/100 examples (12.0%); 23.5% of DAG exact-match failures
- What failed: Exact match failed even though token overlap or semantic similarity is high.
- Improvement target: Normalize aliases, dates, articles, punctuation, and common entity variants before scoring or final output.
- Sample cases: 5a78e9d155429970f5fffdcc: predicted `Rob Parissi` vs gold `Robert "Rob" Parissi`; 5a7602d8554299109176e5fb: predicted `Dawn French` vs gold `Dawn Roma French`; 5ab959905542996be2020497: predicted `1964-1974` vs gold `1964 to 1974`

## Component Signal Counts

These are non-exclusive signals. A single example can appear in several rows, for example when a rate-limited lookup causes downstream missing dependency values.

| Component signal | Severity | Wrong examples | All examples | Interpretation |
|---|---|---:|---:|---|
| Provider / execution rate limiting | Critical | 0 (0.0%) | 0 (0.0%) | The LLM call failed with a rate-limit error, usually causing downstream nodes to miss inputs. |
| Planner schema / dependency contract mismatch | Critical | 0 (0.0%) | 0 (0.0%) | A node referenced a field such as q1.album or q2.county that was absent from the upstream output. |
| Final combination does not consume gathered facts | High | 0 (0.0%) | 0 (0.0%) | The final node had no dependency values or did not use child values in its prompt template. |
| Information gathering / evidence selection failure | High | 30 (30.0%) | 48 (48.0%) | The DAG completed, but cited too little gold evidence or used wrong supporting evidence before producing a wrong answer. |
| Reasoning or comparison over gathered facts failed | Medium | 21 (21.0%) | 52 (52.0%) | The DAG appears to have enough evidence, but the final answer is still wrong. |
| Answer normalization / alias mismatch | Low | 12 (12.0%) | 12 (12.0%) | Exact match failed even though token overlap or semantic similarity is high. |

## Node-Level Signals

Most common node errors:


Failed node task types:


Most common missing dependency fields:


## Direct LLM Comparison

- Matched direct records: 100
- Both correct: 41
- DAG only correct: 8
- Direct only correct: 10
- Both wrong: 41

The DAG trails the direct baseline slightly on exact match for this run: direct-only wins (10) exceed DAG-only wins (8). It also uses many more LLM calls and has a larger structural/execution surface, although structural failures were rare in this 100-item sample.

## AtomicRAG Assessment

AtomicRAG-style decomposition looks useful for this system if it is applied to evidence and intermediate facts, not as a replacement for DAG validation. The paper proposes Atom-Entity Graphs that store self-contained factual atoms instead of coarse text chunks and use graph traversal/filtering to improve retrieval accuracy and reasoning robustness: https://arxiv.org/abs/2604.20844

That maps well to the largest non-provider failure bucket: information gathering and evidence selection. Smaller atomic claims could help lookup nodes isolate the exact supporting facts before synthesis and reduce wrong or incomplete citations.

It would not directly solve the critical planner contract failures: missing `input_map` fields, disconnected final prompts, and rate-limit cascades need schema validation, prompt-template validation, and scheduler/provider hardening first. The recommended order is: fix execution retries and dependency validation, then add atomic evidence extraction for lookup nodes, then add final-node consistency checks over those atoms.

## Method

The analyzer uses deterministic trace signals: node errors, validation errors, structural issues, exact-match/F1/cosine metrics, citation recall, and wrong-supporting-text rate. No LLM judging was required for the ranked counts; the residual ambiguous group is the only bucket that merits sampled LLM or manual review.
