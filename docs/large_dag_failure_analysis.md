# Large Benchmark DAG Failure Analysis

Analyzed DAG run: `runs/benchmarks/hotpotqa-20260605T135513Z-aa346249-5eab-4491-8773-8f5e8a8feb8d.json`
Compared direct run: `runs/benchmarks/hotpotqa-20260605T135513Z-25ddcfd6-6416-42a3-9837-50f9a7aaa7f2.json`

## Summary

- Examples: 1000
- DAG exact match: 0.407; average F1: 0.523; median F1: 0.667
- DAG exact-match failures classified: 593 (59.3%)
- Structural failures: 168 (16.8%)
- Record-level errors: 106 (10.6%)
- Failed node traces: 154
- Average latency: 49.5s; p50 latency: 37.8s

## Ranked Failure Modes

| Rank | Failure mode | Severity | Cases | Rate of all examples | Rate of DAG EM failures |
|---:|---|---|---:|---:|---:|
| 1 | Provider / execution rate limiting | Critical | 57 | 5.7% | 9.6% |
| 2 | Planner schema / dependency contract mismatch | Critical | 5 | 0.5% | 0.8% |
| 3 | Information gathering / evidence selection failure | High | 307 | 30.7% | 51.8% |
| 4 | Reasoning or comparison over gathered facts failed | Medium | 179 | 17.9% | 30.2% |
| 5 | Answer normalization / alias mismatch | Low | 45 | 4.5% | 7.6% |

## Failure Details

### Provider / execution rate limiting

- Severity: Critical
- Frequency: 57/1000 examples (5.7%); 9.6% of DAG exact-match failures
- What failed: The LLM call failed with a rate-limit error, usually causing downstream nodes to miss inputs.
- Improvement target: Add provider-side backoff, concurrency throttling, retry budgets, and resumable per-node execution.
- Sample cases: 5ae7a2e255429952e35ea99e: predicted `` vs gold `Riot Act`; 5ab322b1554299194fa93570: predicted `` vs gold `fictional character`; 5abaf0e055429966062416a3: predicted `` vs gold `early Romantic period`

### Planner schema / dependency contract mismatch

- Severity: Critical
- Frequency: 5/1000 examples (0.5%); 0.8% of DAG exact-match failures
- What failed: A node referenced a field such as q1.album or q2.county that was absent from the upstream output.
- Improvement target: Validate every input_map path against upstream schemas before execution; repair plans or align schemas automatically.
- Sample cases: 5ae1b53b5542997283cd2255: predicted `` vs gold `Geographical Indication tag`; 5a8398405542993344746050: predicted `` vs gold `Limerick`; 5a8babf05542996e8ac8899e: predicted `` vs gold `Daviesia`

### Information gathering / evidence selection failure

- Severity: High
- Frequency: 307/1000 examples (30.7%); 51.8% of DAG exact-match failures
- What failed: The DAG completed, but cited too little gold evidence or used wrong supporting evidence before producing a wrong answer.
- Improvement target: Improve retrieval/evidence atomization and force each lookup node to ground claims in the relevant supporting facts.
- Sample cases: 5a79eef75542996c55b2dca6: predicted `yes` vs gold `Daniel Craig`; 5a7577785542992d0ec05fa0: predicted `yes` vs gold `Tatton Park`; 5ae2b83c5542992decbdcd7f: predicted `Weird Tales is a pulp‑fiction magazine and typically has 128 pages.` vs gold `128 pages`

### Reasoning or comparison over gathered facts failed

- Severity: Medium
- Frequency: 179/1000 examples (17.9%); 30.2% of DAG exact-match failures
- What failed: The DAG appears to have enough evidence, but the final answer is still wrong.
- Improvement target: Add stricter final-node verification, typed intermediate outputs, and answer consistency checks against cited facts.
- Sample cases: 5ae4563e5542996836b02c7f: predicted `Spencer Gifts, which has 600 stores, bought Spirit Halloween in 1999.` vs gold `over 600 stores`; 5a82bdd655429940e5e1a92a: predicted `Phred on Your Head Show, 2002` vs gold `2002`; 5a8e2ba05542995085b373b8: predicted `No, they are in different languages (Tristan und Isolde: German; Ariane et Barbe-bleue: French).` vs gold `no`

### Answer normalization / alias mismatch

- Severity: Low
- Frequency: 45/1000 examples (4.5%); 7.6% of DAG exact-match failures
- What failed: Exact match failed even though token overlap or semantic similarity is high.
- Improvement target: Normalize aliases, dates, articles, punctuation, and common entity variants before scoring or final output.
- Sample cases: 5abe54585542991f66106148: predicted `James A. Garfield` vs gold `James Abram Garfield`; 5a7fb7985542994857a767d2: predicted `Claudio López` vs gold `Claudio Javier López`; 5a8bbfb15542995e66a474f6: predicted `tennis player` vs gold `former tennis player`

## Component Signal Counts

These are non-exclusive signals. A single example can appear in several rows, for example when a rate-limited lookup causes downstream missing dependency values.

| Component signal | Severity | Wrong examples | All examples | Interpretation |
|---|---|---:|---:|---|
| Provider / execution rate limiting | Critical | 57 (5.7%) | 57 (5.7%) | The LLM call failed with a rate-limit error, usually causing downstream nodes to miss inputs. |
| Planner schema / dependency contract mismatch | Critical | 52 (5.2%) | 52 (5.2%) | A node referenced a field such as q1.album or q2.county that was absent from the upstream output. |
| Final combination does not consume gathered facts | High | 62 (6.2%) | 64 (6.4%) | The final node had no dependency values or did not use child values in its prompt template. |
| Information gathering / evidence selection failure | High | 327 (32.7%) | 478 (47.8%) | The DAG completed, but cited too little gold evidence or used wrong supporting evidence before producing a wrong answer. |
| Reasoning or comparison over gathered facts failed | Medium | 204 (20.4%) | 458 (45.8%) | The DAG appears to have enough evidence, but the final answer is still wrong. |
| Answer normalization / alias mismatch | Low | 45 (4.5%) | 45 (4.5%) | Exact match failed even though token overlap or semantic similarity is high. |

## Node-Level Signals

Most common node errors:

- 72: `Too many requests, please try again later.`
- 6: `<empty error>`
- 3: `'q2.county'`
- 2: `'q1.album'`
- 2: `'n1.composer_name'`
- 2: `'q1.city'`
- 2: `'q1.film_title'`
- 2: `'q1.title_organization'`
- 2: `'q1.contest_name'`
- 2: `'q1.opponent_name'`

Failed node task types:

- fact_lookup: 81
- synthesis: 55
- comparison: 18

Most common missing dependency fields:

- q2.county: 3
- q1.album: 2
- n1.composer_name: 2
- q1.city: 2
- q1.film_title: 2
- q1.title_organization: 2
- q1.contest_name: 2
- q1.opponent_name: 2
- q1.commander_name: 2
- q1.show: 2
- q1.actress_name: 2
- q2.title: 1
- q3.victims_killed: 1
- q3.common_structure: 1
- q1.type: 1

## Direct LLM Comparison

- Matched direct records: 1000
- Both correct: 169
- DAG only correct: 238
- Direct only correct: 102
- Both wrong: 491

The DAG improves accuracy over the direct baseline on this run, but it pays for that with many more LLM calls and a larger structural/execution failure surface.

## AtomicRAG Assessment

AtomicRAG-style decomposition looks useful for this system if it is applied to evidence and intermediate facts, not as a replacement for DAG validation. The paper proposes Atom-Entity Graphs that store self-contained factual atoms instead of coarse text chunks and use graph traversal/filtering to improve retrieval accuracy and reasoning robustness: https://arxiv.org/abs/2604.20844

That maps well to the largest non-provider failure bucket: information gathering and evidence selection. Smaller atomic claims could help lookup nodes isolate the exact supporting facts before synthesis and reduce wrong or incomplete citations.

It would not directly solve the critical planner contract failures: missing `input_map` fields, disconnected final prompts, and rate-limit cascades need schema validation, prompt-template validation, and scheduler/provider hardening first. The recommended order is: fix execution retries and dependency validation, then add atomic evidence extraction for lookup nodes, then add final-node consistency checks over those atoms.

## Method

The analyzer uses deterministic trace signals: node errors, validation errors, structural issues, exact-match/F1/cosine metrics, citation recall, and wrong-supporting-text rate. No LLM judging was required for the ranked counts; the residual ambiguous group is the only bucket that merits sampled LLM or manual review.
