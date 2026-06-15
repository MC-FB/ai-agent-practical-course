# Failed Fact-Targeting Experiments, 2026-06-15

This note documents two attempted fixes for factual target selection on the 19 item HotpotQA mini fact-retrieval subset. Both experiments were removed from the runtime path after measurement because neither solved the target-selection failures reliably.

## Baseline Reference

Best clean non-verifier run available in `runs/benchmarks`:

- File: `runs/benchmarks/qwen-constraint-fix-retry-mini-fact-retrieval-20260612.jsonl`
- Examples: 19
- Success / failure / error: 19 / 0 / 0
- Exact match: 0.2632
- F1: 0.4278
- Cosine: 0.6119
- Structural valid rate: 1.0000

Earlier BM25 run for context:

- File: `runs/benchmarks/qwen-bm25-mini-fact-retrieval-20260611.jsonl`
- Examples: 19
- Success / failure / error: 16 / 3 / 0
- Exact match: 0.2105
- F1: 0.3773
- Cosine: 0.4753
- Structural valid rate: 0.8421

## Experiment 1: Post-Hoc Final Answer Verifier

Hypothesis: a final verifier could inspect the DAG answer and cited child evidence, then correct cases where the graph selected the wrong target entity.

Implementation summary:

- Added a verifier call after DAG execution.
- Rendered a separate verification prompt with the original question, final answer, child evidence, and cited snippets.
- Accepted verifier corrections only when confidence was high enough.
- Added verifier trace data and a UI/settings toggle.

Measured results:

- Main run file: `runs/benchmarks/qwen-verifier-v2-mini-fact-retrieval-20260612.jsonl`
- Examples: 19
- Success / failure / error: 16 / 3 / 3
- Exact match: 0.3158
- F1: 0.4080
- Cosine: 0.5310
- Structural valid rate: 1.0000
- Wrong supporting text rate: 0.3810
- Average gold supporting fact recall: 0.5365

Target-case behavior:

- Mark Pavelich: gold `Soviet Union`, predicted `Lake Placid, New York`
- Siwa Oasis / Amazigh: gold `Berber`, predicted `Siwi`
- FEMSA: gold `Fomento Economico Mexicano`, predicted `Coca-Cola FEMSA`

Additional failed-row rerun:

- File: `runs/benchmarks/qwen-verifier-v2-failed3-rerun-20260615.jsonl`
- Examples: 3
- Exact match: 0.0000
- F1: 0.3889
- Cosine: 0.5601
- The verifier did not recover exact answers on the rerun rows.

Conclusion: the verifier improved some aggregate metrics in one run, but it acted as a post-hoc answerer and did not fix the specific target-selection failures. It also added another LLM call and more runtime complexity without making the DAG state more faithful. The verifier has been removed from config, execution, schemas, benchmark accounting, tests, and the Settings UI.

## Experiment 2: Target-Role Planning and Evidence Boosting

Hypothesis: preserving the original answer role inside planner output and node prompts would prevent the graph from collapsing questions to generic fields like location, company, or language.

Implementation summary:

- Added planner instructions for `answer_role`.
- Added event-role, descriptor-matching, and alias/hierarchy prompt patterns.
- Added node trace fields for target role and candidate tables.
- Added descriptor and alias boosts to evidence ranking.

Measured results using the university chair model:

- Command output file: `runs/benchmarks/chair-qwen-target-role-mini-fact-retrieval-20260615.jsonl`
- Model config: `configs/chair-gpt-oss.yaml`, `Qwen/Qwen3.5-122B-A10B` via the cluster endpoint
- Examples: 19
- Success / failure / error: 13 / 6 / 0
- Exact match: 0.2105
- F1: 0.3613
- Cosine: 0.4645
- Structural valid rate: 0.6842
- Structural failure rate: 0.3158
- Wrong supporting text rate: 0.2222
- Average gold supporting fact recall: 0.6009
- Runtime: about 27 minutes

Target-case behavior:

- Mark Pavelich: gold `Soviet Union`, predicted `Lake Placid, New York`
- Siwa Oasis / Amazigh: gold `Berber`, predicted `Berber languages`
- FEMSA: gold `Fomento Economico Mexicano`, prediction empty because the final trace was structurally invalid

Conclusion: this experiment did not improve the score. It fell below both the clean non-verifier reference cosine of 0.6119 and the earlier BM25 cosine of 0.4753. It also increased structural failures. The target-role planner/schema/trace changes and evidence constraint boosts have been removed.

## Invalid OpenRouter Run

The file `runs/benchmarks/qwen-target-role-mini-fact-retrieval-20260615.jsonl` is not a valid experiment result. It used the old OpenRouter path and all 19 examples failed with quota errors. It should be excluded from conclusions.

## Current Code State

The runtime system is back on the non-verifier DAG path with BM25 evidence selection and the final-answer contract. The graph inspection UI changes are kept:

- Failed inspected nodes show an explicit failure reason panel.
- The 10 provided contexts are labeled as expected gold citations or distractors in the graph UI.
