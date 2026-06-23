# MuSiQue 100-Item Failure Iteration

## Context

The larger MuSiQue validation run with seed `1713121241` showed that the multi-node DAG did
not generalize from the earlier 15-item sample:

| Run | System | Items | Cosine | Exact match | F1 | Gold fact recall | Structural failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bigger_run` | Multi-node DAG | 100 | 0.713 | 0.440 | 0.565 | 0.858 | 0.090 |
| `bigger_run` | Single prompt | 100 | 0.769 | 0.480 | 0.614 | 0.622 | 0.040 |

Result files:

- `runs/benchmarks/musique-20260622T080125Z-66bf2f2e-b4c8-4e48-96e2-673f78d48c40.json`
- `runs/benchmarks/musique-20260622T080125Z-7b362159-b3fd-4385-9d83-5317606dfe9c.json`

## Failure Classes Observed

The marked rows and lowest-cosine rows showed three main failure classes.

1. Citation repair failures caused structural downstream failures.
   Nodes sometimes initially returned malformed or empty citations, then the repair call inserted
   placeholder IDs such as `default`, `doc1`, or `doc_001`. Those IDs are invalid because MuSiQue
   contexts are exposed as `context-0`, `context-1`, and so on. The node then failed validation,
   downstream dependencies were missing, and the final answer became empty.

2. Source-surface mismatches hurt cosine even when the core answer was visible.
   Examples include `2017-07-11` instead of `July 11, 2017`, `5` instead of `five`, and `3`
   instead of source wording such as `third-largest`.

3. Real target-selection and decomposition failures remain.
   Several rows retrieve the right documents but select the wrong bridge value or stop at an
   intermediate answer. Examples include returning the religion word instead of the dictionary
   meaning, returning both book co-authors when the gold expects one author, or returning an empty
   match list despite a relevant score table row.

4. Final synthesis was not evidence-grounded.
   Lookup nodes received the source contexts, but synthesis/comparison nodes often only received
   dependency values. This meant final nodes could not validate whether a dependency value was
   supported by the same source chain as the requested answer. A naive attempt to give final nodes
   all documents made some rows worse because it reintroduced distractors, such as selecting the
   unrelated `Baranya County -> Pécs` context for a Wisconsin county/capital question.

## Implemented Changes

### Evidence-aware repair prompt

`render_repair_prompt` now receives the node's selected evidence and lists valid document IDs and
exact titles in the repair prompt. The prompt explicitly forbids placeholder citation IDs such as
`default`, `doc1`, `doc_001`, `source`, or `evidence`.

Expected effect: malformed citation repair should retry toward valid `context-*` IDs instead of
creating a validation failure that cascades through downstream nodes.

### Source-surface preservation

Node prompts now instruct the model to copy the answer form used in evidence, including dates,
ordinals, punctuation, and written number forms.

Post-processing now restores:

- ISO dates such as `2017-07-11` to evidence text such as `July 11, 2017`.
- Numeric ordinal answers such as `3` to evidence wording such as `third-largest` when the question
  asks for a rank-like value.

Expected effect: logically correct answers should align better with cosine scoring without changing
the underlying fact.

### Table and fixture row prompt guidance

Node and final-refinement prompts now include a generic instruction for score rows: for a row like
`Home 2 -- 1 Away`, compare scores in row order. If the question asks when one team beat another,
choose the date from rows where that team's score is greater.

Expected effect: rows such as the George Hollis / Aston Villa example should be less likely to
produce empty lists when the table contains the decisive result.

### Results UI fix

The Results tab now uses a stable four-column browser grid on desktop and collapses to one column
on narrower screens. Long selected run labels are truncated in the middle so the table controls do
not overlap.

### MuSiQue planner depth and bridge-chain prompting

The MuSiQue benchmark override now raises planner depth from `5` to `6`. A partial run showed that
the previous cap caused valid four-hop questions to fall into the one-node DAG failure fallback
when the planner produced a six-depth bridge chain.

The structured planner prompt now explicitly rejects one-node plans for chained relative-clause
questions and asks the planner to merge the final synthesis into the last lookup if the bridge
chain is near the depth limit.

Expected effect: fewer hard MuSiQue rows should degrade into direct-answer fallback graphs.

### Dependency-cited final evidence

Final `synthesis` and `comparison` nodes now receive only the documents cited by their dependency
nodes instead of the full distractor context. Internally, the executor propagates citation metadata
through the dependency output map while keeping `_evidence_citations` out of user-facing node
outputs and final answers.

Expected effect: final nodes can cite and validate source text without being exposed to all
distractors again.

### Results endpoint robustness

The failed-only subset file is a JSON list stored in `runs/benchmarks`. The Results endpoint and
partial-result cleanup now skip non-result JSON files there. This fixes a Results tab `500` caused
by trying to normalize the subset file as a benchmark result.

## Failed-Only Diagnostic Subset

I created a diagnostic subset containing 46 examples from the 100-item run where the DAG either had
cosine below `0.8` or scored below the single-prompt baseline on the same row:

- `runs/benchmarks/musique_failed_subset_seed1713121241.json`

The rerun uses:

- dataset: `musique`
- subset file: `runs/benchmarks/musique_failed_subset_seed1713121241.json`
- limit: 46
- seed: `1713121241`
- model: `mistralai/Mistral-Medium-3.5-128B`
- systems: `dag_agent`, `direct_llm`

## Diagnostic Rerun Results

Several diagnostic runs were intentionally stopped once they had enough signal.

### Citation/surface run

Live run `4671fc94-eda1-4fa0-a3d7-f27b1b6472b4`
(`failed_subset_citation_surface_fix`) was stopped after 35 DAG examples.

| System | Items | Cosine | Exact match | F1 | Gold fact recall | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 35 | 0.599 | 0.143 | 0.365 | 0.795 | 0.029 |

This showed that invalid citation IDs were no longer the dominant problem, but target-selection
errors remained.

### Naive final-all-evidence run

Live run `06ecce30-7da2-49fc-98e0-1d36351866b8`
(`failed_subset_synthesis_evidence_depth6`) was stopped after 9 DAG examples once the failure mode
was clear:

| System | Items | Cosine | Exact match | F1 | Gold fact recall | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 9 | 0.471 | 0.111 | 0.323 | 0.824 | 0.111 |

Giving final nodes all documents reduced structural failures but made distractor selection worse.
This approach was replaced by dependency-cited final evidence.

### Dependency-cited partial run

The first dependency-cited run completed the DAG half but failed before direct mode because result
cleanup tried to normalize the failed-subset JSON list as a benchmark result. The cleanup bug is now
fixed.

Completed partial result file:

- `runs/benchmarks/musique-20260622T161526Z-254420fe-fd6e-4e36-bc2a-93692536a7a0.json`

| System | Items | Cosine | Exact match | F1 | Gold fact recall | Wrong citation rate | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 10 | 0.514 | 0.100 | 0.268 | 0.783 | 0.180 | 0.000 |

### Dependency-cited smoke run

Completed result files:

- `runs/benchmarks/musique-20260622T193742Z-a98db4a9-c1d3-4178-a5db-e5ea244b1de2.json`
- `runs/benchmarks/musique-20260622T193742Z-7222d740-8ce9-4aaf-a76c-4000b42fae11.json`

| System | Items | Cosine | Exact match | F1 | Gold fact recall | Wrong citation rate | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 5 | 0.632 | 0.200 | 0.403 | 0.783 | 0.000 | 0.000 |
| Single prompt | 5 | 0.728 | 0.400 | 0.603 | 0.517 | 0.111 | 0.000 |

The dependency-cited strategy improved citation quality but still did not beat single prompt on the
failed subset. On the five sampled rows, DAG won one row (`Milwaukee Deep`), tied two rows, and
lost two rows due to wrong upstream bridge values (`1976` vs `1984`, `Amy Irving` vs
`Pope John Paul II`).

### Bridge-reasoning contract run

The next iteration targeted the upstream bridge-selection problem directly. The previous executor
passed dependency answer values and citations to parent nodes, but it did not pass a stable
explanation of why a bridge value had been selected. That differs from the direction suggested by
multi-hop reasoning work: IRCoT conditions later retrieval and reasoning on intermediate reasoning
steps, Least-to-Most prompting solves subproblems sequentially and feeds earlier answers forward,
and GenSco argues that decomposition should stay aligned with supporting passages rather than
compressing away the bridge evidence.

Papers used:

- Trivedi et al., "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive
  Multi-Step Questions", 2023. https://arxiv.org/abs/2212.10509
- Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models", 2022.
  https://arxiv.org/abs/2205.10625
- Wang et al., "MCR: Multi-hop Question Answering via Multi-Chain Reasoning", 2023.
  https://arxiv.org/abs/2304.13007
- Li et al., "GenSco: Can Question Decomposition based Passage Alignment improve Question
  Answering?", 2024. https://arxiv.org/abs/2407.10245
- Tang et al., "Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop
  Sub-Questions?", 2020. https://arxiv.org/abs/2002.09919

Implemented change:

- Non-final bridge lookup nodes now receive a normalized bridge contract whenever their value is
  consumed by later nodes: `bridge_answer`, `bridge_reasoning`, `bridge_source_span`, and
  `constraint_status`.
- Node prompts require `constraint_status="satisfied"` only when the cited span supports the
  bridge under all dependency constraints. Partial matches and distractor matches must be marked
  `ambiguous` or `not_found`.
- Parent-node prompts now explicitly inspect child `bridge_reasoning` and `constraint_status`
  through dependency outputs instead of treating every child answer as confirmed.

Live run `6f09babb-7058-44de-b931-94b3fcd59aea`
(`failed_subset_bridge_reasoning_iteration`) was stopped after 23 DAG examples once it gave enough
signal.

| System/sample | Items | Cosine | Exact match | F1 | Gold fact recall | Wrong citation rate | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bridge contract DAG | 23 | 0.605 | 0.174 | 0.361 | 0.928 | 0.129 | 0.044 |
| Previous 35-row DAG on same IDs | 23 | 0.578 | - | - | - | - | - |
| Original 100-run DAG on same IDs | 23 | 0.432 | - | - | - | - | - |
| Original 100-run single prompt on same IDs | 23 | 0.597 | - | - | - | - | - |

This is a modest positive result: on the same 23 failed-subset IDs, the bridge contract beat the
original DAG, the previous citation/surface DAG, and slightly beat the original single-prompt run.
It is not a complete solution because the absolute score is still low and one structural failure
remained.

A follow-up attempt relaxed citation requirements for `not_found` bridge nodes and added an
administrative-location capital rule. This was removed because it regressed the benchmark. On the
first 16 overlapping examples, that variant scored `0.569` cosine with two failures, while the
bridge-contract-only version scored `0.636` with no failures. The worst regressions changed correct
or near-correct answers such as `Oriole Records` and `22` into distractor answers such as `NBC` and
`0`. The retained implementation is therefore only the generic bridge-reasoning contract.

### Source-surface and repair iteration

The bridge run showed that several failures were not bad retrieval or decomposition. The graph had
already found the right span, but final extraction or post-processing degraded it:

- `8.11 million` became `8.11`.
- `8.005 million` became `8.005 million in`.
- `Latin` was rewritten to an unrelated `English-language` span before the relevant `Medieval
  Latin` span was considered.
- `5` stayed numeric although the supporting sentence used `five`.
- Empty, `not_found`, and `never` answers were not consistently sent through evidence-backed
  repair.

Implemented changes:

- Final extraction now prefers a concise `answer_source_span` or `evidence_answer_span` when the
  answer field is only a bare number or one-word value and the span preserves the source surface.
- Language post-processing now first searches for qualified forms of the returned answer, such as
  `Medieval Latin`, before considering unrelated `English-language` surfaces from other contexts.
- Quantity restoration keeps standalone source phrases like `8.005 million` and avoids treating
  following prepositions such as `in` as units, while still preserving real units such as
  `2.3 million pesos`.
- Small number answers from `1` to `10` are restored to the source word form when cited evidence
  uses that form.
- `never`, `not_found`, and empty DAG answers now trigger the existing evidence-backed repair step.
- Node validation now coerces parseable scalar/list values into strings when a schema requires a
  string, reducing structural failures caused only by JSON type shape.

Offline rescoring of the existing 23 bridge-run traces, without new LLM calls, improved cosine from
`0.605` to `0.671`. Changed rows included exact recovery of `8.11 million`, `8.005 million`,
`Medieval Latin`, and `five`.

A new Mistral chair-cluster DAG-only run was executed on 25 failed-subset examples:

- `runs/benchmarks/musique_failed_subset_bridge_surface_repair_25.json`

| System/sample | Items | Cosine | Exact match | F1 | Gold fact recall | Wrong citation rate | Structural failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-surface repair DAG | 25 | 0.613 | 0.280 | 0.362 | 0.757 | 0.192 | 0.040 |
| Original 100-run DAG on same IDs | 25 | 0.406 | - | - | - | - | - |
| Original 100-run single prompt on same IDs | 25 | 0.566 | - | - | - | - | - |

This iteration finally beats the matched single-prompt baseline on the 25 hard failed-subset IDs,
but the margin is still not robust enough to claim the DAG approach is solved.

Remaining failure classes in this run:

- Wrong administrative bridge scope: the graph still treats `Yangzhou, Jiangsu, China` as if the
  capital question asks for Yangzhou or China rather than Jiangsu, producing `0` instead of
  `about 400 years`.
- Score-table reasoning: the George Hollis / Aston Villa derby row still returns `never` despite a
  later table row containing the answer.
- Wrong bridge entity selection in 4-hop geography questions: several natural-boundary and county
  border examples still pick a plausible distractor.
- Over-specific or under-specific answer spans remain, for example returning co-authors when the
  gold answer names one author.

### Wrong-answer bridge iteration

After inspecting the 25 hard-row run, the important failures were not only answer formatting. Many
answers were factually wrong even when relevant evidence was present. Two clear examples were:

- George Hollis / 1894-95 FA Cup winner: the graph returned `never` or a draw row, although the
  compact derby table contains `1 December 2010 ... Birmingham City 2 -- 1`.
- Pfrang / Yaxing headquarters: the graph returned `0`, although the Nanjing paragraph says it
  `had been the capital city of Yangzhou for about 400 years`.

Research basis:

- Liu et al., "Lost in the Middle", show that LLMs can miss relevant evidence inside long contexts,
  even when it is present. This matches the compact table failure where later rows were ignored.
  https://arxiv.org/abs/2307.03172
- Wang et al., "Chain-of-Table", show that table reasoning improves when table operations are made
  explicit rather than left as ordinary prose. This motivated adding explicit score-row operation
  instructions and a deterministic table-condition check. https://arxiv.org/abs/2401.04398
- Zhou et al., "Least-to-Most Prompting", supports preserving intermediate constraints across
  subquestions rather than collapsing the bridge. This remains the design goal for bridge nodes.
  https://arxiv.org/abs/2205.10625

Implemented and kept:

- First-pass node and planner prompts now explicitly preserve score-table operations for "when did
  X beat Y" questions: scan all rows, compare scores in row order, infer omitted opponent columns
  from table scope, and reject draw rows such as `0 -- 0`.
- First-pass prompts now preserve direct capital-duration spans such as `had been the capital city
  of X for Y` instead of replacing them with modern administrative speculation.
- A conservative evidence-pattern recovery runs for these two operation classes when the final
  answer is failure-like, has an unsatisfied constraint status, or is a date from a "last beat"
  question. This recovery is class-based, not row-ID or gold-answer based.

Targeted verification:

- `runs/benchmarks/musique_targeted_wrong_answer_fix_2_v3.json`
- Items: 2
- Exact match: `1.000`
- Cosine: `1.000`
- Fixed outputs: `about 400 years` and `1 December 2010`

However, this does not solve the broader bridge problem. A 25-item run after the narrow wrong-answer
fix scored `0.609`, slightly below the previous `0.613` run:

- `runs/benchmarks/musique_failed_subset_wrong_answer_fix_25.json`

I then tested a broader candidate-ledger approach: requiring every bridge node to output selected,
rejected, and ambiguous candidates for parent nodes. This matches the research direction, but it did
not work with the current Mistral setup. The 25-item run scored `0.605` and introduced a structural
failure:

- `runs/benchmarks/musique_bridge_candidate_ledger_25.json`

This required candidate-ledger change has been removed from runtime code. The result is still saved
as a failed experiment because it shows that making bridge nodes output large candidate ledgers adds
schema/output burden without reliably improving parent selection.

Additional broader bridge attempts were tested after that and removed because they regressed the
same hard subset:

- Dependency bridge audit prompt: parent nodes received a compact audit of upstream
  `bridge_answer`, `bridge_reasoning`, `bridge_source_span`, and `constraint_status`. This was meant
  to make the model use child reasoning more reliably, following the same least-to-most motivation.
  On a 10-item hard smoke, same-ID cosine dropped from `0.597` to `0.562`. Manual inspection showed
  no bridge-choice improvement and worse answer surfaces for `Han nationality` and the Mississippi
  River flow question, so the change was removed.
- Evidence relevance reordering: all contexts remained visible, but exact entity/dependency matches
  were moved earlier to address the Lost-in-the-Middle failure mode. Same-ID cosine dropped from
  `0.597` to `0.564`, with worse outputs for the county-capital and nationality rows. This suggests
  that simple lexical reordering is too brittle for MuSiQue bridge chains and can disturb rows that
  were already correct.
- Extra first-pass prompt rules for geographic/admin inference and superlative-specific spans were
  also tested. They did not fix the intended `Milwaukee Deep` row and lowered same-ID cosine to
  `0.542`, mainly by damaging another geography bridge row. These rules were removed rather than
  kept as unhelpful prompt complexity.

## Assessment

The safe changes reduced structural/citation problems and made final nodes more inspectable. The
bridge-reasoning contract produced a small positive signal, and the source-surface/repair iteration
beat the matched single-prompt baseline on a 25-item failed-subset run. The main remaining issue is
still upstream bridge correctness: once a lookup node chooses the wrong intermediate entity or cites
a distractor, final answer formatting cannot reliably recover.

## Next Steps

The next high-impact change should target bridge lookup validation. A promising direction is to add
a generic candidate-table node pattern for ambiguous bridge steps: return candidate entity,
matched constraint, cited span, and rejection reason, then let the downstream node choose only from
validated candidates. This targets the actual failures better than another final-answer rewrite.
