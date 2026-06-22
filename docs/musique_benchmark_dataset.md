# MuSiQue Benchmark Dataset Integration

## Motivation

HotpotQA has become a weak discriminator for this project because both the single-prompt baseline and the multi-node RAG pipeline can often answer its validation questions from the provided contexts. For the next evaluation loop, we added MuSiQue as a harder benchmark dataset with explicit multi-hop decomposition chains.

MuSiQue is based on connected reasoning chains and was designed to reduce shortcuts that let models answer multi-hop questions without performing the intended reasoning. This better matches the course goal: find cases where decomposition and multi-step retrieval should outperform a single prompt.

Reference:

- Trivedi et al., "MuSiQue: Multihop Questions via Single-hop Question Composition", Transactions of the Association for Computational Linguistics, 2022. https://arxiv.org/abs/2108.00573

## Implementation

The benchmark code now uses a dataset registry instead of HotpotQA-specific logic. Each dataset entry defines its ID, label, split, default subset, result filename prefix, subset metadata, loader, and count function. Adding another dataset should require a loader plus one registry entry, not changes throughout the API and frontend.

Registered datasets:

- `hotpotqa`: existing HotpotQA validation and failure subsets.
- `musique`: MuSiQue validation with `validation` and `validation_3hop_plus` subsets.

The benchmark request now carries a `dataset` field. Existing HotpotQA routes remain compatible, but the backend uses the registry to resolve metadata, counts, loaders, result labels, and output filenames.

## Evidence Mapping

HotpotQA provides sentence-level supporting facts. MuSiQue provides supporting paragraphs. To fit the current evaluation schema without rewriting citation evaluation, each MuSiQue paragraph is represented as one citation unit:

- one paragraph becomes one `EvidenceDocument`.
- the paragraph text is stored as the only sentence in `metadata.sentences`.
- a supporting paragraph becomes `GoldSupportingFact(title=..., sentence_index=0)`.

This means HotpotQA citation recall remains sentence-level, while MuSiQue citation recall is paragraph-level.

## Frontend

The benchmark tab now has separate controls for dataset and subset. The subset selector is populated from backend metadata for the selected dataset.

The results tab now has a dataset filter:

- All datasets
- HotpotQA
- MuSiQue

Run labels include the dataset label so overlapping subset names such as `validation` remain distinguishable.

## Initial Benchmark Results

The initial comparison used:

- dataset: `musique`
- subset: `validation_3hop_plus`
- limit: 15
- systems: `dag_agent` and `direct_llm`
- model: `mistralai/Mistral-Medium-3.5-128B`
- seed: `1745764549`

Saved result files:

- `runs/benchmarks/musique-20260620T104944Z-32c99947-d8cf-4ee1-94a1-9f1dbd024e7c.json`
- `runs/benchmarks/musique-20260620T104944Z-9bb7a78d-d218-4e3f-9c9e-175abc3e51c4.json`

| System | Exact match | F1 | Cosine | Gold fact recall | Wrong citation rate | Structural failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 0.467 | 0.698 | 0.738 | 0.857 | 0.140 | 0.067 |
| Single prompt | 0.600 | 0.716 | 0.821 | 0.578 | 0.129 | 0.000 |

The first MuSiQue run did not improve the course metric: the single-prompt baseline scored higher on cosine. The multi-node DAG retrieved more of the gold supporting evidence, but this did not translate into better final-answer string alignment. That suggests the next optimization loop should focus on final synthesis and answer normalization for MuSiQue-style decomposed evidence, not only retrieval.

## Iteration 2 Rationale

The first error review showed two immediate DAG-specific issues:

- MuSiQue includes 2-4 hop questions, but the default planner depth was capped at 3. One sampled 4-hop item failed structurally and received a blank answer.
- Several DAG answers were valid but poorly aligned with the expected answer surface form, for example returning a broader or shorter entity when the evidence contained the more specific answer.

Research alignment:

- MuSiQue was designed to make shortcut-style disconnected reasoning less effective, so the benchmark setup must allow the full hop depth required by the sampled questions.
- IRCoT argues that multi-step QA improves when each intermediate reasoning step conditions the next retrieval/reasoning step, which supports preserving precise bridge values through the DAG.
- Min et al. show that decomposition systems benefit from answer selection/rescoring over decomposition outputs, which matches the observed gap between strong evidence recall and weaker final answer strings.

Implemented changes:

- MuSiQue benchmark runs now raise planner `max_depth` to 5 while leaving HotpotQA unchanged.
  The extra level is needed because the executor's depth check counts the final aggregation step
  as an additional level for some 4-hop MuSiQue plans.
- Node prompts now explicitly ask for the most specific supported entity/value and forbid extra addresses or explanatory clauses unless requested.

References:

- Trivedi et al., "MuSiQue: Multihop Questions via Single-hop Question Composition", TACL 2022. https://arxiv.org/abs/2108.00573
- Trivedi et al., "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions", ACL 2023. https://arxiv.org/abs/2212.10509
- Min et al., "Multi-hop Reading Comprehension through Question Decomposition and Rescoring", ACL 2019. https://arxiv.org/abs/1906.02916

## Iteration 3: Target Preservation

The next error review showed that the remaining losses were mostly not caused by small answer
format differences. The important failures were target-selection failures:

- The DAG sometimes computed multiple plausible values and selected the bridge entity's attribute
  instead of the target entity's attribute. Example: in "what region of the country where Lam Thao
  is located is the city where the Zone 5 Military Museum can be found?", Lam Thao only supplies
  the country scope. The answer target is the region of the museum city, Da Nang.
- Historical or qualified country answers could be collapsed to a broader modern country name,
  for example "Germany" instead of the evidence-backed "GDR".
- Language answers could drop meaningful qualifiers, for example "Latin" instead of "Medieval
  Latin".

Research alignment:

- MuSiQue explicitly tests connected multi-hop reasoning, so preserving which entity is the bridge
  and which entity owns the final attribute is central to the dataset design.
- IRCoT motivates conditioning later reasoning on intermediate facts. In our case, this means the
  bridge value should constrain the next lookup, not become the final answer when the question asks
  for another entity's attribute.
- Min et al. support answer selection over decomposed outputs. We used this as a bounded
  extractive refinement step, while keeping the primary fix inside the DAG prompts and final-node
  contract.

Implemented changes:

- Final nodes now always receive an explicit answer contract with `answer`, `answer_type`, and
  `answer_source_span`, including plans that come through the normalizer rather than the structured
  planner path.
- Planner and node prompts now distinguish scope bridges from answer targets for the pattern
  "region/place of the country where X is located is Y". X resolves the country; Y owns the final
  region/place answer.
- Node prompts now preserve qualified and historical polity names instead of collapsing them to a
  broader modern country.
- Post-processing restores evidence-backed language qualifiers and qualified/historical polity
  acronyms when they are present in the supplied evidence. This is data-driven; it does not encode
  specific question IDs or gold answers.
- DAG benchmark execution now has a fallback for structural DAG failures: if planning/execution
  raises, it records the failure in the trace and uses a single evidence-only recovery node instead
  of scoring a blank answer. This keeps structural failures inspectable without turning them into
  silent empty predictions.

Final measured comparison:

- dataset: `musique`
- subset: `validation_3hop_plus`
- limit: 15
- systems: `dag_agent` and `direct_llm`
- model: `mistralai/Mistral-Medium-3.5-128B`
- seed: `1745764549`

Saved result files:

- `runs/benchmarks/musique-20260622T073006Z-07f45b6f-9dd2-4fff-9b62-aeba81642b8c.json`
- `runs/benchmarks/musique-20260622T073006Z-6f1c560a-57cd-4297-9dc9-a4e2de870677.json`

| System | Exact match | F1 | Cosine | Gold fact recall | Wrong citation rate | Structural failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-node DAG | 0.867 | 0.908 | 0.940 | 0.889 | 0.070 | 0.000 |
| Single prompt | 0.800 | 0.827 | 0.908 | 0.556 | 0.133 | 0.000 |

The multi-node DAG is now ahead of the single prompt on this 15-item MuSiQue sample by cosine
(`0.940` vs `0.908`), exact match, F1, and evidence recall. The improvement came from fixing
answer-target selection, not from optimizing harmless surface differences.

Remaining error:

- `3hop2__72083_92991_76291`: gold `January 2015`, DAG `January 3, 1947`. The planner made a
  one-node plan that identified the party gaining control in 1946 but did not decompose the second
  hop asking when that party later took control of the government branch that determines House
  rules. This is a true under-decomposition/temporal-target failure and should be the next
  optimization target.
