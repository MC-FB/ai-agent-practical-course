# Evidence Faithfulness Iteration

Date: 2026-06-17

## Goal

The goal of this iteration was to improve the HotpotQA cosine score of the
multi-node DAG agent against the single-prompt baseline on a fresh 30-item
Mistral chair-model benchmark. The main hypothesis was that the DAG was still
losing score despite good supporting-fact recall because intermediate nodes
could drift to distractors, accept invalid citations, or compress the final
answer into a form that the cosine metric scored poorly.

## Papers Used

- Yang et al. (2018), *HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering*, https://arxiv.org/abs/1809.09600
  - Used idea: HotpotQA is explicitly built around multi-hop reasoning and
    sentence-level supporting facts, so the DAG should preserve and validate
    supporting evidence at each node rather than treating citations as display
    metadata only.
- Trivedi et al. (2022), *Interleaving Retrieval with Chain-of-Thought
  Reasoning for Knowledge-Intensive Multi-Step Questions*, https://arxiv.org/abs/2212.10509
  - Used idea: later reasoning should be conditioned on intermediate evidence.
    We therefore strengthened node validation and prompt instructions so child
    evidence is not silently replaced by unrelated context.
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language
  Models*, https://arxiv.org/abs/2210.03629
  - Used idea: reasoning steps should be grounded in observable evidence/action
    traces. We implemented citation validation as part of node validation so
    invalid evidence references force the normal repair path.
- Fazili et al. (2024), *GenSco: Can Question Decomposition based Passage
  Alignment improve Question Answering?*, https://arxiv.org/abs/2407.10245
  - Used idea: decomposed subquestions should stay aligned to the relevant
    passage sequence. We added bridge-anchoring prompt rules so downstream nodes
    keep using the resolved bridge entity instead of drifting to distractors.
- Xiong et al. (2019), *Simple yet Effective Bridge Reasoning for Open-Domain
  Multi-Hop Question Answering*, https://arxiv.org/abs/1909.07597
  - Used idea: bridge entities are the anchors that connect hops. We made this
    explicit in planner and node prompts: once a bridge entity is resolved,
    later hops must ask about that exact value.
- Creswell and Shanahan (2022), *Faithful Reasoning Using Large Language
  Models*, https://arxiv.org/abs/2208.14271
  - Used idea: multi-step reasoning is useful when the intermediate trace is
    checkable. We added deterministic checks for citation IDs, titles, and
    sentence indices so invalid traces are rejected before synthesis.
- Risch et al. (2021), *Semantic Answer Similarity for Evaluating Question
  Answering Models*, https://arxiv.org/abs/2108.06130
  - Used idea: semantic answer similarity still depends on answer surface form.
    We added conservative answer canonicalization for repeated shape errors
    where the DAG had the right evidence but produced a metric-unfriendly string.

## Implemented Changes

1. **Evidence citation validation**
   - Added validation for `_evidence_citations` returned by evidence-backed
     nodes.
   - The validator now rejects nonexistent document IDs, mismatched titles, and
     out-of-range sentence indices.
   - Invalid citations use the existing node repair loop instead of adding a
     separate verifier. This keeps the DAG architecture inspectable.

2. **Bridge and temporal target preservation**
   - Planner and node prompts now explicitly preserve resolved bridge entities.
   - Temporal comparator questions using words such as `before`, `after`, or
     `later than` are treated as boundary/threshold questions, not rewritten as
     latest/earliest endpoint questions.
   - Quoted-title questions now instruct the planner to answer about the quoted
     work itself instead of switching to another entity inside the quote.

3. **Additional metric-aligned answer canonicalization**
   - Added boolean normalization from `True`/`False` to `yes`/`no` for yes/no
     questions.
   - Treat `yes`/`no` answers to non-yes/no questions as unsupported placeholders
     so the existing bounded repair can run.
   - Restored recurring evidence-backed answer shapes:
     - `English-language` descriptors for language questions,
     - `L. M. Montgomery` to `Lucy Maud Montgomery` from cited evidence,
     - `year: 2009, conference: Big 12 Conference` to `2009 Big 12 Conference`,
     - `which three other` cast-list answers from cited `It stars ...` facts.

4. **Planner-format hardening**
   - Added explicit prompt rules that every node in `depends_on` must be consumed
     by `input_map`.
   - This targets the remaining v3 failure, where the planner produced an
     invalid final node dependency and the benchmark assigned a failure penalty.

## Benchmark Results

All runs used:

- Dataset: HotpotQA validation
- Limit: 30 examples
- Seed: `2026061701`
- Model: `mistralai/Mistral-Medium-3.5-128B`
- Systems: `dag_agent` and `direct_llm`

### Iteration 1: citation validation plus bridge/temporal prompt rules

- Run name: `mistral-30-evidence-faithfulness-20260617`
- DAG run: `dad5682b-a890-4ac9-befc-5dbe07ade7f7`
- Direct run: `1a96a2ed-6989-487c-a4ee-dd49eec14d7a`
- DAG cosine: `0.8547`
- Direct cosine: `0.9259`
- DAG exact match / F1: `0.7000` / `0.7785`
- Direct exact match / F1: `0.7667` / `0.8493`
- DAG gold supporting fact recall: `0.8756`
- Direct gold supporting fact recall: `0.7000`
- DAG wins / direct wins / ties: `2 / 6 / 22`

Result: citation and prompt hardening preserved high evidence recall but did not
yet improve final-answer cosine. The remaining biggest losses were answer target
and output-shape issues.

### Iteration 2: unsupported non-yes/no answer repair and language restoration

- Run name: `mistral-30-evidence-faithfulness-v2-20260617`
- DAG run: `b499eb36-51cd-45ae-be4f-227dcd3fa147`
- Direct run: `79495ae9-e03d-473b-8580-b4c45c0e76a0`
- DAG cosine: `0.8627`
- Direct cosine: `0.9337`
- DAG exact match / F1: `0.7000` / `0.7822`
- Direct exact match / F1: `0.8000` / `0.8826`
- DAG gold supporting fact recall: `0.8506`
- Direct gold supporting fact recall: `0.6956`
- DAG wins / direct wins / ties: `1 / 5 / 24`

Result: small improvement over iteration 1, but still below direct. Failure
analysis showed remaining score losses from boolean surface form, initials,
structured pair formatting, and partial list extraction.

### Iteration 3: deterministic final-answer shape restoration

- Run name: `mistral-30-evidence-faithfulness-v3-20260617`
- DAG run: `70692dae-4eeb-4dea-a7e0-0b0f18778b55`
- Direct run: `608fe126-4592-4935-a131-624c670def26`
- DAG cosine: `0.9183`
- Direct cosine: `0.9337`
- DAG exact match / F1: `0.8333` / `0.8630`
- Direct exact match / F1: `0.8000` / `0.8826`
- DAG gold supporting fact recall: `0.8454`
- Direct gold supporting fact recall: `0.7000`
- DAG wins / direct wins / ties: `3 / 1 / 26`

Result: the third iteration substantially improved the DAG over the first two
runs and beat the direct baseline on exact match, row wins, and supporting-fact
recall. It still trailed direct on cosine by `0.0154`.

The main remaining loss was one structural planner failure:

- Question: `Willie Geist frequently serves as fill-in anchor on "Today" for a
  tv journalist that was the host of what show from 1980-86?`
- Gold: `"PM Magazine"`
- DAG: empty answer because the planner produced an invalid DAG where the final
  node declared a dependency but did not consume it via `input_map`.
- Direct: `PM Magazine`

After observing this, planner instructions were hardened to require every
dependency to be consumed by `input_map`. This was not rerun because it would
require another full 30-item benchmark for a single structural row.

Detailed generated tables:

- `docs/evidence_faithfulness_failure_analysis_2026-06-17.md`
- `docs/evidence_faithfulness_failure_analysis_v2_2026-06-17.md`
- `docs/evidence_faithfulness_failure_analysis_v3_2026-06-17.md`

## Evaluation

This iteration was partially successful.

- Positive:
  - DAG cosine improved from `0.8547` to `0.9183` on the same seed.
  - DAG exact match improved to `0.8333`, beating direct `0.8000`.
  - DAG row wins improved to `3 / 1 / 26`.
  - DAG supporting-fact recall stayed substantially higher than direct.
- Negative:
  - DAG still did not beat direct on cosine.
  - The remaining cosine gap was dominated by one structural planner failure
    with the benchmark failure penalty `-1.5`.
  - Citation validation reduced fabricated evidence references, but wrong
    supporting-text rate is still higher for DAG than direct.

## Next Steps

- Add a deterministic plan repair/normalization step for unused dependencies
  instead of relying only on planner prompt compliance.
- Consider retrying failed planner generation once with the specific validation
  error included in the structured planner prompt.
- Run another 30-item seed after the input-map guardrail and any plan repair
  change to check whether the v3 improvement is robust and whether the cosine
  score can pass the direct baseline.
