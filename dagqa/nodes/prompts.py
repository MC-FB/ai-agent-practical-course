from __future__ import annotations

import json
import re
from typing import Any

from dagqa.schemas import DagNode, EvidenceSelection

_TEMPLATE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")


def render_evidence_section(
    supporting_evidence: EvidenceSelection,
    *,
    require_citations: bool = True,
) -> str:
    rendered = "\n\nSupporting evidence documents:\n"
    rendered += (
        "Only some of these documents may be relevant. Use the relevant evidence to answer "
        "the question, ignore distractors, and do not make unsupported factual claims."
    )
    if require_citations:
        rendered += (
            " In _evidence_citations, return only facts that directly determine the values in your "
            "output. Do not cite documents or candidate facts that you merely read, inspected, "
            "considered, or rejected. Do not cite an exhaustive list when only one item is needed. "
            "Every returned field value must be directly supported by one or more citations, "
            "including each item in an array. Do not fill broad lists from partial evidence; "
            "if the evidence supports only one relevant entity or fact, return only that entity "
            "or fact. Preserve the answer type requested by the question and schema: do not "
            'return "yes" or "no" unless the question asks yes/no. '
            "Cite each directly used fact with the exact document ID, title, and zero-based "
            "sentence indices shown below. Never invent document IDs, titles, or sentence "
            "indices; use only the document IDs and sentence numbers printed in this prompt."
        )
    rendered += "\n"
    for document in supporting_evidence.documents:
        rendered += f"\nDocument ID: {document.id}\nTitle: {document.title}\n"
        sentences = document.metadata.get("sentences")
        if isinstance(sentences, list):
            for sentence_index, sentence in enumerate(sentences):
                rendered += f"  ({sentence_index}) {sentence}\n"
        else:
            rendered += f"  (0) {document.text}\n"
    return rendered


def render_node_prompt(
    node: DagNode,
    resolved_question: str,
    dependency_values: dict[str, Any],
    outputs: dict[str, dict[str, Any]] | None = None,
    supporting_evidence: EvidenceSelection | None = None,
) -> str:
    namespaced_outputs = {
        f"{node_id}.{field}": value
        for node_id, values in (outputs or {}).items()
        for field, value in values.items()
    }
    context = {
        "node.id": node.id,
        "node.label": node.label,
        "node.question": node.question,
        "resolved_question": resolved_question,
        "dependencies": json.dumps(dependency_values, indent=2, ensure_ascii=False),
        **namespaced_outputs,
        **dependency_values,
    }
    rendered = _TEMPLATE_RE.sub(
        lambda match: str(context.get(match.group(1), match.group(0))),
        node.prompt.user_template,
    )
    if node.depends_on and dependency_values and "dependencies" not in node.prompt.user_template:
        rendered = (
            "Dependency values available to this node:\n"
            f"{json.dumps(dependency_values, indent=2, ensure_ascii=False)}\n\n"
            f"{rendered}"
        )
    if supporting_evidence is not None:
        rendered += render_evidence_section(supporting_evidence)
    rendered += "\n\nReturn format JSON Schema:\n"
    rendered += json.dumps(node_output_schema(node, supporting_evidence), indent=2)
    rendered += """

Return JSON only. Do not wrap the JSON in markdown fences.
Normalize final answer values:
- yes/no questions: answer with "yes" or "no", not true/false.
- dates: use the natural date form requested by the question when possible.
- numbers: return only the concise number or quantity unless units are part of the answer.
- answer surface: when the evidence gives the answer in a specific written form, copy that
  form exactly. For example, keep "3." if the source says "3.", keep "third" or
  "third-largest" if the source says that, and keep "July 11, 2017" instead of rewriting it
  to "2017-07-11".
- table or fixture rows: when evidence is formatted as a table row with teams/entities and a
  score such as "Home 2 -- 1 Away", compare the numbers in row order. If asked when one team
  beat another, return the date from rows where that team's score is greater.
- compact score tables: scan every row, including later rows. Some tables omit the away/opponent
  column because the table title or question defines the matchup. In that case, infer the omitted
  opponent as the other side in the same table instead of returning never. Treat renamed clubs or
  organizations in the same table as possible aliases when the bridge entity is historical.
  A draw such as "0 -- 0" is not a win and must not be returned for "beat" questions.
- capital-duration questions: if asked how long a place had been the capital/capitol city of
  another resolved location, prefer a direct evidence span like "had been the capital city of X
  for Y". Do not override that span with modern administrative reasoning such as "X is only a city"
  or with a country capital unless the question explicitly asks for the country capital.
- do not return empty, unknown, none, or never if the supplied evidence contains a row or span
  that answers the requested field.
- bridge questions: keep the answer anchored to the entity or value supplied by dependencies.
- bridge contract fields: if the schema contains bridge_answer, bridge_reasoning,
  bridge_source_span, or constraint_status, fill them explicitly. bridge_answer is the value
  downstream nodes may rely on; bridge_reasoning must explain why it satisfies the resolved
  question and dependency constraints; bridge_source_span is the shortest exact support span;
  constraint_status must be exactly "satisfied", "ambiguous", or "not_found".
- set constraint_status to "satisfied" only when the cited evidence supports every required
  dependency constraint. If evidence only partially matches, points to a similarly named
  distractor, or lacks the requested date/place/entity relation, set "ambiguous" or "not_found"
  and do not present the bridge answer as certain.
- parent nodes: inspect dependency bridge_reasoning and constraint_status before using child
  values. If a dependency is ambiguous or not_found, resolve the uncertainty from evidence rather
  than treating the child value as established.
- final synthesis with evidence: cite the sentence or connected sentence chain that supports
  the final answer under the resolved dependency values. Do not answer from a sentence that
  only matches one dependency while contradicting or ignoring another dependency.
- dependency constraints: before returning the final answer, check that the answer span is
  about the resolved target entity from dependencies, not a similarly named distractor or a
  broader category from another document.
- in "region/place of the country where X is located is Y" questions, X is a country/scope
  bridge; answer with the region/place of Y, not the region/place of X.
- entity answers: return the exact specific entity/value requested, not a broader modern successor,
  hypernym, parent region, or shortened form when the evidence gives the more specific answer.
- country/place answers: preserve historical or qualified polity names from evidence titles and
  sentences, including directional qualifiers and acronyms; do not collapse them to a modern or
  broader country name.
- do not append addresses, explanatory clauses, or parenthetical details unless the question asks
  for that extra detail.
- if the schema contains answer_type, first identify what kind of value the original question asks
  for, then ensure answer has that type. Do not return a bridge value of a different type.
- if the schema contains answer_source_span, copy the shortest dependency or evidence span that
  directly supports answer. Preserve modifiers from that span in answer when they change meaning.
- before/after/later than/earlier than/since/until questions: preserve the requested boundary
  value; do not substitute a latest or earliest endpoint unless that is explicitly asked.
- quoted-title questions: answer about the quoted work/title itself, not a different entity
  mentioned inside that title.
- language questions: preserve descriptors such as "English-language"; do not collapse them
  to a person's nationality or to an original-title language.
- if the schema contains an answer field, put the concise final answer there.
"""
    return rendered


def node_output_schema(
    node: DagNode,
    supporting_evidence: EvidenceSelection | None,
) -> dict[str, Any]:
    if supporting_evidence is None:
        return node.output_schema
    schema = json.loads(json.dumps(node.output_schema))
    properties = schema.setdefault("properties", {})
    properties["_evidence_citations"] = {
        "type": "array",
        "description": (
            "Only evidence facts that directly determine the returned values; exclude inspected "
            "or rejected candidates."
        ),
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["document_id", "title", "sentence_indices", "fact"],
            "properties": {
                "document_id": {"type": "string"},
                "title": {"type": "string"},
                "sentence_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                },
                "fact": {"type": "string"},
            },
        },
    }
    required = schema.setdefault("required", [])
    if "_evidence_citations" not in required:
        required.append("_evidence_citations")
    return schema


def render_repair_prompt(
    raw_response: str,
    schema: dict[str, Any],
    errors: list[str],
    supporting_evidence: EvidenceSelection | None = None,
) -> str:
    evidence_section = ""
    if supporting_evidence is not None:
        valid_sources = []
        for document in supporting_evidence.documents:
            valid_sources.append(f"- {document.id}: {document.title}")
        evidence_section = (
            "\nValid evidence document IDs and exact titles:\n"
            + "\n".join(valid_sources)
            + "\nUse only these document_id values and matching titles in _evidence_citations. "
            "Never use placeholder IDs such as default, doc1, doc_001, source, or evidence.\n"
        )

    return f"""Return a corrected JSON object that matches the schema.

Schema:
{json.dumps(schema, indent=2)}

Validation errors:
{chr(10).join(f"- {error}" for error in errors)}
{evidence_section}

Previous response:
{raw_response}
"""
