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
- bridge questions: keep the answer anchored to the entity or value supplied by dependencies.
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


def render_repair_prompt(raw_response: str, schema: dict[str, Any], errors: list[str]) -> str:
    return f"""Return a corrected JSON object that matches the schema.

Schema:
{json.dumps(schema, indent=2)}

Validation errors:
{chr(10).join(f"- {error}" for error in errors)}

Previous response:
{raw_response}
"""
