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
    original_question: str | None = None,
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
    if node.depends_on:
        rendered = (
            render_chain_of_answers(
                original_question=original_question,
                resolved_question=resolved_question,
                outputs=outputs or {},
                dependency_ids=node.depends_on,
            )
            + "\n\n"
            + rendered
        )
    elif original_question:
        rendered = (
            render_question_context_block(
                original_question=original_question,
                resolved_question=resolved_question,
            )
            + "\n\n"
            + rendered
        )
    is_final = bool(
        node.output_schema.get("properties", {}).get("answer_type")
        or node.output_schema.get("properties", {}).get("answer_source_span")
    )
    if supporting_evidence is not None:
        rendered += render_evidence_section(supporting_evidence, require_citations=is_final)
    rendered += "\n\nReturn format JSON Schema:\n"
    rendered += json.dumps(
        node_output_schema(node, supporting_evidence, is_final=is_final), indent=2
    )
    rendered += """

Return JSON only. Do not wrap the JSON in markdown fences.
Normalize answer values:
- yes/no questions: answer "yes" or "no".
- dates: use the natural date form from evidence.
- numbers: return only the concise number or quantity.
- answer surface: copy the exact written form from evidence when possible.
- do not return empty, unknown, none, or never if evidence contains a matching span.
- entity answers: return the specific entity requested, not a broader successor or hypernym.
- preserve historical or qualified names from evidence; do not modernize them.
- if the schema contains answer_type, ensure answer matches that type.
- if the schema contains answer_source_span, copy the shortest supporting span.
"""
    return rendered


def render_question_context_block(
    *,
    original_question: str,
    resolved_question: str,
) -> str:
    lines = [
        "Question context for this node:",
        f"- Original user question: {original_question}",
        f"- Current node question: {resolved_question}",
        "- Answer this node only as a step toward the original question.",
        "- Preserve the original question's scope, comparisons, constraints, and final "
        "answer type.",
    ]
    if _has_scoped_superlative(original_question):
        lines.append(
            "- The original question contains a scoped superlative/comparative. Keep the "
            "superlative inside that scope; do not answer a broader global superlative."
        )
    return "\n".join(lines)


def render_chain_of_answers(
    *,
    original_question: str | None,
    resolved_question: str,
    outputs: dict[str, dict[str, Any]],
    dependency_ids: list[str],
) -> str:
    lines: list[str] = []
    if original_question:
        lines.append(f"Original question: {original_question}")
        lines.append("")
    qa_pairs: list[str] = []
    for dependency_id in dependency_ids:
        output = outputs.get(dependency_id, {})
        answer = output.get("answer") or output.get("bridge_answer") or ""
        if answer:
            qa_pairs.append(f"{dependency_id}: {answer}")
    if qa_pairs:
        lines.append("Answers so far:")
        lines.extend(qa_pairs)
        lines.append("")
    lines.append("Your task: Answer the following question using the evidence below.")
    lines.append(f"Question: {resolved_question}")
    return "\n".join(lines)


def _has_scoped_superlative(question: str) -> bool:
    lowered = question.casefold()
    return bool(
        re.search(
            r"\b(largest|smallest|oldest|youngest|first|last|most|least)\b.+"
            r"\b(where|in|from|of|within|containing|that|which)\b",
            lowered,
        )
    )


def node_output_schema(
    node: DagNode,
    supporting_evidence: EvidenceSelection | None,
    *,
    is_final: bool = False,
) -> dict[str, Any]:
    if supporting_evidence is None:
        return node.output_schema
    schema = json.loads(json.dumps(node.output_schema))
    if is_final:
        properties = schema.setdefault("properties", {})
        properties["_evidence_citations"] = {
            "type": "array",
            "description": (
                "Only evidence facts that directly determine the "
                "returned values; exclude inspected or rejected candidates."
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
