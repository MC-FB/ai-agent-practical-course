from __future__ import annotations

import json
import re
from typing import Any

from dagqa.schemas import DagNode

_TEMPLATE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")


def render_node_prompt(
    node: DagNode,
    resolved_question: str,
    dependency_values: dict[str, Any],
    outputs: dict[str, dict[str, Any]] | None = None,
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
    rendered += "\n\nReturn format JSON Schema:\n"
    rendered += json.dumps(node.output_schema, indent=2)
    rendered += "\n\nReturn JSON only."
    return rendered


def render_repair_prompt(raw_response: str, schema: dict[str, Any], errors: list[str]) -> str:
    return f"""Return a corrected JSON object that matches the schema.

Schema:
{json.dumps(schema, indent=2)}

Validation errors:
{chr(10).join(f"- {error}" for error in errors)}

Previous response:
{raw_response}
"""
