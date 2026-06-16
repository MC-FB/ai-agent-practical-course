from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LOW_COSINE_THRESHOLD = 0.5
GOOD_COSINE_THRESHOLD = 0.8
TIE_EPSILON = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify paired HotpotQA benchmark failures.")
    parser.add_argument("--dag-run", required=True, help="DAG benchmark result JSON")
    parser.add_argument("--direct-run", required=True, help="Direct benchmark result JSON")
    parser.add_argument("--marked", default="runs/benchmarks/.marked/rows.json")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    dag = _load_json(Path(args.dag_run))
    direct = _load_json(Path(args.direct_run))
    marked_rows = _load_marked(Path(args.marked))
    direct_by_id = {record["id"]: record for record in direct.get("records", [])}
    paired = [
        (record, direct_by_id[record["id"]])
        for record in dag.get("records", [])
        if record.get("id") in direct_by_id
    ]

    rows = []
    for dag_record, direct_record in paired:
        rows.append(
            {
                "id": dag_record["id"],
                "question": dag_record.get("question", ""),
                "gold": dag_record.get("gold_answer", ""),
                "dag": dag_record.get("prediction", ""),
                "direct": direct_record.get("prediction", ""),
                "dag_cosine": dag_record.get("cosine_sim", 0),
                "direct_cosine": direct_record.get("cosine_sim", 0),
                "delta": dag_record.get("cosine_sim", 0) - direct_record.get("cosine_sim", 0),
                "category": classify_record(dag_record, direct_record),
                "marked": dag_record["id"] in marked_rows,
            }
        )

    output = render_markdown(dag, direct, rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(output)


def classify_record(dag_record: dict[str, Any], direct_record: dict[str, Any]) -> str:
    question = dag_record.get("question", "")
    gold = str(dag_record.get("gold_answer", ""))
    dag = str(dag_record.get("prediction", ""))
    dag_cosine = float(dag_record.get("cosine_sim") or 0)
    direct_cosine = float(direct_record.get("cosine_sim") or 0)

    category = "other"
    if _is_yes_no_question(question) and gold.casefold() not in {"yes", "no"}:
        category = "bad_or_misaligned_gold"
    elif _looks_like_formatting_issue(dag, gold):
        category = "formatting"
    elif dag.casefold() in {"none", "unknown", "not found", "n/a", "no answer"}:
        category = "pipeline_decomposition"
    elif dag and not _answer_occurs_in_trace(dag, dag_record):
        category = "unsupported_placeholder"
    elif dag_cosine < LOW_COSINE_THRESHOLD and direct_cosine >= GOOD_COSINE_THRESHOLD:
        category = "wrong_factual"
    elif dag_cosine < GOOD_COSINE_THRESHOLD and direct_cosine < GOOD_COSINE_THRESHOLD:
        category = "both_systems_wrong"
    return category


def render_markdown(
    dag: dict[str, Any],
    direct: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    dag_metrics = dag.get("metrics", {})
    direct_metrics = direct.get("metrics", {})
    dag_wins = sum(1 for row in rows if row["delta"] > TIE_EPSILON)
    direct_wins = sum(1 for row in rows if row["delta"] < -TIE_EPSILON)
    ties = len(rows) - dag_wins - direct_wins
    worst = sorted(rows, key=lambda row: row["delta"])[:10]
    marked = [row for row in rows if row["marked"]]

    lines = [
        "# Benchmark Failure Analysis",
        "",
        f"- DAG run: `{dag.get('run_id')}` / `{dag.get('name')}`",
        f"- Direct run: `{direct.get('run_id')}` / `{direct.get('name')}`",
        f"- Compared rows: `{len(rows)}`",
        f"- DAG cosine: `{dag_metrics.get('cosine_sim', 0):.4f}`",
        f"- Direct cosine: `{direct_metrics.get('cosine_sim', 0):.4f}`",
        f"- DAG wins / direct wins / ties: `{dag_wins}` / `{direct_wins}` / `{ties}`",
        "",
        "## Error Buckets",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Worst DAG Deltas", ""])
    lines.extend(_table(worst))
    if marked:
        lines.extend(["", "## Marked Rows", ""])
        lines.extend(_table(marked))
    lines.append("")
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| ID | Category | Delta | Gold | DAG | Direct | Question |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {id} | `{category}` | {delta:.3f} | {gold} | {dag} | {direct} | {question} |".format(
                id=_cell(row["id"]),
                category=row["category"],
                delta=row["delta"],
                gold=_cell(row["gold"]),
                dag=_cell(row["dag"]),
                direct=_cell(row["direct"]),
                question=_cell(row["question"]),
            )
        )
    return lines


def _looks_like_formatting_issue(prediction: str, gold: str) -> bool:
    if not prediction or not gold:
        return False
    if gold.casefold() in prediction.casefold() and prediction.casefold() != gold.casefold():
        return True
    if prediction.startswith("[") and prediction.endswith("]"):
        return True
    return bool(re.fullmatch(r"\d+", prediction) and re.search(r"\b\d+(?:\.\d+)?\s+\w+", gold))


def _answer_occurs_in_trace(answer: str, record: dict[str, Any]) -> bool:
    trace_text = json.dumps(record.get("run_trace") or {}, ensure_ascii=False).casefold()
    return answer.casefold() in trace_text


def _is_yes_no_question(question: str) -> bool:
    first = question.strip().split(maxsplit=1)[0].casefold() if question.strip() else ""
    return first in {"are", "is", "was", "were", "do", "does", "did", "can", "could", "has", "have"}


def _load_marked(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = _load_json(path)
    return {row.get("record_id", "") for row in payload.get("rows", []) if isinstance(row, dict)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    main()
