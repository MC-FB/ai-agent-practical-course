from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_DAG = Path(
    "runs/benchmarks/hotpotqa-20260605T135513Z-aa346249-5eab-4491-8773-8f5e8a8feb8d.json"
)
DEFAULT_DIRECT = Path(
    "runs/benchmarks/hotpotqa-20260605T135513Z-25ddcfd6-6416-42a3-9837-50f9a7aaa7f2.json"
)
DEFAULT_OUTPUT = Path("runs/benchmarks/large_dag_failure_analysis.md")
HIGH_F1_THRESHOLD = 0.8
HIGH_COSINE_THRESHOLD = 0.9
SUPPORT_RECALL_THRESHOLD = 0.67
WRONG_SUPPORT_RATE_THRESHOLD = 0.35
MAX_EXAMPLES_PER_CATEGORY = 5


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    severity: str
    description: str
    recommendation: str


CATEGORIES = {
    "provider_rate_limit": Category(
        "provider_rate_limit",
        "Provider / execution rate limiting",
        "Critical",
        "The LLM call failed with a rate-limit error, usually causing downstream nodes "
        "to miss inputs.",
        "Add provider-side backoff, concurrency throttling, retry budgets, and "
        "resumable per-node execution.",
    ),
    "dependency_contract": Category(
        "dependency_contract",
        "Planner schema / dependency contract mismatch",
        "Critical",
        "A node referenced a field such as q1.album or q2.county that was absent from "
        "the upstream output.",
        "Validate every input_map path against upstream schemas before execution; "
        "repair plans or align schemas automatically.",
    ),
    "final_combination": Category(
        "final_combination",
        "Final combination does not consume gathered facts",
        "High",
        "The final node had no dependency values or did not use child values in its "
        "prompt template.",
        "Require final nodes to reference child outputs and add a structural validator "
        "that rejects disconnected synthesis prompts.",
    ),
    "info_gathering": Category(
        "info_gathering",
        "Information gathering / evidence selection failure",
        "High",
        "The DAG completed, but cited too little gold evidence or used wrong supporting "
        "evidence before producing a wrong answer.",
        "Improve retrieval/evidence atomization and force each lookup node to ground "
        "claims in the relevant supporting facts.",
    ),
    "reasoning_combination": Category(
        "reasoning_combination",
        "Reasoning or comparison over gathered facts failed",
        "Medium",
        "The DAG appears to have enough evidence, but the final answer is still wrong.",
        "Add stricter final-node verification, typed intermediate outputs, and answer "
        "consistency checks against cited facts.",
    ),
    "answer_normalization": Category(
        "answer_normalization",
        "Answer normalization / alias mismatch",
        "Low",
        "Exact match failed even though token overlap or semantic similarity is high.",
        "Normalize aliases, dates, articles, punctuation, and common entity variants "
        "before scoring or final output.",
    ),
    "ambiguous_completed_wrong": Category(
        "ambiguous_completed_wrong",
        "Completed but wrong, ambiguous from trace",
        "Medium",
        "The DAG completed with a wrong answer, but trace metrics do not isolate "
        "retrieval versus reasoning.",
        "Review sampled traces manually or with an LLM judge only for this residual group.",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or "")


def nodes(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("run_trace") or {}
    return list(trace.get("nodes") or [])


def failed_nodes(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in nodes(record) if node.get("status") == "failed" or node.get("error")]


def has_rate_limit(record: dict[str, Any]) -> bool:
    return any(
        "too many requests" in str(node.get("error") or "").lower() for node in failed_nodes(record)
    )


def has_dependency_contract_error(record: dict[str, Any]) -> bool:
    for node in failed_nodes(record):
        validation = node.get("validation") or {}
        errors = list(validation.get("errors") or [])
        error_text = str(node.get("error") or "")
        if any(err.startswith("'") and "." in err for err in errors):
            return True
        if error_text.startswith("'") and "." in error_text:
            return True
    return False


def has_final_combination_issue(record: dict[str, Any]) -> bool:
    issues = record.get("structural_issues") or []
    return any(
        "final trace has no dependency_values" in issue
        or "prompt template does not use child values" in issue
        for issue in issues
    )


def is_normalization_case(record: dict[str, Any]) -> bool:
    if as_float(record.get("exact_match")) >= 1:
        return False
    return (
        as_float(record.get("f1")) >= HIGH_F1_THRESHOLD
        or as_float(record.get("cosine_sim")) >= HIGH_COSINE_THRESHOLD
    )


def is_info_gathering_case(record: dict[str, Any]) -> bool:
    if failed_nodes(record) or has_final_combination_issue(record):
        return False
    recall = as_float(record.get("gold_supporting_fact_recall"))
    wrong_rate = as_float(record.get("wrong_supporting_text_rate"))
    citation_count = as_float(record.get("evidence_citation_count"))
    return (
        recall < SUPPORT_RECALL_THRESHOLD
        or wrong_rate >= WRONG_SUPPORT_RATE_THRESHOLD
        or citation_count == 0
    )


def is_reasoning_combination_case(record: dict[str, Any]) -> bool:
    if failed_nodes(record) or has_final_combination_issue(record):
        return False
    return (
        as_float(record.get("gold_supporting_fact_recall")) >= SUPPORT_RECALL_THRESHOLD
        and as_float(record.get("wrong_supporting_text_rate")) < WRONG_SUPPORT_RATE_THRESHOLD
    )


def classify(record: dict[str, Any]) -> str | None:
    if as_float(record.get("exact_match")) >= 1:
        return None
    checks = [
        ("provider_rate_limit", has_rate_limit),
        ("dependency_contract", has_dependency_contract_error),
        ("final_combination", has_final_combination_issue),
        ("answer_normalization", is_normalization_case),
        ("info_gathering", is_info_gathering_case),
        ("reasoning_combination", is_reasoning_combination_case),
    ]
    for category, check in checks:
        if check(record):
            return category
    return "ambiguous_completed_wrong"


def collect_node_error_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    error_counter: collections.Counter[str] = collections.Counter()
    failed_task_types: collections.Counter[str] = collections.Counter()
    missing_fields: collections.Counter[str] = collections.Counter()
    for record in records:
        for node in failed_nodes(record):
            failed_task_types[str(node.get("task_type") or "unknown")] += 1
            error = str(node.get("error") or "").strip() or "<empty error>"
            if "Too many requests" in error:
                error = "Too many requests, please try again later."
            error_counter[error] += 1
            validation = node.get("validation") or {}
            for err in validation.get("errors") or []:
                if isinstance(err, str) and err.startswith("'") and "." in err:
                    missing_fields[err.strip("'")] += 1
    return {
        "errors": error_counter,
        "failed_task_types": failed_task_types,
        "missing_fields": missing_fields,
    }


def matched_direct_records(direct: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not direct:
        return {}
    return {record_id(record): record for record in direct.get("records") or []}


def summarize(dag: dict[str, Any], direct: dict[str, Any] | None) -> dict[str, Any]:
    records = list(dag.get("records") or [])
    direct_by_id = matched_direct_records(direct)
    categories: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    dag_only_wins = 0
    direct_only_wins = 0
    both_wrong = 0
    both_right = 0

    for record in records:
        category = classify(record)
        if category:
            categories[category] += 1
            if len(examples[category]) < MAX_EXAMPLES_PER_CATEGORY:
                examples[category].append(
                    {
                        "id": record_id(record),
                        "question": str(record.get("question") or ""),
                        "gold": str(record.get("gold_answer") or ""),
                        "prediction": str(record.get("prediction") or ""),
                    }
                )

        direct_record = direct_by_id.get(record_id(record))
        if direct_record:
            dag_right = as_float(record.get("exact_match")) >= 1
            direct_right = as_float(direct_record.get("exact_match")) >= 1
            if dag_right and direct_right:
                both_right += 1
            elif dag_right and not direct_right:
                dag_only_wins += 1
            elif direct_right and not dag_right:
                direct_only_wins += 1
            else:
                both_wrong += 1

    wrong_records = [record for record in records if as_float(record.get("exact_match")) < 1]
    signal_checks = {
        "provider_rate_limit": has_rate_limit,
        "dependency_contract": has_dependency_contract_error,
        "final_combination": has_final_combination_issue,
        "info_gathering": is_info_gathering_case,
        "reasoning_combination": is_reasoning_combination_case,
        "answer_normalization": is_normalization_case,
    }
    f1_values = [as_float(record.get("f1")) for record in records]
    latency_values = [
        as_float(record.get("latency_ms"))
        for record in records
        if record.get("latency_ms") is not None
    ]
    return {
        "records": records,
        "categories": categories,
        "examples": examples,
        "node_stats": collect_node_error_stats(records),
        "signals": {
            key: {
                "all": sum(1 for record in records if check(record)),
                "wrong": sum(1 for record in wrong_records if check(record)),
            }
            for key, check in signal_checks.items()
        },
        "direct_comparison": {
            "matched": len(direct_by_id),
            "both_right": both_right,
            "dag_only_wins": dag_only_wins,
            "direct_only_wins": direct_only_wins,
            "both_wrong": both_wrong,
        },
        "overview": {
            "example_count": len(records),
            "em": mean(as_float(record.get("exact_match")) for record in records),
            "f1": mean(f1_values) if f1_values else 0.0,
            "median_f1": median(f1_values) if f1_values else 0.0,
            "structural_failures": sum(1 for record in records if record.get("structural_failure")),
            "record_errors": sum(1 for record in records if record.get("error")),
            "node_failures": sum(len(failed_nodes(record)) for record in records),
            "avg_latency_ms": mean(latency_values) if latency_values else 0.0,
            "p50_latency_ms": median(latency_values) if latency_values else 0.0,
        },
    }


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def category_rank(category_key: str, count: int) -> tuple[int, int]:
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    category = CATEGORIES[category_key]
    return severity_order.get(category.severity, 9), -count


def render_markdown(dag_path: Path, direct_path: Path | None, summary: dict[str, Any]) -> str:
    records = summary["records"]
    total = len(records)
    categories: collections.Counter[str] = summary["categories"]
    overview = summary["overview"]
    node_stats = summary["node_stats"]
    signals = summary["signals"]
    direct_comparison = summary["direct_comparison"]
    wrong_total = sum(categories.values())
    ranked = sorted(categories.items(), key=lambda item: category_rank(item[0], item[1]))

    lines = [
        "# Large Benchmark DAG Failure Analysis",
        "",
        f"Analyzed DAG run: `{dag_path}`",
    ]
    if direct_path:
        lines.append(f"Compared direct run: `{direct_path}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Examples: {total}",
            (
                f"- DAG exact match: {overview['em']:.3f}; "
                f"average F1: {overview['f1']:.3f}; "
                f"median F1: {overview['median_f1']:.3f}"
            ),
            (f"- DAG exact-match failures classified: {wrong_total} ({pct(wrong_total, total)})"),
            (
                f"- Structural failures: {overview['structural_failures']} "
                f"({pct(overview['structural_failures'], total)})"
            ),
            (
                f"- Record-level errors: {overview['record_errors']} "
                f"({pct(overview['record_errors'], total)})"
            ),
            f"- Failed node traces: {overview['node_failures']}",
            (
                f"- Average latency: {overview['avg_latency_ms'] / 1000:.1f}s; "
                f"p50 latency: {overview['p50_latency_ms'] / 1000:.1f}s"
            ),
            "",
            "## Ranked Failure Modes",
            "",
            "| Rank | Failure mode | Severity | Cases | Rate of all examples | "
            "Rate of DAG EM failures |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for index, (key, count) in enumerate(ranked, start=1):
        category = CATEGORIES[key]
        lines.append(
            f"| {index} | {category.label} | {category.severity} | {count} | "
            f"{pct(count, total)} | {pct(count, wrong_total)} |"
        )

    lines.extend(["", "## Failure Details", ""])
    for key, count in ranked:
        category = CATEGORIES[key]
        lines.extend(
            [
                f"### {category.label}",
                "",
                f"- Severity: {category.severity}",
                (
                    f"- Frequency: {count}/{total} examples ({pct(count, total)}); "
                    f"{pct(count, wrong_total)} of DAG exact-match failures"
                ),
                f"- What failed: {category.description}",
                f"- Improvement target: {category.recommendation}",
            ]
        )
        example_bits = []
        for example in summary["examples"].get(key, [])[:3]:
            example_bits.append(
                f"{example['id']}: predicted `{example['prediction']}` vs gold `{example['gold']}`"
            )
        if example_bits:
            lines.append(f"- Sample cases: {'; '.join(example_bits)}")
        lines.append("")

    lines.extend(
        [
            "## Component Signal Counts",
            "",
            "These are non-exclusive signals. A single example can appear in several "
            "rows, for example when a rate-limited lookup causes downstream missing "
            "dependency values.",
            "",
            "| Component signal | Severity | Wrong examples | All examples | Interpretation |",
            "|---|---|---:|---:|---|",
        ]
    )
    for key in [
        "provider_rate_limit",
        "dependency_contract",
        "final_combination",
        "info_gathering",
        "reasoning_combination",
        "answer_normalization",
    ]:
        category = CATEGORIES[key]
        signal = signals[key]
        lines.append(
            f"| {category.label} | {category.severity} | "
            f"{signal['wrong']} ({pct(signal['wrong'], total)}) | "
            f"{signal['all']} ({pct(signal['all'], total)}) | "
            f"{category.description} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Node-Level Signals",
            "",
            "Most common node errors:",
            "",
        ]
    )
    for error, count in node_stats["errors"].most_common(10):
        lines.append(f"- {count}: `{error}`")
    lines.extend(["", "Failed node task types:", ""])
    for task_type, count in node_stats["failed_task_types"].most_common():
        lines.append(f"- {task_type}: {count}")
    lines.extend(["", "Most common missing dependency fields:", ""])
    for field, count in node_stats["missing_fields"].most_common(15):
        lines.append(f"- {field}: {count}")

    if direct_comparison["matched"]:
        lines.extend(
            [
                "",
                "## Direct LLM Comparison",
                "",
                f"- Matched direct records: {direct_comparison['matched']}",
                f"- Both correct: {direct_comparison['both_right']}",
                f"- DAG only correct: {direct_comparison['dag_only_wins']}",
                f"- Direct only correct: {direct_comparison['direct_only_wins']}",
                f"- Both wrong: {direct_comparison['both_wrong']}",
                "",
                "The DAG improves accuracy over the direct baseline on this run, but "
                "it pays for that with many more LLM calls and a larger "
                "structural/execution failure surface.",
            ]
        )

    lines.extend(
        [
            "",
            "## AtomicRAG Assessment",
            "",
            "AtomicRAG-style decomposition looks useful for this system if it is "
            "applied to evidence and intermediate facts, not as a replacement for "
            "DAG validation. The paper proposes Atom-Entity Graphs that store "
            "self-contained factual atoms instead of coarse text chunks and use "
            "graph traversal/filtering to improve retrieval accuracy and reasoning "
            "robustness: https://arxiv.org/abs/2604.20844",
            "",
            "That maps well to the largest non-provider failure bucket: information "
            "gathering and evidence selection. Smaller atomic claims could help "
            "lookup nodes isolate the exact supporting facts before synthesis and "
            "reduce wrong or incomplete citations.",
            "",
            "It would not directly solve the critical planner contract failures: "
            "missing `input_map` fields, disconnected final prompts, and rate-limit "
            "cascades need schema validation, prompt-template validation, and "
            "scheduler/provider hardening first. The recommended order is: fix "
            "execution retries and dependency validation, then add atomic evidence "
            "extraction for lookup nodes, then add final-node consistency checks "
            "over those atoms.",
            "",
            "## Method",
            "",
            "The analyzer uses deterministic trace signals: node errors, validation "
            "errors, structural issues, exact-match/F1/cosine metrics, citation "
            "recall, and wrong-supporting-text rate. No LLM judging was required "
            "for the ranked counts; the residual ambiguous group is the only bucket "
            "that merits sampled LLM or manual review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze 1000-example HotpotQA DAG benchmark failure modes."
    )
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dag = load_json(args.dag)
    direct = load_json(args.direct) if args.direct and args.direct.exists() else None
    summary = summarize(dag, direct)
    markdown = render_markdown(args.dag, args.direct if direct else None, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)

    print(f"Wrote {args.output}")
    print(json.dumps(summary["overview"], indent=2))
    print("Failure categories:")
    for key, count in sorted(
        summary["categories"].items(), key=lambda item: category_rank(item[0], item[1])
    ):
        print(f"- {CATEGORIES[key].label}: {count}")


if __name__ == "__main__":
    main()
