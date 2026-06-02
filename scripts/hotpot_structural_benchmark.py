from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dagqa.client import DagQaClient
from dagqa.config import AppConfig
from dagqa.eval.hotpot_loader import load_hotpot_examples
from dagqa.eval.metrics import answer_f1, exact_match
from dagqa.planning.validator import validate_plan
from dagqa.schemas import RunTrace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", default="configs/azure-openai.yaml")
    parser.add_argument("--output-dir", default="runs/benchmarks")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.seed, Path(args.config), Path(args.output_dir)))


async def run(limit: int, seed: int | None, config_path: Path, output_dir: Path) -> None:
    seed = seed if seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hotpotqa-structural-{limit}-{run_id}.json"
    cfg = AppConfig.from_file(config_path)
    client = DagQaClient(cfg)
    all_examples = load_hotpot_examples(None)
    rng = random.Random(seed)
    examples = [all_examples[index] for index in rng.sample(range(len(all_examples)), limit)]
    report: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "limit": limit,
        "dataset_size": len(all_examples),
        "config": str(config_path),
        "model": cfg.llm.model,
        "records": [],
        "summary": {},
    }
    _write_report(output_path, report)
    print(f"RUN_ID={run_id}", flush=True)
    print(f"SEED={seed}", flush=True)
    print(f"OUT={output_path}", flush=True)
    started_total = time.perf_counter()

    for index, example in enumerate(examples, 1):
        print(f"[{index}/{limit}] {example.id} {example.question[:110]}", flush=True)
        started = time.perf_counter()
        record: dict[str, Any] = {
            "index": index,
            "id": example.id,
            "question": example.question,
            "gold_answer": example.answer,
        }
        try:
            run_trace = await client.ask(example.question)
            prediction = _extract_answer(run_trace.final_answer)
            structure = _structural_diagnostics(run_trace, cfg)
            record.update(
                {
                    "status": run_trace.status.value,
                    "prediction": prediction,
                    "exact_match": exact_match(prediction, example.answer),
                    "f1": answer_f1(prediction, example.answer),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "llm_call_count": len(run_trace.nodes) + 1,
                    "node_count": len(run_trace.plan.nodes),
                    "wave_count": len(run_trace.waves),
                    "final_answer": run_trace.final_answer,
                    "structure": structure,
                    "error": None,
                }
            )
            print(
                "  -> "
                f"{run_trace.status.value} "
                f"EM={record['exact_match']:.0f} "
                f"F1={record['f1']:.2f} "
                f"nodes={record['node_count']} "
                f"structure={structure['valid']} "
                f"pred={prediction[:80]}",
                flush=True,
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "prediction": "",
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "llm_call_count": None,
                    "node_count": None,
                    "wave_count": None,
                    "final_answer": None,
                    "structure": {"valid": False, "problems": [str(exc)]},
                    "error": str(exc),
                }
            )
            print(f"  -> ERROR {exc}", flush=True)

        report["records"].append(record)
        report["summary"] = _summary(report["records"], started_total)
        _write_report(output_path, report)

    print("SUMMARY", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)


def _extract_answer(final_answer: dict[str, Any] | None) -> str:
    if not final_answer:
        return ""
    if final_answer.get("answer") is not None:
        return str(final_answer["answer"])
    values = [
        value
        for key, value in final_answer.items()
        if key not in {"reasoning", "confidence"} and value is not None
    ]
    if len(values) == 1:
        return str(values[0])
    if "answer" in final_answer and final_answer["answer"] is None:
        return ""
    return ""


def _template_uses_child_values(node: Any) -> bool:
    template = node.prompt.user_template
    if "{dependencies}" in template:
        return True
    if any("{" + name + "}" in template for name in node.input_map):
        return True
    return any("{" + reference + "}" in template for reference in node.input_map.values())


def _structural_diagnostics(run: RunTrace, cfg: AppConfig) -> dict[str, Any]:
    validation = validate_plan(run.plan, cfg.planner)
    by_id = {node.id: node for node in run.plan.nodes}
    by_trace = {trace.node_id: trace for trace in run.nodes}
    problems: list[str] = []
    dependent_nodes = 0
    mapped_edges = 0

    for node in run.plan.nodes:
        if not node.depends_on:
            continue
        dependent_nodes += 1
        if not node.input_map:
            problems.append(f"{node.id}: dependent node has empty input_map")
            continue
        consumed = {reference.split(".", 1)[0] for reference in node.input_map.values()}
        missing = sorted(set(node.depends_on) - consumed)
        if missing:
            problems.append(f"{node.id}: input_map missing dependencies {missing}")
        if not _template_uses_child_values(node):
            problems.append(f"{node.id}: prompt template does not use child values")
        mapped_edges += len(node.input_map)

    final_node = by_id[run.plan.final_node]
    final_trace = by_trace.get(run.plan.final_node)
    if final_node.depends_on and not (final_trace and final_trace.dependency_values):
        problems.append(f"{final_node.id}: final trace has no resolved dependency_values")
    if not final_node.depends_on and len(run.plan.nodes) > 1:
        problems.append(f"{final_node.id}: final node has no dependencies despite multi-node plan")
    if not validation.valid:
        problems.extend(validation.errors)
    if "answer" not in final_node.output_schema.get("properties", {}):
        problems.append(f"{final_node.id}: final schema missing answer field")

    return {
        "valid": not problems,
        "problems": problems,
        "dependent_node_count": dependent_nodes,
        "mapped_edge_count": mapped_edges,
        "final_node": final_node.id,
        "final_depends_on": final_node.depends_on,
        "final_dependency_values": final_trace.dependency_values if final_trace else {},
        "node_summaries": [
            {
                "id": node.id,
                "label": node.label,
                "task_type": node.task_type.value,
                "operation": node.operation.value,
                "depends_on": node.depends_on,
                "input_map": node.input_map,
                "uses_child_values_in_template": (
                    _template_uses_child_values(node) if node.depends_on else None
                ),
                "output_fields": list(node.output_schema.get("properties", {}).keys()),
            }
            for node in run.plan.nodes
        ],
    }


def _summary(records: list[dict[str, Any]], started_total: float) -> dict[str, Any]:
    return {
        "completed": len(records),
        "success_count": sum(1 for record in records if record.get("status") == "succeeded"),
        "error_count": sum(1 for record in records if record.get("error")),
        "structurally_valid_count": sum(
            1 for record in records if record.get("structure", {}).get("valid")
        ),
        "exact_match": sum(record.get("exact_match", 0.0) for record in records) / len(records),
        "f1": sum(record.get("f1", 0.0) for record in records) / len(records),
        "avg_latency_ms": sum(record.get("latency_ms", 0.0) for record in records) / len(records),
        "total_llm_call_count": sum(record.get("llm_call_count") or 0 for record in records),
        "avg_node_count": sum(record.get("node_count") or 0 for record in records) / len(records),
        "total_runtime_ms": (time.perf_counter() - started_total) * 1000,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
