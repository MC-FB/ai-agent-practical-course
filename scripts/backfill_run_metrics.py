"""Backfill newly registered answer metrics into saved benchmark run files.

Runs recorded before a metric joined ``ANSWER_METRICS`` lack its per-record
``metric_scores`` entry and its aggregate value. Predictions, gold answers, and
questions are stored per record, so every answer metric can be recomputed
offline without re-running the LLMs. Mirrors the lazy-fill pattern of
``dagqa.eval.metric_eval.fill_scores``: only missing scores are computed, so a
second pass over an already-backfilled file is a no-op.

Run (from the repo root):
    python scripts/backfill_run_metrics.py
    python scripts/backfill_run_metrics.py "runs/benchmarks/hotpotqa-20260713*.json"

With no arguments, every run in ``runs/benchmarks/`` that has a
``comparison_group_id`` is backfilled (those are the runs shown in the
multi-model comparison table).
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

from dagqa.eval.benchmark import BenchmarkRecord, _aggregate
from dagqa.eval.metrics import ANSWER_METRICS

BENCHMARK_DIR = Path("runs/benchmarks")


def _legacy_scores(record: dict[str, Any]) -> dict[str, float]:
    """Seed the scores dict from legacy top-level keys (pre-``metric_scores`` files)."""
    return {
        metric.name: record[metric.name]
        for metric in ANSWER_METRICS
        if record.get(metric.name) is not None
    }


def _fill_record_scores(record: dict[str, Any]) -> int:
    """Compute any missing metric scores for one record dict; returns how many were added."""
    scores: dict[str, float] = dict(record.get("metric_scores") or _legacy_scores(record))
    missing = [metric for metric in ANSWER_METRICS if metric.name not in scores]
    for metric in missing:
        scores[metric.name] = metric.score(
            record.get("prediction") or "",
            record.get("gold_answer") or "",
            record.get("question") or "",
        )
    record["metric_scores"] = scores
    return len(missing)


def backfill_file(path: Path, data: dict[str, Any]) -> int:
    """Fill missing scores in one loaded run file; returns the number computed."""
    records = data.get("records") or []
    filled = sum(_fill_record_scores(record) for record in records)
    if not filled:
        return 0
    validated = [BenchmarkRecord.model_validate(record) for record in records]
    # Merge with stored values winning: re-aggregating old records with today's
    # aggregation rules can shift historical values (e.g. legacy -1.5 cosine
    # error sentinels), so only keys the file never had are added.
    data["metrics"] = {**_aggregate(validated), **(data.get("metrics") or {})}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return filled


def _expand_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns in-script so the command also works from PowerShell."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if not matches:
            raise SystemExit(f"No files match {pattern!r}")
        paths.extend(Path(match) for match in matches)
    return paths


def _load_run(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("records"):
        return None
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Run files or glob patterns (default: every run with a comparison_group_id)",
    )
    args = parser.parse_args()

    default_mode = not args.patterns
    paths = sorted(BENCHMARK_DIR.glob("*.json")) if default_mode else _expand_paths(args.patterns)

    total_filled = 0
    backfilled_files = 0
    for path in paths:
        data = _load_run(path)
        if data is None:
            continue
        if default_mode and not data.get("comparison_group_id"):
            continue
        filled = backfill_file(path, data)
        if filled:
            backfilled_files += 1
            print(f"{path.name}: computed {filled} scores across {len(data['records'])} records")
        total_filled += filled

    print(f"\nBackfilled {total_filled} scores in {backfilled_files} files")


if __name__ == "__main__":
    main()
