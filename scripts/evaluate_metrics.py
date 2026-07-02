"""Rank answer metrics by agreement with human-annotated score bands.

Reads the annotation store populated from the UI (or an imported CSV), prints a
markdown leaderboard, and writes ``.annotations/evaluation.json`` next to the store.

Run:
    python scripts/evaluate_metrics.py
"""

from __future__ import annotations

import json

from app.api import _benchmark_output_dir, _load_annotation_records
from dagqa.eval.metric_eval import evaluate_metrics


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    records = _load_annotation_records()
    annotated = [record for record in records if record.band is not None]
    results = evaluate_metrics(records)

    print(f"Annotation records: {len(records)} ({len(annotated)} with a score band)\n")
    if not results:
        print("No annotated records with a score band yet — nothing to evaluate.")
        return

    print("| Metric | In-range | 95% CI | Center dist | Spearman | Kendall | n |")
    print("|---|---|---|---|---|---|---|")
    for result in results:
        print(
            f"| {result.name} | {result.in_range_rate:.3f} "
            f"| [{result.ci_low:.3f}, {result.ci_high:.3f}] "
            f"| {result.mean_center_distance:.3f} "
            f"| {_fmt(result.spearman)} | {_fmt(result.kendall)} | {result.n} |"
        )

    output_path = _benchmark_output_dir() / ".annotations" / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "total": len(records),
                "annotated": len(annotated),
                "results": [result.model_dump(mode="json") for result in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
