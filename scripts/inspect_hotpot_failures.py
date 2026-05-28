from __future__ import annotations

import asyncio

from dagqa.client import DagQaClient
from dagqa.config import AppConfig


QUESTIONS = [
    "Where has Kerwin Swint's work appeared that draws 20.5 million unique visitors a month?",
    (
        "At what university can the building that served as the fictional household that "
        "includes Gomez and Morticia be found?"
    ),
    (
        "What census-designated place located in Nassau County, New York has as portions of "
        "it's territory, districts of Brian Curran, New York State assemblyman?"
    ),
]


async def main() -> None:
    client = DagQaClient(AppConfig.from_file("configs/azure-openai.yaml"))
    for question in QUESTIONS:
        print(f"\nQUESTION {question}", flush=True)
        run = await client.ask(question)
        print(f"RUN STATUS {run.status.value} FINAL {run.final_answer}", flush=True)
        print("PLAN", flush=True)
        for node in run.plan.nodes:
            fields = list(node.output_schema.get("properties", {}).keys())
            print(
                node.id,
                node.operation.value,
                "depends",
                node.depends_on,
                "input_map",
                node.input_map,
                "schema",
                fields,
                flush=True,
            )
        print("TRACES", flush=True)
        for trace in run.nodes:
            raw = (trace.raw_response or "")[:500].replace("\n", " ")
            print(
                "TRACE",
                trace.node_id,
                trace.status.value,
                "deps",
                trace.dependency_values,
                "parsed",
                trace.parsed_output,
                "validation",
                trace.validation.model_dump(),
                "error",
                trace.error,
                flush=True,
            )
            if raw:
                print("RAW", raw, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
