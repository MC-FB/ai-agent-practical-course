from __future__ import annotations

from dagqa.planning.validator import scheduler_waves
from dagqa.schemas import DagPlan, SchedulerWave


class Scheduler:
    def build_waves(self, plan: DagPlan) -> list[SchedulerWave]:
        return [
            SchedulerWave(index=index + 1, node_ids=node_ids)
            for index, node_ids in enumerate(scheduler_waves(plan))
        ]
