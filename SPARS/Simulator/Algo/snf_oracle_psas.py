from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Iterable
import math

EPS = 1e-9


@dataclass(frozen=True)
class NodeUse:
    job: dict[str, Any]
    start: float
    finish: float


@dataclass(frozen=True)
class IdleDecision:
    mode: str
    energy: float
    switch_on_at: float | None = None
    off_duration: float | None = None
    on_duration: float | None = None


class SNFOraclePSAS:
    def __init__(
        self,
        machines,
        jobs_manager,
        start_time,
        workload,
        platform_control,
    ):
        if workload is None:
            raise ValueError(
                "SNFOraclePSAS requires the complete workload"
            )

        if platform_control is None:
            raise ValueError(
                "SNFOraclePSAS requires platform_control"
            )

        self.machines = machines
        self.jobs_manager = jobs_manager
        self.state = machines.nodes
        self.start_time = float(start_time)
        self.current_time = float(start_time)
        self.workload = workload
        self.platform_control = platform_control
        self.overrun_policy = platform_control.overrun_policy

        self._oracle_emitted = False
        self._off_duration_cache: dict[
            tuple[int, int], float
        ] = {}
        self._on_duration_cache: dict[
            tuple[int, int], float
        ] = {}

    def set_time(self, current_time):
        self.current_time = float(current_time)

    def schedule(self):
        if self._oracle_emitted:
            return []

        self._oracle_emitted = True
        plan = self._build_complete_plan()

        return [
            {
                "timestamp": float(self.current_time),
                "events": [
                    {
                        "type": "oracle",
                        "plan": plan,
                    }
                ],
            }
        ]

    def _effective_runtime(self, job):
        runtime = float(job["runtime"])
        reqtime = float(job["reqtime"])

        if self.overrun_policy == "continue":
            return runtime

        if self.overrun_policy == "terminate":
            return min(runtime, reqtime)

        raise ValueError(
            f"Unsupported overrun policy: "
            f"{self.overrun_policy!r}"
        )

    def _job_key(self, job):
        return (
            int(job["res"]),
            self._effective_runtime(job),
            int(job["job_id"]),
        )

    def _build_complete_plan(self):
        planning_start = float(self.current_time)

        if self.jobs_manager.active_jobs:
            raise RuntimeError(
                "SNFOraclePSAS must build its plan "
                "before any job starts"
            )

        jobs = []

        for raw_job in self.workload["jobs"]:
            job = dict(raw_job)

            # Private arrival timestamp used only by oracle planning.
            # The arrival event itself is not added to the oracle plan.
            job["subtime"] = (
                self.start_time
                + float(raw_job["subtime"])
            )

            jobs.append(job)

        jobs.sort(
            key=lambda job: (
                float(job["subtime"]),
                int(job["job_id"]),
            )
        )

        node_uses, execution_events = (
            self._build_compute_schedule(
                jobs=jobs,
                planning_start=planning_start,
            )
        )

        power_events = self._build_power_events(
            node_uses=node_uses,
            planning_start=planning_start,
        )

        plan = execution_events + power_events
        plan.sort(key=self._planned_event_key)

        return plan

    def _build_compute_schedule(
        self,
        jobs,
        planning_start,
    ):
        node_ids = sorted(
            int(node_id)
            for node_id in self.state
        )

        for node_id in node_ids:
            node = self.state[node_id]

            if (
                node["state"] != "active"
                or node.get("job_id") is not None
            ):
                raise RuntimeError(
                    "SNFOraclePSAS must build its plan "
                    "while all nodes are active and idle"
                )

        free_nodes = set(node_ids)

        node_idle_since = {
            node_id: float(planning_start)
            for node_id in node_ids
        }

        node_episode = {
            node_id: 0
            for node_id in node_ids
        }

        node_uses = {
            node_id: []
            for node_id in node_ids
        }

        waiting = []
        arrival_index = 0
        active_heap = []
        serial = 0
        execution_events = []
        now = float(planning_start)

        while (
            arrival_index < len(jobs)
            or waiting
            or active_heap
        ):
            while (
                active_heap
                and active_heap[0][0] <= now + EPS
            ):
                finish_time, _, nodes = heapq.heappop(
                    active_heap
                )

                for node_id in nodes:
                    free_nodes.add(node_id)
                    node_idle_since[node_id] = float(
                        finish_time
                    )
                    node_episode[node_id] += 1

            while (
                arrival_index < len(jobs)
                and float(
                    jobs[arrival_index]["subtime"]
                ) <= now + EPS
            ):
                waiting.append(jobs[arrival_index])
                arrival_index += 1

            started_any = False

            while waiting:
                waiting.sort(key=self._job_key)
                job = waiting[0]
                required_nodes = int(job["res"])

                if required_nodes <= 0:
                    raise RuntimeError(
                        f"Job {job['job_id']} has "
                        f"invalid res={required_nodes}"
                    )

                if required_nodes > len(node_ids):
                    raise RuntimeError(
                        f"Job {job['job_id']} requests "
                        f"{required_nodes} nodes, but only "
                        f"{len(node_ids)} exist"
                    )

                if required_nodes > len(free_nodes):
                    break

                selected_nodes = (
                    self._select_nodes_for_job(
                        job=job,
                        candidate_nodes=free_nodes,
                        job_start=now,
                        node_idle_since=node_idle_since,
                        node_episode=node_episode,
                    )
                )

                compute_speed = min(
                    float(
                        self.state[node_id][
                            "compute_speed"
                        ]
                    )
                    for node_id in selected_nodes
                )

                if compute_speed <= 0:
                    raise RuntimeError(
                        "compute_speed must be "
                        "greater than zero"
                    )

                finish_time = (
                    now
                    + self._effective_runtime(job)
                    / compute_speed
                )

                execution_event = dict(job)
                execution_event["type"] = (
                    "execution_start"
                )
                execution_event["nodes"] = list(
                    selected_nodes
                )

                execution_events.append(
                    {
                        "timestamp": float(now),
                        "event": execution_event,
                    }
                )

                for node_id in selected_nodes:
                    free_nodes.remove(node_id)

                    node_uses[node_id].append(
                        NodeUse(
                            job=dict(job),
                            start=float(now),
                            finish=float(finish_time),
                        )
                    )

                serial += 1

                heapq.heappush(
                    active_heap,
                    (
                        float(finish_time),
                        serial,
                        tuple(selected_nodes),
                    ),
                )

                waiting.pop(0)
                started_any = True

            if started_any:
                continue

            next_times = []

            if arrival_index < len(jobs):
                next_times.append(
                    float(
                        jobs[arrival_index]["subtime"]
                    )
                )

            if active_heap:
                next_times.append(
                    float(active_heap[0][0])
                )

            if not next_times:
                if waiting:
                    raise RuntimeError(
                        "Oracle planning stopped "
                        "with unscheduled jobs"
                    )

                break

            next_time = min(next_times)

            if next_time < now - EPS:
                raise RuntimeError(
                    "Oracle virtual time moved backward"
                )

            now = float(next_time)

        return node_uses, execution_events

    def _select_nodes_for_job(
        self,
        job,
        candidate_nodes: Iterable[int],
        job_start,
        node_idle_since,
        node_episode,
    ):
        ranked = []

        for node_id in candidate_nodes:
            decision = self._idle_decision(
                node_id=int(node_id),
                idle_at=float(
                    node_idle_since[node_id]
                ),
                next_use_at=float(job_start),
                episode_id=int(
                    node_episode[node_id]
                ),
            )

            ranked.append(
                (
                    float(decision.energy),
                    int(node_id),
                )
            )

        ranked.sort()

        required_nodes = int(job["res"])

        return [
            node_id
            for _, node_id
            in ranked[:required_nodes]
        ]

    def _sample_transition(
        self,
        node_id,
        from_state,
        to_state,
    ):
        node_name = self.state[node_id][
            "node_name"
        ]

        transition = (
            self.machines.transition_map[
                node_name
            ][
                (from_state, to_state)
            ]
        )

        return float(
            self.platform_control
            .sample_transition_time(transition)
        )

    def _off_duration(
        self,
        node_id,
        episode_id,
    ):
        key = (
            int(node_id),
            int(episode_id),
        )

        if key not in self._off_duration_cache:
            self._off_duration_cache[key] = (
                self._sample_transition(
                    node_id,
                    "active",
                    "switching_off",
                )
                + self._sample_transition(
                    node_id,
                    "switching_off",
                    "sleeping",
                )
            )

        return self._off_duration_cache[key]

    def _on_duration(
        self,
        node_id,
        episode_id,
    ):
        key = (
            int(node_id),
            int(episode_id),
        )

        if key not in self._on_duration_cache:
            self._on_duration_cache[key] = (
                self._sample_transition(
                    node_id,
                    "sleeping",
                    "switching_on",
                )
                + self._sample_transition(
                    node_id,
                    "switching_on",
                    "active",
                )
            )

        return self._on_duration_cache[key]

    def _idle_decision(self, node_id, idle_at, next_use_at, episode_id):
        gap = max(0.0, float(next_use_at) - float(idle_at))
        idle_power = self._state_power(node_id, "active")
        keep_energy = idle_power * gap

        if gap <= 0.0:
            return IdleDecision(mode="idle", energy=keep_energy)

        off_duration = self._off_duration(node_id, episode_id)
        on_duration = self._on_duration(node_id, episode_id)

        if off_duration + on_duration > gap:
            return IdleDecision(mode="idle", energy=keep_energy)

        sleeping_duration = max(0.0, gap - off_duration - on_duration)
        sleep_energy = (
            self._state_power(node_id, "switching_off") * off_duration
            + self._state_power(node_id, "sleeping") * sleeping_duration
            + self._state_power(node_id, "switching_on") * on_duration
        )

        if sleep_energy >= keep_energy:
            return IdleDecision(mode="idle", energy=keep_energy)

        switch_on_at = float(next_use_at) - float(on_duration)
        while switch_on_at + float(on_duration) > float(next_use_at):
            switch_on_at = math.nextafter(switch_on_at, -math.inf)

        if switch_on_at < float(idle_at) + float(off_duration):
            return IdleDecision(mode="idle", energy=keep_energy)

        return IdleDecision(
            mode="sleep",
            energy=float(sleep_energy),
            switch_on_at=float(switch_on_at),
            off_duration=float(off_duration),
            on_duration=float(on_duration),
        )

    def _build_power_events(
        self,
        node_uses,
        planning_start,
    ):
        events = []

        for node_id in sorted(node_uses):
            uses = sorted(
                node_uses[node_id],
                key=lambda use: use.start,
            )

            idle_at = float(planning_start)
            episode_id = 0

            for use in uses:
                decision = self._idle_decision(
                    node_id=node_id,
                    idle_at=idle_at,
                    next_use_at=float(use.start),
                    episode_id=episode_id,
                )

                if decision.mode == "sleep":
                    events.append(
                        self._power_event(
                            timestamp=idle_at,
                            event_type="switch_off",
                            node_id=node_id,
                            duration=(
                                decision.off_duration
                            ),
                        )
                    )

                    events.append(
                        self._power_event(
                            timestamp=(
                                decision.switch_on_at
                            ),
                            event_type="switch_on",
                            node_id=node_id,
                            duration=(
                                decision.on_duration
                            ),
                        )
                    )

                idle_at = float(use.finish)
                episode_id += 1

            events.append(
                self._power_event(
                    timestamp=idle_at,
                    event_type="switch_off",
                    node_id=node_id,
                    duration=self._off_duration(
                        node_id,
                        episode_id,
                    ),
                )
            )

        return events

    @staticmethod
    def _power_event(
        timestamp,
        event_type,
        node_id,
        duration,
    ):
        return {
            "timestamp": float(timestamp),
            "event": {
                "type": event_type,
                "nodes": [int(node_id)],
                "oracle_durations": {
                    int(node_id): float(duration),
                },
            },
        }

    def _state_power(
        self,
        node_id,
        state_name,
    ):
        node = self.state[node_id]

        machine = self.machines.node_specs[
            node["node_name"]
        ]

        power = machine["states"][
            state_name
        ]["power"]

        if power != "from_dvfs":
            return float(power)

        profile = machine["dvfs_profiles"][
            node["dvfs_mode"]
        ]

        if "power_idle" in profile:
            return float(
                profile["power_idle"]
            )

        if "power" in profile:
            return float(profile["power"])

        raise KeyError(
            f"DVFS profile "
            f"{node['dvfs_mode']!r} "
            f"lacks power_idle"
        )

    @staticmethod
    def _planned_event_key(item):
        event_type = item["event"]["type"]

        priority = {
            "execution_start": 0,
            "switch_off": 1,
            "switch_on": 2,
        }.get(event_type, 100)

        return (
            float(item["timestamp"]),
            priority,
        )