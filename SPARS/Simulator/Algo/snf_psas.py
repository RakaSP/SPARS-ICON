# snf_psas.py
import math
import re

from .base_psas import BasePSAS


_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


class SNFPSAS(BasePSAS):
    def __init__(
        self,
        machines,
        jobs_manager,
        start_time,
        timeout,
    ):
        super().__init__(
            machines,
            jobs_manager,
            start_time,
            timeout,
        )
        self.selected_list = []

    @staticmethod
    def _job_key(job):
        return (
            int(job["res"]),
            float(job["reqtime"]),
        )

    @staticmethod
    def _resolve_dvfs_power(
        machine,
        node,
        power_type,
    ):
        """
        Resolve idle or compute power from the node's current DVFS profile.

        Supports the legacy "power" field for older platform files.
        """
        if power_type == "idle":
            power_key = "power_idle"

        elif power_type == "compute":
            power_key = "power_compute"

        else:
            raise ValueError(
                "power_type must be either 'idle' or 'compute'"
            )

        dvfs_mode = node["dvfs_mode"]
        profile = machine["dvfs_profiles"][dvfs_mode]

        if power_key in profile:
            return float(profile[power_key])

        # Backward compatibility.
        if "power" in profile:
            return float(profile["power"])

        raise KeyError(
            f"DVFS profile {dvfs_mode!r} must define "
            f"{power_key!r} or legacy 'power'"
        )

    def schedule(self):
        super().prep_schedule()

        now = float(self.current_time)
        


        # 1. Current SNF commit.
        started_now = self._current_snf_commit()

        # 2. Future SNF plan.
        remaining = [
            job
            for job in self.waiting_queue
            if job["job_id"] not in started_now
        ]

        future_plan = self._future_snf_plan(
            remaining,
            barrier=now,
        )

        self.selected_list = list(future_plan)

        # 3. Wake callbacks.
        self._emit_wake_triggers_from_plan(
            self.selected_list
        )

        if self.timeout is not None:
            super().timeout_policy()

        super().build_callbacks()

        return self.events

    # ---------------- current SNF ----------------

    def _current_snf_commit(self):
        now = float(self.current_time)
        started_now = set()
        node_selection_static = self._build_node_selection_static_data(
            list(self.idle),
            self.next_releases,
        )

        for job in sorted(
            self.waiting_queue[:],
            key=self._job_key,
        ):
            required_nodes = int(job["res"])

            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive "
                    f"resource request: {required_nodes}"
                )

            if len(self.idle) < required_nodes:
                break

            result = self._select_nodes_energy_aware(
                required_nodes=required_nodes,
                _candidates=list(self.idle),
                min_start_time=now,
                node_static_data=node_selection_static,
            )

            if result is None:
                break

            nodes, start_time = result

            if float(start_time) > now + EPS:
                break

            super().allocate(job, nodes)
            started_now.add(job["job_id"])

        return started_now

    # ---------------- future SNF ----------------

    def _future_snf_plan(
        self,
        jobs,
        barrier,
    ):
        planned_releases = {
            node_id: {
                "queue": [
                    dict(segment)
                    for segment in entry["queue"]
                ],
                "release_time": float(
                    entry["release_time"]
                ),
            }
            for node_id, entry
            in self.next_releases.items()
        }

        def _append_planned_compute(
            job,
            selected_node_ids,
            job_start_time,
        ):
            compute_speed = min(
                float(
                    self.state[node_id]["compute_speed"]
                )
                for node_id in selected_node_ids
            )

            walltime = (
                float(job["reqtime"])
                / compute_speed
            )

            finish_time = (
                float(job_start_time)
                + walltime
            )

            phase = (
                f'compute(job={job["job_id"]})'
            )

            for node_id in selected_node_ids:
                entry = planned_releases[node_id]

                entry["queue"].append(
                    {
                        "phase": phase,
                        "start_time": float(
                            job_start_time
                        ),
                        "finish_time": float(
                            finish_time
                        ),
                    }
                )

                entry["release_time"] = float(
                    finish_time
                )

            return float(finish_time)

        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )
        node_selection_static = self._build_node_selection_static_data(
            candidates,
            planned_releases,
        )

        plan = []
        barrier = float(barrier)

        for job in sorted(
            jobs,
            key=self._job_key,
        ):
            required_nodes = int(job["res"])

            if required_nodes <= 0:
                continue

            result = self._select_nodes_energy_aware(
                required_nodes=required_nodes,
                _candidates=candidates,
                min_start_time=barrier,
                release_map=planned_releases,
                node_static_data=node_selection_static,
            )

            if result is None:
                break

            nodes, start_time = result

            finish_time = _append_planned_compute(
                job,
                nodes,
                float(start_time),
            )

            plan.append(
                (
                    job,
                    nodes,
                    float(start_time),
                    float(finish_time),
                )
            )

            barrier = float(start_time)

        return plan

    # ---------------- wake triggers from plan ----------------

    def _emit_wake_triggers_from_plan(
        self,
        plan,
    ):
        now = float(self.current_time)
        earliest_wake = {}

        for job, nodes, start_time, finish_time in plan:
            start_time = float(start_time)

            if start_time <= now + EPS:
                continue

            for node_id in nodes:
                if node_id not in self.sleeping:
                    continue

                lead_time = super()._wake_lead_time(
                    node_id
                )

                wake_time = (
                    start_time
                    - float(lead_time)
                )

                previous_time = earliest_wake.get(
                    node_id
                )

                if (
                    previous_time is None
                    or wake_time < previous_time
                ):
                    earliest_wake[node_id] = wake_time

        if not earliest_wake:
            return

        immediate = [
            node_id
            for node_id, wake_time
            in earliest_wake.items()
            if wake_time <= now + EPS
        ]

        future_times = sorted(
            {
                wake_time
                for wake_time
                in earliest_wake.values()
                if wake_time > now + EPS
            }
        )

        if immediate:
            self.push_event(
                now,
                {
                    "type": "switch_on",
                    "nodes": immediate,
                },
            )

            immediate_set = set(immediate)

            self.sleeping = [
                node_id
                for node_id in self.sleeping
                if node_id not in immediate_set
            ]

            self.switching_on.extend(
                [
                    node_id
                    for node_id in immediate
                    if node_id in self.state
                ]
            )

        for wake_time in future_times:
            self.push_event(
                float(wake_time),
                {
                    "type": "call_me_later_so"
                },
            )

    # ---------------- internals ----------------

    def _remaining_idle_timeout(
        self,
        node_id,
    ):
        if self.timeout is None:
            return math.inf

        timeout_time = self.timeout_list.get(
            node_id
        )

        if timeout_time is None:
            return math.inf

        return float(
            timeout_time - self.current_time
        )

    def _build_node_selection_static_data(
        self,
        candidates,
        release_map,
    ):
        current_time = float(self.current_time)
        static_data = {}

        for node_id in candidates:
            node_release = release_map.get(node_id)
            if node_release is None:
                continue

            release_time = float(node_release["release_time"])
            if math.isinf(release_time):
                continue

            node = self.state[node_id]
            node_name = self.machines.nodes[node_id]["node_name"]
            machine = self.machines.node_specs[node_name]

            if (
                node["state"] == "active"
                and node.get("job_id") is None
            ):
                state_label = "idle"
                state_priority = 0
            elif (
                node["state"] == "active"
                and node.get("job_id") is not None
            ):
                state_label = "computing"
                state_priority = 1
            elif node["state"] == "switching_on":
                state_label = "switching_on"
                state_priority = 2
            else:
                state_label = node["state"]
                state_priority = 3

            base_energy_waste = 0.0
            for queue_entry in node_release["queue"]:
                phase = str(queue_entry["phase"])
                if _COMPUTE_RE.fullmatch(phase):
                    continue

                start_time = float(queue_entry["start_time"])
                finish_time = float(queue_entry["finish_time"])
                duration = (
                    finish_time - current_time
                    if start_time < current_time
                    else finish_time - start_time
                )

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )
                base_energy_waste += float(energy_rate) * float(duration)

            idle_power = machine["states"]["active"]["power"]
            if idle_power == "from_dvfs":
                idle_power = self._resolve_dvfs_power(
                    machine=machine,
                    node=node,
                    power_type="idle",
                )

            static_data[node_id] = {
                "base": float(base_energy_waste),
                "idle": float(idle_power),
                "state_label": state_label,
                "state_priority": int(state_priority),
                "timeout_priority": (
                    -self._remaining_idle_timeout(node_id)
                    if state_label == "idle"
                    else 0.0
                ),
            }

        return static_data

    def _select_nodes_energy_aware(
        self,
        required_nodes,
        _candidates,
        min_start_time=None,
        release_map=None,
        node_static_data=None,
    ):
        if release_map is None:
            release_map = self.next_releases

        candidates = [
            node_id
            for node_id in _candidates
            if (
                node_id in release_map
                and not math.isinf(
                    float(release_map[node_id]["release_time"])
                )
            )
        ]
        if len(candidates) < required_nodes:
            return None

        min_start_time = (
            -math.inf
            if min_start_time is None
            else float(min_start_time)
        )

        if node_static_data is None:
            node_static_data = self._build_node_selection_static_data(
                candidates,
                release_map,
            )

        available_nodes = [
            node_id
            for node_id in candidates
            if node_id in node_static_data
        ]
        if len(available_nodes) < required_nodes:
            return None

        release_times = sorted(
            float(release_map[node_id]["release_time"])
            for node_id in available_nodes
        )
        candidate_time = max(
            min_start_time,
            release_times[required_nodes - 1],
        )

        eligible = []
        for node_id in available_nodes:
            release_time = float(
                release_map[node_id]["release_time"]
            )
            if release_time > candidate_time:
                continue

            data = node_static_data[node_id]
            if data["state_label"] in (
                "switching_off",
                "sleeping",
            ):
                cost = data["base"]
            else:
                cost = (
                    data["base"]
                    + data["idle"]
                    * (candidate_time - release_time)
                )

            eligible.append(
                (
                    float(cost),
                    data["state_priority"],
                    data["timeout_priority"],
                    node_id,
                )
            )

        if len(eligible) < required_nodes:
            return None

        selected_nodes = [
            item[3]
            for item in sorted(eligible)[:required_nodes]
        ]
        return selected_nodes, float(candidate_time)
