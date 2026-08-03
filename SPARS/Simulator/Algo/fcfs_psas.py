import heapq
# fcfs_psas.py
import math
import re
from .base_psas import BasePSAS

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


class FCFSPSAS(BasePSAS):
    def __init__(self, machines, jobs_manager, start_time, timeout):
        super().__init__(machines, jobs_manager, start_time, timeout)
        self.selected_list = []

    @staticmethod
    def _resolve_dvfs_power(machine, node, power_type):
        if power_type == "idle":
            power_key = "power_idle"
        elif power_type == "compute":
            power_key = "power_compute"
        else:
            raise ValueError("power_type must be either 'idle' or 'compute'")

        dvfs_mode = node["dvfs_mode"]
        profile = machine["dvfs_profiles"][dvfs_mode]

        if power_key in profile:
            return float(profile[power_key])

        # Backward compatibility with old platform files.
        if "power" in profile:
            return float(profile["power"])

        raise KeyError(
            f"DVFS profile {dvfs_mode!r} must define "
            f"{power_key!r} or legacy 'power'"
        )

    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        # 1) current FCFS commit
        started_now = self._current_fcfs_commit()

        # 2) future FCFS plan
        remaining = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        future_plan = self._future_fcfs_plan(remaining, barrier=now)

        self.selected_list = list(future_plan)

        # 3) wake callbacks
        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    def _select_nodes_energy_aware_prepared(
        self,
        required_nodes,
        candidates,
        release_times,
        min_start_time,
        node_static_data,
    ):
        """Select nodes using prevalidated candidates and float releases."""
        if len(candidates) < required_nodes:
            return None

        candidate_time = max(
            float(min_start_time),
            heapq.nsmallest(
                required_nodes,
                (release_times[nid] for nid in candidates),
            )[-1],
        )

        eligible = []
        for node_id in candidates:
            release_time = release_times[node_id]
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

            eligible.append((
                float(cost),
                data["state_priority"],
                data["timeout_priority"],
                node_id,
            ))

        if len(eligible) < required_nodes:
            return None

        selected_nodes = [
            item[3]
            for item in heapq.nsmallest(
                required_nodes,
                eligible,
            )
        ]
        return selected_nodes, float(candidate_time)

    def _current_fcfs_commit(self):
        now = float(self.current_time)
        started_now = set()
        idle_candidates = list(self.idle)
        node_selection_static = self._build_node_selection_static_data(
            idle_candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in idle_candidates
            if nid in node_selection_static
        }
        idle_candidates = [
            nid for nid in idle_candidates if nid in release_times
        ]

        for job in self.waiting_queue:
            req = int(job["res"])
            if req <= 0:
                continue

            # FCFS: if head cannot start now, stop.
            if len(idle_candidates) < req:
                break

            res = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=idle_candidates,
                release_times=release_times,
                min_start_time=now,
                node_static_data=node_selection_static,
            )
            if res is None:
                break

            nodes, st = res
            if float(st) > now + EPS:
                break

            super().allocate(job, nodes)
            started_now.add(job["job_id"])

            selected_set = set(nodes)
            idle_candidates = [
                nid
                for nid in idle_candidates
                if nid not in selected_set
            ]

        return started_now

    def _future_fcfs_plan(self, jobs, barrier):
        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        node_selection_static = self._build_node_selection_static_data(
            candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in candidates
            if nid in node_selection_static
        }
        candidates = [
            nid for nid in candidates if nid in release_times
        ]

        plan = []
        barrier = float(barrier)
        state = self.state

        for job in jobs:
            req = int(job["res"])
            if req <= 0:
                continue

            res = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=candidates,
                release_times=release_times,
                min_start_time=barrier,
                node_static_data=node_selection_static,
            )
            if res is None:
                break

            nodes, st = res
            st = float(st)
            compute_speed = min(
                float(state[nid]["compute_speed"])
                for nid in nodes
            )
            ft = st + (float(job["reqtime"]) / compute_speed)
            plan.append((job, nodes, st, float(ft)))

            barrier = st

        return plan

    def _emit_wake_triggers_from_plan(self, plan):
        now = float(self.current_time)

        sleeping_set = set(self.sleeping)
        wake_lead_by_node = {}
        earliest_wake = {}
        for job, nodes, st, ft in plan:
            st = float(st)
            if st <= now + EPS:
                continue
            for nid in nodes:
                if nid not in sleeping_set:
                    continue
                lead = wake_lead_by_node.get(nid)
                if lead is None:
                    lead = float(super()._wake_lead_time(nid))
                    wake_lead_by_node[nid] = lead
                wake_time = st - lead
                prev = earliest_wake.get(nid)
                if prev is None or wake_time < prev:
                    earliest_wake[nid] = wake_time

        if not earliest_wake:
            return

        immediate = [nid for nid, t in earliest_wake.items() if t <= now + EPS]
        future_times = sorted({t for nid, t in earliest_wake.items() if t > now + EPS})

        if immediate:
            self.push_event(now, {"type": "switch_on", "nodes": immediate})
            imm_set = set(immediate)
            self.sleeping = [n for n in self.sleeping if n not in imm_set]
            self.switching_on.extend([nid for nid in immediate if nid in self.state])

        for t in future_times:
            self.push_event(float(t), {"type": "call_me_later_so"})

    def _remaining_idle_timeout(self, node_id):
        if self.timeout is None:
            return math.inf
        t = self.timeout_list.get(node_id)
        return float(t - self.current_time) if t is not None else math.inf

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
                if start_time < current_time:
                    duration = finish_time - current_time
                else:
                    duration = finish_time - start_time

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )

                base_energy_waste += (
                    float(energy_rate) * float(duration)
                )

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

        if min_start_time is None:
            min_start_time = -math.inf
        else:
            min_start_time = float(min_start_time)

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

        earliest_releases = heapq.nsmallest(
            required_nodes,
            (
                float(release_map[node_id]["release_time"])
                for node_id in available_nodes
            ),
        )
        candidate_time = max(
            min_start_time,
            earliest_releases[-1],
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
            for item in heapq.nsmallest(
                required_nodes,
                eligible,
            )
        ]
        return selected_nodes, float(candidate_time)
