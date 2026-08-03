from __future__ import annotations

import csv
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .base_psas import BasePSAS
from .snf_psas import SNFPSAS


_COMPUTE_RE = re.compile(r"^compute\(job=(\d+)\)$")
EPS = 1e-9


class SNFICON(SNFPSAS):
    """Shortest-node-first scheduler with Markov spare-node control.

    ``alpha`` and ``beta`` must put seconds and joules on comparable scales.
    For example, if one second of waiting is considered equivalent to 50 kJ,
    use ``alpha=1`` and ``beta=1 / 50_000``.
    """

    def __init__(
        self,
        machines,
        jobs_manager,
        start_time,
        monitor,
        timeout,
        alpha,
        beta,
        arrival_rate,
        service_rate,
        resource_pmf,
        min_spare_nodes,
        max_spare_nodes,
        initial_arrival_rate,
        arrival_ema_alpha,
        service_ema_alpha,
        resource_ema_alpha,
        runtime_log_sigma_floor,
        max_decision_horizon,
        spare_arrival_window,
        immediate_shutdown_on_stale_completion,
        respect_timeout,
        markov_fallback_enabled,
        markov_window_seconds,
        markov_min_samples,
        markov_max_samples,
        markov_cv_tolerance,
        markov_max_lag1_autocorrelation,
        markov_ks_multiplier,
        markov_require_service_samples,
        markov_ks_param,
        decision_log_enabled,
        decision_log_path,
    ):
        super().__init__(
            machines,
            jobs_manager,
            start_time,
            timeout,
        )

        if monitor is None:
            raise ValueError("monitor must not be None")
        self.monitor = monitor

        self.alpha = self._nonnegative_float("alpha", alpha)
        self.beta = self._nonnegative_float("beta", beta)
        if self.alpha <= 0.0 and self.beta <= 0.0:
            raise ValueError("At least one of alpha and beta must be positive")

        self._configured_arrival_rate = self._optional_positive_float(
            "arrival_rate", arrival_rate
        )
        self._configured_service_rate = self._optional_positive_float(
            "service_rate", service_rate
        )
        self.initial_arrival_rate = self._positive_float(
            "initial_arrival_rate", initial_arrival_rate
        )
        self.arrival_ema_alpha = self._unit_interval_float(
            "arrival_ema_alpha", arrival_ema_alpha
        )
        self.service_ema_alpha = self._unit_interval_float(
            "service_ema_alpha", service_ema_alpha
        )
        self.resource_ema_alpha = self._unit_interval_float(
            "resource_ema_alpha", resource_ema_alpha
        )
        self.runtime_log_sigma_floor = self._nonnegative_float(
            "runtime_log_sigma_floor", runtime_log_sigma_floor
        )
        self.max_decision_horizon = self._positive_float(
            "max_decision_horizon", max_decision_horizon
        )
        self.spare_arrival_window = self._optional_nonnegative_float(
            "spare_arrival_window", spare_arrival_window
        )
        self.immediate_shutdown_on_stale_completion = bool(
            immediate_shutdown_on_stale_completion
        )
        
        self.markov_ks_param = markov_ks_param

        self.markov_fallback_enabled = bool(markov_fallback_enabled)
        self.markov_window_seconds = self._positive_float(
            "markov_window_seconds", markov_window_seconds
        )
        self.markov_min_samples = int(markov_min_samples)
        self.markov_max_samples = int(markov_max_samples)
        if self.markov_min_samples < 3:
            raise ValueError("markov_min_samples must be at least 3")
        if self.markov_max_samples < self.markov_min_samples:
            raise ValueError(
                "markov_max_samples must be greater than or equal to "
                "markov_min_samples"
            )
        self.markov_cv_tolerance = self._nonnegative_float(
            "markov_cv_tolerance", markov_cv_tolerance
        )
        self.markov_max_lag1_autocorrelation = self._nonnegative_float(
            "markov_max_lag1_autocorrelation",
            markov_max_lag1_autocorrelation,
        )
        if self.markov_max_lag1_autocorrelation > 1.0:
            raise ValueError(
                "markov_max_lag1_autocorrelation must not exceed 1"
            )
        self.markov_ks_multiplier = self._positive_float(
            "markov_ks_multiplier", markov_ks_multiplier
        )
        self.markov_require_service_samples = bool(
            markov_require_service_samples
        )

        total_nodes = len(self.state)
        requested_min_spare_nodes = int(min_spare_nodes)
        if requested_min_spare_nodes < 0:
            raise ValueError("min_spare_nodes must be non-negative")
        self.min_spare_nodes = min(requested_min_spare_nodes, total_nodes)
        self.respect_timeout = bool(respect_timeout)

        if max_spare_nodes is None:
            self.max_spare_nodes = total_nodes
        else:
            requested_max_spare_nodes = int(max_spare_nodes)
            if requested_max_spare_nodes < 0:
                raise ValueError("max_spare_nodes must be non-negative")
            if requested_max_spare_nodes < requested_min_spare_nodes:
                raise ValueError(
                    "max_spare_nodes must be greater than or equal to "
                    "min_spare_nodes"
                )
            self.max_spare_nodes = min(
                requested_max_spare_nodes,
                total_nodes,
            )

        self._fixed_resource_pmf = self._normalise_pmf(resource_pmf)

        self.arrives_count = 0
        self.completion_count = 0

        self._last_arrival_event_time: Optional[float] = None
        self._ema_interarrival = 1.0 / self.initial_arrival_rate
        self._ema_requested_service_time: Optional[float] = None
        self._ema_service_time: Optional[float] = None

        self._recent_interarrivals = deque(maxlen=self.markov_max_samples)
        self._recent_service_times = deque(maxlen=self.markov_max_samples)

        self._ema_log_runtime_ratio: Optional[float] = None
        self._ema_log_runtime_ratio_sq: Optional[float] = None
        self._runtime_ratio_samples = 0

        self._resource_weights: Dict[int, float] = {}
        self._scheduled_wake_callbacks = set()

        # Local visibility for jobs allocated during the current schedule call.
        self._started_this_schedule: Dict[int, Dict[str, object]] = {}

        self._first_schedule = True
        self.selected_list = []
        self.last_decision: Dict[str, object] = {}

        self.decision_log_enabled = bool(decision_log_enabled)
        self.decision_log_path = self._resolve_decision_log_path(
            decision_log_path
        )

        if (
            self.decision_log_enabled
            and self.decision_log_path is None
        ):
            raise ValueError(
                "decision_log_enabled=True, but decision_log_path "
                "was not configured"
            )

        self._decision_log_index = 0
        self._previous_logged_fallback: Optional[bool] = None
        self._previous_logged_gate: Optional[bool] = None
        if self.decision_log_enabled and self.decision_log_path is not None:
            self._initialise_decision_log()

        self._previous_node_jobs = self._node_job_snapshot()

    # ------------------------------------------------------------------
    # Public scheduling entry point
    # ------------------------------------------------------------------

    def prep_schedule(self):
        super().prep_schedule()

        arrival_log = self._monitor_log("jobs_arrival_log")
        completion_log = self._monitor_log("jobs_execution_log")

        arrival_total = len(arrival_log)
        completion_total = len(completion_log)

        if arrival_total < self.arrives_count:
            raise RuntimeError(
                "monitor.jobs_arrival_log shrank from "
                f"{self.arrives_count} to {arrival_total}"
            )
        if completion_total < self.completion_count:
            raise RuntimeError(
                "monitor.jobs_execution_log shrank from "
                f"{self.completion_count} to {completion_total}"
            )

        new_arrival_count = arrival_total - self.arrives_count
        new_completion_count = completion_total - self.completion_count

        # Save the new records for diagnostics
        new_arrival_records = list(arrival_log[self.arrives_count:arrival_total])
        new_completion_records = list(
            completion_log[self.completion_count:completion_total]
        )

        self.arrives_count = arrival_total
        self.completion_count = completion_total

        return {
            "new_arrival": new_arrival_count > 0,
            "new_completion": new_completion_count > 0,
            "new_arrival_count": new_arrival_count,
            "new_completion_count": new_completion_count,
            "new_arrival_records": new_arrival_records,
            "new_completion_records": new_completion_records,
        }

    def schedule(self):
        monitor_events = self.prep_schedule()
        now = float(self.current_time)
        released_by_completion = self._nodes_released_since_last_schedule()

        new_arrival_records = list(monitor_events["new_arrival_records"])
        new_completion_records = list(monitor_events["new_completion_records"])
        self._observe_arrivals(new_arrival_records, now)
        self._observe_completions(new_completion_records)

        markovianity = self._recent_markovianity(now)
        if (
            self.markov_fallback_enabled
            and not markovianity["well_markovian"]
        ):
            return self._schedule_snf_fallback(
                monitor_events=monitor_events,
                markovianity=markovianity,
                released_by_completion=released_by_completion,
            )

        decision_event = (
            self._first_schedule
            or bool(monitor_events["new_arrival"])
            or bool(monitor_events["new_completion"])
        )

        self._started_this_schedule = {}
        started_now, allocated_now = self._current_snf_commit()
        remaining_jobs = [
            job
            for job in self.waiting_queue
            if job.get("job_id") not in started_now
        ]

        mandatory_wake, mandatory_diag = self._ensure_waiting_capacity(
            remaining_jobs
        )

        spare_arrival_gate = self._spare_arrival_gate(now)
        spare_diag = None
        immediate_completion_shutdown = None

        if not remaining_jobs and decision_event:
            if spare_arrival_gate["allowed"]:
                spare_diag = self._optimise_and_apply_spare_target(
                    protected_nodes=set(mandatory_wake)
                )
            else:
                if (
                    self.immediate_shutdown_on_stale_completion
                    and bool(monitor_events["new_completion"])
                ):
                    immediate_completion_shutdown = (
                        self._switch_off_stale_completion_nodes(
                            released_nodes=released_by_completion,
                            allocated_nodes=allocated_now,
                            protected_nodes=set(mandatory_wake),
                        )
                    )

                immediate_nodes = (
                    immediate_completion_shutdown["switch_off"]
                    if immediate_completion_shutdown is not None
                    else []
                )
                spare_diag = {
                    "suppressed": True,
                    "reason": "no_recent_job_arrival",
                    "target": len(self.idle) + len(self.switching_on),
                    "wake": [],
                    "switch_off": list(immediate_nodes),
                    "arrival_age": spare_arrival_gate["arrival_age"],
                    "arrival_window": spare_arrival_gate["arrival_window"],
                    "time": now,
                }

        spare_timeout_diag = None
        if not remaining_jobs and self.timeout is not None:
            spare_timeout_diag = self._apply_spare_timeout_policy()

        self.last_decision = {
            "time": now,
            "new_arrival": bool(monitor_events["new_arrival"]),
            "new_completion": bool(monitor_events["new_completion"]),
            "new_arrival_count": monitor_events["new_arrival_count"],
            "new_completion_count": monitor_events["new_completion_count"],
            "new_arrivals": [
                job.get("job_id") for job in new_arrival_records
            ],
            "started_now": sorted(started_now),
            "remaining_queue": [
                job.get("job_id") for job in remaining_jobs
            ],
            "released_by_completion": list(released_by_completion),
            "mandatory": mandatory_diag,
            "spare": spare_diag,
            "spare_arrival_gate": spare_arrival_gate,
            "immediate_completion_shutdown": immediate_completion_shutdown,
            "spare_timeout": spare_timeout_diag,
            "arrival_rate": self._arrival_rate(),
            "service_rate": self._service_rate(),
            "runtime_ratio_median": self._runtime_ratio_median(),
            "runtime_log_ratio_mean": self._ema_log_runtime_ratio,
            "runtime_log_ratio_second_moment": (
                self._ema_log_runtime_ratio_sq
            ),
            "runtime_ratio_samples": self._runtime_ratio_samples,
            "active_job_predictions": self._active_job_predictions(),
            "resource_pmf": self._resource_pmf(),
            "policy_mode": "markov_power",
            "markovianity": markovianity,
        }
        self._append_decision_log(
            decision=self.last_decision,
            fallback=False,
            gate=spare_arrival_gate,
            decision_event=decision_event,
        )

        next_node_jobs = self._node_job_snapshot()
        for node_id, previous_job_id in self._previous_node_jobs.items():
            if (
                previous_job_id is not None
                and next_node_jobs.get(node_id) is None
                and node_id not in self.idle
                and node_id not in self.switching_off
                and node_id not in self.sleeping
            ):
                next_node_jobs[node_id] = previous_job_id

        for node_id, job_id in allocated_now.items():
            next_node_jobs[node_id] = job_id

        self._previous_node_jobs = next_node_jobs
        self._first_schedule = False
        return self.events

    def _schedule_snf_fallback(
        self,
        *,
        monitor_events: Mapping[str, object],
        markovianity: Mapping[str, object],
        released_by_completion: Sequence[int],
    ):
        """Run the baseline SNFPSAS

        The fallback intentionally skips all speculative Markov spare actions.
        Queue ordering, future planning, wake triggers, timeout handling, and
        callback construction follow :class:`SNFPSAS`.
        """
        now = float(self.current_time)
        self._started_this_schedule = {}

        started_now = SNFPSAS._current_snf_commit(self)
        allocated_now: Dict[int, int] = {}

        for bucket in self.events:
            if abs(float(bucket["timestamp"]) - now) > EPS:
                continue
            for event in bucket.get("events", []):
                if event.get("type") != "execution_start":
                    continue
                job_id = int(event["job_id"])
                nodes = [int(node_id) for node_id in event.get("nodes", [])]
                allocated_now.update({node_id: job_id for node_id in nodes})
                self._started_this_schedule[job_id] = {
                    "job_id": job_id,
                    "res": int(event.get("res", len(nodes))),
                    "reqtime": float(event.get("reqtime", 0.0)),
                    "start_time": now,
                    "nodes": list(nodes),
                }

        remaining_jobs = [
            job
            for job in self.waiting_queue
            if job.get("job_id") not in started_now
        ]
        future_plan = SNFPSAS._future_snf_plan(
            self,
            remaining_jobs,
            barrier=now,
        )
        self.selected_list = list(future_plan)

        event_counts = {
            float(bucket["timestamp"]): len(bucket.get("events", []))
            for bucket in self.events
        }
        SNFPSAS._emit_wake_triggers_from_plan(self, self.selected_list)

        if self.timeout is not None:
            BasePSAS.timeout_policy(self)
        BasePSAS.build_callbacks(self)

        immediate_wake = []
        future_wake_callbacks = []
        for bucket in self.events:
            timestamp = float(bucket["timestamp"])
            start = event_counts.get(timestamp, 0)
            for event in bucket.get("events", [])[start:]:
                if event.get("type") == "switch_on":
                    immediate_wake.extend(event.get("nodes", []))
                elif event.get("type") == "call_me_later_so":
                    future_wake_callbacks.append(timestamp)
                    self._scheduled_wake_callbacks.add(timestamp)

        spare_arrival_gate = self._spare_arrival_gate(now)
        decision_event = (
            self._first_schedule
            or bool(monitor_events["new_arrival"])
            or bool(monitor_events["new_completion"])
        )

        self.last_decision = {
            "time": now,
            "policy_mode": "snf_fallback",
            "markovianity": dict(markovianity),
            "spare_arrival_gate": spare_arrival_gate,
            "new_arrival": bool(monitor_events["new_arrival"]),
            "new_completion": bool(monitor_events["new_completion"]),
            "new_arrival_count": int(
                monitor_events["new_arrival_count"]
            ),
            "new_completion_count": int(
                monitor_events["new_completion_count"]
            ),
            "started_now": sorted(int(job_id) for job_id in started_now),
            "remaining_queue": [
                job.get("job_id") for job in remaining_jobs
            ],
            "released_by_completion": list(released_by_completion),
            "mandatory": {
                "planned_jobs": [
                    {
                        "job_id": job.get("job_id"),
                        "nodes": list(nodes),
                        "start_time": float(start_time),
                        "finish_time": float(finish_time),
                    }
                    for job, nodes, start_time, finish_time in future_plan
                ],
                "woken": sorted(set(immediate_wake)),
                "future_wake_callbacks": sorted(
                    set(future_wake_callbacks)
                ),
            },
            "spare": {
                "suppressed": True,
                "reason": "recent_workload_not_well_markovian",
            },
            "arrival_rate": self._arrival_rate(),
            "service_rate": self._service_rate(),
            "resource_pmf": self._resource_pmf(),
        }
        self._append_decision_log(
            decision=self.last_decision,
            fallback=True,
            gate=spare_arrival_gate,
            decision_event=decision_event,
        )

        next_node_jobs = self._node_job_snapshot()
        for node_id, previous_job_id in self._previous_node_jobs.items():
            if (
                previous_job_id is not None
                and next_node_jobs.get(node_id) is None
                and node_id not in self.idle
                and node_id not in self.switching_off
                and node_id not in self.sleeping
            ):
                next_node_jobs[node_id] = previous_job_id
        for node_id, job_id in allocated_now.items():
            next_node_jobs[node_id] = job_id

        self._previous_node_jobs = next_node_jobs
        self._first_schedule = False
        return self.events

    def _prune_markov_history(self, now: float) -> None:
        # Discard Markov-classification samples older than the last T seconds
        cutoff = float(now) - self.markov_window_seconds
        for history in (
            self._recent_interarrivals,
            self._recent_service_times,
        ):
            while history and float(history[0][0]) < cutoff:
                history.popleft()

    @staticmethod
    def _window_values(history) -> List[float]:
        # Return sample values from a timestamped ``(time, value)`` window
        return [float(value) for _, value in history]

    def _recent_markovianity(self, now: float) -> Dict[str, object]:
        #Classify workload observations from the last T simulation seconds
        now = float(now)
        self._prune_markov_history(now)

        arrival = self._series_markovianity(
            self._window_values(self._recent_interarrivals),
            name="interarrival",
            required=True,
        )
        service = self._series_markovianity(
            self._window_values(self._recent_service_times),
            name="service",
            required=self.markov_require_service_samples,
        )

        well_markovian = bool(
            arrival["accepted"]
            and (
                service["accepted"]
                or not self.markov_require_service_samples
            )
        )
        reasons = []
        if not arrival["accepted"]:
            reasons.append(f"arrival:{arrival['reason']}")
        if self.markov_require_service_samples and not service["accepted"]:
            reasons.append(f"service:{service['reason']}")

        return {
            "well_markovian": well_markovian,
            "reason": "accepted" if well_markovian else ";".join(reasons),
            "window_seconds": self.markov_window_seconds,
            "window_start": now - self.markov_window_seconds,
            "window_end": now,
            "minimum_samples": self.markov_min_samples,
            "maximum_samples": self.markov_max_samples,
            "arrival": arrival,
            "service": service,
        }

    def _series_markovianity(
        self,
        samples: Sequence[float],
        *,
        name: str,
        required: bool,
    ) -> Dict[str, object]:
        values = [
            float(value)
            for value in samples
            if math.isfinite(float(value)) and float(value) >= 0.0
        ]
        sample_count = len(values)
        base = {
            "name": name,
            "required": bool(required),
            "sample_count": sample_count,
        }

        if not required and sample_count < self.markov_min_samples:
            return {
                **base,
                "accepted": True,
                "reason": "optional_insufficient_samples_in_time_window",
                "failed_checks": [],
            }

        if sample_count < self.markov_min_samples:
            return {
                **base,
                "accepted": False,
                "reason": "insufficient_samples_in_time_window",
                "failed_checks": [
                    "insufficient_samples_in_time_window",
                ],
            }

        mean = sum(values) / sample_count
        if mean <= EPS:
            return {
                **base,
                "accepted": False,
                "reason": "non_positive_mean",
                "failed_checks": ["non_positive_mean"],
                "mean": float(mean),
            }

        variance = sum((value - mean) ** 2 for value in values) / sample_count
        coefficient_of_variation = math.sqrt(max(0.0, variance)) / mean

        left = values[:-1]
        right = values[1:]
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (x - left_mean) * (y - right_mean)
            for x, y in zip(left, right)
        )
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        lag1_autocorrelation = (
            numerator / denominator if denominator > EPS else 1.0
        )

        sorted_values = sorted(values)
        ks_distance = 0.0
        for index, value in enumerate(sorted_values, start=1):
            fitted_cdf = 1.0 - math.exp(-value / mean)
            empirical_before = (index - 1) / sample_count
            empirical_after = index / sample_count
            ks_distance = max(
                ks_distance,
                abs(empirical_before - fitted_cdf),
                abs(empirical_after - fitted_cdf),
            )

        cv_ok = (
            abs(coefficient_of_variation - 1.0)
            <= self.markov_cv_tolerance
        )
        autocorrelation_ok = (
            abs(lag1_autocorrelation)
            <= self.markov_max_lag1_autocorrelation
        )
        ks_limit = (
            self.markov_ks_multiplier
            * self.markov_ks_param
            / math.sqrt(sample_count)
        )
        ks_ok = ks_distance <= ks_limit
        accepted = bool(cv_ok and autocorrelation_ok and ks_ok)

        failed_checks = []
        if not cv_ok:
            failed_checks.append("cv")
        if not autocorrelation_ok:
            failed_checks.append("lag1_autocorrelation")
        if not ks_ok:
            failed_checks.append("exponential_ks")

        return {
            **base,
            "accepted": accepted,
            "reason": "accepted" if accepted else ",".join(failed_checks),
            "failed_checks": list(failed_checks),
            "mean": float(mean),
            "coefficient_of_variation": float(coefficient_of_variation),
            "cv_range": [
                max(0.0, 1.0 - self.markov_cv_tolerance),
                1.0 + self.markov_cv_tolerance,
            ],
            "lag1_autocorrelation": float(lag1_autocorrelation),
            "max_abs_lag1_autocorrelation": (
                self.markov_max_lag1_autocorrelation
            ),
            "ks_distance": float(ks_distance),
            "ks_limit": float(ks_limit),
        }

    def _node_job_snapshot(self) -> Dict[int, object]:
        # Return scheduler-visible job ownership for every node
        return {
            node_id: (
                node.get("job_id")
                if node.get("state") == "active"
                else None
            )
            for node_id, node in self.state.items()
        }

    def _nodes_released_since_last_schedule(self) -> List[int]:
        # Find nodes that changed from computing to active-and-idle
        released = []
        for node_id, previous_job_id in self._previous_node_jobs.items():
            if previous_job_id is None:
                continue
            node = self.state.get(node_id, {})
            if (
                node.get("state") == "active"
                and node.get("job_id") is None
                and node_id in self.idle
            ):
                released.append(node_id)
        return sorted(released)

    def _switch_off_stale_completion_nodes(
        self,
        *,
        released_nodes: Sequence[int],
        allocated_nodes: Iterable[int],
        protected_nodes: Iterable[int],
    ) -> Dict[str, object]:
        """Immediately switch off nodes released while the gate is closed.

        Nodes reused by an immediately-started job, selected for mandatory
        wake-up, or no longer idle are excluded.  The ordinary idle timeout is
        intentionally bypassed for the remaining nodes.
        """
        excluded = set(allocated_nodes) | set(protected_nodes)
        candidates = [
            node_id
            for node_id in released_nodes
            if node_id not in excluded and node_id in self.idle
        ]
        switched_off = self._emit_switch_off(
            candidates,
            ignore_timeout=True,
        )
        return {
            "enabled": True,
            "released_nodes": sorted(set(released_nodes)),
            "excluded_nodes": sorted(
                node_id for node_id in set(released_nodes) if node_id in excluded
            ),
            "switch_off": list(switched_off),
            "reason": "arrival_gate_closed_after_completion",
        }

    def _spare_arrival_gate(self, now: float) -> Dict[str, object]:
        """Report whether speculative spare preparation is currently allowed.

        ``spare_arrival_window=None`` preserves the original Markov policy.
        Otherwise, the gate opens only after a monitor-confirmed arrival and
        remains open for the configured number of simulation seconds.
        """
        window = self.spare_arrival_window
        last_arrival = self._last_arrival_event_time

        if window is None:
            return {
                "enabled": False,
                "allowed": True,
                "last_arrival_time": last_arrival,
                "arrival_age": None,
                "arrival_window": None,
                "reason": "disabled",
            }

        if last_arrival is None:
            return {
                "enabled": True,
                "allowed": False,
                "last_arrival_time": None,
                "arrival_age": None,
                "arrival_window": float(window),
                "reason": "no_arrival_observed",
            }

        age = max(0.0, float(now) - float(last_arrival))
        allowed = age <= float(window) + EPS
        return {
            "enabled": True,
            "allowed": bool(allowed),
            "last_arrival_time": float(last_arrival),
            "arrival_age": float(age),
            "arrival_window": float(window),
            "reason": "recent_arrival" if allowed else "arrival_window_expired",
        }

    def _apply_spare_timeout_policy(self) -> Dict[str, object]:
        previous_counts = {
            float(bucket["timestamp"]): len(bucket.get("events", []))
            for bucket in self.events
        }
        super().timeout_policy()

        switch_off = []
        callbacks = []
        for bucket in self.events:
            timestamp = float(bucket["timestamp"])
            start = previous_counts.get(timestamp, 0)
            for event in bucket.get("events", [])[start:]:
                event_type = event.get("type")
                if event_type == "switch_off":
                    switch_off.extend(event.get("nodes", []))
                elif event_type == "call_me_later_to":
                    callbacks.append(timestamp)

        return {
            "timeout": float(self.timeout),
            "switch_off": sorted(set(switch_off)),
            "next_callback": min(callbacks) if callbacks else None,
        }

    # ------------------------------------------------------------------
    # Persistent policy-decision trace
    # ------------------------------------------------------------------

    DECISION_LOG_FIELDS = [
        "decision_index",
        "current_time",
        "fallback",
        "fallback_changed",
        "fallback_enabled",
        "fallback_reason",
        "gate",
        "gate_changed",
        "gate_enabled",
        "gate_reason",
        "gate_relevant",
        "policy_mode",
        "trigger",
        "decision_event",
        "markov_well",
        "markov_reason",
        "markov_window_seconds",
        "arrival_markov_accepted",
        "service_markov_accepted",
        "arrival_sample_count",
        "service_sample_count",
        "arrival_coefficient_of_variation",
        "arrival_lag1_autocorrelation",
        "arrival_ks_distance",
        "arrival_failed_checks",
        "service_coefficient_of_variation",
        "service_lag1_autocorrelation",
        "service_ks_distance",
        "service_failed_checks",
        "new_arrival",
        "new_completion",
        "new_arrival_count",
        "new_completion_count",
        "waiting_jobs",
        "started_jobs",
        "idle_nodes",
        "sleeping_nodes",
        "switching_on_nodes",
        "switching_off_nodes",
        "computing_nodes",
        "arrival_age",
        "arrival_window",
        "arrival_rate",
        "service_rate",
        "spare_suppressed",
        "spare_reason",
        "spare_target",
        "wake_count",
        "switch_off_count",
    ]

    def _resolve_decision_log_path(
        self,
        configured_path: Optional[str],
    ) -> Optional[Path]:
        if configured_path:
            return Path(configured_path).expanduser().resolve()

        output_dir = os.environ.get("SPARS_OUTPUT_DIR")
        if output_dir:
            return (
                Path(output_dir).expanduser().resolve()
                / "scheduler_decision_log.csv"
            )

        for owner in (self.monitor, self.jobs_manager):
            for attribute in (
                "output_path",
                "output_dir",
                "result_path",
                "results_path",
            ):
                candidate = getattr(owner, attribute, None)
                if not candidate:
                    continue
                candidate_path = Path(candidate).expanduser().resolve()
                if candidate_path.suffix:
                    candidate_path = candidate_path.parent
                return candidate_path / "scheduler_decision_log.csv"

        return None

    def _initialise_decision_log(self) -> None:
        path = self.decision_log_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=self.DECISION_LOG_FIELDS,
            ).writeheader()

    @staticmethod
    def _decision_trigger(
        decision: Mapping[str, object],
        *,
        first_schedule: bool,
    ) -> str:
        triggers = []
        if first_schedule:
            triggers.append("initial")
        if bool(decision.get("new_arrival")):
            triggers.append("arrival")
        if bool(decision.get("new_completion")):
            triggers.append("completion")
        return "+".join(triggers) if triggers else "callback"

    def _append_decision_log(
        self,
        *,
        decision: Mapping[str, object],
        fallback: bool,
        gate: Mapping[str, object],
        decision_event: bool,
    ) -> None:
        if not self.decision_log_enabled:
            return

        path = self.decision_log_path
        if path is None:
            return

        markov = decision.get("markovianity") or {}
        arrival_markov = markov.get("arrival") or {}
        service_markov = markov.get("service") or {}
        spare = decision.get("spare") or {}
        mandatory = decision.get("mandatory") or {}

        gate_allowed = bool(gate.get("allowed", True))
        fallback = bool(fallback)
        fallback_changed = (
            self._previous_logged_fallback is None
            or fallback != self._previous_logged_fallback
        )
        gate_changed = (
            self._previous_logged_gate is None
            or gate_allowed != self._previous_logged_gate
        )

        remaining_queue = decision.get("remaining_queue") or []
        started_now = decision.get("started_now") or []
        wake_nodes = []
        switch_off_nodes = []
        if isinstance(spare, Mapping):
            wake_nodes = spare.get("wake") or []
            switch_off_nodes = spare.get("switch_off") or []
        if not wake_nodes and isinstance(mandatory, Mapping):
            wake_nodes = mandatory.get("woken") or []

        first_schedule = self._decision_log_index == 0
        row = {
            "decision_index": self._decision_log_index,
            "current_time": float(decision.get("time", self.current_time)),
            "fallback": fallback,
            "fallback_changed": fallback_changed,
            "fallback_enabled": self.markov_fallback_enabled,
            "fallback_reason": (
                markov.get("reason", "") if fallback else ""
            ),
            "gate": gate_allowed,
            "gate_changed": gate_changed,
            "gate_enabled": bool(gate.get("enabled", False)),
            "gate_reason": gate.get("reason", ""),
            "gate_relevant": bool(
                not fallback
                and decision_event
                and len(remaining_queue) == 0
            ),
            "policy_mode": decision.get("policy_mode", ""),
            "trigger": self._decision_trigger(
                decision,
                first_schedule=first_schedule,
            ),
            "decision_event": bool(decision_event),
            "markov_well": bool(markov.get("well_markovian", False)),
            "markov_reason": markov.get("reason", ""),
            "markov_window_seconds": markov.get("window_seconds", ""),
            "arrival_markov_accepted": arrival_markov.get("accepted", ""),
            "service_markov_accepted": service_markov.get("accepted", ""),
            "arrival_sample_count": arrival_markov.get("sample_count", ""),
            "service_sample_count": service_markov.get("sample_count", ""),
            "arrival_coefficient_of_variation": arrival_markov.get("coefficient_of_variation", ""),
            "arrival_lag1_autocorrelation": arrival_markov.get("lag1_autocorrelation", ""),
            "arrival_ks_distance": arrival_markov.get("ks_distance", ""),
            "arrival_failed_checks": ",".join(arrival_markov.get("failed_checks", [])),
            "service_coefficient_of_variation": service_markov.get("coefficient_of_variation", ""),
            "service_lag1_autocorrelation": service_markov.get("lag1_autocorrelation", ""),
            "service_ks_distance": service_markov.get("ks_distance", ""),
            "service_failed_checks": ",".join(service_markov.get("failed_checks", [])),
            "new_arrival": bool(decision.get("new_arrival", False)),
            "new_completion": bool(decision.get("new_completion", False)),
            "new_arrival_count": int(decision.get("new_arrival_count", 0)),
            "new_completion_count": int(
                decision.get("new_completion_count", 0)
            ),
            "waiting_jobs": len(remaining_queue),
            "started_jobs": len(started_now),
            "idle_nodes": len(self.idle),
            "sleeping_nodes": len(self.sleeping),
            "switching_on_nodes": len(self.switching_on),
            "switching_off_nodes": len(self.switching_off),
            "computing_nodes": len(self.computing),
            "arrival_age": gate.get("arrival_age", ""),
            "arrival_window": gate.get("arrival_window", ""),
            "arrival_rate": decision.get("arrival_rate", ""),
            "service_rate": decision.get("service_rate", ""),
            "spare_suppressed": (
                bool(spare.get("suppressed", False))
                if isinstance(spare, Mapping)
                else False
            ),
            "spare_reason": (
                spare.get("reason", "")
                if isinstance(spare, Mapping)
                else ""
            ),
            "spare_target": (
                spare.get("target", "")
                if isinstance(spare, Mapping)
                else ""
            ),
            "wake_count": len(wake_nodes),
            "switch_off_count": len(switch_off_nodes),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.DECISION_LOG_FIELDS,
            )
            writer.writerow(row)

        self._previous_logged_fallback = fallback
        self._previous_logged_gate = gate_allowed
        self._decision_log_index += 1

    def _monitor_log(self, attribute: str):
        if not hasattr(self.monitor, attribute):
            raise AttributeError(
                f"monitor must expose {attribute!r} for SNFPSAS event detection"
            )
        log = getattr(self.monitor, attribute)
        if log is None:
            raise ValueError(f"monitor.{attribute} must not be None")
        if not hasattr(log, "__len__") or not hasattr(log, "__getitem__"):
            raise TypeError(
                f"monitor.{attribute} must be a sequence-like object"
            )
        return log
    
    # Immediate scheduling and mandatory wake-up
    
    @staticmethod
    def _job_key(job):
        return (
            int(job["res"]),
            float(job["reqtime"]),
            int(job.get("job_id", -1)),
        )

    def _current_snf_commit(self):
        # Start all SNF-ordered jobs that fit on currently idle nodes
        now = float(self.current_time)
        started_now = set()
        allocated_nodes = {}
        node_selection_static = self._build_node_selection_static_data(
            list(self.idle),
            self.next_releases,
        )

        for job in sorted(self.waiting_queue[:], key=self._job_key):
            required_nodes = int(job["res"])

            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive resource request: "
                    f"{required_nodes}"
                )
            if required_nodes > len(self.state):
                raise RuntimeError(
                    f"Job {job['job_id']} requests {required_nodes} nodes, "
                    f"but the platform has only {len(self.state)}"
                )

            # SNF sorts by node count.  If the smallest remaining job does not
            # fit, no later job can fit either.
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
            job_id = int(job["job_id"])
            started_now.add(job_id)
            self._started_this_schedule[job_id] = {
                "job_id": job_id,
                "res": required_nodes,
                "reqtime": float(job["reqtime"]),
                "start_time": float(start_time),
                "nodes": list(nodes),
            }
            for node_id in nodes:
                allocated_nodes[node_id] = job_id

        return started_now, allocated_nodes

    def _ensure_waiting_capacity(self, jobs):
        # Plan queued jobs sequentially and emit only required wake actions.
        now = float(self.current_time)

        # Preserve callback-set pruning and the exact empty diagnostics without
        # constructing a prediction release map when there is no queued work.
        if not jobs:
            self.selected_list = []
            immediate_wake, callback_times = self._emit_wake_triggers_from_plan(
                []
            )
            return immediate_wake, {
                "required_nodes": 0,
                "planned_jobs": [],
                "woken": list(immediate_wake),
                "future_wake_callbacks": list(callback_times),
                "estimated_wait": 0.0,
                "estimated_energy": 0.0,
            }

        future_plan = self._future_snf_plan(jobs, barrier=now)
        self.selected_list = list(future_plan)
        immediate_wake, callback_times = self._emit_wake_triggers_from_plan(
            future_plan
        )

        return immediate_wake, {
            "required_nodes": sum(int(job["res"]) for job in jobs),
            "planned_jobs": [
                {
                    "job_id": job.get("job_id"),
                    "nodes": list(nodes),
                    "start_time": float(start_time),
                    "finish_time": float(finish_time),
                }
                for job, nodes, start_time, finish_time in future_plan
            ],
            "woken": list(immediate_wake),
            "future_wake_callbacks": list(callback_times),
            "estimated_wait": sum(
                max(0.0, float(start_time) - now)
                for _, _, start_time, _ in future_plan
            ),
            "estimated_energy": sum(
                self._wake_transition_energy(node_id)
                for node_id in immediate_wake
            ),
        }

    def _future_snf_plan(self, jobs, barrier):
        # BasePSAS release bounds can be based on requested wall time.  Replace
        # running compute releases with this policy's job-specific learned
        # runtime predictions before planning queued work.
        planned_releases = self._prediction_release_map()

        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        # During one future plan, only compute segments and release times are
        # appended to ``planned_releases``.  Compute segments are intentionally
        # excluded from energy-waste cost, while node state, idle power, and
        # timeout priority remain unchanged.  Precompute those invariant values
        # once instead of rebuilding them for every queued job.
        node_selection_static = self._build_node_selection_static_data(
            candidates,
            planned_releases,
        )

        plan = []
        barrier = float(barrier)

        for job in sorted(jobs, key=self._job_key):
            required_nodes = int(job["res"])
            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive resource request: "
                    f"{required_nodes}"
                )
            if required_nodes > len(self.state):
                raise RuntimeError(
                    f"Job {job['job_id']} requests {required_nodes} nodes, "
                    f"but the platform has only {len(self.state)}"
                )

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
            compute_speed = min(
                self._active_compute_speed(node_id)
                for node_id in nodes
            )
            if compute_speed <= 0.0:
                raise RuntimeError("compute_speed must be > 0")

            predicted_duration = self._predicted_job_duration(
                job,
                compute_speed=compute_speed,
            )
            finish_time = float(start_time) + predicted_duration
            phase = f'compute(job={job["job_id"]})'

            for node_id in nodes:
                entry = planned_releases[node_id]
                entry["queue"].append(
                    {
                        "phase": phase,
                        "start_time": float(start_time),
                        "finish_time": float(finish_time),
                    }
                )
                entry["release_time"] = float(finish_time)

            plan.append(
                (job, list(nodes), float(start_time), float(finish_time))
            )
            barrier = float(start_time)

        return plan

    def _emit_wake_triggers_from_plan(self, plan):
        now = float(self.current_time)
        self._scheduled_wake_callbacks = {
            timestamp
            for timestamp in self._scheduled_wake_callbacks
            if timestamp > now + EPS
        }

        earliest_wake = {}
        for _, nodes, start_time, _ in plan:
            for node_id in nodes:
                if node_id not in self.sleeping:
                    continue

                wake_time = (
                    float(start_time)
                    - self._wake_lead_time_safe(node_id)
                )
                previous = earliest_wake.get(node_id)
                if previous is None or wake_time < previous:
                    earliest_wake[node_id] = wake_time

        immediate = sorted(
            node_id
            for node_id, wake_time in earliest_wake.items()
            if wake_time <= now + EPS
        )
        self._emit_switch_on(immediate)

        future_times = sorted(
            {
                float(wake_time)
                for wake_time in earliest_wake.values()
                if wake_time > now + EPS
            }
        )

        emitted_callbacks = []
        for wake_time in future_times:
            if wake_time in self._scheduled_wake_callbacks:
                continue
            self.push_event(wake_time, {"type": "call_me_later_so"})
            self._scheduled_wake_callbacks.add(wake_time)
            emitted_callbacks.append(wake_time)

        return immediate, emitted_callbacks

    # ------------------------------------------------------------------
    # Markov spare target
    # ------------------------------------------------------------------

    def _optimise_and_apply_spare_target(self, protected_nodes: set):
        """Enumerate feasible spare targets and apply the minimum-cost plan."""
        del protected_nodes

        now = float(self.current_time)
        arrival_rate = self._arrival_rate()
        service_rate = self._service_rate()

        active_job_ids = {
            node.get("job_id")
            for node in self.state.values()
            if node.get("state") == "active"
            and node.get("job_id") is not None
        }

        next_completion = self._next_completion_delay()
        if math.isfinite(next_completion) and next_completion > EPS:
            completion_hazard = 1.0 / next_completion
        else:
            completion_hazard = len(active_job_ids) * service_rate

        total_hazard = arrival_rate + completion_hazard
        horizon_cap = min(
            self.max_decision_horizon,
            self._next_timeout_delay(),
        )

        if total_hazard <= EPS:
            horizon = horizon_cap
            probability_next_event_is_arrival = 0.0
            gamma_for_residual = 1.0 / max(EPS, horizon_cap)
        else:
            horizon = (
                -math.expm1(-total_hazard * horizon_cap)
                / total_hazard
            )
            probability_next_event_is_arrival = (
                arrival_rate / total_hazard
            )
            gamma_for_residual = total_hazard

        expired_idle = []
        if self.respect_timeout and self.timeout is not None:
            expired_idle = [
                node_id
                for node_id in self.idle
                if (
                    self.timeout_list.get(node_id) is not None
                    and now + EPS
                    >= float(self.timeout_list[node_id])
                )
            ]
        expired_idle_set = set(expired_idle)

        forced_idle = [
            node_id
            for node_id in self.idle
            if (
                node_id not in expired_idle_set
                and not self._can_switch_off(node_id)
            )
        ]
        forced_idle_set = set(forced_idle)
        optional_idle = sorted(
            (
                node_id
                for node_id in self.idle
                if (
                    node_id not in forced_idle_set
                    and node_id not in expired_idle_set
                )
            ),
            key=lambda node_id: (
                self._idle_power(node_id),
                -self._wake_lead_time_safe(node_id),
                node_id,
            ),
        )
        ranked_sleeping = sorted(
            self.sleeping,
            key=lambda node_id: (
                self.alpha * self._wake_lead_time_safe(node_id)
                + self.beta * self._wake_transition_energy(node_id),
                self._wake_lead_time_safe(node_id),
                self._wake_transition_energy(node_id),
                node_id,
            ),
        )

        forced_warm = len(self.switching_on) + len(forced_idle)
        maximum_realisable = (
            len(self.switching_on)
            + len(forced_idle)
            + len(optional_idle)
            + len(self.sleeping)
        )
        minimum_target = max(
            forced_warm,
            min(self.min_spare_nodes, maximum_realisable),
        )
        maximum_target = max(
            minimum_target,
            min(self.max_spare_nodes, maximum_realisable),
        )

        pmf = self._resource_pmf()
        candidates = []

        for target in range(minimum_target, maximum_target + 1):
            plan = self._build_spare_plan(
                target=target,
                forced_idle=forced_idle,
                optional_idle=optional_idle,
                forced_switch_off=expired_idle,
                ranked_sleeping=ranked_sleeping,
            )

            expected_wait = self._expected_power_wait(
                plan=plan,
                resource_pmf=pmf,
                event_hazard=gamma_for_residual,
            )
            expected_wait *= probability_next_event_is_arrival

            expected_energy = self._expected_plan_energy(plan, horizon)
            objective = self.alpha * expected_wait + self.beta * expected_energy

            candidates.append(
                {
                    "target": target,
                    "plan": plan,
                    "expected_wait": float(expected_wait),
                    "expected_energy": float(expected_energy),
                    "objective": float(objective),
                }
            )

        best = min(
            candidates,
            key=lambda item: (
                item["objective"],
                item["expected_wait"],
                item["expected_energy"],
                item["target"],
            ),
        )
        plan = best["plan"]

        self._emit_switch_on(plan["wake"])
        self._emit_switch_off(plan["switch_off"])

        return {
            "target": best["target"],
            "wake": list(plan["wake"]),
            "switch_off": list(plan["switch_off"]),
            "keep_idle": list(plan["keep_idle"]),
            "expected_wait": best["expected_wait"],
            "expected_energy": best["expected_energy"],
            "objective": best["objective"],
            "decision_horizon": float(horizon),
            "horizon_cap": float(horizon_cap),
            "predicted_next_completion": (
                float(next_completion)
                if math.isfinite(next_completion)
                else None
            ),
            "probability_next_event_is_arrival": float(
                probability_next_event_is_arrival
            ),
            "candidate_costs": [
                {
                    "target": item["target"],
                    "expected_wait": item["expected_wait"],
                    "expected_energy": item["expected_energy"],
                    "objective": item["objective"],
                }
                for item in candidates
            ],
            "time": now,
        }

    def _next_completion_delay(self) -> float:
        remaining = [
            float(item["predicted_remaining"])
            for item in self._active_job_predictions()
            if math.isfinite(float(item["predicted_remaining"]))
        ]
        return min(remaining) if remaining else math.inf

    def _next_timeout_delay(self) -> float:
        if self.timeout is None:
            return self.max_decision_horizon

        now = float(self.current_time)
        delays = [
            float(expiry) - now
            for node_id, expiry in self.timeout_list.items()
            if (
                node_id in self.idle
                and float(expiry) > now + EPS
            )
        ]
        return (
            min(delays)
            if delays
            else self.max_decision_horizon
        )

    def _build_spare_plan(
        self,
        *,
        target: int,
        forced_idle: Sequence[int],
        optional_idle: Sequence[int],
        forced_switch_off: Sequence[int] = (),
        ranked_sleeping: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[int]]:
        switching_on = list(self.switching_on)
        forced_idle = list(forced_idle)
        optional_idle = list(optional_idle)
        forced_switch_off = list(forced_switch_off)

        base_warm = len(switching_on) + len(forced_idle)
        additional_needed = max(0, int(target) - base_warm)

        keep_optional_count = min(additional_needed, len(optional_idle))
        keep_optional = optional_idle[:keep_optional_count]
        switch_off = forced_switch_off + optional_idle[keep_optional_count:]

        still_needed = additional_needed - keep_optional_count
        if ranked_sleeping is None:
            wake = self._select_sleeping_to_wake(still_needed)
        else:
            wake = list(ranked_sleeping[:still_needed])

        wake_set = set(wake)
        return {
            "keep_idle": forced_idle + keep_optional,
            "switching_on": switching_on,
            "wake": wake,
            "switch_off": switch_off,
            "sleep": [
                node_id
                for node_id in self.sleeping
                if node_id not in wake_set
            ],
            "switching_off": list(self.switching_off),
        }

    def _expected_power_wait(
        self,
        *,
        plan: Mapping[str, Sequence[int]],
        resource_pmf: Mapping[int, float],
        event_hazard: float,
    ) -> float:
        """Expected power-induced delay for the next arriving job.

        For each requested node count r, work can start after the r-th
        fastest non-computing node is ready.  Nodes intentionally left cold are
        assumed to begin waking at the arrival instant.  Nodes being switched
        off must finish that transition before they can wake again.

        Queueing delay caused by currently computing nodes is intentionally not
        included because it is not controllable by the power action.  Requests
        larger than the current non-computing capacity are therefore truncated
        to that capacity; this adds the same omitted compute-delay constant to
        every candidate and does not change the selected target.
        """
        warm_delays = []
        warm_delays.extend(0.0 for _ in plan["keep_idle"])
        warm_delays.extend(
            self._expected_residual_delay(
                self._remaining_switch_on_time(nid), event_hazard
            )
            for nid in plan["switching_on"]
        )
        warm_delays.extend(
            self._expected_residual_delay(
                self._wake_lead_time_safe(nid), event_hazard
            )
            for nid in plan["wake"]
        )

        cold_delays = []
        cold_delays.extend(
            self._wake_lead_time_safe(nid) for nid in plan["sleep"]
        )
        cold_delays.extend(
            self._wake_lead_time_safe(nid)
            + self._expected_residual_delay(
                self._switch_off_duration(nid), event_hazard
            )
            for nid in plan["switch_off"]
        )
        cold_delays.extend(
            self._wake_lead_time_safe(nid)
            + self._expected_residual_delay(
                self._remaining_switch_off_time(nid), event_hazard
            )
            for nid in plan["switching_off"]
        )

        readiness = sorted(float(x) for x in warm_delays + cold_delays)
        if not readiness:
            return 0.0

        expected = 0.0
        for requested, probability in resource_pmf.items():
            r = max(1, min(int(requested), len(readiness)))
            expected += float(probability) * readiness[r - 1]

        return float(expected)

    def _expected_plan_energy(
        self,
        plan: Mapping[str, Sequence[int]],
        horizon: float,
    ) -> float:
        # Expected non-compute energy until the next decision event
        horizon = max(0.0, float(horizon))
        energy = 0.0

        for nid in plan["keep_idle"]:
            energy += self._idle_power(nid) * horizon

        for nid in plan["switching_on"]:
            rem = self._remaining_switch_on_time(nid)
            transition = min(rem, horizon)
            energy += self._state_power(nid, "switching_on") * transition
            energy += self._idle_power(nid) * max(0.0, horizon - transition)

        for nid in plan["wake"]:
            energy += self._wake_energy_over_horizon(nid, horizon)

        for nid in plan["sleep"]:
            energy += self._state_power(nid, "sleeping") * horizon

        for nid in plan["switch_off"]:
            duration = self._switch_off_duration(nid)
            transition = min(duration, horizon)
            energy += self._state_power(nid, "switching_off") * transition
            energy += self._state_power(nid, "sleeping") * max(
                0.0, horizon - transition
            )

        for nid in plan["switching_off"]:
            rem = self._remaining_switch_off_time(nid)
            transition = min(rem, horizon)
            energy += self._state_power(nid, "switching_off") * transition
            energy += self._state_power(nid, "sleeping") * max(
                0.0, horizon - transition
            )

        return float(energy)

    # ------------------------------------------------------------------
    # Online Markov model
    # ------------------------------------------------------------------

    def _observe_arrivals(self, arrival_records, now: float) -> None:
        # Update EMA statistics from every monitor arrival record
        if not arrival_records:
            return

        speeds = [
            float(node.get("compute_speed", 1.0))
            for node in self.state.values()
            if float(node.get("compute_speed", 1.0)) > 0.0
        ]
        representative_speed = (
            sum(speeds) / len(speeds)
            if speeds
            else 1.0
        )

        records = sorted(
            arrival_records,
            key=lambda job: (
                float(job.get("subtime", now)),
                int(job.get("job_id", -1)),
            ),
        )

        for job in records:
            event_time = float(job.get("subtime", now))

            if (
                self._last_arrival_event_time is not None
                and event_time + EPS < self._last_arrival_event_time
            ):
                raise RuntimeError(
                    "Arrival log is not time ordered: "
                    f"{event_time} < {self._last_arrival_event_time}"
                )

            if self._last_arrival_event_time is not None:
                interarrival = max(
                    0.0,
                    event_time - self._last_arrival_event_time,
                )
                self._ema_interarrival = self._ema_update(
                    self._ema_interarrival,
                    interarrival,
                    self.arrival_ema_alpha,
                )
                # Associate an interarrival interval with its later arrival.
                # It is therefore retained while that arrival lies in the
                # current last-T observation window.
                self._recent_interarrivals.append(
                    (float(event_time), float(interarrival))
                )

            self._last_arrival_event_time = event_time

            requested_nodes = int(job["res"])
            if requested_nodes <= 0:
                raise ValueError("Requested node can't be <= 0")
            self._update_resource_ema(requested_nodes)

            reqtime = float(job.get("reqtime", 0.0))
            if reqtime > 0.0:
                requested_service = reqtime / representative_speed
                self._ema_requested_service_time = self._ema_update(
                    self._ema_requested_service_time,
                    requested_service,
                    self.service_ema_alpha,
                )

    def _observe_completions(self, completion_records) -> None:
        # Learn actual service time and actual/requested runtime ratio.
        records = sorted(
            completion_records,
            key=lambda job: (
                float(job.get("finish_time"))
                if job.get("finish_time") is not None
                else math.inf,
                int(job.get("job_id", -1)),
            ),
        )
        for job in records:
            start_time = job.get("start_time")
            finish_time = job.get("finish_time")
            if start_time is None or finish_time is None:
                continue

            duration = float(finish_time) - float(start_time)
            if duration <= EPS or not math.isfinite(duration):
                continue

            self._ema_service_time = self._ema_update(
                self._ema_service_time,
                duration,
                self.service_ema_alpha,
            )
            # Associate service duration with completion time so the
            # classifier uses completions observed within the last T seconds.
            self._recent_service_times.append(
                (float(finish_time), float(duration))
            )

            requested_duration = self._requested_walltime(job)
            if requested_duration is None or requested_duration <= EPS:
                continue

            ratio = duration / requested_duration
            if ratio <= EPS or not math.isfinite(ratio):
                continue

            log_ratio = math.log(ratio)
            self._ema_log_runtime_ratio = self._ema_update_signed(
                self._ema_log_runtime_ratio,
                log_ratio,
                self.service_ema_alpha,
            )
            self._ema_log_runtime_ratio_sq = self._ema_update(
                self._ema_log_runtime_ratio_sq,
                log_ratio * log_ratio,
                self.service_ema_alpha,
            )
            self._runtime_ratio_samples += 1

    @staticmethod
    def _phase_job_id(phase) -> Optional[int]:
        match = _COMPUTE_RE.fullmatch(str(phase))
        return int(match.group(1)) if match is not None else None

    def _active_compute_speed(self, node_id: int) -> float:
        # Return a node's active-state compute speed.
        node = self.state[node_id]
        current = float(node.get("compute_speed", 0.0))
        if node.get("state") == "active" and current > EPS:
            return current

        machine = self._machine_for(node_id)
        active = machine["states"]["active"]
        speed = active.get("compute_speed")
        if speed == "from_dvfs":
            speed = machine["dvfs_profiles"][node["dvfs_mode"]][
                "compute_speed"
            ]
        speed = float(speed)
        if speed <= EPS:
            raise RuntimeError(
                f"Active compute speed for node {node_id} must be > 0"
            )
        return speed

    def _requested_walltime(
        self,
        job: Mapping[str, object],
        *,
        compute_speed: Optional[float] = None,
    ) -> Optional[float]:
        # Return this job's requested wall-clock duration.
        start_time = job.get("start_time")
        for key in ("req_finish_time", "requested_finish_time"):
            requested_finish = job.get(key)
            if start_time is not None and requested_finish is not None:
                duration = float(requested_finish) - float(start_time)
                if duration > EPS and math.isfinite(duration):
                    return duration

        reqtime = job.get("reqtime")
        if reqtime is None:
            return None
        reqtime = float(reqtime)
        if reqtime <= EPS or not math.isfinite(reqtime):
            return None

        if compute_speed is None:
            nodes = list(job.get("nodes", []))
            speeds = [
                self._active_compute_speed(int(node_id))
                for node_id in nodes
                if int(node_id) in self.state
            ]
            if speeds:
                compute_speed = min(speeds)

        if compute_speed is None or compute_speed <= EPS:
            compute_speed = 1.0

        duration = reqtime / float(compute_speed)
        return duration if duration > EPS and math.isfinite(duration) else None

    def _runtime_ratio_median(self) -> float:
        # Return exp(E[log(actual/requested)]), or 1.0 at cold start
        if self._ema_log_runtime_ratio is None:
            return 1.0
        return max(EPS, math.exp(min(700.0, self._ema_log_runtime_ratio)))

    def _runtime_ratio_distribution(self):
        if (
            self._ema_log_runtime_ratio is None
            or self._ema_log_runtime_ratio_sq is None
        ):
            return None
        mu = float(self._ema_log_runtime_ratio)
        variance = max(
            0.0,
            float(self._ema_log_runtime_ratio_sq) - mu * mu,
        )
        sigma = max(self.runtime_log_sigma_floor, math.sqrt(variance))
        return mu, sigma

    def _predicted_job_duration(
        self,
        job: Mapping[str, object],
        *,
        compute_speed: Optional[float] = None,
    ) -> float:
        requested = self._requested_walltime(
            job,
            compute_speed=compute_speed,
        )
        if requested is None:
            if self._ema_service_time is not None:
                return max(EPS, float(self._ema_service_time))
            raise ValueError(
                f"Job {job.get('job_id')} has no valid requested duration"
            )
        return max(EPS, requested * self._runtime_ratio_median())

    @staticmethod
    def _normal_log_survival(z: float) -> float:
        z = float(z)
        if z < 8.0:
            survival = 0.5 * math.erfc(z / math.sqrt(2.0))
            return math.log(max(1e-300, survival))
        inv2 = 1.0 / (z * z)
        correction = max(EPS, 1.0 - inv2 + 3.0 * inv2 * inv2)
        return (
            -0.5 * z * z
            - math.log(z)
            - 0.5 * math.log(2.0 * math.pi)
            + math.log(correction)
        )

    @classmethod
    def _lognormal_log_survival(
        cls,
        log_mu: float,
        log_sigma: float,
        threshold: float,
    ) -> float:
        if threshold <= 0.0:
            return 0.0
        z = (math.log(float(threshold)) - log_mu) / max(EPS, log_sigma)
        return cls._normal_log_survival(z)

    @classmethod
    def _lognormal_mean_excess(
        cls,
        log_mu: float,
        log_sigma: float,
        threshold: float,
    ) -> float:
        # E[T-threshold | T>threshold] for log-normal total runtime T.
        threshold = max(0.0, float(threshold))
        sigma = max(EPS, float(log_sigma))
        if threshold <= EPS:
            return max(
                EPS,
                math.exp(min(700.0, log_mu + 0.5 * sigma * sigma)),
            )

        z = (math.log(threshold) - log_mu) / sigma
        log_denominator = cls._normal_log_survival(z)
        log_numerator = cls._normal_log_survival(z - sigma)
        log_conditional_mean = (
            log_mu
            + 0.5 * sigma * sigma
            + log_numerator
            - log_denominator
        )
        conditional_mean = math.exp(min(700.0, log_conditional_mean))
        return max(EPS, conditional_mean - threshold)

    def _scheduler_active_jobs(self) -> Dict[int, Dict[str, object]]:
        active: Dict[int, Dict[str, object]] = {}

        for job in getattr(self.jobs_manager, "active_jobs", []):
            job_id = int(job["job_id"])
            active[job_id] = dict(job)
            active[job_id]["job_id"] = job_id
            active[job_id]["nodes"] = list(job.get("nodes", []))

        for job_id, job in self._started_this_schedule.items():
            active[int(job_id)] = dict(job)
            active[int(job_id)]["nodes"] = list(job.get("nodes", []))

        now = float(self.current_time)
        for node_id in self.computing:
            job_id = self.state[node_id].get("job_id")
            if job_id is None:
                entry = self.next_releases.get(node_id, {})
                for segment in entry.get("queue", []):
                    candidate = self._phase_job_id(segment.get("phase"))
                    if candidate is not None:
                        job_id = candidate
                        break
            if job_id is None:
                continue

            job_id = int(job_id)
            record = active.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "start_time": now,
                    "nodes": [],
                },
            )
            record.setdefault("nodes", [])
            if node_id not in record["nodes"]:
                record["nodes"].append(node_id)

            entry = self.next_releases.get(node_id, {})
            for segment in entry.get("queue", []):
                if self._phase_job_id(segment.get("phase")) != job_id:
                    continue
                segment_start = float(segment.get("start_time", now))
                record["start_time"] = min(
                    float(record.get("start_time", segment_start)),
                    segment_start,
                )
                if "reqtime" not in record:
                    segment_finish = float(
                        segment.get("finish_time", segment_start)
                    )
                    requested_duration = segment_finish - segment_start
                    if requested_duration > EPS:
                        record["requested_finish_time"] = segment_finish
                break

        for record in active.values():
            record["nodes"] = sorted(set(record.get("nodes", [])))
        return active

    def _predict_active_job_remaining(
        self,
        job: Mapping[str, object],
    ) -> Dict[str, object]:
        now = float(self.current_time)
        start_time = float(job.get("start_time", now))
        elapsed = max(0.0, now - start_time)
        requested = self._requested_walltime(job)

        if requested is None:
            predicted_total = (
                float(self._ema_service_time)
                if self._ema_service_time is not None
                else self.max_decision_horizon
            )
            remaining = max(EPS, predicted_total - elapsed)
            source = "absolute_service_fallback"
        else:
            distribution = self._runtime_ratio_distribution()
            if distribution is None:
                predicted_total = requested
                remaining = max(EPS, predicted_total - elapsed)
                if elapsed >= predicted_total:
                    # No learned ratio exists yet.  Stay conservative, but keep
                    # the fallback proportional to this job's own request.
                    remaining = max(EPS, requested)
                source = "requested_duration_cold_start"
            else:
                ratio_mu, ratio_sigma = distribution
                log_total_mu = math.log(max(EPS, requested)) + ratio_mu
                predicted_total = requested * math.exp(
                    min(700.0, ratio_mu)
                )
                remaining = self._lognormal_mean_excess(
                    log_total_mu,
                    ratio_sigma,
                    elapsed,
                )
                source = "conditional_lognormal_runtime_ratio"

        return {
            "job_id": int(job["job_id"]),
            "nodes": list(job.get("nodes", [])),
            "requested_duration": requested,
            "elapsed": float(elapsed),
            "predicted_total_duration": float(predicted_total),
            "predicted_remaining": float(remaining),
            "prediction_source": source,
        }

    def _active_job_predictions(self) -> List[Dict[str, object]]:
        predictions = [
            self._predict_active_job_remaining(job)
            for job in self._scheduler_active_jobs().values()
        ]
        return sorted(predictions, key=lambda item: item["job_id"])

    def _prediction_release_map(self):
        release_map = {
            node_id: {
                "queue": [
                    dict(segment) for segment in entry.get("queue", [])
                ],
                "release_time": float(
                    entry.get("release_time", self.current_time)
                ),
            }
            for node_id, entry in self.next_releases.items()
        }

        now = float(self.current_time)
        for prediction in self._active_job_predictions():
            job_id = int(prediction["job_id"])
            predicted_release = now + float(
                prediction["predicted_remaining"]
            )
            for node_id in prediction["nodes"]:
                entry = release_map.get(node_id)
                if entry is None:
                    continue
                matched = False
                for segment in entry["queue"]:
                    if self._phase_job_id(segment.get("phase")) == job_id:
                        segment["finish_time"] = predicted_release
                        matched = True
                if not matched:
                    entry["queue"].append(
                        {
                            "phase": f"compute(job={job_id})",
                            "start_time": now - float(prediction["elapsed"]),
                            "finish_time": predicted_release,
                        }
                    )
                entry["release_time"] = predicted_release

        return release_map

    def _arrival_rate(self) -> float:
        if self._configured_arrival_rate is not None:
            return self._configured_arrival_rate
        return 1.0 / max(EPS, self._ema_interarrival)

    def _service_rate(self) -> float:
        if self._configured_service_rate is not None:
            return self._configured_service_rate

        mean_service = (
            self._ema_service_time
            if self._ema_service_time is not None
            else self._ema_requested_service_time
        )
        if mean_service is None:
            return 0.0

        return 1.0 / max(EPS, mean_service)

    def _resource_pmf(self) -> Dict[int, float]:
        if self._fixed_resource_pmf is not None:
            return dict(self._fixed_resource_pmf)

        total = sum(self._resource_weights.values())
        if total <= EPS:
            return {1: 1.0}

        return {
            int(requested_nodes): float(weight) / total
            for requested_nodes, weight
            in sorted(self._resource_weights.items())
            if weight > EPS
        }

    @staticmethod
    def _ema_update(
        previous: Optional[float],
        sample: float,
        smoothing: float,
    ) -> float:
        sample = max(0.0, float(sample))
        if previous is None:
            return sample
        return (
            float(smoothing) * sample
            + (1.0 - float(smoothing)) * float(previous)
        )

    @staticmethod
    def _ema_update_signed(
        previous: Optional[float],
        sample: float,
        smoothing: float,
    ) -> float:
        sample = float(sample)
        if previous is None:
            return sample
        return (
            float(smoothing) * sample
            + (1.0 - float(smoothing)) * float(previous)
        )

    def _update_resource_ema(self, requested_nodes: int) -> None:
        if not self._resource_weights:
            self._resource_weights[requested_nodes] = 1.0
            return

        keep = 1.0 - self.resource_ema_alpha
        for key in list(self._resource_weights):
            self._resource_weights[key] *= keep
            if self._resource_weights[key] <= EPS:
                del self._resource_weights[key]

        self._resource_weights[requested_nodes] = (
            self._resource_weights.get(requested_nodes, 0.0)
            + self.resource_ema_alpha
        )

    # ------------------------------------------------------------------
    # Node choice and event emission
    # ------------------------------------------------------------------

    def _select_sleeping_to_wake(self, count: int) -> List[int]:
        count = max(0, int(count))
        if count == 0:
            return []

        ranked = sorted(
            self.sleeping,
            key=lambda nid: (
                self.alpha * self._wake_lead_time_safe(nid)
                + self.beta * self._wake_transition_energy(nid),
                self._wake_lead_time_safe(nid),
                self._wake_transition_energy(nid),
                nid,
            ),
        )
        return ranked[:count]

    def _emit_switch_on(self, nodes: Iterable[int]) -> None:
        nodes = list(dict.fromkeys(nodes))
        nodes = [nid for nid in nodes if nid in self.sleeping]
        if not nodes:
            return

        self.push_event(
            float(self.current_time),
            {"type": "switch_on", "nodes": nodes},
        )
        node_set = set(nodes)
        self.sleeping = [nid for nid in self.sleeping if nid not in node_set]
        for nid in nodes:
            if nid not in self.switching_on:
                self.switching_on.append(nid)

    def _emit_switch_off(
        self,
        nodes: Iterable[int],
        *,
        ignore_timeout: bool = False,
    ) -> List[int]:
        nodes = list(dict.fromkeys(nodes))
        nodes = [
            nid
            for nid in nodes
            if (
                nid in self.idle
                and (ignore_timeout or self._can_switch_off(nid))
            )
        ]
        if not nodes:
            return []

        self.push_event(
            float(self.current_time),
            {"type": "switch_off", "nodes": nodes},
        )
        node_set = set(nodes)
        self.idle = [nid for nid in self.idle if nid not in node_set]
        for nid in nodes:
            if nid not in self.switching_off:
                self.switching_off.append(nid)
        super().remove_from_timeout_list(nodes)
        return list(nodes)

    def _can_switch_off(self, node_id: int) -> bool:
        if node_id not in self.idle:
            return False
        if not self.respect_timeout or self.timeout is None:
            return True
        expiry = self.timeout_list.get(node_id)
        if expiry is None:
            return False
        return float(self.current_time) + EPS >= float(expiry)

    def configure_scheduler_output_files(cfg):
        if cfg["run"]["algorithm"] != "snf_icon":
            return

        output_dir = Path(cfg["paths"]["output"]).resolve()
        algo_config = cfg["run"].setdefault(
            "algo_config",
            {},
        )

        algo_config["decision_log_enabled"] = True
        algo_config["decision_log_path"] = str(
            output_dir / "scheduler_decision_log.csv"
        )
        
    # ------------------------------------------------------------------
    # Power and transition helpers
    # ------------------------------------------------------------------

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
        if "power" in profile:
            return float(profile["power"])
        raise KeyError(
            f"DVFS profile {dvfs_mode!r} must define {power_key!r} "
            "or legacy 'power'"
        )

    def _machine_for(self, node_id: int):
        node_name = self.machines.nodes[node_id]["node_name"]
        return self.machines.node_specs[node_name]

    def _state_power(self, node_id: int, state_name: str) -> float:
        node = self.state[node_id]
        machine = self._machine_for(node_id)
        value = machine["states"][state_name]["power"]
        if value == "from_dvfs":
            # Non-compute phases count as waste and use the idle DVFS power.
            return self._resolve_dvfs_power(machine, node, "idle")
        return float(value)

    def _idle_power(self, node_id: int) -> float:
        return self._state_power(node_id, "active")

    def _transition_time_safe(
        self, node_id: int, from_state: str, to_state: str
    ) -> float:
        try:
            return max(0.0, float(super()._transition_time(
                node_id, from_state, to_state
            )))
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _wake_lead_time_safe(self, node_id: int) -> float:
        try:
            return max(0.0, float(super()._wake_lead_time(node_id)))
        except (KeyError, TypeError, ValueError):
            return (
                self._transition_time_safe(
                    node_id, "sleeping", "switching_on"
                )
                + self._transition_time_safe(
                    node_id, "switching_on", "active"
                )
            )

    def _switch_off_duration(self, node_id: int) -> float:
        return self._transition_time_safe(
            node_id, "switching_off", "sleeping"
        )

    def _remaining_switch_on_time(self, node_id: int) -> float:
        queued = self._remaining_phase_time(node_id, "switching_on")
        if queued is not None:
            return queued

        # Fallback for a simulator state that has not yet been mirrored into
        # next_releases.  state_start_time is optional in the attached base.
        node = self.state[node_id]
        total = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        started = float(node.get("state_start_time", self.current_time))
        elapsed = max(0.0, float(self.current_time) - started)
        return max(0.0, total - elapsed)

    def _remaining_switch_off_time(self, node_id: int) -> float:
        queued = self._remaining_phase_time(node_id, "switching_off")
        if queued is not None:
            return queued

        node = self.state[node_id]
        total = self._switch_off_duration(node_id)
        started = float(node.get("state_start_time", self.current_time))
        elapsed = max(0.0, float(self.current_time) - started)
        return max(0.0, total - elapsed)

    def _remaining_phase_time(
        self,
        node_id: int,
        phase_name: str,
    ) -> Optional[float]:
        entry = self.next_releases.get(node_id)
        if not entry:
            return None
        now = float(self.current_time)
        for segment in entry.get("queue", []):
            if str(segment.get("phase")) != phase_name:
                continue
            start = float(segment.get("start_time", now))
            finish = float(segment.get("finish_time", now))
            if start <= now + EPS and finish > now:
                return max(0.0, finish - now)
        return None

    def _wake_transition_energy(self, node_id: int) -> float:
        sleep_delay = self._transition_time_safe(
            node_id, "sleeping", "switching_on"
        )
        switch_delay = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        return (
            self._state_power(node_id, "sleeping") * sleep_delay
            + self._state_power(node_id, "switching_on") * switch_delay
        )

    def _remaining_switch_on_energy(self, node_id: int) -> float:
        return (
            self._state_power(node_id, "switching_on")
            * self._remaining_switch_on_time(node_id)
        )

    def _wake_energy_over_horizon(self, node_id: int, horizon: float) -> float:
        remaining = max(0.0, float(horizon))
        energy = 0.0

        sleep_delay = self._transition_time_safe(
            node_id, "sleeping", "switching_on"
        )
        duration = min(sleep_delay, remaining)
        energy += self._state_power(node_id, "sleeping") * duration
        remaining -= duration

        switch_delay = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        duration = min(switch_delay, remaining)
        energy += self._state_power(node_id, "switching_on") * duration
        remaining -= duration

        energy += self._idle_power(node_id) * remaining
        return float(energy)

    @staticmethod
    def _expected_residual_delay(duration: float, hazard: float) -> float:
        """E[(duration - T)^+] for T ~ Exp(hazard)."""
        duration = max(0.0, float(duration))
        hazard = max(EPS, float(hazard))
        return max(
            0.0,
            duration - (1.0 - math.exp(-hazard * duration)) / hazard,
        )

    # ------------------------------------------------------------------
    # Existing energy-aware idle-node selector
    # ------------------------------------------------------------------

    def _remaining_idle_timeout(self, node_id):
        if self.timeout is None:
            return math.inf
        timeout_time = self.timeout_list.get(node_id)
        if timeout_time is None:
            return math.inf
        return float(timeout_time - self.current_time)

    def _build_node_selection_static_data(self, candidates, release_map):
        """Precompute selector values that stay fixed during one plan.

        Future planning appends only ``compute(job=...)`` segments and updates
        release times.  Compute segments do not contribute to non-compute
        energy waste, so the values returned here remain valid for every job
        in the same future plan.
        """
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
            machine = self._machine_for(node_id)

            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
                state_priority = 0
            elif node["state"] == "active" and node.get("job_id") is not None:
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
                duration = max(0.0, duration)

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )
                base_energy_waste += float(energy_rate) * duration

            static_data[node_id] = {
                "base": float(base_energy_waste),
                "idle": self._idle_power(node_id),
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
            -math.inf if min_start_time is None else float(min_start_time)
        )

        if node_static_data is None:
            node_static_data = self._build_node_selection_static_data(
                candidates,
                release_map,
            )

        # The original implementation checked every unique release time until
        # at least ``required_nodes`` nodes were eligible.  That first feasible
        # time is exactly max(min_start_time, r-th earliest release), where r is
        # ``required_nodes``.  Computing it directly preserves the same start
        # time while avoiding repeated full-node scans.
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
            release_time = float(release_map[node_id]["release_time"])
            if release_time > candidate_time:
                continue

            data = node_static_data[node_id]
            if data["state_label"] in ("switching_off", "sleeping"):
                cost = data["base"]
            else:
                cost = data["base"] + data["idle"] * (
                    candidate_time - release_time
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_pmf(
        pmf: Optional[Mapping[int, float]],
    ) -> Optional[Dict[int, float]]:
        if pmf is None:
            return None

        cleaned: Dict[int, float] = {}
        for raw_req, raw_probability in pmf.items():
            req = int(raw_req)
            probability = float(raw_probability)
            if req <= 0:
                raise ValueError("resource_pmf keys must be positive integers")
            if probability < 0.0 or not math.isfinite(probability):
                raise ValueError(
                    "resource_pmf probabilities must be finite and non-negative"
                )
            if probability > 0.0:
                cleaned[req] = cleaned.get(req, 0.0) + probability

        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("resource_pmf must contain positive probability")
        return {req: probability / total for req, probability in cleaned.items()}

    @staticmethod
    def _nonnegative_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    @staticmethod
    def _positive_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @staticmethod
    def _unit_interval_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1]")
        return value

    @classmethod
    def _optional_positive_float(
        cls, name: str, value: Optional[float]
    ) -> Optional[float]:
        if value is None:
            return None
        return cls._positive_float(name, value)
    @classmethod
    def _optional_nonnegative_float(
        cls, name: str, value: Optional[float]
    ) -> Optional[float]:
        if value is None:
            return None
        return cls._nonnegative_float(name, value)
    
