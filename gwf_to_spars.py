#!/usr/bin/env python3
"""Convert Grid Workload Format (GWF) traces to SPARS workload JSON.

The converter targets the workload schema used by SPARS-Pub:

    {
      "nb_res": 475,
      "jobs": [
        {
          "job_id": 1,
          "res": 1,
          "subtime": 0,
          "user_id": "U2004S1",
          "reqtime": 259200,
          "runtime": 138467,
          "profile": "default"
        }
      ],
      "profiles": {
        "default": {"cpu": 1, "com": 0, "type": "parallel_homogeneous"}
      }
    }

By default, only GWF status=1 jobs with positive actual runtimes are kept.
Submission times are shifted so the first retained job arrives at time 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


GWF_FIELD_COUNT = 29


@dataclass(frozen=True)
class RawJob:
    job_id: int | str
    submit_time: int
    runtime: int
    allocated_processors: int
    requested_processors: int
    requested_time: int
    status: int
    user_id: str
    executable_id: str


@dataclass
class Counters:
    data_rows: int = 0
    malformed_rows: int = 0
    skipped_status: int = 0
    skipped_runtime: int = 0
    skipped_resources: int = 0
    skipped_by_offset: int = 0
    selected_jobs: int = 0


def parse_int(value: str, *, field: str, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"line {line_number}: invalid integer in {field}: {value!r}"
        ) from exc


def parse_job(line: str, line_number: int) -> RawJob:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != GWF_FIELD_COUNT:
        raise ValueError(
            f"line {line_number}: expected {GWF_FIELD_COUNT} tab-separated fields, "
            f"found {len(fields)}"
        )

    raw_job_id = fields[0]
    try:
        job_id: int | str = int(raw_job_id)
    except ValueError:
        job_id = raw_job_id

    return RawJob(
        job_id=job_id,
        submit_time=parse_int(fields[1], field="SubmitTime", line_number=line_number),
        runtime=parse_int(fields[3], field="RunTime", line_number=line_number),
        allocated_processors=parse_int(
            fields[4], field="NProcs", line_number=line_number
        ),
        requested_processors=parse_int(
            fields[7], field="ReqNProcs", line_number=line_number
        ),
        requested_time=parse_int(fields[8], field="ReqTime", line_number=line_number),
        status=parse_int(fields[10], field="Status", line_number=line_number),
        user_id=fields[11],
        executable_id=fields[13],
    )


def read_header_processors(path: Path) -> int | None:
    pattern = re.compile(r"^\s*#\s*processors\s*:\s*(\d+)\b", re.IGNORECASE)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                break
            match = pattern.match(line)
            if match:
                processors = int(match.group(1))
                return processors if processors > 0 else None
    return None


def parse_statuses(value: str) -> set[int] | None:
    normalized = value.strip().lower()
    if normalized in {"all", "*"}:
        return None
    aliases = {
        "completed": 1,
        "success": 1,
        "successful": 1,
        "failed": 0,
        "cancelled": 5,
        "canceled": 5,
    }
    statuses: set[int] = set()
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if token in aliases:
            statuses.add(aliases[token])
        else:
            try:
                statuses.add(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid status {token!r}; use completed, failed, cancelled, all, "
                    "or comma-separated numeric codes"
                ) from exc
    if not statuses:
        raise argparse.ArgumentTypeError("at least one status is required")
    return statuses


def choose_resources(job: RawJob, mode: str) -> int:
    if mode == "requested":
        return (
            job.requested_processors
            if job.requested_processors > 0
            else job.allocated_processors
        )
    if mode == "allocated":
        return (
            job.allocated_processors
            if job.allocated_processors > 0
            else job.requested_processors
        )
    if mode == "max":
        return max(job.requested_processors, job.allocated_processors)
    raise AssertionError(f"unexpected resource mode: {mode}")


def choose_runtime(job: RawJob, policy: str) -> int | None:
    if job.runtime > 0:
        return job.runtime
    if policy == "skip":
        return None
    if policy == "requested":
        return job.requested_time if job.requested_time > 0 else None
    if policy == "minimum":
        return 1
    raise AssertionError(f"unexpected runtime policy: {policy}")


def profile_id_for(job: RawJob, mode: str) -> str:
    if mode == "single":
        return "default"
    if mode == "executable":
        return job.executable_id if job.executable_id not in {"", "-1"} else "default"
    raise AssertionError(f"unexpected profile mode: {mode}")


def iter_selected_jobs(
    path: Path,
    *,
    statuses: set[int] | None,
    resource_mode: str,
    runtime_policy: str,
    profile_mode: str,
    skip: int,
    limit: int | None,
    strict: bool,
    counters: Counters | None = None,
) -> Iterator[dict[str, int | str]]:
    retained_before_slice = 0
    emitted = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            if counters is not None:
                counters.data_rows += 1

            try:
                raw = parse_job(line, line_number)
            except ValueError:
                if strict:
                    raise
                if counters is not None:
                    counters.malformed_rows += 1
                continue

            if statuses is not None and raw.status not in statuses:
                if counters is not None:
                    counters.skipped_status += 1
                continue

            runtime = choose_runtime(raw, runtime_policy)
            if runtime is None or runtime <= 0:
                if counters is not None:
                    counters.skipped_runtime += 1
                continue

            resources = choose_resources(raw, resource_mode)
            if resources <= 0:
                if counters is not None:
                    counters.skipped_resources += 1
                continue

            reqtime = raw.requested_time if raw.requested_time > 0 else runtime
            user_id = raw.user_id if raw.user_id not in {"", "-1"} else "unknown"

            if retained_before_slice < skip:
                retained_before_slice += 1
                if counters is not None:
                    counters.skipped_by_offset += 1
                continue

            if limit is not None and emitted >= limit:
                break

            retained_before_slice += 1
            emitted += 1
            if counters is not None:
                counters.selected_jobs += 1

            yield {
                "job_id": raw.job_id,
                "res": resources,
                "subtime": raw.submit_time,
                "user_id": user_id,
                "reqtime": reqtime,
                "runtime": runtime,
                "profile": profile_id_for(raw, profile_mode),
            }


def json_dump_compact(value: object, handle: TextIO) -> None:
    json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))


def write_spars_json(
    output_path: Path,
    *,
    nb_res: int,
    jobs: Iterator[dict[str, int | str]],
    origin: int,
    profiles: dict[str, dict[str, int | str]],
    pretty: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            if pretty:
                handle.write("{\n  \"nb_res\": ")
                handle.write(str(nb_res))
                handle.write(",\n  \"jobs\": [")
            else:
                handle.write('{"nb_res":')
                handle.write(str(nb_res))
                handle.write(',"jobs":[')

            first = True
            for job in jobs:
                normalized = dict(job)
                normalized["subtime"] = int(normalized["subtime"]) - origin
                if normalized["subtime"] < 0:
                    raise ValueError(
                        "a normalized submission time became negative; input may not be "
                        "chronologically sorted"
                    )

                if not first:
                    handle.write(",")
                if pretty:
                    handle.write("\n    ")
                    json.dump(normalized, handle, ensure_ascii=False, separators=(", ", ": "))
                else:
                    json_dump_compact(normalized, handle)
                first = False
                count += 1

            if pretty:
                if count:
                    handle.write("\n  ")
                handle.write("],\n  \"profiles\": ")
                json.dump(profiles, handle, ensure_ascii=False, indent=2)
                handle.write("\n}\n")
            else:
                handle.write('],"profiles":')
                json_dump_compact(profiles, handle)
                handle.write("}\n")

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 29-field Grid Workload Format trace to SPARS JSON."
    )
    parser.add_argument("input", type=Path, help="input .gwf file")
    parser.add_argument("output", type=Path, help="output .json file")
    parser.add_argument(
        "--statuses",
        type=parse_statuses,
        default={1},
        help="statuses to keep (default: completed; examples: all, 1, 0,1,5)",
    )
    parser.add_argument(
        "--resource-field",
        choices=("requested", "allocated", "max"),
        default="requested",
        help="GWF processor field used for SPARS res (default: requested)",
    )
    parser.add_argument(
        "--runtime-policy",
        choices=("skip", "requested", "minimum"),
        default="skip",
        help=(
            "handling for missing/non-positive RunTime: skip, use ReqTime, or use 1 "
            "second (default: skip)"
        ),
    )
    parser.add_argument(
        "--time-origin",
        choices=("first", "absolute"),
        default="first",
        help="shift first retained submission to zero, or preserve timestamps",
    )
    parser.add_argument(
        "--nb-res",
        type=int,
        default=None,
        help="platform node count; defaults to GWF '# Processors:' or max job size",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("single", "executable"),
        default="single",
        help="one placeholder profile, or one profile per GWF ExecutableID",
    )
    parser.add_argument("--skip", type=int, default=0, help="skip N usable jobs")
    parser.add_argument("--limit", type=int, default=None, help="write at most N jobs")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop on malformed rows instead of skipping them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        raise FileNotFoundError(f"input file not found: {args.input}")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.nb_res is not None and args.nb_res <= 0:
        raise ValueError("--nb-res must be positive")

    # First pass: collect only conversion metadata. This keeps memory use nearly
    # constant even for multi-million-row GWA traces.
    counters = Counters()
    first_submit: int | None = None
    max_job_resources = 0
    profile_ids: set[str] = set()

    for job in iter_selected_jobs(
        args.input,
        statuses=args.statuses,
        resource_mode=args.resource_field,
        runtime_policy=args.runtime_policy,
        profile_mode=args.profile_mode,
        skip=args.skip,
        limit=args.limit,
        strict=args.strict,
        counters=counters,
    ):
        if first_submit is None:
            first_submit = int(job["subtime"])
        max_job_resources = max(max_job_resources, int(job["res"]))
        profile_ids.add(str(job["profile"]))

    if first_submit is None:
        raise ValueError("no usable jobs matched the selected conversion rules")

    origin = first_submit if args.time_origin == "first" else 0

    header_processors = read_header_processors(args.input)
    nb_res = args.nb_res
    if nb_res is None:
        nb_res = header_processors if header_processors is not None else max_job_resources
    if nb_res < max_job_resources:
        raise ValueError(
            f"nb_res={nb_res} is smaller than the largest job request "
            f"({max_job_resources})"
        )

    profiles = {
        profile_id: {"cpu": 1, "com": 0, "type": "parallel_homogeneous"}
        for profile_id in sorted(profile_ids)
    }

    # Second pass: stream jobs directly into the JSON array.
    count = write_spars_json(
        args.output,
        nb_res=nb_res,
        jobs=iter_selected_jobs(
            args.input,
            statuses=args.statuses,
            resource_mode=args.resource_field,
            runtime_policy=args.runtime_policy,
            profile_mode=args.profile_mode,
            skip=args.skip,
            limit=args.limit,
            strict=args.strict,
            counters=None,
        ),
        origin=origin,
        profiles=profiles,
        pretty=args.pretty,
    )

    print(f"Wrote {count:,} jobs to {args.output}")
    print(f"nb_res={nb_res}; time_origin={origin}; max_job_res={max_job_resources}")
    print(
        "Rows: "
        f"data={counters.data_rows:,}, malformed={counters.malformed_rows:,}, "
        f"status_skipped={counters.skipped_status:,}, "
        f"runtime_skipped={counters.skipped_runtime:,}, "
        f"resource_skipped={counters.skipped_resources:,}, "
        f"offset_skipped={counters.skipped_by_offset:,}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
