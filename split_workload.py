import json
from pathlib import Path


def split_workload(
    input_path,
    output_path,
    start_percent=None,
    finish_percent=None,
    start_job=0,
    job_count=None,
    shift_subtime=False,
    reindex_jobs=False,
):
    """
    Select jobs using either:

    1. Percentage range:
        start_percent=0.25,
        finish_percent=0.75

    2. Number of jobs:
        start_job=100,
        job_count=500

    When job_count is provided, start_percent and finish_percent
    are ignored.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        workload = json.load(f)

    jobs = sorted(
        workload["jobs"],
        key=lambda job: job["subtime"],
    )

    n = len(jobs)

    if n == 0:
        raise ValueError("workload contains no jobs")

    # Select by number of jobs.
    if job_count is not None:
        if job_count <= 0:
            raise ValueError("job_count must be > 0")

        # Negative start_job counts from the end.
        if start_job < 0:
            start_idx = n + start_job
        else:
            start_idx = start_job

        finish_idx = start_idx + job_count

        if start_idx < 0 or start_idx >= n:
            raise ValueError(
                f"start_job={start_job} is outside the workload "
                f"containing {n} jobs"
            )

        if finish_idx > n:
            raise ValueError(
                f"Requested indices {start_idx} to {finish_idx - 1}, "
                f"but workload only contains {n} jobs"
            )

        selection_description = (
            f"{job_count} jobs starting from index {start_idx}"
        )
    # Select by percentage.
    else:
        if start_percent is None or finish_percent is None:
            raise ValueError(
                "Provide either job_count, or both "
                "start_percent and finish_percent"
            )

        if not 0.0 <= start_percent < finish_percent <= 1.0:
            raise ValueError(
                "Percentages must satisfy: "
                "0.0 <= start_percent < finish_percent <= 1.0"
            )

        start_idx = int(n * start_percent)
        finish_idx = int(n * finish_percent)

        if finish_percent == 1.0:
            finish_idx = n

        if start_idx >= finish_idx:
            raise ValueError(
                "The selected percentage range contains 0 jobs"
            )

        selection_description = (
            f"{start_percent:.1%} to {finish_percent:.1%}"
        )

    selected_jobs = jobs[start_idx:finish_idx]

    if shift_subtime:
        if start_idx == 0:
            shift_amount = selected_jobs[0]["subtime"]
        else:
            # Preserve the interarrival time between the preceding job
            # and the first selected job.
            shift_amount = jobs[start_idx - 1]["subtime"]
    else:
        shift_amount = 0

    new_jobs = []

    for i, job in enumerate(selected_jobs):
        new_job = job.copy()

        if shift_subtime:
            new_job["subtime"] -= shift_amount

        if reindex_jobs:
            new_job["job_id"] = i + 1

        new_jobs.append(new_job)

    new_workload = workload.copy()
    new_workload["jobs"] = new_jobs

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            new_workload,
            f,
            indent=4,
        )

    print(f"Total jobs:       {n}")
    print(f"Selection:        {selection_description}")
    print(f"Selected indices: {start_idx} to {finish_idx - 1}")
    print(f"Saved jobs:       {len(new_jobs)}")
    print(f"Subtime shifted:  {shift_amount}")
    print(f"Output:           {output_path}")


split_workload(
    input_path="workloads/json/DAS2-fs1-2003-1.json",
    output_path="workloads/json/DAS2-fs1-3001-4000.json",
    start_job=3000,
    job_count=1000,
    shift_subtime=True,
    reindex_jobs=False,
)