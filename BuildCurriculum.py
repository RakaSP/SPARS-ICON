import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde


def load_workload(path):
    with open(path, "r", encoding="utf-8") as file:
        workload = json.load(file)

    jobs = sorted(
        workload["jobs"],
        key=lambda job: (
            job["subtime"],
            job["job_id"],
        ),
    )

    if not jobs:
        raise ValueError("Real workload contains no jobs")

    return workload, jobs


def get_real_samples(jobs):
    subtimes = np.asarray(
        [job["subtime"] for job in jobs],
        dtype=np.float64,
    )

    runtimes = np.asarray(
        [job["runtime"] for job in jobs],
        dtype=np.float64,
    )

    interarrivals = np.diff(
        np.concatenate((
            np.asarray([0.0]),
            subtimes,
        ))
    )

    if np.any(interarrivals < 0):
        raise ValueError(
            "Job subtimes must be non-decreasing"
        )

    if np.any(runtimes < 0):
        raise ValueError(
            "Job runtimes cannot be negative"
        )

    mean_interarrival = float(
        np.mean(interarrivals)
    )

    mean_runtime = float(
        np.mean(runtimes)
    )

    if mean_interarrival <= 0:
        raise ValueError(
            "Mean interarrival time must be greater than 0"
        )

    if mean_runtime <= 0:
        raise ValueError(
            "Mean runtime must be greater than 0"
        )

    return {
        "interarrivals": interarrivals,
        "runtimes": runtimes,
        "mean_interarrival": mean_interarrival,
        "mean_runtime": mean_runtime,
        "arrival_rate": 1.0 / mean_interarrival,
        "service_rate": 1.0 / mean_runtime,
    }


def sample_exponential(mean_value, count, rng):
    return rng.exponential(
        scale=mean_value,
        size=count,
    )


def sample_kde(values, count, rng):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    log_values = np.log1p(values)

    if (
        len(log_values) < 2
        or np.std(log_values) <= 1e-12
    ):
        return rng.choice(
            values,
            size=count,
            replace=True,
        )

    kde = gaussian_kde(log_values)
    sampled_values = []

    while len(sampled_values) < count:
        remaining = count - len(sampled_values)
        draw_count = max(remaining * 2, 32)

        sampled_log = kde.resample(
            draw_count,
            seed=rng,
        ).reshape(-1)

        sampled = np.expm1(sampled_log)
        sampled = sampled[
            np.isfinite(sampled)
            & (sampled >= 0)
        ]

        sampled_values.extend(
            sampled.tolist()
        )

    return np.asarray(
        sampled_values[:count],
        dtype=np.float64,
    )


def to_integer_interarrivals(values):
    values = np.rint(values)
    values = np.clip(
        values,
        0,
        None,
    )

    return values.astype(np.int64)


def to_integer_runtimes(values):
    values = np.rint(values)
    values = np.clip(
        values,
        0,
        None,
    )

    return values.astype(np.int64)


def build_workload(
    template_jobs,
    interarrivals,
    runtimes,
):
    if len(template_jobs) != len(interarrivals):
        raise ValueError(
            "Number of jobs and interarrivals do not match"
        )

    if len(template_jobs) != len(runtimes):
        raise ValueError(
            "Number of jobs and runtimes do not match"
        )

    integer_interarrivals = (
        to_integer_interarrivals(
            interarrivals
        )
    )

    integer_runtimes = (
        to_integer_runtimes(
            runtimes
        )
    )

    subtimes = np.cumsum(
        integer_interarrivals
    )

    generated_jobs = []

    for index, template in enumerate(
        template_jobs
    ):
        generated_jobs.append({
            "job_id": index + 1,
            "res": int(template["res"]),
            "subtime": int(subtimes[index]),
            "reqtime": int(template["reqtime"]),
            "runtime": int(integer_runtimes[index]),
            "profile": str(
                template.get(
                    "profile",
                    "100",
                )
            ),
            "user_id": int(
                template.get(
                    "user_id",
                    0,
                )
            ),
        })

    nb_res = max(
        job["res"]
        for job in generated_jobs
    )

    return {
        "nb_res": nb_res,
        "jobs": generated_jobs,
    }


def save_workload(workload, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            workload,
            file,
            indent=4,
        )


def calculate_generated_stats(workload):
    jobs = workload["jobs"]

    subtimes = np.asarray(
        [job["subtime"] for job in jobs],
        dtype=np.float64,
    )

    runtimes = np.asarray(
        [job["runtime"] for job in jobs],
        dtype=np.float64,
    )

    interarrivals = np.diff(
        np.concatenate((
            np.asarray([0.0]),
            subtimes,
        ))
    )

    mean_interarrival = float(
        np.mean(interarrivals)
    )

    mean_runtime = float(
        np.mean(runtimes)
    )

    return {
        "number_of_jobs": len(jobs),
        "nb_res": workload["nb_res"],
        "mean_interarrival_time": mean_interarrival,
        "arrival_rate": (
            1.0 / mean_interarrival
            if mean_interarrival > 0
            else None
        ),
        "mean_runtime": mean_runtime,
        "service_rate": (
            1.0 / mean_runtime
            if mean_runtime > 0
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "real_workload",
        help="Path to the real workload JSON",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    real_path = Path(
        args.real_workload
    ).resolve()

    if args.output_dir is None:
        output_dir = real_path.parent
    else:
        output_dir = Path(
            args.output_dir
        ).resolve()

    _, real_jobs = load_workload(
        real_path
    )

    real_samples = get_real_samples(
        real_jobs
    )

    job_count = len(real_jobs)

    seed_sequence = np.random.SeedSequence(
        args.seed
    )

    simple_seed, complex_seed = (
        seed_sequence.spawn(2)
    )

    simple_rng = np.random.default_rng(
        simple_seed
    )

    complex_rng = np.random.default_rng(
        complex_seed
    )

    simple_interarrivals = (
        sample_exponential(
            real_samples[
                "mean_interarrival"
            ],
            job_count,
            simple_rng,
        )
    )

    simple_runtimes = (
        sample_exponential(
            real_samples[
                "mean_runtime"
            ],
            job_count,
            simple_rng,
        )
    )

    complex_interarrivals = sample_kde(
        real_samples["interarrivals"],
        job_count,
        complex_rng,
    )

    complex_runtimes = sample_kde(
        real_samples["runtimes"],
        job_count,
        complex_rng,
    )

    simple_workload = build_workload(
        real_jobs,
        simple_interarrivals,
        simple_runtimes,
    )

    complex_workload = build_workload(
        real_jobs,
        complex_interarrivals,
        complex_runtimes,
    )

    simple_path = (
        output_dir
        / f"{real_path.stem}_simple.json"
    )

    complex_path = (
        output_dir
        / f"{real_path.stem}_complex.json"
    )

    summary_path = (
        output_dir
        / f"{real_path.stem}_dataset_summary.json"
    )

    save_workload(
        simple_workload,
        simple_path,
    )

    save_workload(
        complex_workload,
        complex_path,
    )

    summary = {
        "seed": args.seed,
        "real_workload": str(real_path),
        "simple_workload": str(simple_path),
        "complex_workload": str(complex_path),
        "real": {
            "number_of_jobs": job_count,
            "nb_res": max(
                job["res"]
                for job in real_jobs
            ),
            "mean_interarrival_time": (
                real_samples[
                    "mean_interarrival"
                ]
            ),
            "arrival_rate": (
                real_samples[
                    "arrival_rate"
                ]
            ),
            "mean_runtime": (
                real_samples[
                    "mean_runtime"
                ]
            ),
            "service_rate": (
                real_samples[
                    "service_rate"
                ]
            ),
        },
        "simple": calculate_generated_stats(
            simple_workload
        ),
        "complex": calculate_generated_stats(
            complex_workload
        ),
        "complex_distribution": {
            "interarrival": "log1p Gaussian KDE",
            "runtime": "log1p Gaussian KDE",
        },
        "curriculum": [
            {
                "stage": 1,
                "dataset": str(simple_path),
                "epochs": "X",
            },
            {
                "stage": 2,
                "dataset": str(real_path),
                "epochs": "X",
            },
            {
                "stage": 3,
                "dataset": str(complex_path),
                "epochs": "X",
            },
        ],
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("Real dataset:")
    print(" jobs:", job_count)
    print(
        " mean interarrival:",
        real_samples["mean_interarrival"],
    )
    print(
        " arrival rate:",
        real_samples["arrival_rate"],
    )
    print(
        " mean runtime:",
        real_samples["mean_runtime"],
    )
    print(
        " service rate:",
        real_samples["service_rate"],
    )
    print("\nGenerated:")
    print(" simple:", simple_path)
    print(" complex:", complex_path)
    print(" summary:", summary_path)


if __name__ == "__main__":
    main()