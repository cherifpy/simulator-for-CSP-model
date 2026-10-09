"""One-off generator for a small Poisson-arrival live-simulation instance (jobs.json +
infrastructure.csv), in the same format xp_online_grid5000.py expects (and the same "mixte x2"
sizing convention as workloads/poisson_mixte_x2_20j_50n/: task_duration in [200,600]s,
dataset_size in [4096,409600]MB) -- just at a smaller scale (10 jobs / 20 nodes) for quickly
iterating on controlLoopReplicationFactor.

Usage:
    python3 exps/generate_poisson_instance.py --out workloads/poisson_mixte_x2_10j_20n \
        --nb-jobs 10 --nb-nodes 20 --seed 700
"""
import argparse
import csv
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--nb-jobs", type=int, default=10)
    parser.add_argument("--nb-nodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--nb-tasks-range", type=int, nargs=2, default=[1, 20])
    parser.add_argument("--task-duration-range", type=float, nargs=2, default=[200, 600])
    parser.add_argument("--dataset-size-range", type=float, nargs=2, default=[4096, 409600])
    parser.add_argument("--bandwidth-range", type=float, nargs=2, default=[25, 1280])
    parser.add_argument("--compute-range", type=float, nargs=2, default=[0.5, 8])
    parser.add_argument("--energy-range", type=float, nargs=2, default=[0.1, 2.1])
    parser.add_argument("--storage-range", type=float, nargs=2, default=[8000, 550000])
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for job_id in range(args.nb_jobs):
        jobs.append({
            "job_id": job_id,
            "id_dataset": job_id,
            "nb_tasks": random.randint(*args.nb_tasks_range),
            "task_duration": round(random.uniform(*args.task_duration_range)),
            "dataset_size": round(random.uniform(*args.dataset_size_range)),
            "arriving_time": 0.0,  # ignored by jobsInjectorBasedOnLambdaPoisson (poisson=True)
            "type_of_job": None,
        })
    with open(os.path.join(args.out, "jobs.json"), "w") as f:
        json.dump(jobs, f, indent=2)

    with open(os.path.join(args.out, "infrastructure.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"])
        for _ in range(args.nb_nodes):
            writer.writerow([
                round(random.uniform(*args.bandwidth_range)),
                round(random.uniform(*args.compute_range), 6),
                round(random.uniform(*args.energy_range), 6),
                round(random.uniform(*args.storage_range)),
            ])

    print(f"Wrote {args.nb_jobs} jobs / {args.nb_nodes} nodes to {args.out}")


if __name__ == "__main__":
    main()
