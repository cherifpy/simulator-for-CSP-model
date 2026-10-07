"""Generates a FRESH random scenario (jobs + infrastructure) at the start of each iteration --
no file-based instance, nothing read from workloads/ -- instead of replaying the same fixed
jobs.json/infrastructure.csv every time. Each iteration:
  1. Draws n_existing uniformly from --n-existing-range, then generates n_existing existing jobs
     + 1 new job (nb_tasks/task_duration from config.json's own ranges, dataset_size from
     --dataset-size-range) and --nb-nodes compute nodes (bandwidth/compute_capacity/storage
     from realistic ranges, wide enough storage headroom that the generated dataset sizes are
     always placeable).
  2. Writes that scenario to its own throwaway instance directory (jobs.json +
     infrastructure.csv), then invokes the EXISTING, already-validated xp_dataset_size_sweep.py
     as a subprocess against it -- not a reimplementation of state-A construction or the 3
     approaches' own logic, just a fresh instance to point it at. All 3 approaches
     (incremental/online_biobj/hybrid) run against the SAME generated scenario within one
     iteration (xp_dataset_size_sweep.py's own run_tier already guarantees this -- state A is
     built once per invocation, reused for every approach).
  3. Repeats --n-iterations times, each with its own fresh scenario (distinct seed), writing
     results to its own subdirectory.

After all iterations, aggregates every iteration's jobs_detail_by_run.csv/transfers_detail_by_run.csv
into one combined CSV each, tagged with iteration + the generated n_existing, so the N iterations
give a genuine distribution (mean/std/max/min/CDF) per approach -- not a single data point.

Run with the venv python:
    python3 xp_generated_scenarios.py --n-iterations 10 --n-existing-range 5 15 \\
        --dataset-size-range 10240 102400 --nb-nodes 50 --results-root <dir>
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../simulator/exps
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATOR_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, "env", "bin", "python3")

# Same ranges config.json itself uses for job generation (min/max_nb_tasks_per_job,
# min/max_task_duration_sec) -- kept in sync with that file's own values, not re-derived.
NB_TASKS_RANGE = (1, 20)
TASK_DURATION_RANGE = (100, 150)
# Node characteristic ranges, matching inst-20J-50N's own empirical spread (bandwidth 12-799,
# compute_capacity/computation_nodes 1.0-9.7) -- storage_capacity's ceiling is set well above any
# --dataset-size-range this script will be asked to use, so a generated job is always placeable
# on at least one node regardless of which range is passed.
BANDWIDTH_RANGE = (12, 800)
COMPUTE_CAPACITY_RANGE = (1.0, 9.7)
STORAGE_HEADROOM_FACTOR = 1.3  # storage ceiling = dataset-size-range's own max * this factor


def generate_scenario(rng, n_existing, nb_nodes, existing_dataset_size_range, existing_task_duration_range,
                       new_job_dataset_size_range=None, new_job_task_duration_range=None,
                       bandwidth_range=BANDWIDTH_RANGE, compute_capacity_range=COMPUTE_CAPACITY_RANGE):
    """Returns (existing_jobs_raw, new_job_raw, infra_rows) -- same shapes
    xp_dataset_size_sweep.py's build_state_a expects from jobs.json/infrastructure.csv.

    existing_* and new_job_* ranges are SEPARATE (new_job_* defaults to existing_* when unset, the
    original i.i.d.-everything behavior) so a scenario can draw a deliberately heavy existing batch
    -- enough load to leave the infrastructure genuinely congested once state A is built -- while
    the new arrival itself stays an ordinary job, instead of the new job's own size being what
    makes or breaks congestion. This is what lets a joint reschedule have real room to help: the
    system is already tight BEFORE the new job shows up, not because the new job happens to be
    unusually large."""
    new_job_dataset_size_range = new_job_dataset_size_range or existing_dataset_size_range
    new_job_task_duration_range = new_job_task_duration_range or existing_task_duration_range

    existing_jobs_raw = []
    arriving_time = 0.0
    for i in range(n_existing):
        arriving_time += rng.uniform(5, 50)
        existing_jobs_raw.append({
            "job_id": i,
            "id_dataset": i,
            "nb_tasks": rng.randint(*NB_TASKS_RANGE),
            "task_duration": rng.randint(*existing_task_duration_range),
            "dataset_size": rng.randint(*existing_dataset_size_range),
            "arriving_time": arriving_time,
            "type_of_job": None,
        })
    arriving_time += rng.uniform(5, 50)
    new_job_raw = {
        "job_id": n_existing,
        "id_dataset": n_existing,
        "nb_tasks": rng.randint(*NB_TASKS_RANGE),
        "task_duration": rng.randint(*new_job_task_duration_range),
        "dataset_size": rng.randint(*new_job_dataset_size_range),
        "arriving_time": arriving_time,
        "type_of_job": None,
    }

    storage_ceiling = int(max(existing_dataset_size_range[1], new_job_dataset_size_range[1]) * STORAGE_HEADROOM_FACTOR)
    infra_rows = []
    for _ in range(nb_nodes):
        infra_rows.append({
            "bandwidth": rng.randint(*bandwidth_range),
            "computation_nodes": round(rng.uniform(*compute_capacity_range), 6),
            # Overwritten by generateHeterogeneousInfrastructureEquilibre's own
            # random.uniform(0.1, 2.1) regardless of what's written here -- kept for a complete,
            # self-describing CSV, not because it's actually read as the final value.
            "energy_consumption": round(rng.uniform(0.1, 2.1), 6),
            "storage_capacity": rng.randint(2050, storage_ceiling),
        })
    return existing_jobs_raw, new_job_raw, infra_rows


def write_instance(instance_dir, existing_jobs_raw, new_job_raw, infra_rows):
    os.makedirs(instance_dir, exist_ok=True)
    with open(os.path.join(instance_dir, "jobs.json"), "w") as f:
        json.dump(existing_jobs_raw + [new_job_raw], f, indent=2)
    with open(os.path.join(instance_dir, "infrastructure.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"])
        writer.writeheader()
        writer.writerows(infra_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-existing-range", type=int, nargs=2, default=(5, 15),
                         help="n_existing is drawn uniformly from this range (inclusive) each iteration.")
    parser.add_argument("--nb-nodes", type=int, default=50)
    parser.add_argument("--dataset-size-range", type=int, nargs=2, default=(10240, 102400),
                         help="Dataset size range (MB) for EXISTING jobs (and the new job too, "
                              "unless --new-job-dataset-size-range overrides it) -- default "
                              "matches the 'mixed'/'full' tier's own span. Storage capacity's own "
                              "ceiling scales with the larger of this and the new-job range "
                              "automatically (see STORAGE_HEADROOM_FACTOR) -- no separate flag "
                              "needed for it.")
    parser.add_argument("--task-duration-range", type=int, nargs=2, default=TASK_DURATION_RANGE,
                         help="Per-task duration range (s) for EXISTING jobs (and the new job too, "
                              "unless --new-job-task-duration-range overrides it).")
    parser.add_argument("--new-job-dataset-size-range", type=int, nargs=2, default=None,
                         help="Dataset size range (MB) for the NEW job only -- defaults to "
                              "--dataset-size-range (the original i.i.d.-everything behavior). "
                              "Set this LOWER than --dataset-size-range to build a 'stress then "
                              "arrival' scenario: a deliberately heavy existing batch saturates "
                              "the infrastructure, then an ORDINARY new job arrives into that "
                              "already-congested state -- the regime where a joint reschedule has "
                              "real room to help, unlike i.i.d. scenarios where congestion right "
                              "at the new job's own arrival is rare by construction.")
    parser.add_argument("--new-job-task-duration-range", type=int, nargs=2, default=None,
                         help="Per-task duration range (s) for the NEW job only -- defaults to "
                              "--task-duration-range. See --new-job-dataset-size-range.")
    parser.add_argument("--approaches", nargs="+", default=["incremental", "epsilon", "hybrid"],
                         choices=["incremental", "epsilon", "hybrid"])
    parser.add_argument("--seed", type=int, default=42, help="Base seed; iteration i uses seed+i.")
    parser.add_argument("--incremental-time-limit", type=int, default=30)
    parser.add_argument("--epsilon-time-limit", type=int, default=600)
    parser.add_argument("--epsilon-fraction", type=float, default=0.1)
    parser.add_argument("--epsilon-phase1-fraction", type=float, default=0.833333333333)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--hybrid-max-budget", type=float, default=600)
    parser.add_argument("--hybrid-incremental-time-limit", type=float, default=30)
    parser.add_argument("--adaptive-gate-metric", choices=["new_job", "max", "mean"], default="max")
    parser.add_argument("--adaptive-f1-relative-margin", type=float, default=None)
    parser.add_argument("--adaptive-f1-dynamic-margin", action="store_true")
    parser.add_argument("--adaptive-selection-metric", choices=["new_job", "max"], default="max")
    parser.add_argument("--state-a-time-limit", type=int, default=30)
    parser.add_argument("--iteration-offset", type=int, default=0,
                         help="Global index of the first iteration this invocation runs (iter_N "
                              "directories and the seed are both offset by this). Lets --n-"
                              "iterations 1 --iteration-offset K be submitted as its own OAR job "
                              "for K in 0..N-1 -- true parallelism across separate nodes, instead "
                              "of one job running all N iterations sequentially -- while writing "
                              "into the SAME --results-root as if one sequential run had produced "
                              "it. Each such job only aggregates the iteration(s) IT ran; run with "
                              "--n-iterations matching the full count (and this flag at its "
                              "default 0) once all the parallel jobs are done to build the final "
                              "combined_jobs_detail.csv/combined_transfers_detail.csv over "
                              "whatever iter_* directories exist by then (see the aggregation step "
                              "below, which globs rather than assuming a single sequential run).")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    os.makedirs(args.results_root, exist_ok=True)
    instances_root = os.path.join(args.results_root, "_generated_instances")

    for i in range(args.iteration_offset, args.iteration_offset + args.n_iterations):
        rng = random.Random(args.seed + i)
        n_existing = rng.randint(*args.n_existing_range)
        existing_jobs_raw, new_job_raw, infra_rows = generate_scenario(
            rng, n_existing, args.nb_nodes, tuple(args.dataset_size_range), tuple(args.task_duration_range),
            tuple(args.new_job_dataset_size_range) if args.new_job_dataset_size_range else None,
            tuple(args.new_job_task_duration_range) if args.new_job_task_duration_range else None)

        instance_dir = os.path.join(instances_root, f"iter_{i}")
        write_instance(instance_dir, existing_jobs_raw, new_job_raw, infra_rows)

        iter_results_dir = os.path.join(args.results_root, f"iter_{i}")
        os.makedirs(iter_results_dir, exist_ok=True)
        run_cwd = os.path.join(args.results_root, ".run_cwd", f"iter_{i}")
        os.makedirs(os.path.join(run_cwd, "utils", "model", "inputs"), exist_ok=True)
        os.makedirs(os.path.join(run_cwd, "utils", "model", "outputs"), exist_ok=True)
        os.makedirs(os.path.join(run_cwd, "utils", "model", "bin"), exist_ok=True)
        for shared in ("lib", "src"):
            link_path = os.path.join(run_cwd, "utils", "model", shared)
            if not os.path.islink(link_path):
                os.symlink(os.path.join(SIMULATOR_DIR, "utils", "model", shared), link_path)

        cmd = [
            VENV_PYTHON, os.path.join(SCRIPT_DIR, "xp_dataset_size_sweep.py"),
            "--instance-dir", instance_dir,
            "--nb-nodes", str(args.nb_nodes),
            "--n-existing", str(n_existing),
            "--new-job-index", str(n_existing),
            "--tiers", "full",
            # --full-range drives EXISTING jobs' own dataset_size redraw inside build_state_a
            # (job_size_rng) -- the new job's draw is separately controlled by
            # --new-job-dataset-size-range below (added 2026-10-04 specifically so these two can
            # differ; previously this script always passed the same combined range for both,
            # which silently made the "stress existing jobs, keep the new job ordinary" design
            # impossible -- xp_dataset_size_sweep.py redraws EVERY job's dataset_size from a range
            # itself, ignoring whatever write_instance put in jobs.json).
            "--full-range", str(args.dataset_size_range[0]), str(args.dataset_size_range[1]),
            "--repeats", "1",
            "--approaches", *args.approaches,
            "--incremental-time-limit", str(args.incremental_time_limit),
            "--epsilon-time-limit", str(args.epsilon_time_limit),
            "--epsilon-fraction", str(args.epsilon_fraction),
            "--epsilon-phase1-fraction", str(args.epsilon_phase1_fraction),
            "--hybrid-alpha", str(args.hybrid_alpha),
            "--hybrid-max-budget", str(args.hybrid_max_budget),
            "--hybrid-incremental-time-limit", str(args.hybrid_incremental_time_limit),
            "--adaptive-gate-metric", args.adaptive_gate_metric,
            "--adaptive-selection-metric", args.adaptive_selection_metric,
            "--state-a-time-limit", str(args.state_a_time_limit),
            "--seed", str(args.seed + i),
            "--results-dir", iter_results_dir,
        ]
        if args.adaptive_f1_relative_margin is not None:
            cmd += ["--adaptive-f1-relative-margin", str(args.adaptive_f1_relative_margin)]
        if args.adaptive_f1_dynamic_margin:
            cmd += ["--adaptive-f1-dynamic-margin"]
        if args.new_job_dataset_size_range:
            cmd += ["--new-job-dataset-size-range",
                    str(args.new_job_dataset_size_range[0]), str(args.new_job_dataset_size_range[1])]
        print(f"### iteration {i}: n_existing={n_existing}, seed={args.seed + i} -> {iter_results_dir} ###", flush=True)
        env = dict(os.environ)
        env["SIMULATOR_RUN_CWD"] = run_cwd
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"### iteration {i} FAILED (exit code {result.returncode}) -- continuing with remaining iterations ###",
                  file=sys.stderr, flush=True)

    # --- Aggregate every iteration's per-job/per-transfer CSVs into one combined file each,
    # tagged with the iteration index and that iteration's own generated n_existing, for a real
    # cross-iteration distribution analysis. ---
    jobs_frames, transfers_frames = [], []
    for iter_dir in sorted(glob.glob(os.path.join(args.results_root, "iter_*"))):
        i = int(os.path.basename(iter_dir).split("_", 1)[1])
        jobs_path = os.path.join(iter_dir, "jobs_detail_by_run.csv")
        transfers_path = os.path.join(iter_dir, "transfers_detail_by_run.csv")
        if os.path.isfile(jobs_path):
            dj = pd.read_csv(jobs_path)
            dj["iteration"] = i
            jobs_frames.append(dj)
        if os.path.isfile(transfers_path):
            dt = pd.read_csv(transfers_path)
            dt["iteration"] = i
            transfers_frames.append(dt)

    if jobs_frames:
        pd.concat(jobs_frames, ignore_index=True).to_csv(
            os.path.join(args.results_root, "combined_jobs_detail.csv"), index=False)
    if transfers_frames:
        pd.concat(transfers_frames, ignore_index=True).to_csv(
            os.path.join(args.results_root, "combined_transfers_detail.csv"), index=False)

    print(f"### Done: {len(jobs_frames)} iterations (across all iter_* directories found under "
          f"--results-root) produced jobs_detail_by_run.csv ###",
          flush=True)
    print(f"### Combined CSVs written to {args.results_root}/combined_jobs_detail.csv and combined_transfers_detail.csv ###",
          flush=True)


if __name__ == "__main__":
    main()
