"""Generalizes the one-off "simultaneous arrival" scenario (built by hand earlier) into a proper
N-iteration sweep, varying nb_nodes and n_existing across iterations instead of a single fixed
case. Per iteration:
  1. Generate n_existing jobs that all arrive at t=0 (not spread out -- a genuine simultaneous
     burst), sized from --task-duration-range / --dataset-size-range (defaults are DOUBLED from
     this script's own earlier "normal" baseline, per the 2026-10-04 request for jobs that stay
     busy for hours, not minutes). nb_nodes is drawn per iteration from --nb-nodes-list.
  2. PROBE: build State A alone (--state-a-time-limit budget) via a throwaway --approaches
     incremental invocation of xp_dataset_size_sweep.py, then read its own state_A task rows to
     get each existing job's committed finish time (arrival=0, so finish time IS its flow time).
     The new job's own placeholder arrival doesn't affect this solve at all (state A is built
     purely from the existing jobs, independent of when the new job shows up) -- see that
     script's build_state_a for why.
  3. Set the new job's arriving_time = max(those finish times) / 2 -- i.e. right in the middle of
     the busiest existing job's lifetime, a genuinely congested moment instead of an arbitrary
     fixed offset. The new job itself is forced to the LARGEST value of every parameter's own
     range (never nb_tasks=1 -- a single-task job can't be split across nodes, which blunts any
     escalation variant that works by redistributing a job's OWN tasks).
  4. REAL RUN: rewrite jobs.json with that arrival time, then invoke xp_dataset_size_sweep.py
     again with --approaches incremental hybrid and whatever hybrid tuning flags were passed
     through (alpha, budgets, gate/selection metric, margin, stability-CV veto).

Run with the venv python, one oarsub job per iteration (see the launch commands used alongside
this script -- it does not parallelize iterations itself, each invocation handles exactly one).
"""
import argparse
import json
import csv
import os
import random
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../simulator/exps
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SIMULATOR_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, "env", "bin", "python3")

NB_TASKS_RANGE = (1, 20)  # existing jobs only -- not doubled, it's a count, not a size


def generate_existing_jobs(rng, n_existing, task_duration_range, dataset_size_range):
    jobs = []
    for i in range(n_existing):
        jobs.append({
            "job_id": i, "id_dataset": i,
            "nb_tasks": rng.randint(*NB_TASKS_RANGE),
            "task_duration": rng.randint(*task_duration_range),
            "dataset_size": rng.randint(*dataset_size_range),
            "arriving_time": 0.0,
            "type_of_job": None,
        })
    return jobs


def draw_new_job_sizes(rng, nb_tasks_range, task_duration_range, dataset_size_range, mode="upper-quarter"):
    """mode='upper-quarter' (original behavior): big but variable from one scenario to the next --
    draws from the upper quarter of each range instead of pinning every iteration to the exact
    same max, so each iteration gets its own distinct but consistently-large new job (nb_tasks
    never 1, since the whole point there was a job big enough to split across nodes).
    mode='same-tier' (2026-10-05, for the size-tier x n_existing grid): draws from the FULL range
    exactly like an existing job -- the new job is just another member of whatever tier
    (small/large/mixed) is being tested, not deliberately oversized relative to it."""
    if mode == "same-tier":
        nb_tasks = rng.randint(*nb_tasks_range)
        task_duration = rng.randint(*task_duration_range)
        dataset_size = rng.randint(*dataset_size_range)
        return nb_tasks, task_duration, dataset_size

    def upper_quarter(lo, hi):
        return lo + int(0.75 * (hi - lo)), hi
    nb_tasks = rng.randint(*upper_quarter(*nb_tasks_range))
    task_duration = rng.randint(*upper_quarter(*task_duration_range))
    dataset_size = rng.randint(*upper_quarter(*dataset_size_range))
    return nb_tasks, task_duration, dataset_size


def make_new_job_raw(n_existing, nb_tasks, task_duration, dataset_size, arriving_time):
    return {
        "job_id": n_existing, "id_dataset": n_existing,
        "nb_tasks": nb_tasks,
        "task_duration": task_duration,
        "dataset_size": dataset_size,
        "arriving_time": arriving_time,
        "type_of_job": None,
    }


def write_instance(instance_dir, existing_jobs, new_job_raw, rng, nb_nodes, storage_ceiling):
    os.makedirs(instance_dir, exist_ok=True)
    with open(os.path.join(instance_dir, "jobs.json"), "w") as f:
        json.dump(existing_jobs + [new_job_raw], f, indent=2)
    infra_rows = []
    for _ in range(nb_nodes):
        infra_rows.append({
            "bandwidth": rng.randint(12, 800),
            "computation_nodes": round(rng.uniform(1.0, 9.7), 6),
            "energy_consumption": round(rng.uniform(0.1, 2.1), 6),
            "storage_capacity": rng.randint(2050, storage_ceiling),
        })
    with open(os.path.join(instance_dir, "infrastructure.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"])
        writer.writeheader()
        writer.writerows(infra_rows)


def run_cwd_for(results_root, tag):
    run_cwd = os.path.join(results_root, ".run_cwd", tag)
    for sub in ("inputs", "outputs", "bin"):
        os.makedirs(os.path.join(run_cwd, "utils", "model", sub), exist_ok=True)
    for shared in ("lib", "src"):
        link_path = os.path.join(run_cwd, "utils", "model", shared)
        if not os.path.islink(link_path):
            os.symlink(os.path.join(SIMULATOR_DIR, "utils", "model", shared), link_path)
    return run_cwd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iteration", type=int, required=True, help="Which single iteration to run (0-based).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-existing-range", type=int, nargs=2, default=(5, 15),
                    help="Used only when --n-existing is unset -- n_existing is then drawn "
                         "uniformly from this range per iteration.")
    p.add_argument("--n-existing", type=int, default=None,
                    help="Exact n_existing for this iteration, overriding --n-existing-range -- "
                         "for a grid sweep over specific load levels (e.g. 1/3/6/8/10) instead "
                         "of a random draw.")
    p.add_argument("--nb-nodes-list", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    p.add_argument("--new-job-mode", choices=["upper-quarter", "same-tier"], default="upper-quarter",
                    help="'upper-quarter' (default): new job forced into the upper quarter of "
                         "each range, deliberately bigger than a typical existing job. "
                         "'same-tier': new job drawn from the exact same range as existing jobs "
                         "-- just another member of whatever tier is being tested.")
    p.add_argument("--task-duration-range", type=int, nargs=2, default=(200, 300))
    p.add_argument("--dataset-size-range", type=int, nargs=2, default=(20480, 204800))
    p.add_argument("--state-a-time-limit", type=int, default=180)
    p.add_argument("--incremental-time-limit", type=int, default=300)
    p.add_argument("--hybrid-incremental-time-limit", type=float, default=300)
    p.add_argument("--hybrid-alpha", type=float, default=0.25)
    p.add_argument("--hybrid-max-budget", type=float, default=600)
    p.add_argument("--adaptive-gate-metric", choices=["new_job", "max", "mean"], default="max")
    p.add_argument("--adaptive-selection-metric", choices=["new_job", "max"], default="max")
    p.add_argument("--adaptive-f1-relative-margin", type=float, default=None)
    p.add_argument("--adaptive-f1-dynamic-margin", action="store_true")
    p.add_argument("--adaptive-f1-stability-cv-threshold", type=float, default=None)
    p.add_argument("--adaptive-new-job-objective", action="store_true")
    p.add_argument("--adaptive-degradation-cap-pct", type=float, default=None)
    p.add_argument("--adaptive-no-bi-objective", action="store_true")
    p.add_argument("--results-root", required=True)
    args = p.parse_args()

    i = args.iteration
    seed_i = args.seed + i
    rng = random.Random(seed_i)
    n_existing = args.n_existing if args.n_existing is not None else rng.randint(*args.n_existing_range)
    nb_nodes = args.nb_nodes_list[i % len(args.nb_nodes_list)]
    storage_ceiling = int(args.dataset_size_range[1] * 1.3)

    existing_jobs = generate_existing_jobs(rng, n_existing, args.task_duration_range, args.dataset_size_range)
    new_nb_tasks, new_task_duration, new_dataset_size = draw_new_job_sizes(
        rng, NB_TASKS_RANGE, args.task_duration_range, args.dataset_size_range, mode=args.new_job_mode)

    instance_dir = os.path.join(args.results_root, "_instances", f"iter_{i}")
    iter_results_dir = os.path.join(args.results_root, f"iter_{i}")
    os.makedirs(iter_results_dir, exist_ok=True)

    print(f"### iter {i}: seed={seed_i}, n_existing={n_existing}, nb_nodes={nb_nodes}, "
          f"new_job=(nb_tasks={new_nb_tasks}, task_duration={new_task_duration}, dataset_size={new_dataset_size}) ###",
          flush=True)

    # --- Phase 1: probe state A alone to learn its own max flow time (new job's placeholder
    # arrival is irrelevant to this solve -- see module docstring). ---
    placeholder_new_job = make_new_job_raw(n_existing, new_nb_tasks, new_task_duration,
                                            new_dataset_size, arriving_time=1.0)
    write_instance(instance_dir, existing_jobs, placeholder_new_job, rng, nb_nodes, storage_ceiling)

    probe_dir = os.path.join(iter_results_dir, "_probe")
    os.makedirs(probe_dir, exist_ok=True)
    probe_cmd = [
        VENV_PYTHON, os.path.join(SCRIPT_DIR, "xp_dataset_size_sweep.py"),
        "--instance-dir", instance_dir, "--nb-nodes", str(nb_nodes),
        "--n-existing", str(n_existing), "--new-job-index", str(n_existing),
        "--tiers", "full", "--full-range", str(args.dataset_size_range[0]), str(args.dataset_size_range[1]),
        "--repeats", "1", "--approaches", "incremental",
        "--incremental-time-limit", "30",
        "--state-a-time-limit", str(args.state_a_time_limit),
        "--seed", str(seed_i), "--results-dir", probe_dir,
    ]
    env = dict(os.environ)
    env["SIMULATOR_RUN_CWD"] = run_cwd_for(args.results_root, f"iter_{i}_probe")
    print(f"### iter {i}: probing State A (budget={args.state_a_time_limit}s) ###", flush=True)
    result = subprocess.run(probe_cmd, env=env)
    if result.returncode != 0:
        print(f"### iter {i} PROBE FAILED (exit {result.returncode}) -- aborting this iteration ###",
              file=sys.stderr, flush=True)
        return

    probe_tasks = pd.read_csv(os.path.join(probe_dir, "tasks_detail_by_run.csv"))
    state_a_tasks = probe_tasks[probe_tasks.approach == "state_A"]
    finish_by_job = state_a_tasks.groupby("job_id")["end"].max()
    max_flow = float(finish_by_job.max())
    new_arrival = max_flow / 2.0
    print(f"### iter {i}: State A max flow time = {max_flow:.1f}s -> new job arrives at t={new_arrival:.1f}s ###",
          flush=True)

    # --- Phase 2: rewrite the new job's arrival time, then run the real incremental+hybrid comparison. ---
    real_new_job = make_new_job_raw(n_existing, new_nb_tasks, new_task_duration,
                                     new_dataset_size, arriving_time=new_arrival)
    with open(os.path.join(instance_dir, "jobs.json"), "w") as f:
        json.dump(existing_jobs + [real_new_job], f, indent=2)

    real_cmd = [
        VENV_PYTHON, os.path.join(SCRIPT_DIR, "xp_dataset_size_sweep.py"),
        "--instance-dir", instance_dir, "--nb-nodes", str(nb_nodes),
        "--n-existing", str(n_existing), "--new-job-index", str(n_existing),
        "--tiers", "full", "--full-range", str(args.dataset_size_range[0]), str(args.dataset_size_range[1]),
        "--new-job-dataset-size-range", str(new_dataset_size), str(new_dataset_size),
        "--repeats", "1", "--approaches", "incremental", "hybrid",
        "--incremental-time-limit", str(args.incremental_time_limit),
        "--hybrid-incremental-time-limit", str(args.hybrid_incremental_time_limit),
        "--hybrid-alpha", str(args.hybrid_alpha),
        "--hybrid-max-budget", str(args.hybrid_max_budget),
        "--adaptive-gate-metric", args.adaptive_gate_metric,
        "--adaptive-selection-metric", args.adaptive_selection_metric,
        "--state-a-time-limit", str(args.state_a_time_limit),
        "--seed", str(seed_i), "--results-dir", iter_results_dir,
    ]
    if args.adaptive_f1_relative_margin is not None:
        real_cmd += ["--adaptive-f1-relative-margin", str(args.adaptive_f1_relative_margin)]
    if args.adaptive_f1_dynamic_margin:
        real_cmd += ["--adaptive-f1-dynamic-margin"]
    if args.adaptive_f1_stability_cv_threshold is not None:
        real_cmd += ["--adaptive-f1-stability-cv-threshold", str(args.adaptive_f1_stability_cv_threshold)]
    if args.adaptive_new_job_objective:
        real_cmd += ["--adaptive-new-job-objective"]
    if args.adaptive_degradation_cap_pct is not None:
        real_cmd += ["--adaptive-degradation-cap-pct", str(args.adaptive_degradation_cap_pct)]
    if args.adaptive_no_bi_objective:
        real_cmd += ["--adaptive-no-bi-objective"]

    env = dict(os.environ)
    env["SIMULATOR_RUN_CWD"] = run_cwd_for(args.results_root, f"iter_{i}_real")
    print(f"### iter {i}: real run (incremental + hybrid) -> {iter_results_dir} ###", flush=True)
    result = subprocess.run(real_cmd, env=env)
    if result.returncode != 0:
        print(f"### iter {i} REAL RUN FAILED (exit {result.returncode}) ###", file=sys.stderr, flush=True)
        return

    with open(os.path.join(iter_results_dir, "scenario_meta.json"), "w") as f:
        json.dump({"iteration": i, "seed": seed_i, "n_existing": n_existing, "nb_nodes": nb_nodes,
                    "state_a_max_flow": max_flow, "new_job_arrival": new_arrival,
                    "new_job_nb_tasks": new_nb_tasks, "new_job_task_duration": new_task_duration,
                    "new_job_dataset_size": new_dataset_size}, f, indent=2)
    print(f"### iter {i}: done ###", flush=True)


if __name__ == "__main__":
    main()
