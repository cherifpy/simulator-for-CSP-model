"""
Standalone, portable experiment runner meant to be launched on Grid5000 (or any machine): it
takes every parameter from the command line and resolves every path relative to this file's own
location, so the whole `simulator/` folder can be copied anywhere (a different machine, a
different username/home directory) and this script keeps working unmodified.

This is the same simulation pipeline as exps/xp_compare_online_vs_incremental.py, generalized so
a single run (one approach, one instance, one solver time budget) can be launched per Grid5000
job -- convenient for running many configurations in parallel across a cluster allocation.

Example:
    python3 xp_online_grid5000.py --approach online --instance-dir /path/to/inst-20J-50N \
        --nb-jobs 20 --nb-nodes 50 --solver-time-limit 60 --lambda-rate 60 \
        --results-dir /path/to/results/online_60s

    python3 xp_online_grid5000.py --approach incremental --instance-dir /path/to/inst-50J-50N \
        --nb-jobs 50 --nb-nodes 50 --solver-time-limit 20 --lambda-rate 100
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from datetime import date

# Resolve the simulator/ root relative to this file, not to any hardcoded developer path --
# this file lives at <simulator>/exps/xp_online_grid5000.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
if SIMULATOR_DIR not in sys.path:
    sys.path.append(SIMULATOR_DIR)

from simulator import (
    simulatorForOptimalPerfsUsingCSPOnline,
    generateHeterogeneousInfrastructureEquilibre,
    save_results_to_csv,
    configure_logging,
)
from master_node_with_heterogeneous_nodes_csp import (
    SchedulingUsingCSPOnline,
    SchedulingUsingCSPOnlineMultiObj,
    SchedulingUsingCSPIncremental,
    SchedulingUsingCSPIncrementalFreeNodesOnly,
    SchedulingUsingCSPAdaptive,
)
from utils.plots import plot_gantt_chart
from utils.run_export import start_recording

logger = logging.getLogger(__name__)

APPROACHES = {
    "online": SchedulingUsingCSPOnline,
    # Bi-objective Online: same full-replan approach, but each replan solves the epsilon-
    # constraint problem (phase 1: max flow time, phase 2: minimize transfer energy within
    # epsilon_fraction of phase 1's result) via MainOnlineMultiObj.java. See
    # SchedulingUsingCSPOnlineMultiObj in master_node_with_heterogeneous_nodes_csp.py.
    "online_biobj": SchedulingUsingCSPOnlineMultiObj,
    "incremental": SchedulingUsingCSPIncremental,
    # Hard filter: a node is only a candidate if nothing is ongoing on it right now (queued
    # backlog doesn't disqualify it -- nodes_free_time already accounts for that). Known to be
    # able to starve or, for a very large dataset on a lightly-provisioned instance, hit a
    # genuine infeasibility edge case in the Java model's node-domain fallback -- see notes in
    # master_node_with_heterogeneous_nodes_csp.py around restrict_to_free_nodes.
    "incremental_free_nodes_only": SchedulingUsingCSPIncrementalFreeNodesOnly,
    # Incremental-first escalation: place each new job with Incremental to get F1, then --
    # budgeted at adaptive_alpha * F1 seconds, using an infra snapshot taken only after that
    # wait -- optionally escalate to online_biobj's MainOnlineMultiObj for the same single job.
    # Escalation is kept only if it beats F1 by more than adaptive_alpha (same knob for the
    # budget and the acceptance bar). See SchedulingUsingCSPAdaptive's docstring for the design.
    "adaptive": SchedulingUsingCSPAdaptive,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", required=True, choices=sorted(APPROACHES.keys()),
                         help="Scheduling approach to run.")
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing this instance's jobs.json and infrastructure.csv.")
    parser.add_argument("--nb-jobs", required=True, type=int, help="Number of jobs in the instance.")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes in the instance.")
    parser.add_argument("--solver-time-limit", required=True, type=int,
                         help="CSP solver time budget per solve, in seconds.")
    parser.add_argument("--lambda-rate", type=int, default=60,
                         help="Mean inter-arrival time for the Poisson job injector (default: 60).")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write results. Default: "
                              "<simulator>/results-grid5000/<approach>_<nb_jobs>j-<nb_nodes>n_<solver_time_limit>s_<date>.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--skip-gantt", action="store_true", help="Skip generating the gantt chart PNG.")
    parser.add_argument("--epsilon-fraction", type=float, default=0.1,
                         help="online_biobj only: max flow time slack allowed in phase 2, as a "
                              "fraction of phase 1's result (default: 0.1 = 10%%).")
    parser.add_argument("--epsilon-phase1-fraction", type=float, default=0.5,
                         help="online_biobj only: fraction of --solver-time-limit given to phase "
                              "1 (max flow time); the rest goes to phase 2 (energy) (default: 0.5).")
    parser.add_argument("--epsilon-max-cap", type=float, default=None,
                         help="online_biobj only: absolute ceiling on the max-flow-time cap phase "
                              "2 is allowed to accept, regardless of epsilon_fraction (default: none).")
    parser.add_argument("--adaptive-alpha", type=float, default=0.2,
                         help="adaptive only: escalation search budget as a fraction of "
                              "Incremental's own predicted flow time F1, and the minimum relative "
                              "gain over F1 required to keep the escalated plan -- same knob for "
                              "both (default: 0.2 = 20%%).")
    parser.add_argument("--adaptive-max-budget", type=float, default=1200,
                         help="adaptive only: hard ceiling on the escalation search budget "
                              "(adaptive_alpha * F1), in seconds -- keeps a rare very-large-F1 "
                              "job from running unboundedly long (default: 1200 = 20min).")
    parser.add_argument("--freeze-large-jobs-threshold", type=float, default=None,
                         help="online/online_biobj/adaptive: a not-finished job already resident "
                              "somewhere whose dataset_size (MB) is at or above this threshold is "
                              "frozen for the solve -- no new replica, no move, kept exactly where "
                              "it is (see SchedulingUsingCSPOnline docs / frozen_jobs.txt). "
                              "Default: unset (no job frozen, identical to before this existed).")
    return parser.parse_args()


def run(args):
    master_class = APPROACHES[args.approach]
    if args.approach == "online_biobj":
        master_class.epsilon_fraction = args.epsilon_fraction
        master_class.epsilon_phase1_fraction = args.epsilon_phase1_fraction
        master_class.epsilon_max_cap = args.epsilon_max_cap
    if args.approach == "adaptive":
        master_class.adaptive_alpha = args.adaptive_alpha
        master_class.adaptive_max_budget_s = args.adaptive_max_budget

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["total_nb_jobs"] = args.nb_jobs
    config["total_nb_compute_nodes"] = args.nb_nodes
    config["jobs_file_path"] = os.path.join(args.instance_dir, "jobs.json")
    config["solver_time_limit_s"] = args.solver_time_limit
    config["lambda_rate"] = args.lambda_rate
    config["adaptive_alpha"] = args.adaptive_alpha
    config["adaptive_max_budget_s"] = args.adaptive_max_budget
    config["freeze_large_jobs_threshold_mb"] = args.freeze_large_jobs_threshold

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000",
        f"{args.approach}_{args.nb_jobs}j-{args.nb_nodes}n_{args.solver_time_limit}s_{date.today().isoformat()}",
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"### Running '{args.approach}' on {args.nb_jobs} jobs / {args.nb_nodes} nodes "
          f"(solver_time_limit={args.solver_time_limit}s, lambda_rate={args.lambda_rate}) ###", flush=True)
    print(f"### instance: {args.instance_dir}", flush=True)
    print(f"### results:  {results_dir}", flush=True)

    random.seed(args.seed)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(
        config, path=os.path.join(args.instance_dir, "infrastructure.csv"))

    # Everything a later analysis could need is saved with the run itself (see utils/run_export.py):
    # parameters, infrastructure, the solver's complete console output, and -- for EVERY replan -- its
    # decisions, log and raw Java input/output files under solver_archive/, next to the usual
    # infos_on_* csv files and events_history.json below.
    start_recording(
        results_dir,
        params={**vars(args), "master_class": master_class.__name__,
                "java_main_class": getattr(master_class, "java_main_class", None),
                "multi_objective": getattr(master_class, "multi_objective", None),
                "epsilon_fraction": getattr(master_class, "epsilon_fraction", None),
                "epsilon_phase1_fraction": getattr(master_class, "epsilon_phase1_fraction", None),
                "epsilon_max_cap": getattr(master_class, "epsilon_max_cap", None),
                "adaptive_alpha": getattr(master_class, "adaptive_alpha", None),
                "adaptive_max_budget_s": getattr(master_class, "adaptive_max_budget_s", None),
                "escalation_java_main_class": getattr(master_class, "escalation_java_main_class", None),
                "freeze_large_jobs_threshold_mb": args.freeze_large_jobs_threshold},
        config=config, nodes_config=nodes_config)

    random.seed(args.seed)
    results, _ = simulatorForOptimalPerfsUsingCSPOnline(
        config=config, jobs=[], overlap=True, poisson=True, varying_load=False,
        nodes_config=nodes_config, master_class=master_class,
    )

    save_results_to_csv(logger, results, results_dir, "")
    with open(os.path.join(results_dir, "events_history.json"), "w") as f:
        json.dump(results.events_history, f)

    if not args.skip_gantt:
        gantt_path = os.path.join(results_dir, "gantt.png")
        plot_gantt_chart(results.events_history, args.nb_nodes,
                          title=f"{args.approach} ({args.nb_jobs}j-{args.nb_nodes}n)", save_path=gantt_path)

    print(f"### DONE: total wall time = {results.total_wall_time:.1f}", flush=True)
    return results_dir


def verify_storage(results_dir, instance_dir):
    """Ground truth for storage-occupancy: rebuild real per-node occupancy from actual transfer
    completions and deletions as they executed in the simulation, and flag any instant where it
    exceeds the node's capacity. Returns the list of violations (empty if none)."""
    with open(os.path.join(instance_dir, "jobs.json")) as f:
        jobs = json.load(f)
    job_size = {j["job_id"]: j["dataset_size"] for j in jobs}

    node_capacity = {}
    with open(os.path.join(instance_dir, "infrastructure.csv")) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            node_capacity[i] = float(row["storage_capacity"]) if "storage_capacity" in row else float("inf")

    with open(os.path.join(results_dir, "events_history.json")) as f:
        events = json.load(f)

    arrival = {}
    for e in events:
        if e["type"] == "transfer":
            key = (e["node_id"], e["job_id"])
            if key not in arrival or e["end"] < arrival[key]:
                arrival[key] = e["end"]

    departures = {}
    for e in events:
        if e["type"] == "deletion":
            departures.setdefault((e["node_id"], e["job_id"]), []).append(e["time"])

    events_per_node = {}
    for (node, job), t_in in arrival.items():
        size = job_size[job]
        events_per_node.setdefault(node, []).append((t_in, size, f"+job{job}"))
        if (node, job) in departures:
            events_per_node[node].append((min(departures[(node, job)]), -size, f"-job{job}"))

    violations = []
    for node, evts in events_per_node.items():
        evts.sort(key=lambda e: e[0])
        occupied = 0.0
        cap = node_capacity.get(node, float("inf"))
        for t, delta, label in evts:
            occupied += delta
            if occupied > cap + 1e-6:
                violations.append((node, t, occupied, cap, label))
    return violations


def main():
    args = parse_args()
    configure_logging(logging.WARNING)
    results_dir = run(args)

    violations = verify_storage(results_dir, args.instance_dir)
    print()
    print("=" * 70)
    if not violations:
        print("STORAGE CHECK: OK -- no violation at any instant, on any node.")
    else:
        print(f"STORAGE CHECK: {len(violations)} violation(s) found:")
        for node, t, occupied, cap, label in violations:
            print(f"  node_{node} at t={t:.2f}: occupied={occupied:.1f} > capacity={cap:.1f} ({label})")
    print("=" * 70)


if __name__ == "__main__":
    main()
