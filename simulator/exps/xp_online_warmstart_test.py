"""
Tests whether Online's usual underperformance vs. Incremental (seen at high infra occupancy in
xp_occupancy_sweep.py -- Online several times worse than Incremental even at a 5min budget) comes
from its search simply not reliably REACHING a good solution within budget, rather than full
replanning being inherently worse.

For each occupancy level: builds state A synthetically (same method as xp_occupancy_sweep.py --
one existing job per occupied node, no CSP solve involved, so occupancy at T is exact), then
solves the new job's placement THREE ways from that SAME state:
  1. incremental      -- new job alone (the baseline Online is compared against).
  2. online_cold      -- Online's usual full replan, cold search (MainOnline.java).
  3. online_warmstart -- the SAME full replan, but the search is seeded with a warm-start
                         solution: every existing job's remaining tasks stay EXACTLY where state
                         A already had them, and the new job is placed exactly as Incremental
                         placed it (MainOnlineWarmStart.java). This point is ALWAYS feasible in
                         Online's (larger) search space and reproduces Incremental's own objective
                         value exactly -- so online_warmstart's result is a real regression if it
                         does WORSE than incremental, and shows the search actually improving on
                         it if it does better. Only the search seed differs from online_cold; the
                         constraint model is identical between the two Java programs.

Every path is resolved relative to this file's own location, so the whole `simulator/` folder
can be copied to Grid5000 (or anywhere) and this script keeps working unmodified.

Example:
    python3 xp_online_warmstart_test.py --instance-dir /path/to/inst-20J-50N --nb-nodes 50 \\
        --occupancy-fractions 1.0 0.75 0.5 0.25 \\
        --online-time-limit 300 --incremental-time-limit 30
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import random
import sys
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
if SIMULATOR_DIR not in sys.path:
    sys.path.append(SIMULATOR_DIR)

import simpy
from classes.tracker import Tracker
from classes.job import Job
from compute_node import ComputeNode
from master_node_with_heterogeneous_nodes_csp import SchedulingUsingCSPOnline, SchedulingUsingCSPIncremental
from utils.modelCSP import schedulingUsingJavaCSP, MODEL_DIR
from utils.plots import plot_gantt_chart
from simulator import generateHeterogeneousInfrastructureEquilibre, configure_logging
# --state-a-mode solved reuses xp_single_decision_grid5000.py's own build_state_a() (ONE direct
# joint CSP solve over --n-existing real jobs) rather than re-duplicating that carefully-tuned
# construction a third time -- it already carries the arrival-time lower-bound fix, per-task
# Finished/Started marking, and node-occupancy population this script's synthetic path hand-rolls.
from exps.xp_single_decision_grid5000 import build_state_a as build_solved_state_a

logger = logging.getLogger(__name__)

NEW_JOB_ID_BASE = 10_000_000

WARM_START_PATH = os.path.join(MODEL_DIR, "inputs", "warm_start.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing infrastructure.csv. In --state-a-mode "
                              "synthetic (default), jobs.json is not used -- every job is "
                              "synthetic. In --state-a-mode solved, jobs.json IS used (real "
                              "arriving_time/dataset_size/nb_tasks/task_duration).")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes to use.")
    parser.add_argument("--state-a-mode", choices=["synthetic", "solved"], default="synthetic",
                         help="'synthetic' (default): build state A directly, no CSP solve, one "
                              "existing job per occupied node -- sweeps --occupancy-fractions, "
                              "runs the full 5-way comparison (incremental + 4 online variants) "
                              "at each level. 'solved': build state A the SAME way as "
                              "xp_single_decision_grid5000.py -- ONE direct joint CSP solve over "
                              "--n-existing real jobs from jobs.json (budget --state-a-time-limit) "
                              "-- then run ONLY incremental + online_obj3_warmstart on top of it "
                              "(--occupancy-fractions is ignored; occupancy isn't directly "
                              "controllable in this mode, it's whatever that solve converges to).")
    parser.add_argument("--occupancy-fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25],
                         help="[synthetic mode only] Fractions of --nb-nodes to make genuinely busy "
                              "at T, one run each (default: 1.0 0.75 0.5 0.25).")
    parser.add_argument("--n-existing", type=int, default=5,
                         help="[solved mode only] Number of jobs (from the start of jobs.json) used "
                              "to build state A (default: 5).")
    parser.add_argument("--new-job-index", type=int, default=None,
                         help="[solved mode only] Index into jobs.json of the new job (default: "
                              "--n-existing, i.e. the job right after state A).")
    parser.add_argument("--state-a-time-limit", type=int, default=60,
                         help="[solved mode only] CSP solver time budget (seconds) for the ONE "
                              "joint solve that builds state A (default: 60, i.e. 1 minute).")
    parser.add_argument("--occupied-nodes-fraction", type=float, default=1.0,
                         help="[solved mode only] Fraction of --nb-nodes eligible for state A's "
                              "own solve (default: 1.0). See xp_single_decision_grid5000.py.")
    parser.add_argument("--lambda-rate", type=int, default=100,
                         help="[solved mode only] Present for config compatibility (default: 100).")
    parser.add_argument("--freeze-at", type=float, default=1000.0,
                         help="Synthetic 'now' (T) when the new job arrives (default: 1000.0).")
    parser.add_argument("--occupied-nb-tasks", type=int, default=3,
                         help="Tasks per synthetic existing (occupying) job (default: 3).")
    parser.add_argument("--occupied-task-duration", type=int, default=300,
                         help="Duration in seconds of each task of a synthetic existing job (default: 300).")
    parser.add_argument("--occupied-dataset-size", type=int, default=5120,
                         help="Dataset size in MB for each synthetic existing job (default: 5120), "
                              "clipped down to whatever its own node's real storage_capacity allows.")
    parser.add_argument("--new-job-dataset-size", type=int, default=5120,
                         help="Dataset size in MB for the synthetic new job (default: 5120).")
    parser.add_argument("--new-job-nb-tasks", type=int, default=15,
                         help="Number of tasks for the synthetic new job (default: 15).")
    parser.add_argument("--new-job-task-duration", type=int, default=100,
                         help="Per-task duration in seconds for the synthetic new job (default: 100).")
    parser.add_argument("--online-time-limit", required=True, type=int,
                         help="CSP solver time budget (seconds), used identically for both "
                              "online_cold and online_warmstart.")
    parser.add_argument("--incremental-time-limit", required=True, type=int,
                         help="CSP solver time budget (seconds) for Incremental's placement solve.")
    parser.add_argument("--solved-approaches", nargs="+",
                         choices=["online_cold", "online_warmstart", "online_obj3_cold", "online_obj3_warmstart",
                                  "online_mean_cold", "online_mean_warmstart"],
                         default=["online_obj3_warmstart"],
                         help="[solved mode only] Which online_* approach(es) to run against the "
                              "solved state A, alongside incremental (always run). Default: "
                              "online_obj3_warmstart alone. online_cold/online_warmstart use "
                              "objective 2 (max flow time, all jobs); online_obj3_* use objective "
                              "3 (the new job's own flow time alone).")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write per-fraction results + the summary. Default: "
                              "<simulator>/results-grid5000/online_warmstart_test_<nb_nodes>n_<date>.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def build_synthetic_state_a(config, args, occupancy_fraction):
    """Identical construction to xp_occupancy_sweep.py's build_synthetic_state_a: one synthetic
    existing job per occupied node (task 0 already in progress at T, fixed; the rest queued
    right after it, still reconsiderable), no CSP solve involved. See that script's module
    docstring for the full rationale."""
    random.seed(args.seed)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(
        config, path=os.path.join(args.instance_dir, "infrastructure.csv"))

    env = simpy.Environment()
    tracker = Tracker(env)
    master = SchedulingUsingCSPOnline(env, [], tracker, config, overlap=True)
    compute_nodes = [
        ComputeNode(env, i, master, bandwidth=nodes_config[i]['bandwidth'],
                    compute_capacity=nodes_config[i]['computation_nodes'],
                    energy_consumption=nodes_config[i]['energy_consumption'],
                    storage_capacity=nodes_config[i].get('storage_capacity', float('inf')))
        for i in range(args.nb_nodes)
    ]
    master.compute_nodes = compute_nodes
    master.nb_nodes = len(compute_nodes)
    for i in range(len(compute_nodes)):
        key = f'node_{i}'
        master.ongoing_transfers.setdefault(key, None)
        master.ongoing_works.setdefault(key, None)
        master.transfers.setdefault(key, [])
        master.works.setdefault(key, [])
        master.deletions.setdefault(key, [])

    freeze_at = args.freeze_at
    occupied_count = max(0, min(args.nb_nodes, round(args.nb_nodes * occupancy_fraction)))
    half = args.occupied_task_duration / 2.0

    existing_jobs = []
    state_a_works = {f'node_{i}': [] for i in range(args.nb_nodes)}
    state_a_transfers = {f'node_{i}': [] for i in range(args.nb_nodes)}

    for node_id in range(occupied_count):
        job_id = node_id
        node_dataset_size = min(args.occupied_dataset_size, int(compute_nodes[node_id].storage_capacity))
        job = Job(job_id, args.occupied_task_duration, args.occupied_nb_tasks, node_dataset_size)
        job.arriving_time = freeze_at - half
        existing_jobs.append(job)
        tracker.register_job(job_id, job.arriving_time)

        t0_start, t0_end = freeze_at - half, freeze_at + half
        job.tasks[0].status = "Started"
        state_a_works[f'node_{node_id}'].append((job_id, node_id, 0, t0_start, t0_end, args.occupied_task_duration))
        master.ongoing_works[f'node_{node_id}'] = (job_id, node_id, 0, t0_start, t0_end, args.occupied_task_duration)

        state_a_transfers[f'node_{node_id}'].append((job_id, node_id, t0_start - 1, t0_start, 1))
        master.replicas_locations[job_id] = [node_id]

        prev_end = t0_end
        for k in range(1, args.occupied_nb_tasks):
            k_start, k_end = prev_end, prev_end + args.occupied_task_duration
            state_a_works[f'node_{node_id}'].append((job_id, node_id, k, k_start, k_end, args.occupied_task_duration))
            master.works[f'node_{node_id}'].append((job_id, node_id, k, k_start, k_end, args.occupied_task_duration))
            prev_end = k_end

    master.jobs = existing_jobs
    master._state_a_works = state_a_works
    master._state_a_transfers = state_a_transfers
    master._state_a_finish = {
        job.job_id: max(w[4] for w in state_a_works[f'node_{job.job_id}'])
        for job in existing_jobs
    }

    not_finished_jobs = list(existing_jobs)
    env.run(until=freeze_at)

    new_job = Job(NEW_JOB_ID_BASE, args.new_job_task_duration, args.new_job_nb_tasks, args.new_job_dataset_size)
    new_job.arriving_time = env.now
    master.tracker.register_job(new_job.job_id, env.now)

    isolated_ids = {j.job_id for j in not_finished_jobs} | {new_job.job_id}
    print(f"### Synthetic state A: {occupied_count}/{args.nb_nodes} nodes occupied "
          f"({occupancy_fraction * 100:.0f}%) at T={freeze_at:.2f}, {occupied_count} existing "
          f"jobs (all still running by construction) ###", flush=True)
    return master, new_job, isolated_ids, not_finished_jobs


def compute_transfer_energy(transfers_, master):
    sender = master._config.get('master_energy_consumption', 0.0)
    network = master._config.get('network_energy_per_transfer', 0.0)
    total = 0.0
    detail = []
    for entries in transfers_.values():
        for job_id, node_index, start_abs, end_abs, duration in entries:
            receiver = master.compute_nodes[node_index].energy_consumption
            energy = sender + receiver * duration + network
            total += energy
            detail.append({"job_id": job_id, "node": node_index, "duration": duration, "energy": energy})
    return total, detail


def summarize_flow_times(master, new_job, isolated_ids, jobs_to_reschedule, nb_nodes, works_, transfers_):
    """Shared flow-time/wait-time bookkeeping for any of the three approaches, given their raw
    works_/transfers_ output."""
    finish = {j.job_id: None for j in jobs_to_reschedule}
    for key in [f'node_{i}' for i in range(nb_nodes)]:
        for w in works_.get(key, []):
            job_id, node_index, task_index, start_abs, end_abs, duration = w
            if job_id in finish:
                finish[job_id] = end_abs if finish[job_id] is None else max(finish[job_id], end_abs)

    flow_by_job = {}
    for j in master.jobs + [new_job]:
        if j.job_id in finish and finish[j.job_id] is not None:
            flow_by_job[j.job_id] = finish[j.job_id] - j.arriving_time
        else:
            cf = master._state_a_finish.get(j.job_id)
            if cf is not None:
                flow_by_job[j.job_id] = cf - j.arriving_time
    flow_times = list(flow_by_job.values())
    isolated_flow_times = [ft for jid, ft in flow_by_job.items() if jid in isolated_ids]

    new_job_start = None
    for key in [f'node_{i}' for i in range(nb_nodes)]:
        for w in works_.get(key, []):
            job_id, node_index, task_index, start_abs, end_abs, duration = w
            if job_id == new_job.job_id:
                new_job_start = start_abs if new_job_start is None else min(new_job_start, start_abs)
    wait_time = (new_job_start - new_job.arriving_time) if new_job_start is not None else None
    transfer_energy_total, transfer_energy_detail = compute_transfer_energy(transfers_, master)

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": flow_by_job.get(new_job.job_id),
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "flow_time_by_job": flow_by_job,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": transfer_energy_total,
        "transfer_energy_detail": transfer_energy_detail,
        "transfers": transfers_,
        "works": works_,
        "replicas_locations": master.replicas_locations,
    }


def run_incremental(master, new_job, isolated_ids, nb_nodes, now, replicas_locations):
    """New job alone -- also the source of the warm start's own placement for the new job."""
    jobs_to_reschedule = [new_job]
    nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(master, master.ongoing_transfers, master.ongoing_works)
    master.java_main_class = 'MainIncremental'
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    if not transfers_ or not works_:
        return None, {}, {}
    result = summarize_flow_times(master, new_job, isolated_ids, jobs_to_reschedule, nb_nodes, works_, transfers_)
    return result, transfers_, works_


def build_warm_start(jobs_to_reschedule, not_finished_jobs, new_job, master, incremental_transfers, incremental_works, now):
    """Warm-start solution for Online's search: every not-finished existing job's remaining tasks
    stay EXACTLY where state A already had them (same node, same local schedule -- no transfer
    entry needed, their data is already resident there), and the new job is placed exactly as
    Incremental placed it. This point is always feasible in Online's own search space (a superset
    of "touch nothing, place only the new job") and reproduces Incremental's objective value
    exactly, so seeding the search with it can only help, never hurt, Online's final result --
    other than the search-time overhead of applying it."""
    sorted_jobs = sorted(jobs_to_reschedule, key=lambda j: j.job_id)
    job_index = {job.job_id: idx for idx, job in enumerate(sorted_jobs)}

    def local_task_index_map(job):
        # Mirrors schedulingUsingJavaCSP's own jobs_data filter: only "NotStarted" tasks are ever
        # sent to the CSP, renumbered 0..k in that same relative order.
        return {orig: local for local, orig in
                enumerate(idx for idx, t in enumerate(job.tasks) if t.status == "NotStarted")}

    job_placements = []
    transfers_ws = []

    for job in not_finished_jobs:
        if job.job_id not in job_index:
            continue
        i = job_index[job.job_id]
        remap = local_task_index_map(job)
        for entries in master._state_a_works.values():
            for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                if job_id != job.job_id or task_index not in remap:
                    continue
                job_placements.append({
                    "job_index": i, "task_index": remap[task_index],
                    "node": int(node_index), "start": int(round(start_abs - now)),
                })

    if new_job.job_id in job_index:
        i = job_index[new_job.job_id]
        for entries in incremental_works.values():
            for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                if job_id != new_job.job_id:
                    continue
                job_placements.append({
                    "job_index": i, "task_index": int(task_index),
                    "node": int(node_index), "start": int(round(start_abs - now)),
                })
        for entries in incremental_transfers.values():
            for job_id, node_index, start_abs, end_abs, duration in entries:
                if job_id != new_job.job_id:
                    continue
                transfers_ws.append({
                    "job_index": i, "node": int(node_index),
                    "start": int(round(start_abs - now)),
                })

    warm_start = {"job_placements": job_placements, "transfers": transfers_ws}
    with open(WARM_START_PATH, "w") as f:
        json.dump(warm_start, f)
    print(f"### Warm start written: {len(job_placements)} job_placement(s), {len(transfers_ws)} "
          f"transfer(s) ({WARM_START_PATH}) ###", flush=True)
    return warm_start


def run_online(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
               java_main_class, time_limit, objective_choice):
    """objective_choice: 0=sum flow time (all jobs), 1=max flow time (all jobs), 2=the new job's
    own flow time alone. Always passed explicitly here (never left to each Java file's own
    differing default -- MainOnline.java defaults to 1, MainOnlineWarmStart.java to 0) so cold
    and warm-started runs are always compared on the exact same objective."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
    master._config["solver_time_limit_s"] = time_limit
    master.java_main_class = java_main_class
    master.objective_choice = objective_choice
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    if not transfers_ or not works_:
        return None
    return summarize_flow_times(master, new_job, isolated_ids, jobs_to_reschedule, nb_nodes, works_, transfers_)


def build_gantt_events(master, result, freeze_at):
    if result is None:
        return []
    reschedule_ids = set(result["jobs_to_reschedule"])
    events = []
    for job_id, node_index, task_index, start_abs, end_abs, duration in (
        e for entries in master._state_a_works.values() for e in entries
    ):
        if job_id in reschedule_ids and start_abs > freeze_at:
            continue
        events.append({"type": "processing", "node_id": node_index, "start": start_abs,
                        "end": end_abs, "job_id": job_id, "task_id": task_index})
    for job_id, node_index, start_abs, end_abs, duration in (
        e for entries in master._state_a_transfers.values() for e in entries
    ):
        if job_id in reschedule_ids and start_abs > freeze_at:
            continue
        events.append({"type": "transfer", "node_id": node_index, "start": start_abs,
                        "end": end_abs, "job_id": job_id})
    for job_id, node_index, task_index, start_abs, end_abs, duration in (
        e for entries in result["works"].values() for e in entries
    ):
        events.append({"type": "processing", "node_id": node_index, "start": start_abs,
                        "end": end_abs, "job_id": job_id, "task_id": task_index})
    for job_id, node_index, start_abs, end_abs, duration in (
        e for entries in result["transfers"].values() for e in entries
    ):
        events.append({"type": "transfer", "node_id": node_index, "start": start_abs,
                        "end": end_abs, "job_id": job_id})
    return events


METRIC_ROWS = [("New job's wait time", "wait_time_new_job"),
               ("New job's flow time", "flow_time_new_job"),
               ("Mean flow time (all jobs)", "mean_flow_time_all"),
               ("Max flow time (all jobs)", "max_flow_time_all"),
               ("Transfer energy (this solve)", "transfer_energy_total")]

APPROACHES = ["incremental", "online_cold", "online_warmstart",
              "online_obj3_cold", "online_obj3_warmstart",
              "online_mean_cold", "online_mean_warmstart"]

# objective_choice for each online_* approach: 0=sum flow time (all jobs) -- equivalent to
# minimizing the MEAN flow time, since the batch size is fixed, so this is what "online_mean_*"
# uses; 1=max flow time (all jobs); 2=the new job's own flow time alone. Passed explicitly every
# time (see run_online's docstring) so cold vs warm-started is always an apples-to-apples
# comparison on the SAME objective -- MainOnline.java and MainOnlineWarmStart.java used to
# default to different ones (1 vs 0) when this wasn't specified.
ONLINE_JAVA_CLASS = {
    "online_cold": "MainOnline", "online_warmstart": "MainOnlineWarmStart",
    "online_obj3_cold": "MainOnline", "online_obj3_warmstart": "MainOnlineWarmStart",
    "online_mean_cold": "MainOnline", "online_mean_warmstart": "MainOnlineWarmStart",
}
ONLINE_OBJECTIVE = {
    "online_cold": 1, "online_warmstart": 1,
    "online_obj3_cold": 2, "online_obj3_warmstart": 2,
    "online_mean_cold": 0, "online_mean_warmstart": 0,
}


def solve_and_compare(master, new_job, isolated_ids, not_finished_jobs, args, out_dir, label, approaches, freeze_at):
    """Runs INCREMENTAL (always -- it's also the warm start's own source) plus whichever subset
    of the online_* approaches is requested, all against the SAME already-built state (synthetic
    or solved), and saves/prints the comparison. `approaches` is a subset of APPROACHES minus
    "incremental" (which is implicit)."""
    os.makedirs(out_dir, exist_ok=True)
    now = master.env.now
    replicas_locations = master.replicas_locations

    print(f"\n### [{label}] Solving new job's (id={new_job.job_id}) placement -- INCREMENTAL, "
          f"{args.incremental_time_limit}s budget ###", flush=True)
    master._config["solver_time_limit_s"] = args.incremental_time_limit
    incremental_result, incremental_transfers, incremental_works = run_incremental(
        master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations)
    print("incremental result:", {k: v for k, v in (incremental_result or {}).items()
                                   if k not in ("transfers", "works", "replicas_locations")}, flush=True)

    warm_start = None
    needs_warmstart = any(ONLINE_JAVA_CLASS[a] == "MainOnlineWarmStart" for a in approaches)
    if incremental_result is not None and needs_warmstart:
        jobs_to_reschedule = [new_job] + not_finished_jobs
        warm_start = build_warm_start(jobs_to_reschedule, not_finished_jobs, new_job, master,
                                       incremental_transfers, incremental_works, now)

    results = {"incremental": incremental_result}
    for approach in approaches:
        java_main_class = ONLINE_JAVA_CLASS[approach]
        objective_choice = ONLINE_OBJECTIVE[approach]
        if java_main_class == "MainOnlineWarmStart" and warm_start is None:
            print(f"### Skipping {approach}: incremental found no solution to warm-start from ###", flush=True)
            results[approach] = None
            continue
        print(f"\n### [{label}] Solving new job's placement -- "
              f"{approach.upper()} (java={java_main_class}, objective={objective_choice}), "
              f"{args.online_time_limit}s budget ###", flush=True)
        result = run_online(master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations,
                             not_finished_jobs, java_main_class, args.online_time_limit, objective_choice)
        print(f"{approach} result:", {k: v for k, v in (result or {}).items()
                                       if k not in ("transfers", "works", "replicas_locations")}, flush=True)
        results[approach] = result
    ran_approaches = ["incremental"] + list(approaches)

    for approach, result in results.items():
        approach_dir = os.path.join(out_dir, approach)
        os.makedirs(approach_dir, exist_ok=True)
        results_path = os.path.join(approach_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "approach": approach,
                "label": label,
                "nb_nodes": args.nb_nodes,
                "freeze_at": freeze_at,
                "isolated_ids": sorted(isolated_ids),
                "result": result,
            }, f, indent=2)
        gantt_events = build_gantt_events(master, result, freeze_at)
        if gantt_events:
            gantt_path = os.path.join(approach_dir, "gantt.png")
            plot_gantt_chart(gantt_events, args.nb_nodes,
                              title=f"{approach} -- {label} + new job",
                              save_path=gantt_path, freeze_at=freeze_at)

    col_width = 20
    total_width = 30 + col_width * len(ran_approaches)
    print("\n" + "=" * total_width)
    print(f"### {label} -- {' vs '.join(ran_approaches)} ###")
    print(f"{'Metric':<30}" + "".join(f"{a:>{col_width}}" for a in ran_approaches))
    print("=" * total_width)
    for row_label, key in METRIC_ROWS:
        cells = []
        for approach in ran_approaches:
            r = results[approach]
            v = r.get(key) if r else None
            cells.append(('%.2f' % v) if v is not None else 'N/A')
        print(f"{row_label:<30}" + "".join(f"{c:>{col_width}}" for c in cells))
    print("=" * total_width)
    return results


def main_synthetic(args, config, results_dir):
    """Sweeps --occupancy-fractions, running the full 5-way comparison at each exact level."""
    all_results = {}
    for fraction in args.occupancy_fractions:
        tag = f"occ{round(fraction * 100)}pct"
        out_dir = os.path.join(results_dir, tag)
        master, new_job, isolated_ids, not_finished_jobs = build_synthetic_state_a(config, args, fraction)
        label = f"{fraction * 100:.0f}% occupied"
        approaches = ["online_cold", "online_warmstart", "online_obj3_cold", "online_obj3_warmstart",
                      "online_mean_cold", "online_mean_warmstart"]
        all_results[fraction] = solve_and_compare(master, new_job, isolated_ids, not_finished_jobs, args,
                                                    out_dir, label, approaches, args.freeze_at)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "nb_nodes": args.nb_nodes,
            "occupancy_fractions": args.occupancy_fractions,
            "online_time_limit_s": args.online_time_limit,
            "incremental_time_limit_s": args.incremental_time_limit,
            "by_fraction": {
                str(frac): {
                    approach: ({k: v for k, v in r.items() if k not in ("transfers", "works", "replicas_locations")}
                               if r else None)
                    for approach, r in results.items()
                }
                for frac, results in all_results.items()
            },
        }, f, indent=2)
    print(f"\n### Summary written to {summary_path} ###", flush=True)

    print("\n" + "=" * 100)
    print("### FULL SWEEP SUMMARY (mean flow time / max flow time / energy) ###")
    header = f"{'Occupancy':<12}"
    for approach in APPROACHES:
        header += f"{approach + ' mean-FT':>20}{approach + ' max-FT':>18}{approach + ' energy':>16}"
    print(header)
    print("=" * 100)
    for fraction, results in all_results.items():
        row = f"{fraction * 100:>10.0f}% "
        for approach in APPROACHES:
            r = results.get(approach)
            mean_ft = ('%.2f' % r['mean_flow_time_all']) if r and r.get('mean_flow_time_all') is not None else 'N/A'
            max_ft = ('%.2f' % r['max_flow_time_all']) if r and r.get('max_flow_time_all') is not None else 'N/A'
            energy = ('%.2f' % r['transfer_energy_total']) if r and r.get('transfer_energy_total') is not None else 'N/A'
            row += f"{mean_ft:>20}{max_ft:>18}{energy:>16}"
        print(row)
    print("=" * 100)

    try:
        import matplotlib.pyplot as plt
        fractions_sorted = sorted(all_results.keys(), reverse=True)
        x_labels = [f"{f * 100:.0f}%" for f in fractions_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for approach, color in zip(APPROACHES, ['#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#d62728', '#17becf', '#8c564b']):
            mean_vals = [all_results[f][approach]["mean_flow_time_all"] if all_results[f][approach] else None for f in fractions_sorted]
            axes[0].plot(x_labels, mean_vals, marker='o', label=approach, color=color)
            energy_vals = [all_results[f][approach]["transfer_energy_total"] if all_results[f][approach] else None for f in fractions_sorted]
            axes[1].plot(x_labels, energy_vals, marker='o', label=approach, color=color)

        axes[0].set_xlabel("Infra occupancy at T")
        axes[0].set_ylabel("Mean flow time (all jobs)")
        axes[0].set_title("Solution quality vs occupancy")
        axes[0].legend()

        axes[1].set_xlabel("Infra occupancy at T")
        axes[1].set_ylabel("Transfer energy")
        axes[1].set_title("Energy cost vs occupancy")
        axes[1].legend()

        fig.suptitle(f"Incremental vs Online (max-flow, obj3=new job's flow time) -- {args.nb_nodes} nodes")
        plt.tight_layout()
        chart_path = os.path.join(results_dir, "warmstart_test_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"### Chart written to {chart_path} ###", flush=True)
    except Exception as e:
        print(f"### Could not generate chart: {e} ###", flush=True)


def main_solved(args, config, results_dir):
    """Builds state A the SAME way as xp_single_decision_grid5000.py -- ONE direct joint CSP
    solve over --n-existing real jobs from jobs.json, budget --state-a-time-limit -- then runs
    incremental + --solved-approaches on top of it (occupancy isn't directly controllable here;
    it's whatever that solve converges to)."""
    master, new_job, isolated_ids, not_finished_jobs = build_solved_state_a(config, args, results_dir)
    freeze_at = new_job.arriving_time
    label = f"solved state A ({args.n_existing}j, {args.state_a_time_limit}s budget)"
    results = solve_and_compare(master, new_job, isolated_ids, not_finished_jobs, args,
                                 results_dir, label, args.solved_approaches, freeze_at)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "state_a_mode": "solved",
            "nb_nodes": args.nb_nodes,
            "n_existing": args.n_existing,
            "state_a_time_limit_s": args.state_a_time_limit,
            "online_time_limit_s": args.online_time_limit,
            "incremental_time_limit_s": args.incremental_time_limit,
            "freeze_at": freeze_at,
            "results": {approach: ({k: v for k, v in r.items()
                                     if k not in ("transfers", "works", "replicas_locations")} if r else None)
                        for approach, r in results.items()},
        }, f, indent=2)
    print(f"\n### Summary written to {summary_path} ###", flush=True)


def main():
    args = parse_args()
    configure_logging(logging.WARNING)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000", f"online_warmstart_test_{args.nb_nodes}n_{date.today().isoformat()}")
    os.makedirs(results_dir, exist_ok=True)

    if args.state_a_mode == "solved":
        main_solved(args, config, results_dir)
    else:
        main_synthetic(args, config, results_dir)


if __name__ == "__main__":
    main()
