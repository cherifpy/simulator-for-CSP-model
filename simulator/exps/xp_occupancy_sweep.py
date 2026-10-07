"""
Controlled infra-occupancy sweep: at the moment T a new job arrives, directly (synthetically)
place one "existing" job per occupied node -- NO CSP solve is involved in building this state --
so the fraction of nodes genuinely busy at T is set EXACTLY (e.g. 100%, 75%, 50%, 25%) instead of
hoping a joint CSP solve happens to leave that many jobs still running by T (which, at real
instance scale, has repeatedly proven unreliable/slow to control in the sibling
xp_single_decision_grid5000.py script).

Each synthetic existing job gets exactly --occupied-nb-tasks tasks: task 0 is already IN
PROGRESS, straddling T (fixed/unmovable -- this is what makes the node genuinely "occupied" at
the freeze point), and every task after it is NOT yet started -- genuinely reconsiderable by an
Online replan, and left exactly where state A's own (synthetic) plan already queued it --
right after task 0, same node -- for Incremental's "everything else untouched" baseline.

Then the SAME new job (fully synthetic too, sized via --new-job-*) is solved via the REAL CSP
solver, both Online-style (joint replan of the new job + every occupied job's remaining tasks)
and Incremental-style (new job alone), once per occupancy level in --occupancy-fractions. Reports
and plots how the gap between them evolves as occupancy increases.

Only the infrastructure (bandwidth/compute/storage per node) comes from a real instance's
infrastructure.csv (via --instance-dir) -- jobs.json is never read; every job in this experiment
is synthetic.

Every path is resolved relative to this file's own location, so the whole `simulator/` folder
can be copied to Grid5000 (or anywhere) and this script keeps working unmodified.

Example (Grid5000, 50 nodes, sweep 100/75/50/25% occupancy, online @ 1h, incremental @ 30s):
    python3 xp_occupancy_sweep.py --instance-dir /path/to/inst-20J-50N --nb-nodes 50 \\
        --occupancy-fractions 1.0 0.75 0.5 0.25 \\
        --online-time-limit 3600 --incremental-time-limit 30 \\
        --results-dir /path/to/results/occupancy_sweep
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
from utils.modelCSP import schedulingUsingJavaCSP
from utils.run_export import start_recording
from utils.plots import plot_gantt_chart
from simulator import generateHeterogeneousInfrastructureEquilibre, configure_logging

logger = logging.getLogger(__name__)

NEW_JOB_ID_BASE = 10_000_000  # clear of any occupied-node job id (0..nb_nodes-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing infrastructure.csv (jobs.json is not used -- "
                              "every job here is synthetic).")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes to use.")
    parser.add_argument("--occupancy-fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25],
                         help="Fractions of --nb-nodes to make genuinely busy at T, one run each "
                              "(default: 1.0 0.75 0.5 0.25). occupied_count = round(nb_nodes * "
                              "fraction); occupied nodes are the first occupied_count of the node "
                              "list each time (fresh state per fraction, not cumulative).")
    parser.add_argument("--freeze-at", type=float, default=1000.0,
                         help="Synthetic 'now' (T) when the new job arrives (default: 1000.0).")
    parser.add_argument("--occupied-nb-tasks", type=int, default=3,
                         help="Tasks per synthetic existing (occupying) job: task 0 is already in "
                              "progress at T, the rest are not-yet-started and genuinely "
                              "reconsiderable by Online (default: 3).")
    parser.add_argument("--occupied-task-duration", type=int, default=300,
                         help="Duration in seconds of each task of a synthetic existing job "
                              "(default: 300). Task 0 straddles T symmetrically: "
                              "[T - duration/2, T + duration/2].")
    parser.add_argument("--occupied-dataset-size", type=int, default=5120,
                         help="Dataset size in MB for each synthetic existing job (default: 5120, "
                              "i.e. 5GB). Already resident on its node by construction.")
    parser.add_argument("--new-job-dataset-size", type=int, default=5120,
                         help="Dataset size in MB for the synthetic new job (default: 5120).")
    parser.add_argument("--new-job-nb-tasks", type=int, default=15,
                         help="Number of tasks for the synthetic new job (default: 15).")
    parser.add_argument("--new-job-task-duration", type=int, default=100,
                         help="Per-task duration in seconds for the synthetic new job (default: 100).")
    parser.add_argument("--online-time-limit", required=True, type=int,
                         help="CSP solver time budget (seconds) for Online's placement solve.")
    parser.add_argument("--incremental-time-limit", required=True, type=int,
                         help="CSP solver time budget (seconds) for Incremental's placement solve.")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write per-fraction results + the summary. Default: "
                              "<simulator>/results-grid5000/occupancy_sweep_<nb_nodes>n_<date>.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def build_synthetic_state_a(config, args, occupancy_fraction):
    """Directly constructs state A -- one synthetic existing job per occupied node, no CSP solve
    involved -- so the fraction of nodes busy at T is EXACT, not a hopeful outcome of a joint
    solve. See module docstring for the per-job task layout."""
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
        # Ghost storage entries are unconditional (a fixed height on that node's cumulative
        # capacity constraint, not gated by any placement choice) -- an occupied job whose
        # dataset_size exceeds ITS OWN node's real storage_capacity makes the model infeasible
        # from construction alone, regardless of solver budget (real infra is heterogeneous:
        # e.g. this instance has nodes as small as ~2GB and as large as 100GB). Clip to what
        # this specific node can actually hold.
        node_dataset_size = min(args.occupied_dataset_size, int(compute_nodes[node_id].storage_capacity))
        job = Job(job_id, args.occupied_task_duration, args.occupied_nb_tasks, node_dataset_size)
        job.arriving_time = freeze_at - half
        existing_jobs.append(job)
        tracker.register_job(job_id, job.arriving_time)

        # Task 0: already in progress, straddling T -- fixed/unmovable. This alone is what makes
        # this node genuinely "occupied" at the freeze point.
        t0_start, t0_end = freeze_at - half, freeze_at + half
        job.tasks[0].status = "Started"
        state_a_works[f'node_{node_id}'].append((job_id, node_id, 0, t0_start, t0_end, args.occupied_task_duration))
        master.ongoing_works[f'node_{node_id}'] = (job_id, node_id, 0, t0_start, t0_end, args.occupied_task_duration)

        # A synthetic, already-finished transfer right before task 0 -- purely so the gantt shows
        # this job's data landing on the node. The data is already resident by construction.
        state_a_transfers[f'node_{node_id}'].append((job_id, node_id, t0_start - 1, t0_start, 1))
        master.replicas_locations[job_id] = [node_id]

        # Remaining tasks: NOT yet started (Task class default) -- genuinely reconsiderable by an
        # Online replan. Under state A's own (synthetic) plan they're simply queued right after
        # task 0, same node, back-to-back -- exactly what Incremental's untouched baseline keeps.
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

    # Every synthetic existing job is "still running" (not finished) at T by construction -- its
    # last task was deliberately queued to end well after T.
    not_finished_jobs = list(existing_jobs)

    env.run(until=freeze_at)  # 0 processes registered -- SimPy just fast-forwards the clock

    new_job = Job(NEW_JOB_ID_BASE, args.new_job_task_duration, args.new_job_nb_tasks, args.new_job_dataset_size)
    new_job.arriving_time = env.now
    master.tracker.register_job(new_job.job_id, env.now)

    isolated_ids = {j.job_id for j in not_finished_jobs} | {new_job.job_id}
    print(f"### Synthetic state A: {occupied_count}/{args.nb_nodes} nodes occupied "
          f"({occupancy_fraction * 100:.0f}%) at T={freeze_at:.2f}, {occupied_count} existing "
          f"jobs (all still running by construction) ###", flush=True)
    return master, new_job, isolated_ids, not_finished_jobs


def compute_transfer_energy(transfers_, master):
    """Same formula as Tracker.log_transfer -- this test bypasses the live simulation loop
    entirely, so proposed transfers never flow through the tracker's own accounting."""
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


def run_online_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs):
    """Mirrors SchedulingUsingCSPOnline.schedulingNewJob(): jointly replan the new job + every
    occupied job's not-yet-started remainder."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnline'
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    if not transfers_ or not works_:
        return None

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
        "deletions": deletions_,
        "replicas_locations": master.replicas_locations,
    }


def run_incremental_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs=None):
    """Mirrors SchedulingUsingCSPIncremental.schedulingNewJob(): place the new job alone; every
    occupied job keeps whatever state A already committed it to, untouched."""
    jobs_to_reschedule = [new_job]
    nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(master, master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainIncremental'
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    if not transfers_ or not works_:
        return None

    new_job_finish = None
    for key in [f'node_{i}' for i in range(nb_nodes)]:
        for w in works_.get(key, []):
            job_id, node_index, task_index, start_abs, end_abs, duration = w
            if job_id == new_job.job_id:
                new_job_finish = end_abs if new_job_finish is None else max(new_job_finish, end_abs)

    flow_by_job = {new_job.job_id: (new_job_finish - new_job.arriving_time)} if new_job_finish is not None else {}
    for j in master.jobs:
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
        "deletions": deletions_,
        "replicas_locations": master.replicas_locations,
    }


def build_gantt_events(master, result, freeze_at):
    """Same policy as xp_single_decision_grid5000.py's build_gantt_events: keep state A's own
    placement for everything already committed by T even for a rescheduled job; only the portion
    that was still genuinely open at T (start_abs > freeze_at) is superseded by this solve's own
    result."""
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
               ("Mean flow time (ISOLATED)", "mean_flow_time_isolated"),
               ("Max flow time (ISOLATED)", "max_flow_time_isolated"),
               ("Transfer energy (this solve)", "transfer_energy_total")]


def run_one_fraction(config, args, occupancy_fraction, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    master, new_job, isolated_ids, not_finished_jobs = build_synthetic_state_a(config, args, occupancy_fraction)
    now = master.env.now
    replicas_locations = master.replicas_locations

    results = {}
    for approach, time_limit, solve_fn in [
        ("online", args.online_time_limit, run_online_style),
        ("incremental", args.incremental_time_limit, run_incremental_style),
    ]:
        approach_dir = os.path.join(out_dir, approach)
        os.makedirs(approach_dir, exist_ok=True)
        master._config["solver_time_limit_s"] = time_limit
        print(f"\n### [{occupancy_fraction * 100:.0f}% occupied] Solving new job's (id={new_job.job_id}) "
              f"placement -- {approach.upper()} style, {time_limit}s budget ###", flush=True)
        result = solve_fn(master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations, not_finished_jobs)
        result_summary = {k: v for k, v in result.items()
                           if k not in ("transfers", "works", "deletions", "replicas_locations", "transfer_energy_detail")} if result else None
        print(f"{approach} result ({time_limit}s):", result_summary, flush=True)

        results_path = os.path.join(approach_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "approach": approach,
                "occupancy_fraction": occupancy_fraction,
                "solver_time_limit_s": time_limit,
                "nb_nodes": args.nb_nodes,
                "freeze_at": args.freeze_at,
                "isolated_ids": sorted(isolated_ids),
                "result": result,
            }, f, indent=2)
        print(f"### Results written to {results_path} ###", flush=True)

        gantt_events = build_gantt_events(master, result, args.freeze_at)
        if gantt_events:
            gantt_path = os.path.join(approach_dir, "gantt.png")
            plot_gantt_chart(gantt_events, args.nb_nodes,
                              title=f"{approach} -- {occupancy_fraction * 100:.0f}% occupied + new job",
                              save_path=gantt_path, freeze_at=args.freeze_at)
            print(f"### Gantt chart written to {gantt_path} ###", flush=True)

        results[approach] = result

    print("\n" + "=" * 70)
    print(f"### {occupancy_fraction * 100:.0f}% OCCUPANCY -- online ({args.online_time_limit}s) vs "
          f"incremental ({args.incremental_time_limit}s) ###")
    print(f"{'Metric':<30}{'online':>20}{'incremental':>20}")
    print("=" * 70)
    for row_label, key in METRIC_ROWS:
        online_v = results["online"].get(key) if results["online"] else None
        incr_v = results["incremental"].get(key) if results["incremental"] else None
        online_s = ('%.2f' % online_v) if online_v is not None else 'N/A'
        incr_s = ('%.2f' % incr_v) if incr_v is not None else 'N/A'
        print(f"{row_label:<30}{online_s:>20}{incr_s:>20}")
    print("=" * 70)
    return results


def main():
    args = parse_args()
    configure_logging(logging.WARNING)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000", f"occupancy_sweep_{args.nb_nodes}n_{date.today().isoformat()}")
    os.makedirs(results_dir, exist_ok=True)

    start_recording(results_dir, params=args, config=config)

    all_results = {}
    for fraction in args.occupancy_fractions:
        tag = f"occ{round(fraction * 100)}pct"
        out_dir = os.path.join(results_dir, tag)
        all_results[fraction] = run_one_fraction(config, args, fraction, out_dir)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "nb_nodes": args.nb_nodes,
            "occupancy_fractions": args.occupancy_fractions,
            "freeze_at": args.freeze_at,
            "online_time_limit_s": args.online_time_limit,
            "incremental_time_limit_s": args.incremental_time_limit,
            "by_fraction": {
                str(frac): {
                    approach: ({k: v for k, v in r.items() if k not in
                                ("transfers", "works", "deletions", "replicas_locations", "transfer_energy_detail")}
                               if r else None)
                    for approach, r in results.items()
                }
                for frac, results in all_results.items()
            },
        }, f, indent=2)
    print(f"\n### Summary written to {summary_path} ###", flush=True)

    print("\n" + "=" * 90)
    print("### FULL SWEEP SUMMARY ###")
    header = f"{'Occupancy':<12}"
    for approach in ("online", "incremental"):
        header += f"{approach + ' mean-FT':>18}{approach + ' max-FT':>18}{approach + ' energy':>18}"
    print(header)
    print("=" * 90)
    for fraction, results in all_results.items():
        row = f"{fraction * 100:>10.0f}% "
        for approach in ("online", "incremental"):
            r = results.get(approach)
            mean_ft = ('%.2f' % r['mean_flow_time_all']) if r and r.get('mean_flow_time_all') is not None else 'N/A'
            max_ft = ('%.2f' % r['max_flow_time_all']) if r and r.get('max_flow_time_all') is not None else 'N/A'
            energy = ('%.2f' % r['transfer_energy_total']) if r and r.get('transfer_energy_total') is not None else 'N/A'
            row += f"{mean_ft:>18}{max_ft:>18}{energy:>18}"
        print(row)
    print("=" * 90)

    try:
        import matplotlib.pyplot as plt
        fractions_sorted = sorted(all_results.keys(), reverse=True)
        online_mean = [all_results[f]["online"]["mean_flow_time_all"] if all_results[f]["online"] else None for f in fractions_sorted]
        incr_mean = [all_results[f]["incremental"]["mean_flow_time_all"] if all_results[f]["incremental"] else None for f in fractions_sorted]
        online_energy = [all_results[f]["online"]["transfer_energy_total"] if all_results[f]["online"] else None for f in fractions_sorted]
        incr_energy = [all_results[f]["incremental"]["transfer_energy_total"] if all_results[f]["incremental"] else None for f in fractions_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x_labels = [f"{f * 100:.0f}%" for f in fractions_sorted]
        axes[0].plot(x_labels, online_mean, marker='o', label='online')
        axes[0].plot(x_labels, incr_mean, marker='o', label='incremental')
        axes[0].set_xlabel("Infra occupancy at T")
        axes[0].set_ylabel("Mean flow time (all jobs)")
        axes[0].set_title("Solution quality vs occupancy")
        axes[0].legend()

        axes[1].plot(x_labels, online_energy, marker='o', label='online')
        axes[1].plot(x_labels, incr_energy, marker='o', label='incremental')
        axes[1].set_xlabel("Infra occupancy at T")
        axes[1].set_ylabel("Transfer energy")
        axes[1].set_title("Energy cost vs occupancy")
        axes[1].legend()

        fig.suptitle(f"Online vs Incremental across infra occupancy ({args.nb_nodes} nodes)")
        plt.tight_layout()
        chart_path = os.path.join(results_dir, "occupancy_sweep_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"### Sweep chart written to {chart_path} ###", flush=True)
    except Exception as e:
        print(f"### Could not generate sweep chart: {e} ###", flush=True)


if __name__ == "__main__":
    main()
