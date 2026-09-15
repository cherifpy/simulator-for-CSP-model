"""
Standalone, portable Grid5000 launcher for the controlled single-decision-point test: build
state A as a single, direct JOINT CSP solve over the --n-existing jobs (their real
dataset_size/nb_tasks/task_duration/arriving_time from the instance), with node free times at 0
and no pre-existing replicas -- a genuine one-shot "offline" placement decision for those N jobs
(NOT a replay of N sequential live-simulation arrivals). Then inject exactly ONE new job and
solve ITS placement with a given approach (online-style joint replan, or incremental-style: the
new job alone) and a given solver time budget. Reports the new job's wait/flow time plus the
mean/max flow time over ALL jobs (state A + new job) -- since nothing has actually "finished" in
this offline framing (state A's own solve just decided a placement, none of it physically
executed), every existing job is legitimately part of the comparison scope, not just a subset.

Every path is resolved relative to this file's own location, so the whole `simulator/` folder
can be copied to Grid5000 (or anywhere) and this script keeps working unmodified.

Example (2h budget on Grid5000, Online-style, N=15 existing jobs, job index 15 is the new one):
    python3 xp_single_decision_grid5000.py --approach online \
        --instance-dir /path/to/inst-20J-50N --nb-nodes 50 --n-existing 15 \
        --solver-time-limit 7200 --lambda-rate 100 \
        --results-dir /path/to/results/single_decision_online_7200s

    python3 xp_single_decision_grid5000.py --approach incremental \
        --instance-dir /path/to/inst-20J-50N --nb-nodes 50 --n-existing 15 \
        --solver-time-limit 30 --lambda-rate 100
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
from utils.plots import plot_gantt_chart
from simulator import generateHeterogeneousInfrastructureEquilibre, configure_logging

logger = logging.getLogger(__name__)

JAVA_MAIN_CLASS_BY_APPROACH = {
    "online": "MainOnline",
    "incremental": "MainIncremental",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", required=True, choices=sorted(JAVA_MAIN_CLASS_BY_APPROACH.keys()),
                         help="How to solve the single new job's placement: 'online' jointly "
                              "replans it with every existing job that still has unstarted tasks; "
                              "'incremental' places it alone, leaving existing jobs untouched.")
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing this instance's jobs.json and infrastructure.csv.")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes in the instance.")
    parser.add_argument("--n-existing", type=int, default=15,
                         help="Number of jobs (from the start of jobs.json) used to build state A, "
                              "i.e. already 'in progress' before the new job arrives (default: 15).")
    parser.add_argument("--new-job-index", type=int, default=None,
                         help="Index into jobs.json of the single new job injected after state A is "
                              "frozen (default: --n-existing, i.e. the job right after state A).")
    parser.add_argument("--solver-time-limit", required=True, type=int,
                         help="CSP solver time budget for the new job's placement solve, in seconds "
                              "(e.g. 7200 for a 2h Grid5000 run). Only applies to that final solve -- "
                              "the state-A-building solve always uses --state-a-time-limit.")
    parser.add_argument("--state-a-time-limit", type=int, default=30,
                         help="CSP solver time budget for the ONE joint solve that builds state A "
                              "over all --n-existing jobs at once (default: 30s).")
    parser.add_argument("--lambda-rate", type=int, default=100,
                         help="Present for config compatibility; unused now that state A is a single "
                              "direct joint solve rather than a live-simulation replay with injected "
                              "arrivals (default: 100).")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write results.json. Default: "
                              "<simulator>/results-grid5000/single_decision_<approach>_<n_existing>j-"
                              "<nb_nodes>n_<solver_time_limit>s_<date>.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def build_state_a(config, args, results_dir):
    """Builds state A as ONE direct joint CSP solve over all --n-existing jobs at once (their real
    dataset_size/nb_tasks/task_duration/arriving_time), with node free times at 0 and no
    pre-existing replicas -- a genuine one-shot "offline" placement, not a replay of N sequential
    live-simulation arrivals. env.now is fast-forwarded directly to the moment the new job would
    arrive via env.run(until=...) with NO processes registered (SimPy just advances the clock;
    nothing else happens), since schedulingUsingJavaCSP reads master.env.now directly everywhere
    (its scheduling_start_time parameter is unused) and the rest of this script uses that same
    value as its "now" reference throughout."""
    with open(os.path.join(args.instance_dir, "jobs.json")) as f:
        all_jobs_raw = json.load(f)
    existing_jobs_raw = all_jobs_raw[:args.n_existing]
    new_job_index = args.new_job_index if args.new_job_index is not None else args.n_existing
    new_job_raw = all_jobs_raw[new_job_index]
    freeze_at = new_job_raw["arriving_time"]

    with open(os.path.join(results_dir, "state_a_jobs.json"), "w") as f:
        json.dump(existing_jobs_raw, f)

    config = dict(config)
    config["total_nb_jobs"] = args.n_existing
    config["total_nb_compute_nodes"] = args.nb_nodes
    config["solver_time_limit_s"] = args.state_a_time_limit
    config["lambda_rate"] = args.lambda_rate

    random.seed(args.seed)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(
        config, path=os.path.join(args.instance_dir, "infrastructure.csv"))

    env = simpy.Environment()
    env.run(until=freeze_at)
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
    # Constructed with compute_nodes=[] above (ComputeNode needs an already-existing master to
    # reference), so __init__ never got a chance to size these per-node dicts -- do it now, same
    # as __init__ would have. Without this, nodesFreeTimeIncremental's direct `self.transfers[key]`
    # indexing (no .get()) raises KeyError.
    for i in range(len(compute_nodes)):
        key = f'node_{i}'
        master.ongoing_transfers.setdefault(key, None)
        master.ongoing_works.setdefault(key, None)
        master.transfers.setdefault(key, [])
        master.works.setdefault(key, [])
        master.deletions.setdefault(key, [])

    existing_jobs = []
    for raw in existing_jobs_raw:
        j = Job(raw["job_id"], raw["task_duration"], raw["nb_tasks"], raw["dataset_size"])
        j.arriving_time = raw["arriving_time"]
        existing_jobs.append(j)
        tracker.register_job(j.job_id, raw["arriving_time"])
    master.jobs = existing_jobs

    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
    master.java_main_class = 'MainOnline'
    print(f"### Building state A: ONE joint solve over jobs 0..{args.n_existing - 1} "
          f"(solver_time_limit={args.state_a_time_limit}s, node free times=0, no pre-existing "
          f"replicas) ###", flush=True)
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, existing_jobs, {}, nodes_free_time, freeze_at)

    state_a_finish = {}
    if works_:
        for entries in works_.values():
            for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                state_a_finish[job_id] = end_abs if job_id not in state_a_finish else max(state_a_finish[job_id], end_abs)
    master._state_a_finish = state_a_finish
    master._state_a_works = works_ or {}
    master._state_a_transfers = transfers_ or {}
    if len(state_a_finish) < args.n_existing:
        print(f"### WARNING: state A solve only placed {len(state_a_finish)}/{args.n_existing} jobs "
              f"-- possibly infeasible or cut short within the time budget ###", flush=True)
    print(f"### State A built. env.now={env.now:.2f}, jobs placed={len(state_a_finish)}/{args.n_existing} ###",
          flush=True)

    new_job = Job(new_job_raw["job_id"], new_job_raw["task_duration"], new_job_raw["nb_tasks"], new_job_raw["dataset_size"])
    new_job.arriving_time = env.now
    master.tracker.register_job(new_job.job_id, env.now)

    # Nothing has actually "finished" in this offline framing -- state A's solve just DECIDED a
    # placement for these jobs, none of them physically executed. So every existing job (plus the
    # new one) is legitimately part of the comparison scope, unlike the live-replay version where
    # some jobs could already be done by the freeze point.
    isolated_ids = {j.job_id for j in existing_jobs} | {new_job.job_id}
    print(f"### Isolated comparison scope: all {len(isolated_ids)} jobs (state A + new job) ###", flush=True)
    return master, new_job, isolated_ids


def committed_finish_time(master, nb_nodes, job_id):
    """Already-decided completion time for an existing job that this solve does NOT re-plan --
    read directly from state A's own solve result (there's no live execution history to fall back
    on in this offline-style construction)."""
    return master._state_a_finish.get(job_id)


def compute_transfer_energy(transfers_, master):
    """Same formula as Tracker.log_transfer (energy = sender + receiver * transfer_time +
    network), applied here to a solve's PROPOSED transfers -- this test bypasses the live
    simulation loop entirely (it calls schedulingUsingJavaCSP directly and never actually
    executes the resulting plan), so those transfers never flow through the tracker's own
    accounting and would otherwise be invisible to any energy comparison."""
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


def run_online_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations):
    """Mirrors SchedulingUsingCSPOnline.schedulingNewJob(): jointly replan the new job + every
    existing job that still has unstarted tasks."""
    jobs_to_reschedule = [new_job] + master.getRunningJobs()
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
    reoptimized_flow_times = []
    historical_flow_times = []
    for j in master.jobs + [new_job]:
        if j.job_id in finish and finish[j.job_id] is not None:
            ft = finish[j.job_id] - j.arriving_time
            flow_by_job[j.job_id] = ft
            reoptimized_flow_times.append(ft)
        else:
            cf = committed_finish_time(master, nb_nodes, j.job_id)
            if cf is not None:
                ft = cf - j.arriving_time
                flow_by_job[j.job_id] = ft
                historical_flow_times.append(ft)
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
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "n_jobs_in_mean": len(flow_times),
        "flow_times_detail": flow_times,
        "flow_time_by_job": flow_by_job,
        "n_reoptimized": len(reoptimized_flow_times),
        "sum_reoptimized": sum(reoptimized_flow_times) if reoptimized_flow_times else None,
        "mean_reoptimized": (sum(reoptimized_flow_times) / len(reoptimized_flow_times)) if reoptimized_flow_times else None,
        "n_historical": len(historical_flow_times),
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": transfer_energy_total,
        "transfer_energy_detail": transfer_energy_detail,
        "transfers": transfers_,
        "works": works_,
        "deletions": deletions_,
        "replicas_locations": master.replicas_locations,
    }


def run_incremental_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations):
    """Mirrors SchedulingUsingCSPIncremental.schedulingNewJob(): place the new job alone; every
    existing job keeps whatever it was already committed to in state A, untouched."""
    jobs_to_reschedule = [new_job]
    # master is a SchedulingUsingCSPOnline instance (state A is always built via Online now);
    # nodesFreeTimeIncremental is only DEFINED on the Incremental subclass, but only ever
    # touches attributes (transfers/works/compute_nodes/env) that any instance already has,
    # so calling it unbound on our Online instance is safe and avoids needing a second master.
    nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(master, master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainIncremental'
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)

    if not transfers_ or not works_:
        return None

    new_job_finish = None
    new_job_start = None
    for key in [f'node_{i}' for i in range(nb_nodes)]:
        for w in works_.get(key, []):
            job_id, node_index, task_index, start_abs, end_abs, duration = w
            if job_id == new_job.job_id:
                new_job_finish = end_abs if new_job_finish is None else max(new_job_finish, end_abs)
                new_job_start = start_abs if new_job_start is None else min(new_job_start, start_abs)

    flow_by_job = {}
    reoptimized_flow_times = []
    historical_flow_times = []
    for j in master.jobs + [new_job]:
        if j.job_id == new_job.job_id:
            if new_job_finish is not None:
                ft = new_job_finish - j.arriving_time
                flow_by_job[j.job_id] = ft
                reoptimized_flow_times.append(ft)
        else:
            cf = committed_finish_time(master, nb_nodes, j.job_id)
            if cf is not None:
                ft = cf - j.arriving_time
                flow_by_job[j.job_id] = ft
                historical_flow_times.append(ft)
    flow_times = list(flow_by_job.values())
    isolated_flow_times = [ft for jid, ft in flow_by_job.items() if jid in isolated_ids]

    wait_time = (new_job_start - new_job.arriving_time) if new_job_start is not None else None
    transfer_energy_total, transfer_energy_detail = compute_transfer_energy(transfers_, master)
    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (new_job_finish - new_job.arriving_time) if new_job_finish is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "n_jobs_in_mean": len(flow_times),
        "flow_times_detail": flow_times,
        "n_reoptimized": len(reoptimized_flow_times),
        "sum_reoptimized": sum(reoptimized_flow_times) if reoptimized_flow_times else None,
        "mean_reoptimized": (sum(reoptimized_flow_times) / len(reoptimized_flow_times)) if reoptimized_flow_times else None,
        "n_historical": len(historical_flow_times),
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "flow_time_by_job": flow_by_job,
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": transfer_energy_total,
        "transfer_energy_detail": transfer_energy_detail,
        "transfers": transfers_,
        "works": works_,
        "deletions": deletions_,
        "replicas_locations": master.replicas_locations,
    }


def print_jobs_replicas_and_schedule(master, new_job, isolated_ids, result):
    """Prints full per-job, per-replica and per-task/per-transfer detail -- same information
    that used to be visible only as raw object reprs / the Java model's own stdout, now spelled
    out explicitly and also captured in the persisted results.json (schedule detail is only
    solved for the jobs actually in jobs_to_reschedule; every other job's placement is exactly
    what state A already committed and is not repeated here)."""
    print("\n### Jobs (state A + new job) ###", flush=True)
    print(f"{'job_id':>8}{'nb_tasks':>10}{'task_duration':>15}{'dataset_size':>14}{'arriving_time':>15}{'isolated?':>11}")
    for j in sorted(master.jobs + [new_job], key=lambda j: j.job_id):
        task_duration = j.tasks[0].duration if j.tasks else None
        print(f"{j.job_id:>8}{j.nb_tasks:>10}{task_duration!s:>15}{j.dataset_size:>14}"
              f"{j.arriving_time:>15.2f}{'yes' if j.job_id in isolated_ids else 'no':>11}")

    print("\n### Replicas locations (after this solve) ###", flush=True)
    for job_id, nodes in sorted(master.replicas_locations.items(), key=lambda kv: kv[0]):
        print(f"  job {job_id}: nodes {nodes}")

    if result is None:
        print("\n(no schedule detail -- solve returned no solution)")
        return

    print(f"\n### Schedule detail for the re-solved jobs {result['jobs_to_reschedule']} ###", flush=True)
    for key in sorted(result["works"].keys(), key=lambda k: int(k.split('_')[1])):
        for job_id, node_index, task_index, start_abs, end_abs, duration in result["works"][key]:
            print(f"  {key} - job {job_id} - task {task_index} - start: {start_abs:.2f} - end: {end_abs:.2f}")
    for key in sorted(result["transfers"].keys(), key=lambda k: int(k.split('_')[1])):
        for entry in result["transfers"][key]:
            print(f"  {key} - transfer {entry}")
    for key in sorted(result["deletions"].keys(), key=lambda k: int(k.split('_')[1])):
        for entry in result["deletions"][key]:
            print(f"  {key} - deletion {entry}")

    print("\n### Flow time by job (this solve) ###", flush=True)
    for job_id, ft in sorted(result["flow_time_by_job"].items()):
        print(f"  job {job_id}: flow_time={ft:.2f}"
              f"{'  <- new job' if job_id == new_job.job_id else ''}"
              f"{'  [isolated]' if job_id in isolated_ids else ''}")


def build_gantt_events(master, result):
    """Full final schedule as plot_gantt_chart's flat event list: state A's own placement for
    every job it decided, with any job actually touched by THIS solve (jobs_to_reschedule)
    overridden by its result instead -- for Online that's the new job + every still-open
    existing job it re-planned; for Incremental it's just the new job, so every other job's
    state-A placement is untouched here too."""
    if result is None:
        return []
    reschedule_ids = set(result["jobs_to_reschedule"])
    events = []
    for source_works, source_transfers, skip_ids in (
        (master._state_a_works, master._state_a_transfers, reschedule_ids),
        (result["works"], result["transfers"], set()),
    ):
        for entries in source_works.values():
            for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                if job_id in skip_ids:
                    continue
                events.append({"type": "processing", "node_id": node_index, "start": start_abs,
                                "end": end_abs, "job_id": job_id, "task_id": task_index})
        for entries in source_transfers.values():
            for job_id, node_index, start_abs, end_abs, duration in entries:
                if job_id in skip_ids:
                    continue
                events.append({"type": "transfer", "node_id": node_index, "start": start_abs,
                                "end": end_abs, "job_id": job_id})
    return events


def main():
    args = parse_args()
    configure_logging(logging.WARNING)

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000",
        f"single_decision_{args.approach}_{args.n_existing}j-{args.nb_nodes}n_"
        f"{args.solver_time_limit}s_{date.today().isoformat()}",
    )
    os.makedirs(results_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    master, new_job, isolated_ids = build_state_a(config, args, results_dir)
    now = master.env.now
    replicas_locations = master.replicas_locations

    master._config["solver_time_limit_s"] = args.solver_time_limit
    print(f"\n### Solving new job's (id={new_job.job_id}) placement -- {args.approach.upper()} style, "
          f"{args.solver_time_limit}s budget ###", flush=True)
    solve_fn = run_online_style if args.approach == "online" else run_incremental_style
    result = solve_fn(master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations)
    result_summary = {k: v for k, v in result.items()
                       if k not in ("transfers", "works", "deletions", "replicas_locations", "transfer_energy_detail")} if result else None
    print(f"{args.approach} result ({args.solver_time_limit}s):", result_summary, flush=True)

    print_jobs_replicas_and_schedule(master, new_job, isolated_ids, result)

    print("\n" + "=" * 70)
    label = f"{args.approach} ({args.solver_time_limit}s)"
    print(f"{'Metric':<30}{label:>25}")
    print("=" * 70)
    for row_label, key in [("New job's wait time", "wait_time_new_job"),
                            ("New job's flow time", "flow_time_new_job"),
                            ("Mean flow time (all jobs)", "mean_flow_time_all"),
                            ("Max flow time (all jobs)", "max_flow_time_all"),
                            ("Mean flow time (ISOLATED)", "mean_flow_time_isolated"),
                            ("Max flow time (ISOLATED)", "max_flow_time_isolated"),
                            ("Transfer energy (this solve)", "transfer_energy_total")]:
        v = result.get(key) if result else None
        print(f"{row_label:<30}{('%.2f' % v) if v is not None else 'N/A':>25}")
    print("=" * 70)

    jobs_info = [
        {
            "job_id": j.job_id,
            "nb_tasks": j.nb_tasks,
            "task_duration": j.tasks[0].duration if j.tasks else None,
            "dataset_size": j.dataset_size,
            "arriving_time": j.arriving_time,
            "isolated": j.job_id in isolated_ids,
        }
        for j in sorted(master.jobs + [new_job], key=lambda j: j.job_id)
    ]

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "approach": args.approach,
            "solver_time_limit_s": args.solver_time_limit,
            "state_a_time_limit_s": args.state_a_time_limit,
            "n_existing": args.n_existing,
            "new_job_index": args.new_job_index if args.new_job_index is not None else args.n_existing,
            "new_job_id": new_job.job_id,
            "nb_nodes": args.nb_nodes,
            "instance_dir": args.instance_dir,
            "isolated_ids": sorted(isolated_ids),
            "jobs": jobs_info,
            "replicas_locations": master.replicas_locations,
            "result": result,
        }, f, indent=2)
    print(f"\n### Results written to {results_path} ###", flush=True)

    gantt_events = build_gantt_events(master, result)
    if gantt_events:
        gantt_path = os.path.join(results_dir, "gantt.png")
        plot_gantt_chart(gantt_events, args.nb_nodes,
                          title=f"{args.approach} -- state A ({args.n_existing}j) + new job {new_job.job_id}",
                          save_path=gantt_path)
        print(f"### Gantt chart written to {gantt_path} ###", flush=True)
    else:
        print("### No gantt chart: no schedule to plot (solve returned no solution) ###", flush=True)


if __name__ == "__main__":
    main()
