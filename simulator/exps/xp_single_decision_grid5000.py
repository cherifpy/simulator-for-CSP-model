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
    parser.add_argument("--occupied-nodes-fraction", type=float, default=1.0,
                         help="Fraction of --nb-nodes eligible for state A's own solve (default: "
                              "1.0, i.e. all of them). Use this to directly control infra occupancy "
                              "instead of --n-existing/--state-a-time-limit alone -- e.g. 5 existing "
                              "jobs restricted to the first 50%% of nodes (--occupied-nodes-fraction "
                              "0.5) genuinely fills that half, rather than hoping a joint solve over "
                              "all N nodes happens to leave some jobs still running by the freeze "
                              "point. The new job's own solve still sees every node -- only state A's "
                              "construction is restricted to the first "
                              "round(nb_nodes * occupied_nodes_fraction) nodes.")
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
    live-simulation arrivals.

    Critically, this solve runs with env.now=0 (each job's OWN real arriving_time as its actual
    lower bound, absolute times coming out exactly as if state A had really executed from t=0) --
    NOT with env.now already at the freeze point. schedulingUsingJavaCSP reads master.env.now
    directly everywhere to turn Java's local times into absolute ones, so solving at env.now=T
    would wrongly anchor every job's schedule to start at-or-after T, making nothing ever appear
    "already finished" by the time the new job arrives (jobs that arrived long ago would still be
    scheduled to start in the future). Solving at env.now=0 first lets each job's real committed
    finish time fall wherever the solver decides -- some existing jobs will genuinely have
    finished before the freeze point T, some will still be mid-execution -- exactly like the
    original live-simulation version, just computed with one direct solve instead of a replay.
    Only AFTER solving is env.now advanced to T (via env.run(until=T) with no processes
    registered, so SimPy just advances the clock) for the new job's own arrival and the final
    solve."""
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

    env = simpy.Environment()  # starts at now=0 -- do NOT advance it before the state-A solve
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

    # Restrict state A's OWN solve to a PREFIX of the node list (nb_occupied nodes), so the
    # --n-existing jobs are forced to genuinely fill that fraction of the infra, rather than
    # spreading across all --nb-nodes and leaving occupancy up to how the solver happens to
    # behave. Node indices returned by this restricted solve equal the real global node index
    # exactly (it's a prefix, not an arbitrary subset), so no remapping is needed. Restored to the
    # full node list right after, since the final new-job solve should see every node -- the
    # un-occupied ones are genuinely free.
    nb_occupied = max(1, min(args.nb_nodes, round(args.nb_nodes * args.occupied_nodes_fraction)))
    master.compute_nodes = compute_nodes[:nb_occupied]

    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
    master.java_main_class = 'MainOnline'
    print(f"### Building state A: ONE joint solve over jobs 0..{args.n_existing - 1}, restricted to "
          f"the first {nb_occupied}/{args.nb_nodes} nodes ({args.occupied_nodes_fraction * 100:.0f}% "
          f"of the infra) (solver_time_limit={args.state_a_time_limit}s, node free times=0, no "
          f"pre-existing replicas, solved at env.now=0 so each job's real timeline comes out as if "
          f"it had actually executed) ###", flush=True)
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, existing_jobs, {}, nodes_free_time, 0)
    master.compute_nodes = compute_nodes  # restore full node list for the final new-job solve

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

    # A job is "finished by T" iff its own committed schedule has it fully done (last task's end)
    # at or before the freeze point -- computed from state A's OWN decided timeline, not from any
    # live status flag (nothing here ever actually executes via SimPy's event loop).
    not_finished_ids = {j.job_id for j in existing_jobs
                         if state_a_finish.get(j.job_id) is None or state_a_finish[j.job_id] > freeze_at}
    not_finished_jobs = [j for j in existing_jobs if j.job_id in not_finished_ids]
    print(f"### At freeze point T={freeze_at:.2f}: {args.n_existing - len(not_finished_ids)} of "
          f"{args.n_existing} existing jobs already finished per state A's own schedule; "
          f"{len(not_finished_ids)} still running (job_ids={sorted(not_finished_ids)}) ###",
          flush=True)

    # Reflect state A's own per-TASK schedule onto each not-yet-finished job's actual Task
    # objects: anything already committed by T becomes "Finished" (task fully done before T) or
    # "Started" (in progress right at T), leaving only genuinely not-yet-started tasks as
    # "NotStarted". schedulingUsingJavaCSP (utils/modelCSP.py) only ever batches a job's
    # "NotStarted" tasks into the CSP, so this is what makes an online replan reconsider just the
    # REMAINING work -- without it, every task.status here is still the Task class's default
    # ("NotStarted", since these Job/Task objects never run through SimPy's live event loop in
    # this offline-style construction), so a replan would silently re-decide the job's ENTIRE
    # placement from scratch at T, discarding whatever state A had already committed before T.
    job_by_id = {j.job_id: j for j in existing_jobs}
    for key, entries in (works_ or {}).items():
        for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
            if job_id not in not_finished_ids:
                continue
            task_obj = job_by_id[job_id].tasks[task_index]
            if end_abs <= freeze_at:
                task_obj.status = "Finished"
            elif start_abs <= freeze_at < end_abs:
                task_obj.status = "Started"
            # else: stays "NotStarted" (default) -- genuinely still reschedulable

    # Reflect state A's own decisions for the not-yet-finished jobs as REAL node occupancy at the
    # freeze point, so nodesFreeTime()/nodesFreeTimeIncremental() correctly see which nodes are
    # busy (and until when) instead of treating every node as free -- exactly what a live replay
    # would have given for free via ongoing_transfers/ongoing_works/transfers/works.
    for key, entries in (works_ or {}).items():
        for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
            if job_id not in not_finished_ids or end_abs <= freeze_at:
                continue
            if start_abs <= freeze_at < end_abs:
                master.ongoing_works[f'node_{node_index}'] = (job_id, node_index, task_index, start_abs, end_abs, duration)
            else:
                master.works[f'node_{node_index}'].append((job_id, node_index, task_index, start_abs, end_abs, duration))
    for key, entries in (transfers_ or {}).items():
        for job_id, node_index, start_abs, end_abs, duration in entries:
            if job_id not in not_finished_ids or end_abs <= freeze_at:
                continue
            if start_abs <= freeze_at < end_abs:
                master.ongoing_transfers[f'node_{node_index}'] = (job_id, node_index, start_abs, end_abs, duration)
            else:
                master.transfers[f'node_{node_index}'].append((job_id, node_index, start_abs, end_abs, duration))

    env.run(until=freeze_at)
    print(f"### State A built. env.now={env.now:.2f}, jobs placed={len(state_a_finish)}/{args.n_existing} ###",
          flush=True)

    new_job = Job(new_job_raw["job_id"], new_job_raw["task_duration"], new_job_raw["nb_tasks"], new_job_raw["dataset_size"])
    new_job.arriving_time = env.now
    master.tracker.register_job(new_job.job_id, env.now)

    # Only the new job plus existing jobs still running at T could actually differ between
    # approaches -- an already-finished job's outcome is fixed no matter what, so including it
    # would only dilute the comparison.
    isolated_ids = not_finished_ids | {new_job.job_id}
    print(f"### Isolated comparison scope: {len(isolated_ids)} jobs (new job + still-running "
          f"existing jobs) ###", flush=True)
    return master, new_job, isolated_ids, not_finished_jobs


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


def run_online_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs):
    """Mirrors SchedulingUsingCSPOnline.schedulingNewJob(): jointly replan the new job + every
    existing job that still has unstarted tasks. Which existing jobs count as "still running" is
    passed in directly (not_finished_jobs, from state A's own committed schedule vs the freeze
    point) rather than via master.getRunningJobs() -- that method reads live job.status/task.status
    flags that are never updated here (nothing actually executes through SimPy's event loop in
    this offline-style construction), so it would incorrectly treat every existing job as still
    running regardless of whether state A's own schedule already finished it."""
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


def run_incremental_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs=None):
    """Mirrors SchedulingUsingCSPIncremental.schedulingNewJob(): place the new job alone; every
    existing job keeps whatever it was already committed to in state A, untouched. not_finished_jobs
    is accepted (and ignored) purely so this shares a call signature with run_online_style."""
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


def build_gantt_events(master, result, freeze_at):
    """Full final schedule as plot_gantt_chart's flat event list: state A's own placement for
    every job it decided, EXCEPT the portion of a rescheduled job (jobs_to_reschedule) that was
    still genuinely open at T (start_abs > freeze_at, i.e. not yet started -- the only part the
    CSP was actually free to re-decide) -- that portion comes from this solve's own result
    instead. Anything already committed by T (finished, or in progress right at T) keeps its
    state-A placement even for a rescheduled job, since that work physically already happened
    and this solve never touched it (see the task.status pre-marking in build_state_a). For
    Incremental, jobs_to_reschedule is just the new job, so every existing job's state-A
    placement is untouched here regardless of freeze_at."""
    if result is None:
        return []
    reschedule_ids = set(result["jobs_to_reschedule"])
    events = []
    for job_id, node_index, task_index, start_abs, end_abs, duration in (
        e for entries in master._state_a_works.values() for e in entries
    ):
        if job_id in reschedule_ids and start_abs > freeze_at:
            continue  # this task was still open at T -- superseded by result's own placement
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

    master, new_job, isolated_ids, not_finished_jobs = build_state_a(config, args, results_dir)
    now = master.env.now
    replicas_locations = master.replicas_locations

    master._config["solver_time_limit_s"] = args.solver_time_limit
    print(f"\n### Solving new job's (id={new_job.job_id}) placement -- {args.approach.upper()} style, "
          f"{args.solver_time_limit}s budget ###", flush=True)
    solve_fn = run_online_style if args.approach == "online" else run_incremental_style
    result = solve_fn(master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations, not_finished_jobs)
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

    gantt_events = build_gantt_events(master, result, new_job.arriving_time)
    if gantt_events:
        gantt_path = os.path.join(results_dir, "gantt.png")
        plot_gantt_chart(gantt_events, args.nb_nodes,
                          title=f"{args.approach} -- state A ({args.n_existing}j) + new job {new_job.job_id}",
                          save_path=gantt_path, freeze_at=new_job.arriving_time)
        print(f"### Gantt chart written to {gantt_path} ###", flush=True)
    else:
        print("### No gantt chart: no schedule to plot (solve returned no solution) ###", flush=True)


if __name__ == "__main__":
    main()
