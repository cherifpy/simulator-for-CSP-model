"""
Standalone, portable Grid5000 launcher for the controlled single-decision-point test, extended to
THREE approaches: build state A as a single, direct JOINT CSP solve over the --n-existing jobs
(their real dataset_size/nb_tasks/task_duration/arriving_time from the instance), with node free
times at 0 and no pre-existing replicas -- a genuine one-shot "offline" placement decision for
those N jobs (NOT a replay of N sequential live-simulation arrivals). Then inject exactly ONE new
job and solve ITS placement with incremental / online_biobj / hybrid (or all three, from the SAME
frozen state A) and report, for each: the new job's wait/flow time, the mean/max flow time over
ALL jobs (state A + new job), the transfer energy, AND the real wall-clock SCHEDULING TIME the
solve itself took (absent from the original online/incremental-only xp_single_decision_grid5000.py
and its results_analysis.ipynb).

- incremental: places the new job alone (MainIncremental), every existing job untouched.
- online_biobj: jointly replans the new job + every existing job with unstarted tasks
  (MainOnlineMultiObj, epsilon-constraint bi-objective: phase 1 minimizes max flow time, phase 2
  minimizes transfer energy within --epsilon-fraction of phase 1's result).
- hybrid: mirrors SchedulingUsingCSPAdaptiveJoint.schedulingNewJob's own escalation exactly (same
  code, not a reimplementation) -- probes Incremental for F1 (bounded by
  --hybrid-incremental-time-limit), then runs the REAL 4-way parallel escalation
  (_timedParallelEscalation: freeze_below_median / freeze_above_median / nofreeze / warm_nofreeze)
  budgeted at min(--hybrid-alpha * F1, --hybrid-max-budget) seconds. Reached by shallow-copying
  the state-A master and reassigning its class to SchedulingUsingCSPAdaptiveJoint (same technique
  _timedParallelEscalation itself already uses for its own per-variant proxies) -- state A itself
  is always built the same Online/MainOnline way regardless of which approach(es) later solve the
  new job, so this only ever affects the new job's own solve.

Every path is resolved relative to this file's own location, so the whole `simulator/` folder can
be copied to Grid5000 (or anywhere) and this script keeps working unmodified.

Example (Grid5000, 50 nodes, N=15 existing jobs, job index 15 is the new one, all 3 approaches
from the same frozen state A, budgets matching the live workload experiment):
    python3 xp_single_decision_3way_grid5000.py --approach all \\
        --instance-dir /path/to/inst-20J-50N --nb-nodes 50 --n-existing 15 \\
        --incremental-time-limit 180 --online-biobj-time-limit 180 \\
        --hybrid-incremental-time-limit 30 --hybrid-alpha 0.2 --hybrid-max-budget 180 \\
        --results-dir /path/to/results/single_decision_3way_15j
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import copy
import json
import logging
import random
import sys
import time
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
if SIMULATOR_DIR not in sys.path:
    sys.path.append(SIMULATOR_DIR)

import simpy
from classes.tracker import Tracker
from classes.job import Job
from compute_node import ComputeNode
from master_node_with_heterogeneous_nodes_csp import (
    SchedulingUsingCSPOnline, SchedulingUsingCSPIncremental, SchedulingUsingCSPAdaptiveJoint,
)
from utils.modelCSP import schedulingUsingJavaCSP
from utils.run_export import start_recording
from utils.plots import plot_gantt_chart
from simulator import generateHeterogeneousInfrastructureEquilibre, configure_logging

logger = logging.getLogger(__name__)

APPROACHES = ("incremental", "online_biobj", "hybrid")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", required=True, choices=sorted(APPROACHES) + ["all"],
                         help="How to solve the single new job's placement. 'all' builds state A "
                              "ONCE (per --state-a-time-limit) and solves the new job's placement "
                              "with all three approaches from that SAME frozen state, so the "
                              "comparison isn't muddied by independently-rebuilt state A's.")
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing this instance's jobs.json and infrastructure.csv.")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes in the instance.")
    parser.add_argument("--n-existing", type=int, default=15,
                         help="Number of jobs (from the start of jobs.json) used to build state A, "
                              "i.e. already 'in progress' before the new job arrives (default: 15).")
    parser.add_argument("--new-job-index", type=int, default=None,
                         help="Index into jobs.json of the single new job injected after state A is "
                              "frozen (default: --n-existing, i.e. the job right after state A).")
    parser.add_argument("--incremental-time-limit", type=int, default=None,
                         help="Incremental's own solver time budget in seconds. Required if "
                              "--approach is incremental or all.")
    parser.add_argument("--online-biobj-time-limit", type=int, default=None,
                         help="online_biobj's fixed solver time budget in seconds (split into "
                              "phase1/phase2 by --epsilon-phase1-fraction, like every replan in "
                              "the live simulator). Required if --approach is online_biobj or all.")
    parser.add_argument("--hybrid-incremental-time-limit", type=int, default=None,
                         help="Hybrid's F1 probe budget in seconds (bounds the cheap Incremental "
                              "solve used only to ESTIMATE the new job's own flow time before "
                              "deciding the escalation budget). Required if --approach is hybrid "
                              "or all.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.2,
                         help="Hybrid's escalation budget as a fraction of F1: "
                              "budget=min(alpha*F1, --hybrid-max-budget) (default: 0.2).")
    parser.add_argument("--hybrid-max-budget", type=int, default=None,
                         help="Hybrid's escalation budget cap in seconds. Required if --approach "
                              "is hybrid or all.")
    parser.add_argument("--epsilon-fraction", type=float, default=0.05,
                         help="online_biobj/hybrid: phase 2 (minimize energy) may not worsen "
                              "phase 1's own max flow time by more than this fraction (default: 0.05).")
    parser.add_argument("--epsilon-phase1-fraction", type=float, default=0.75,
                         help="online_biobj/hybrid: fraction of the solver time budget given to "
                              "phase 1 (minimize max flow time); the rest goes to phase 2 "
                              "(minimize energy) (default: 0.75).")
    parser.add_argument("--state-a-time-limit", type=int, default=30,
                         help="CSP solver time budget for the ONE joint solve that builds state A "
                              "over all --n-existing jobs at once (default: 30s).")
    parser.add_argument("--occupied-nodes-fraction", type=float, default=1.0,
                         help="Fraction of --nb-nodes eligible for state A's own solve (default: "
                              "1.0, i.e. all of them). The new job's own solve still sees every "
                              "node -- only state A's construction is restricted to the first "
                              "round(nb_nodes * occupied_nodes_fraction) nodes.")
    parser.add_argument("--lambda-rate", type=int, default=100,
                         help="Present for config compatibility; unused now that state A is a single "
                              "direct joint solve rather than a live-simulation replay with injected "
                              "arrivals (default: 100).")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write results.json. Default: "
                              "<simulator>/results-grid5000/single_decision_3way_<approach>_"
                              "<n_existing>j-<nb_nodes>n_<date>.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    need_incremental = args.approach in ("incremental", "all")
    need_online_biobj = args.approach in ("online_biobj", "all")
    need_hybrid = args.approach in ("hybrid", "all")
    if need_incremental and args.incremental_time_limit is None:
        parser.error("--incremental-time-limit is required for --approach incremental/all")
    if need_online_biobj and args.online_biobj_time_limit is None:
        parser.error("--online-biobj-time-limit is required for --approach online_biobj/all")
    if need_hybrid and (args.hybrid_incremental_time_limit is None or args.hybrid_max_budget is None):
        parser.error("--hybrid-incremental-time-limit and --hybrid-max-budget are required for "
                      "--approach hybrid/all")
    return args


def build_state_a(config, args, results_dir):
    """Builds state A as ONE direct joint CSP solve over all --n-existing jobs at once (their real
    dataset_size/nb_tasks/task_duration/arriving_time), with node free times at 0 and no
    pre-existing replicas -- a genuine one-shot "offline" placement, not a replay of N sequential
    live-simulation arrivals. Always built via plain Online/MainOnline, independently of which
    approach(es) later solve the new job -- see the module docstring.

    Critically, this solve runs with env.now=0 (each job's OWN real arriving_time as its actual
    lower bound, absolute times coming out exactly as if state A had really executed from t=0) --
    NOT with env.now already at the freeze point. schedulingUsingJavaCSP reads master.env.now
    directly everywhere to turn Java's local times into absolute ones, so solving at env.now=T
    would wrongly anchor every job's schedule to start at-or-after T, making nothing ever appear
    "already finished" by the time the new job arrives. Only AFTER solving is env.now advanced to
    T (via env.run(until=T) with no processes registered, so SimPy just advances the clock)."""
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

    nb_occupied = max(1, min(args.nb_nodes, round(args.nb_nodes * args.occupied_nodes_fraction)))
    master.compute_nodes = compute_nodes[:nb_occupied]

    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
    master.java_main_class = 'MainOnline'
    print(f"### Building state A: ONE joint solve over jobs 0..{args.n_existing - 1}, restricted to "
          f"the first {nb_occupied}/{args.nb_nodes} nodes ({args.occupied_nodes_fraction * 100:.0f}% "
          f"of the infra) (solver_time_limit={args.state_a_time_limit}s, node free times=0, no "
          f"pre-existing replicas, solved at env.now=0) ###", flush=True)
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, existing_jobs, {}, nodes_free_time, 0)
    master.compute_nodes = compute_nodes  # restore full node list for the final new-job solve(s)

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

    not_finished_ids = {j.job_id for j in existing_jobs
                         if state_a_finish.get(j.job_id) is None or state_a_finish[j.job_id] > freeze_at}
    not_finished_jobs = [j for j in existing_jobs if j.job_id in not_finished_ids]
    print(f"### At freeze point T={freeze_at:.2f}: {args.n_existing - len(not_finished_ids)} of "
          f"{args.n_existing} existing jobs already finished per state A's own schedule; "
          f"{len(not_finished_ids)} still running (job_ids={sorted(not_finished_ids)}) ###",
          flush=True)

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

    isolated_ids = not_finished_ids | {new_job.job_id}
    print(f"### Isolated comparison scope: {len(isolated_ids)} jobs (new job + still-running "
          f"existing jobs) ###", flush=True)
    return master, new_job, isolated_ids, not_finished_jobs


def committed_finish_time(master, nb_nodes, job_id):
    """Already-decided completion time for an existing job that a given solve does NOT re-plan --
    read directly from state A's own solve result (there's no live execution history to fall back
    on in this offline-style construction)."""
    return master._state_a_finish.get(job_id)


def compute_transfer_energy(transfers_, master):
    """Same formula as Tracker.log_transfer (energy = sender + receiver * transfer_time +
    network), applied here to a solve's PROPOSED transfers -- this test bypasses the live
    simulation loop entirely, so those transfers never flow through the tracker's own accounting
    and would otherwise be invisible to any energy comparison."""
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


def _finalize_result(master, new_job, isolated_ids, nb_nodes, jobs_to_reschedule,
                      transfers_, works_, deletions_, scheduling_time_s):
    """Shared by all three approaches: turns a solve's raw (transfers_, works_, deletions_) into
    the same metrics dict (wait/flow time, mean/max over all jobs and the isolated subset, energy)
    plus scheduling_time_s -- the real wall-clock the solve itself took, the one thing missing
    from the original online/incremental-only version of this script."""
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
        "scheduling_time_s": scheduling_time_s,
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


def run_incremental_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations,
                           not_finished_jobs, time_limit):
    """Mirrors SchedulingUsingCSPIncremental.schedulingNewJob(): place the new job alone; every
    existing job keeps whatever it was already committed to in state A, untouched."""
    jobs_to_reschedule = [new_job]
    nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(
        master, master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainIncremental'
    master._config['solver_time_limit_s'] = time_limit
    t0 = time.time()
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    scheduling_time_s = time.time() - t0

    if not transfers_ or not works_:
        return None
    return _finalize_result(master, new_job, isolated_ids, nb_nodes, jobs_to_reschedule,
                             transfers_, works_, deletions_, scheduling_time_s)


def run_online_biobj_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations,
                            not_finished_jobs, time_limit, epsilon_fraction, epsilon_phase1_fraction):
    """Mirrors SchedulingUsingCSPOnlineMultiObj.schedulingNewJob() (inherited, unmodified, from
    SchedulingUsingCSPOnline): jointly replan the new job + every existing job that still has
    unstarted tasks, via the epsilon-constraint bi-objective solver (MainOnlineMultiObj) -- phase 1
    minimizes max flow time, phase 2 minimizes transfer energy within epsilon_fraction of phase
    1's own result. A FIXED time_limit is used for every call, exactly like the live simulator's
    online_biobj (no F1/adaptive budget -- that's hybrid-only)."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnlineMultiObj'
    master._config['solver_time_limit_s'] = time_limit
    master._config['epsilon_fraction'] = epsilon_fraction
    master._config['epsilon_phase1_fraction'] = epsilon_phase1_fraction
    t0 = time.time()
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, jobs_to_reschedule, replicas_locations, nodes_free_time, now)
    scheduling_time_s = time.time() - t0

    if not transfers_ or not works_:
        return None
    return _finalize_result(master, new_job, isolated_ids, nb_nodes, jobs_to_reschedule,
                             transfers_, works_, deletions_, scheduling_time_s)


def _drive_to_completion(gen, what):
    """Drives a charge_thinking_time-aware generator (_placeSingleJobIncremental /
    _timedParallelEscalation) to its return value WITHOUT a live SimPy env.run() loop. Safe only
    because the caller sets config['charge_thinking_time']=False first, so the generator body
    never actually reaches a `yield self.env.timeout(...)` -- it runs straight through to
    `return`, which next() surfaces as StopIteration.value exactly like `yield from` would."""
    try:
        next(gen)
    except StopIteration as e:
        return e.value
    raise RuntimeError(f"{what}: generator unexpectedly yielded -- charge_thinking_time should be "
                        f"False in this offline single-decision context")


def run_hybrid_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations,
                      not_finished_jobs, incremental_time_limit, alpha, max_budget,
                      epsilon_fraction, epsilon_phase1_fraction):
    """Mirrors SchedulingUsingCSPAdaptiveJoint.schedulingNewJob()'s escalation branch EXACTLY --
    same code, not a reimplementation: probe Incremental for F1 (_placeSingleJobIncremental,
    bounded by incremental_time_limit), then run the real 4-way parallel escalation
    (_timedParallelEscalation) budgeted at min(alpha*F1, max_budget) seconds. Reached by
    shallow-copying master and reassigning its class to SchedulingUsingCSPAdaptiveJoint: state A
    itself was always built the plain Online/MainOnline way (see build_state_a/module docstring),
    so this only ever affects how the new job's own placement gets solved. charge_thinking_time is
    turned off (this offline script tracks its own wall-clock scheduling_time_s instead, and there
    is no live SimPy env.run() loop here for env.timeout() to advance anyway)."""
    hybrid_master = copy.copy(master)
    hybrid_master.__class__ = SchedulingUsingCSPAdaptiveJoint
    hybrid_master._config = dict(master._config)
    hybrid_master._config['charge_thinking_time'] = False
    hybrid_master._config['epsilon_fraction'] = epsilon_fraction
    hybrid_master._config['epsilon_phase1_fraction'] = epsilon_phase1_fraction
    hybrid_master._config['parallel_warm_cold_escalation'] = True
    hybrid_master._config['incremental_time_limit_s'] = incremental_time_limit
    hybrid_master._config['adaptive_alpha'] = alpha
    hybrid_master._config['adaptive_max_budget_s'] = max_budget

    t0 = time.time()
    inc_transfers, inc_works, inc_deletions, f1 = _drive_to_completion(
        hybrid_master._placeSingleJobIncremental(new_job), "F1 probe")
    t_f1 = time.time() - t0
    print(f"### Hybrid: F1 probe took {t_f1:.2f}s, F1={f1} ###", flush=True)

    if f1 is None:
        return None

    budget = min(alpha * f1, max_budget)
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = hybrid_master.nodesFreeTime(hybrid_master.ongoing_transfers, hybrid_master.ongoing_works)

    print(f"### Hybrid: escalation budget=min({alpha}*{f1:.2f}, {max_budget})={budget:.2f}s, "
          f"4-way parallel escalation starting ###", flush=True)
    t1 = time.time()
    transfers_, works_, deletions_ = _drive_to_completion(
        hybrid_master._timedParallelEscalation(jobs_to_reschedule, not_finished_jobs, replicas_locations,
                                                nodes_free_time, now, budget),
        "escalation")
    t_escalation = time.time() - t1
    scheduling_time_s = t_f1 + t_escalation

    if not transfers_ or not works_:
        # Mirrors the live schedulingNewJob's own fallback: escalation found nothing usable, fall
        # back to the F1 probe's own (single-job) placement.
        print("### Hybrid: escalation found no solution, falling back to F1 probe's placement ###", flush=True)
        if not inc_transfers or not inc_works:
            return None
        result = _finalize_result(hybrid_master, new_job, isolated_ids, nb_nodes, [new_job],
                                   inc_transfers, inc_works, inc_deletions, scheduling_time_s)
        if result is not None:
            result.update({"hybrid_f1": f1, "hybrid_budget_s": budget, "hybrid_t_f1_s": t_f1,
                           "hybrid_t_escalation_s": t_escalation, "hybrid_fallback_to_incremental": True})
        return result

    result = _finalize_result(hybrid_master, new_job, isolated_ids, nb_nodes, jobs_to_reschedule,
                               transfers_, works_, deletions_, scheduling_time_s)
    if result is not None:
        result.update({"hybrid_f1": f1, "hybrid_budget_s": budget, "hybrid_t_f1_s": t_f1,
                       "hybrid_t_escalation_s": t_escalation, "hybrid_fallback_to_incremental": False})
    return result


def print_jobs_replicas_and_schedule(master, new_job, isolated_ids, result):
    """Prints full per-job, per-replica and per-task/per-transfer detail. `master` here is always
    the ORIGINAL state-A master (not a hybrid_master copy) -- job list/replicas_locations printed
    are identical either way (hybrid_master shares them by reference), only jobs_to_reschedule's
    schedule detail comes from `result`."""
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
    """Full final schedule as plot_gantt_chart's flat event list -- see the original script's own
    docstring for this function; unchanged here."""
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
               ("Transfer energy (this solve)", "transfer_energy_total"),
               ("Scheduling time (s)", "scheduling_time_s")]

SOLVERS = {
    "incremental": lambda master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs, p:
        run_incremental_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations,
                               not_finished_jobs, p["incremental_time_limit"]),
    "online_biobj": lambda master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs, p:
        run_online_biobj_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                                p["online_biobj_time_limit"], p["epsilon_fraction"], p["epsilon_phase1_fraction"]),
    "hybrid": lambda master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs, p:
        run_hybrid_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                          p["hybrid_incremental_time_limit"], p["hybrid_alpha"], p["hybrid_max_budget"],
                          p["epsilon_fraction"], p["epsilon_phase1_fraction"]),
}


def solve_and_save(approach, out_dir, args, master, new_job, isolated_ids, not_finished_jobs, now,
                    replicas_locations, params):
    """Runs ONE approach's placement solve against the given (already-built) state A, prints its
    schedule/metrics, and saves results.json + gantt.png under out_dir. Safe to call more than
    once against the SAME master for a --approach all run: none of the three solve functions
    mutate master's own accumulated state (works/transfers/ongoing_*/replicas_locations) or the
    Job/Task objects' status beyond what build_state_a already fixed once -- hybrid works off its
    own shallow copy for the same reason."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n### Solving new job's (id={new_job.job_id}) placement -- {approach.upper()} style ###", flush=True)
    result = SOLVERS[approach](master, new_job, isolated_ids, args.nb_nodes, now, replicas_locations,
                                not_finished_jobs, params)
    result_summary = {k: v for k, v in result.items()
                       if k not in ("transfers", "works", "deletions", "replicas_locations", "transfer_energy_detail")} if result else None
    print(f"{approach} result:", result_summary, flush=True)

    print_jobs_replicas_and_schedule(master, new_job, isolated_ids, result)

    print("\n" + "=" * 70)
    label = approach
    print(f"{'Metric':<30}{label:>25}")
    print("=" * 70)
    for row_label, key in METRIC_ROWS:
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

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "approach": approach,
            "params": params,
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
        gantt_path = os.path.join(out_dir, "gantt.png")
        plot_gantt_chart(gantt_events, args.nb_nodes,
                          title=f"{approach} -- state A ({args.n_existing}j) + new job {new_job.job_id}",
                          save_path=gantt_path, freeze_at=new_job.arriving_time)
        print(f"### Gantt chart written to {gantt_path} ###", flush=True)
    else:
        print("### No gantt chart: no schedule to plot (solve returned no solution) ###", flush=True)
    return result


def main():
    args = parse_args()
    configure_logging(logging.WARNING)

    params = {
        "incremental_time_limit": args.incremental_time_limit,
        "online_biobj_time_limit": args.online_biobj_time_limit,
        "hybrid_incremental_time_limit": args.hybrid_incremental_time_limit,
        "hybrid_alpha": args.hybrid_alpha,
        "hybrid_max_budget": args.hybrid_max_budget,
        "epsilon_fraction": args.epsilon_fraction,
        "epsilon_phase1_fraction": args.epsilon_phase1_fraction,
    }
    approach_plan = list(APPROACHES) if args.approach == "all" else [args.approach]

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000",
        f"single_decision_3way_{args.approach}_{args.n_existing}j-{args.nb_nodes}n_"
        f"{date.today().isoformat()}",
    )
    os.makedirs(results_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    start_recording(results_dir, params=args, config=config)

    master, new_job, isolated_ids, not_finished_jobs = build_state_a(config, args, results_dir)
    now = master.env.now
    replicas_locations = master.replicas_locations

    results_by_approach = {}
    for approach in approach_plan:
        out_dir = results_dir if len(approach_plan) == 1 else os.path.join(results_dir, approach)
        results_by_approach[approach] = solve_and_save(
            approach, out_dir, args, master, new_job, isolated_ids, not_finished_jobs, now,
            replicas_locations, params)

    if len(approach_plan) > 1:
        print("\n" + "=" * 70)
        print("### COMPARISON -- same state A, different approach for the new job ###")
        col_labels = list(results_by_approach.keys())
        print(f"{'Metric':<30}" + "".join(f"{lbl:>25}" for lbl in col_labels))
        print("=" * 70)
        for row_label, key in METRIC_ROWS:
            cells = []
            for result in results_by_approach.values():
                v = result.get(key) if result else None
                cells.append(('%.2f' % v) if v is not None else 'N/A')
            print(f"{row_label:<30}" + "".join(f"{c:>25}" for c in cells))
        print("=" * 70)


if __name__ == "__main__":
    main()
