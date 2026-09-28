"""
Controlled single-decision-point test, swept over the NEW job's dataset size (small / medium /
large), to show how the cost of Online's "reconsideration" -- jointly re-planning every existing
job that still has unstarted tasks, instead of just placing the new job alone like Incremental
does -- scales with the size of the data being introduced.

For each size tier: build state A as ONE direct joint CSP solve over --n-existing jobs -- node
free times at 0, no pre-existing replicas, a genuine one-shot "offline" placement, not a replay
of N sequential live-simulation arrivals -- ONCE per tier, with EVERY one of those jobs' own
dataset_size ALSO redrawn from that tier's size_range (not just the new job's -- the whole
infrastructure's data volumes shift with the tier, existing jobs included), then solve the SAME
new job's placement (with a freshly-drawn dataset_size from that same range) BOTH Online-style
and Incremental-style from that one state. Reusing one state-A build across all repeats and both
approaches is safe: schedulingUsingJavaCSP never mutates the master's persistent state
(works/ongoing_works/replicas_locations) -- it only reads it and returns a proposed solution --
so calling it many times in a row for the same snapshot doesn't let one solve contaminate
another's starting point.

Reports, per tier and per approach: the new job's wait/flow time, mean/max flow time (all jobs
and the ISOLATED subset that can actually differ between approaches), plus a derived
"reconsideration cost" = Online's isolated mean minus Incremental's, and as a percentage.

Every path is resolved relative to this file's own location, so the whole `simulator/` folder
can be copied to Grid5000 (or anywhere) and this script keeps working unmodified.

Example:
    python3 xp_dataset_size_sweep.py --instance-dir /path/to/inst-20J-50N --nb-nodes 50 \
        --n-existing 15 --solver-time-limit 30 --lambda-rate 100 \
        --small-dataset-size 1024 --medium-dataset-size 5120 --large-dataset-size 20480
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import copy
import csv
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
from master_node_with_heterogeneous_nodes_csp import (
    SchedulingUsingCSPOnline, SchedulingUsingCSPIncremental, SchedulingUsingCSPAdaptiveJoint,
)
from utils.modelCSP import schedulingUsingJavaCSP
from utils.run_export import (
    TASKS_FIELDS, TRANSFERS_FIELDS, DELETIONS_FIELDS, TRAJECTORY_FIELDS,
    run_captured, snapshot_solver_io, write_rows, plan_rows, parse_solver_log, start_recording,
    solver_model_dir,
)
from simulator import generateHeterogeneousInfrastructureEquilibre, configure_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-dir", required=True,
                         help="Directory containing this instance's jobs.json and infrastructure.csv.")
    parser.add_argument("--nb-nodes", required=True, type=int, help="Number of compute nodes in the instance.")
    parser.add_argument("--n-existing", type=int, default=15,
                         help="Number of jobs (from the start of jobs.json) used to build state A, "
                              "i.e. already 'in progress' before the new job arrives (default: 15).")
    parser.add_argument("--new-job-index", type=int, default=None,
                         help="Index into jobs.json of the single new job injected after state A is "
                              "frozen (default: --n-existing). Its dataset_size is overridden by "
                              "each tier below; its nb_tasks/task_duration/arriving_time are kept.")
    # dataset_size is in MB (see config.json's min/max_dataset_size_MB) -- defaults below are
    # 1-10GB / 15-40GB / 50-100GB. NOTE: on inst-20J-50N, node storage capacities top out at
    # 81920 MB (80GB) -- any large-tier draw above that (the top ~40% of 50-100GB) has NO node
    # that can hold it at all, so that solve will come back infeasible (result=None). That's a
    # real, meaningful finding (data too big for the infrastructure), not a bug, but it does mean
    # a chunk of the large tier's repeats will show N/A rather than a comparable number -- narrow
    # --large-range to e.g. 51200 81920 first if you want every large-tier draw to be feasible.
    parser.add_argument("--small-range", type=int, nargs=2, metavar=("LOW", "HIGH"), default=(1024, 10240),
                         help="Small tier dataset_size range in MB, inclusive (default: 1024 10240, i.e. 1-10GB).")
    parser.add_argument("--medium-range", type=int, nargs=2, metavar=("LOW", "HIGH"), default=(15360, 40960),
                         help="Medium tier dataset_size range in MB, inclusive (default: 15360 40960, i.e. 15-40GB).")
    parser.add_argument("--large-range", type=int, nargs=2, metavar=("LOW", "HIGH"), default=(51200, 102400),
                         help="Large tier dataset_size range in MB, inclusive (default: 51200 102400, i.e. "
                              "50-100GB -- see the infeasibility note above).")
    parser.add_argument("--full-range", type=int, nargs=2, metavar=("LOW", "HIGH"), default=(10240, 102400),
                         help="Full-range tier dataset_size range in MB, inclusive (default: 10240 102400, "
                              "i.e. 10-100GB) -- one single draw range spanning small through large, for a "
                              "run that doesn't split by tier.")
    parser.add_argument("--tiers", nargs="+", choices=["small", "medium", "large", "full"],
                         default=["small", "medium", "large"],
                         help="Which tier(s) to actually run (default: all three). E.g. "
                              "--tiers medium to test only the medium size range, skipping the "
                              "small/large solves entirely.")
    parser.add_argument("--repeats", type=int, default=5,
                         help="Number of trials per tier, each drawing a fresh dataset_size uniformly "
                              "from that tier's range -- for heterogeneity within a tier instead of a "
                              "single fixed value (default: 5). State A is built ONCE per tier and "
                              "reused across its repeats (safe: solving never mutates it), so this "
                              "only adds solves, not state-A rebuilds.")
    parser.add_argument("--solver-time-limit", type=int, default=30,
                         help="CSP solver time budget (seconds) for Online's placement solve, per "
                              "tier (default: 30). Also Incremental's, unless --incremental-time-limit "
                              "is given separately.")
    parser.add_argument("--incremental-time-limit", type=int, default=None,
                         help="CSP solver time budget (seconds) for Incremental's placement solve, "
                              "per tier. Defaults to --solver-time-limit if not given -- set this "
                              "separately when Online needs a much larger budget than Incremental "
                              "(e.g. --solver-time-limit 7200 --incremental-time-limit 60).")
    parser.add_argument("--approaches", nargs="+",
                         choices=["online", "online_warmstart", "online_biobj_warmstart", "incremental", "epsilon", "hybrid"],
                         default=["online", "incremental"],
                         help="Which approach(es) to actually run per tier (default: online + "
                              "incremental). Use --approaches epsilon alone to run ONLY the "
                              "epsilon-constraint approach against a fresh state A built with "
                              "the SAME --seed/--n-existing/tiers as a prior online+incremental "
                              "run -- the per-tier job draws are then identical (deterministic "
                              "given the same seed), making the two runs' numbers directly "
                              "comparable without re-solving online/incremental. The 4 approaches "
                              "meant for regular comparison are: incremental (new job alone), "
                              "online_warmstart (mono-objective joint reconsideration, warm-started), "
                              "online_biobj_warmstart (same but bi-objective: max flow time then "
                              "energy), and hybrid (Incremental-gated escalation to "
                              "online_biobj_warmstart with pre-processing). online/epsilon are the "
                              "older non-warm-started variants, kept for reference.")
    parser.add_argument("--epsilon-time-limit", type=int, default=None,
                         help="epsilon-constraint approach only: TOTAL solver budget (seconds, both "
                              "phases together, split by --epsilon-phase1-fraction). Defaults to "
                              "--solver-time-limit -- set separately to reproduce e.g. Online at 2h "
                              "but epsilon at 4h (2h per phase).")
    parser.add_argument("--epsilon-fraction", type=float, default=0.1,
                         help="epsilon-constraint approach only: how much worse than the optimal "
                              "max flow time (phase 1) the final solution is allowed to be, as a "
                              "fraction (default: 0.1, i.e. 10%%) -- see MainOnlineMultiObj.java.")
    parser.add_argument("--epsilon-phase1-fraction", type=float, default=0.75,
                         help="epsilon-constraint approach only: fraction of --solver-time-limit "
                              "given to phase 1 (minimize max flow time); the rest goes to phase "
                              "2 (minimize energy under the flow-time cap). Default: 0.75.")
    parser.add_argument("--epsilon-max-cap", type=float, default=None,
                         help="epsilon-constraint approach only: an absolute ceiling on phase 2's "
                              "max-flow-time cap, e.g. a baseline approach's own max_flow_time_all "
                              "from a prior run of this same tier/scenario -- prevents "
                              "--epsilon-fraction's RELATIVE slack from drifting past what that "
                              "baseline already achieves for free when phase 1 itself under-"
                              "converges (seen on the large tier with a 1h/1h split: the relative "
                              "cap ended up +18.5%% over Online's own 2h result). Default: no ceiling.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.2,
                         help="hybrid approach only: escalation budget = min(hybrid_alpha * F1, "
                              "--hybrid-max-budget) seconds, where F1 is Incremental's own flow "
                              "time for the new job alone. Default: 0.2 (20%% of F1) -- matches the "
                              "live simulator's own default.")
    parser.add_argument("--hybrid-max-budget", type=float, default=1200,
                         help="hybrid approach only: hard ceiling (seconds) on the escalation budget "
                              "(hybrid_alpha * F1). Default: 1200 = 20min.")
    parser.add_argument("--hybrid-incremental-time-limit", type=float, default=15,
                         help="hybrid approach only: solver budget (seconds) for its own Incremental "
                              "F1 probe (_placeSingleJobIncremental). Independent of "
                              "--incremental-time-limit (which governs the standalone incremental "
                              "approach). Default: 15.")
    parser.add_argument("--state-a-time-limit", type=int, default=30,
                         help="CSP solver time budget for the ONE joint solve that builds state A over "
                              "all --n-existing jobs at once, per tier (default: 30s).")
    parser.add_argument("--lambda-rate", type=int, default=100,
                         help="Present for config compatibility; unused now that state A is a single "
                              "direct joint solve rather than a live-simulation replay (default: 100).")
    parser.add_argument("--config", default=os.path.join(SIMULATOR_DIR, "config.json"),
                         help="Path to the base config.json to start from (default: <simulator>/config.json).")
    parser.add_argument("--results-dir", default=None,
                         help="Where to write results.json. Default: "
                              "<simulator>/results-grid5000/dataset_size_sweep_<n_existing>j-"
                              "<nb_nodes>n_<solver_time_limit>s_<date>.")
    parser.add_argument("--arrival-lambda", type=float, default=None,
                         help="If given, the --n-existing jobs' arrival times are NOT taken from the "
                              "instance's jobs.json but generated as a Poisson process (mean inter-"
                              "arrival gap = this value, same time unit as arriving_time): job i's "
                              "arrival = sum of i exponential(1/lambda) draws from 0 -- small values "
                              "pack jobs closely together. The new job's own arrival is one more such "
                              "gap after the last existing job, guaranteeing it arrives strictly after "
                              "state A's own timeline. Omit to keep the instance's real arrival times "
                              "(default behaviour, unchanged).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()
    if args.incremental_time_limit is None:
        args.incremental_time_limit = args.solver_time_limit
    if args.epsilon_time_limit is None:
        args.epsilon_time_limit = args.solver_time_limit
    return args


def build_state_a(config, args, results_dir, tier_name, size_range, tier_index):
    """Builds state A as ONE direct joint CSP solve over all --n-existing jobs at once (their
    real dataset_size/nb_tasks/task_duration/arriving_time), with node free times at 0 and no
    pre-existing replicas -- a genuine one-shot "offline" placement, not a replay of N sequential
    live-simulation arrivals. Rebuilt fresh per tier so each tier's comparison starts from an
    identical, uncontaminated state A.

    Solved at env.now=0 (each job's own real arriving_time as its actual lower bound), NOT at the
    freeze point -- schedulingUsingJavaCSP turns Java's local times into absolute ones via
    master.env.now, so solving at env.now=T would wrongly anchor every job to start at-or-after T,
    making nothing ever appear already finished by the freeze point. Only after solving is env.now
    advanced to T (via env.run(until=T) with no processes registered, so SimPy just advances the
    clock) for the new job's own arrival and the final solve."""
    with open(os.path.join(args.instance_dir, "jobs.json")) as f:
        all_jobs_raw = json.load(f)
    existing_jobs_raw = all_jobs_raw[:args.n_existing]
    new_job_index = args.new_job_index if args.new_job_index is not None else args.n_existing
    new_job_raw = dict(all_jobs_raw[new_job_index])

    # --arrival-lambda: replace the instance's own (fixed) arrival timestamps with a fresh Poisson
    # process -- job i's arrival = sum of i exponential(1/lambda) inter-arrival gaps from 0, so a
    # smaller lambda packs the --n-existing jobs closer together. The new job's own arrival is one
    # MORE such gap after the last existing job, which keeps it always strictly after state A's own
    # timeline (same invariant `freeze_at = new_job_raw["arriving_time"]` relied on before). Own
    # random.Random instance, seeded distinctly from job_size_rng below and from the infra/global
    # random module, so this draw doesn't shift when other seeded draws change.
    if args.arrival_lambda is not None:
        arrival_rng = random.Random(args.seed * 4000 + tier_index)
        t = 0.0
        generated_arrivals = []
        for _ in range(args.n_existing):
            t += arrival_rng.expovariate(1.0 / args.arrival_lambda)
            generated_arrivals.append(t)
        for raw, arrival in zip(existing_jobs_raw, generated_arrivals):
            raw["arriving_time"] = arrival
        new_job_raw["arriving_time"] = generated_arrivals[-1] + arrival_rng.expovariate(1.0 / args.arrival_lambda) if generated_arrivals else arrival_rng.expovariate(1.0 / args.arrival_lambda)

    freeze_at = new_job_raw["arriving_time"]


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

    # Every existing job's dataset_size is ALSO redrawn from this tier's size_range (not just the
    # new job's) -- own random.Random instance (not the global `random` module, which
    # generateHeterogeneousInfrastructureEquilibre's own random.uniform calls above already
    # consumed from) so this draw is deterministic and independent of infra generation, seeded
    # distinctly per tier so different tiers don't share a draw sequence.
    job_size_rng = random.Random(args.seed * 2000 + tier_index)
    existing_jobs = []
    for raw in existing_jobs_raw:
        dataset_size = job_size_rng.randint(*size_range)
        j = Job(raw["job_id"], raw["task_duration"], raw["nb_tasks"], dataset_size)
        j.arriving_time = raw["arriving_time"]
        existing_jobs.append(j)
        tracker.register_job(j.job_id, raw["arriving_time"])
    master.jobs = existing_jobs

    with open(os.path.join(results_dir, f"state_a_jobs_{tier_name}.json"), "w") as f:
        json.dump([{"job_id": j.job_id, "dataset_size": j.dataset_size, "nb_tasks": j.nb_tasks,
                    "task_duration": j.tasks[0].duration if j.tasks else None,
                    "arriving_time": j.arriving_time} for j in existing_jobs], f)

    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
    master.java_main_class = 'MainOnline'
    # Explicit rather than relying on MainOnline.java's own default (1 = max flow time) -- state
    # A's own construction is also an Online-style joint solve, so it gets the same objective as
    # the final new-job solve below for consistency.
    master.objective_choice = 1
    print(f"### [{tier_name}] Building state A: ONE joint solve over jobs 0..{args.n_existing - 1} "
          f"(solver_time_limit={args.state_a_time_limit}s, node free times=0, no pre-existing "
          f"replicas, solved at env.now=0) ###", flush=True)
    (transfers_, works_, deletions_), state_a_log, state_a_wall = run_captured(
        schedulingUsingJavaCSP, master, existing_jobs, {}, nodes_free_time, 0)
    master._state_a_plan = {"transfers": transfers_ or {}, "works": works_ or {}, "deletions": deletions_ or {}}
    master._state_a_log, master._state_a_wall = state_a_log, state_a_wall
    master._state_a_nodes_free_time = dict(nodes_free_time)
    master._nodes_config = nodes_config
    master._freeze_at = freeze_at

    state_a_finish = {}
    if works_:
        for entries in works_.values():
            for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
                state_a_finish[job_id] = end_abs if job_id not in state_a_finish else max(state_a_finish[job_id], end_abs)
    master._state_a_finish = state_a_finish
    if len(state_a_finish) < args.n_existing:
        print(f"### [{tier_name}] WARNING: state A solve only placed {len(state_a_finish)}/"
              f"{args.n_existing} jobs -- possibly infeasible or cut short within the time budget ###",
              flush=True)

    # A job is "finished by T" iff its own committed schedule has it fully done at or before the
    # freeze point -- computed from state A's OWN decided timeline, not a live status flag
    # (nothing here ever actually executes through SimPy's event loop).
    not_finished_ids = {j.job_id for j in existing_jobs
                         if state_a_finish.get(j.job_id) is None or state_a_finish[j.job_id] > freeze_at}
    not_finished_jobs = [j for j in existing_jobs if j.job_id in not_finished_ids]
    print(f"### [{tier_name}] At freeze point T={freeze_at:.2f}: {args.n_existing - len(not_finished_ids)} "
          f"of {args.n_existing} existing jobs already finished; {len(not_finished_ids)} still running "
          f"(job_ids={sorted(not_finished_ids)}) ###", flush=True)

    # Reflect state A's own per-TASK schedule onto each not-yet-finished job's actual Task
    # objects: anything already committed by T becomes "Finished" or "Started", leaving only
    # genuinely not-yet-started tasks as "NotStarted". schedulingUsingJavaCSP only ever batches a
    # job's "NotStarted" tasks into the CSP, so this is what makes an online replan reconsider
    # just the REMAINING work -- without it, every task here is still "NotStarted" (the Task
    # class default, since these Job/Task objects never run through SimPy's live event loop), so
    # a replan would silently re-decide the ENTIRE job's placement from scratch at T, discarding
    # whatever state A had already committed before T. See xp_single_decision_grid5000.py's
    # matching fix/comment.
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
    # busy (and until when) instead of treating every node as free.
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
    print(f"### [{tier_name}] State A built. env.now={env.now:.2f}, "
          f"jobs placed={len(state_a_finish)}/{args.n_existing} ###", flush=True)

    return master, new_job_raw, not_finished_jobs


def make_new_job(new_job_raw, dataset_size, now):
    """A fresh Job for this tier: same id/nb_tasks/task_duration/arrival as the instance's real
    next job, but with dataset_size overridden to the tier's value (propagates to every one of
    its Task objects too, since Job's constructor builds them from this same parameter)."""
    new_job = Job(new_job_raw["job_id"], new_job_raw["task_duration"], new_job_raw["nb_tasks"], dataset_size)
    new_job.arriving_time = now
    return new_job


def committed_finish_time(master, nb_nodes, job_id):
    """Already-decided completion time for an existing job that this solve does NOT re-plan --
    read directly from state A's own solve result (there's no live execution history to fall
    back on in this offline-style construction)."""
    return master._state_a_finish.get(job_id)


def compute_transfer_energy(transfers_, master):
    """Same formula as Tracker.log_transfer (energy = sender + receiver * transfer_time +
    network), applied to a solve's PROPOSED transfers -- this test bypasses the live simulation
    loop (it calls schedulingUsingJavaCSP directly and never actually executes the resulting
    plan), so those transfers never flow through the tracker's own accounting."""
    sender = master._config.get('master_energy_consumption', 0.0)
    network = master._config.get('network_energy_per_transfer', 0.0)
    total = 0.0
    for entries in transfers_.values():
        for job_id, node_index, start_abs, end_abs, duration in entries:
            receiver = master.compute_nodes[node_index].energy_consumption
            total += sender + receiver * duration + network
    return total


# ---------------------------------------------------------------------------------------------
# Full per-run export. Every run (state A once per tier, then each approach x repeat) saves ALL it
# decided and observed, so no later analysis needs a re-run: every task placement, every transfer
# (with its energy split), every replica, every deletion, the per-job timeline, the node free times
# the solver saw, the solver's own convergence trajectory + raw log, and the raw Java input/output
# files of that solve. Layout: <results_dir>/runs/<tier>_state_A/ and
# <results_dir>/runs/<tier>_r<repeat>_<approach>/ (written as soon as each solve ends, so a crash or
# walltime cut later never loses the runs already done), plus consolidated <table>_detail_by_run.csv
# files at the end (one row per tier x repeat x approach x record; state A appears as
# approach="state_A", repeat=-1).
# ---------------------------------------------------------------------------------------------
JOBS_FIELDS = ["job_id", "is_new", "rescheduled", "isolated", "finished_before_T", "nb_tasks", "task_duration",
               "job_dataset_size", "arriving_time", "starting_time", "finishing_time", "wait_time", "flow_time",
               "last_task_end", "nb_replicas", "total_transfer_time", "total_transfer_energy"]
REPLICAS_FIELDS = ["job_id", "node", "data_size", "transfer_start", "transfer_end", "transfer_time",
                   "transfer_energy", "nb_tasks", "time_of_use", "first_task_start", "last_task_end",
                   "deletion_time", "origin"]
NODES_FREE_FIELDS = ["node", "free_time"]
DETAIL_TABLES = {
    "jobs": JOBS_FIELDS, "tasks": TASKS_FIELDS, "transfers": TRANSFERS_FIELDS, "replicas": REPLICAS_FIELDS,
    "deletions": DELETIONS_FIELDS, "trajectory": TRAJECTORY_FIELDS, "nodes_free_time": NODES_FREE_FIELDS,
}


def merge_plans(state_a, solve, rescheduled_ids, freeze_at):
    """The schedule an approach ends up with = state A's committed part + this solve's plan. State A
    entries are kept unless this solve superseded them: same (job, task) / (job, node) key, or a
    not-yet-started (start > T) entry of a job this solve re-planned. `origin` says which is which."""
    solve_tasks = {(r["job_id"], r["task_id"]) for r in solve["tasks"]}
    solve_transfers = {(r["job_id"], r["node"]) for r in solve["transfers"]}
    solve_deletions = {(r["job_id"], r["node"]) for r in solve["deletions"]}

    def keep(row, key, solved, time_key):
        if key in solved:
            return False
        return not (row["job_id"] in rescheduled_ids and row[time_key] > freeze_at)

    return {
        "tasks": [r for r in state_a["tasks"] if keep(r, (r["job_id"], r["task_id"]), solve_tasks, "start")] + solve["tasks"],
        "transfers": [r for r in state_a["transfers"] if keep(r, (r["job_id"], r["node"]), solve_transfers, "start")] + solve["transfers"],
        "deletions": [r for r in state_a["deletions"] if keep(r, (r["job_id"], r["node"]), solve_deletions, "deletion_time")] + solve["deletions"],
    }


def build_jobs_table(jobs, new_job_id, flow_by_job, schedule, rescheduled_ids, isolated_ids, state_a_finish, freeze_at):
    """One row per job (this project's infos_on_jobs.csv spirit): timeline, flow time, wait, replicas."""
    rows = []
    for j in jobs:
        tasks = [t for t in schedule["tasks"] if t["job_id"] == j.job_id]
        transfers = [t for t in schedule["transfers"] if t["job_id"] == j.job_id]
        flow = flow_by_job.get(j.job_id)
        start = min((t["start"] for t in tasks), default=None)
        sa_fin = state_a_finish.get(j.job_id)
        rows.append({
            "job_id": j.job_id, "is_new": j.job_id == new_job_id, "rescheduled": j.job_id in rescheduled_ids,
            "isolated": j.job_id in isolated_ids,
            "finished_before_T": (sa_fin is not None and sa_fin <= freeze_at) if j.job_id != new_job_id else False,
            "nb_tasks": len(j.tasks), "task_duration": j.tasks[0].duration if j.tasks else None,
            "job_dataset_size": j.dataset_size, "arriving_time": j.arriving_time, "starting_time": start,
            "finishing_time": (j.arriving_time + flow) if flow is not None else None,
            "wait_time": (start - j.arriving_time) if start is not None else None, "flow_time": flow,
            "last_task_end": max((t["end"] for t in tasks), default=None),
            "nb_replicas": len({t["node"] for t in transfers}),
            "total_transfer_time": sum(t["duration"] for t in transfers),
            "total_transfer_energy": sum(t["energy"] for t in transfers),
        })
    return rows


def build_replicas_table(jobs_by_id, schedule):
    """One row per replica (a job's dataset copied onto a node): its transfer, the tasks that used it,
    and when it was dropped -- this project's infos_on_replicas.csv spirit."""
    tasks_by = {}
    for t in schedule["tasks"]:
        tasks_by.setdefault((t["job_id"], t["node"]), []).append(t)
    deleted_at = {(d["job_id"], d["node"]): d["deletion_time"] for d in schedule["deletions"]}
    rows = []
    for tr in schedule["transfers"]:
        key = (tr["job_id"], tr["node"])
        used = tasks_by.get(key, [])
        rows.append({
            "job_id": tr["job_id"], "node": tr["node"], "data_size": jobs_by_id[tr["job_id"]].dataset_size,
            "transfer_start": tr["start"], "transfer_end": tr["end"], "transfer_time": tr["duration"],
            "transfer_energy": tr["energy"], "nb_tasks": len(used), "time_of_use": sum(t["duration"] for t in used),
            "first_task_start": min((t["start"] for t in used), default=None),
            "last_task_end": max((t["end"] for t in used), default=None),
            "deletion_time": deleted_at.get(key), "origin": tr["origin"],
        })
    return rows


def write_run_folder(run_dir, detail, summary, log_text):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "solver.log"), "w") as f:
        f.write(log_text)
    for table, fields in DETAIL_TABLES.items():
        write_rows(os.path.join(run_dir, f"{table}.csv"), fields, detail.get(table, []))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


def finalize_state_a(master, tier_name, runs_dir, log_text, wall_s, args, not_finished_ids):
    """Saves state A (the one joint solve every approach starts from) as its own run folder."""
    run_dir = os.path.join(runs_dir, f"{tier_name}_state_A")
    snapshot_solver_io(run_dir)
    freeze_at = master._freeze_at
    schedule = plan_rows(master, master._state_a_plan, "state_A")
    trajectory, stats = parse_solver_log(log_text)
    flow_by_job = {jid: fin - next(j.arriving_time for j in master.jobs if j.job_id == jid)
                   for jid, fin in master._state_a_finish.items()}
    detail = {
        "jobs": build_jobs_table(master.jobs, None, flow_by_job, schedule, set(), not_finished_ids,
                                 master._state_a_finish, freeze_at),
        "tasks": schedule["tasks"], "transfers": schedule["transfers"], "deletions": schedule["deletions"],
        "replicas": build_replicas_table({j.job_id: j for j in master.jobs}, schedule),
        "trajectory": trajectory,
        "nodes_free_time": [{"node": k, "free_time": v} for k, v in master._state_a_nodes_free_time.items()],
    }
    summary = {"kind": "state_A", "tier": tier_name, "freeze_at": freeze_at, "n_existing": args.n_existing,
               "solver_time_limit_s": args.state_a_time_limit, "wall_time_s": wall_s, "solver": stats,
               "n_jobs_placed": len(master._state_a_finish),
               "energy_constants": {"master_energy_consumption": master._config.get('master_energy_consumption', 0.0),
                                    "network_energy_per_transfer": master._config.get('network_energy_per_transfer', 0.0)}}
    write_run_folder(run_dir, detail, summary, log_text)
    return detail


def finalize_run(master, new_job, res, log_text, wall_s, run_dir, tier_name, repeat, dataset_size, approach,
                 time_limit, isolated_ids, args):
    """Builds and saves one approach's complete record; returns (aggregates dict for results.json,
    detail dict for the consolidated CSVs). Infeasible runs (res is None) still keep their log + raw I/O."""
    snapshot_solver_io(run_dir)
    trajectory, stats = parse_solver_log(log_text)
    meta = {"kind": "approach_run", "tier": tier_name, "repeat": repeat, "dataset_size": dataset_size,
            "approach": approach, "freeze_at": master._freeze_at, "solver_time_limit_s": time_limit,
            "wall_time_s": wall_s, "solver": stats, "new_job_id": new_job.job_id, "n_existing": args.n_existing,
            "epsilon_fraction": args.epsilon_fraction if approach in ("epsilon", "online_biobj_warmstart", "hybrid") else None,
            "epsilon_phase1_fraction": args.epsilon_phase1_fraction if approach in ("epsilon", "online_biobj_warmstart", "hybrid") else None,
            "epsilon_max_cap": args.epsilon_max_cap if approach in ("epsilon", "online_biobj_warmstart", "hybrid") else None,
            "hybrid_alpha": args.hybrid_alpha if approach == "hybrid" else None,
            "hybrid_max_budget": args.hybrid_max_budget if approach == "hybrid" else None,
            "hybrid_incremental_time_limit": args.hybrid_incremental_time_limit if approach == "hybrid" else None}
    if res is None:
        write_run_folder(run_dir, {"trajectory": trajectory}, {**meta, "result": None}, log_text)
        return None, None
    # Surfaced into the aggregate results.json (not just this run's own summary.json) so it's
    # directly readable next to mean/max_flow_time_all -- the one stat missing from the original
    # online/incremental-only version of this sweep and its results_analysis.ipynb.
    res["scheduling_time_s"] = wall_s
    plan = res.pop("_plan")
    freeze_at = master._freeze_at
    rescheduled = set(res["jobs_to_reschedule"])
    schedule = merge_plans(plan_rows(master, master._state_a_plan, "state_A"), plan_rows(master, plan, "solve"),
                           rescheduled, freeze_at)
    all_jobs = master.jobs + [new_job]
    detail = {
        "jobs": build_jobs_table(all_jobs, new_job.job_id, plan["flow_by_job"], schedule, rescheduled,
                                 isolated_ids, master._state_a_finish, freeze_at),
        "tasks": schedule["tasks"], "transfers": schedule["transfers"], "deletions": schedule["deletions"],
        "replicas": build_replicas_table({j.job_id: j for j in all_jobs}, schedule),
        "trajectory": trajectory,
        "nodes_free_time": [{"node": k, "free_time": v} for k, v in plan["nodes_free_time"].items()],
        "meta": meta,
    }
    write_run_folder(run_dir, detail, {**meta, "result": res}, log_text)
    return res, detail


def run_online_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs):
    """Mirrors SchedulingUsingCSPOnline.schedulingNewJob(): jointly replan the new job + every
    existing job that still has unstarted tasks. not_finished_jobs (from state A's own committed
    schedule vs the freeze point) is passed in directly rather than via master.getRunningJobs(),
    which reads live status flags never updated here."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnline'
    master.objective_choice = 1  # max flow time (all jobs in this batch) -- explicit, not relied on as a default
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
            cf = committed_finish_time(master, nb_nodes, j.job_id)
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

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
    }


def build_warm_start_for_reconsideration(master, new_job, not_finished_jobs, nb_nodes, now, replicas_locations,
                                          new_job_placement=None):
    """Writes warm_start.json to seed Online's search with a solution AT LEAST heuristically as
    good as "leave every not-finished job exactly where state A already had it, and place the new
    job however Incremental would" -- see MainOnlineWarmStart.java's own warm-start reader for the
    exact mechanism (IntDomainLast value-selector bias, not a hard incumbent -- see the comment on
    run_online_warmstart_style below for why this is a heuristic nudge, not a proof).

    job_index in the written JSON must match jobs_to_reschedule's own sort-by-job_id order (the
    same order _schedulingUsingJavaCSP_impl exports jobs_data in). task_index must be the LOCAL
    index within THIS solve's own NotStarted-only export (position i in
    [t for t in job.tasks if t.status == "NotStarted"]), NOT the task's true/global task_id --
    mixing these up silently applies a hint to the wrong task (mirrors the exact bug fixed in
    modelCSP.py's toDict() task_id remapping; the same local/true distinction applies here, on the
    way IN this time instead of on the way out).

    new_job_placement: optional (inc_transfers, inc_works) already computed elsewhere (e.g.
    hybrid's own F1 Incremental probe) -- when given, this function reuses it directly instead of
    running its own throwaway Incremental solve for the new job, avoiding a redundant JVM launch."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    sorted_jobs = sorted(jobs_to_reschedule, key=lambda j: j.job_id)
    job_index = {j.job_id: idx for idx, j in enumerate(sorted_jobs)}

    job_placements, transfers_ws = [], []

    # (a) not-finished existing jobs: reuse state A's OWN committed placement for their remaining
    # (NotStarted) tasks -- there is no "last replan" to fall back on in this frozen-state-A
    # protocol (state A itself is the only prior decision), converted to the LOCAL (relative-to-T)
    # time frame the solver expects.
    for job in not_finished_jobs:
        not_started_true_ids = [t.task_id for t in job.tasks if t.status == "NotStarted"]
        local_index_of = {true_id: k for k, true_id in enumerate(not_started_true_ids)}
        for entries in master._state_a_plan["works"].values():
            for e_job_id, e_node, e_task_id, e_start, e_end, e_dur in entries:
                if e_job_id == job.job_id and e_task_id in local_index_of:
                    job_placements.append({
                        "job_index": job_index[job.job_id], "task_index": local_index_of[e_task_id],
                        "node": int(e_node), "start": int(round(e_start - now)),
                    })
        for entries in master._state_a_plan["transfers"].values():
            for e_job_id, e_node, e_start, e_end, e_dur in entries:
                if e_job_id == job.job_id:
                    transfers_ws.append({
                        "job_index": job_index[job.job_id], "node": int(e_node),
                        "start": int(round(e_start - now)),
                    })

    # (b) the new job: reuse an already-computed Incremental placement if given, else ask
    # Incremental ourselves, using the exact state this replan itself sees (same
    # nodes_free_time/replicas_locations Online's own joint solve is about to use).
    if new_job_placement is not None:
        inc_transfers, inc_works = new_job_placement
    else:
        nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)
        orig_java_main_class = master.java_main_class
        try:
            master.java_main_class = 'MainIncremental'
            inc_transfers, inc_works, _ = schedulingUsingJavaCSP(master, [new_job], replicas_locations, nodes_free_time, now)
        finally:
            master.java_main_class = orig_java_main_class
    for entries in (inc_works or {}).values():
        for e_job_id, e_node, e_task_id, e_start, e_end, e_dur in entries:
            if e_job_id == new_job.job_id:
                # New job is exported fresh (no filtering, every task NotStarted) -- local index ==
                # true index directly, no remapping needed.
                job_placements.append({
                    "job_index": job_index[new_job.job_id], "task_index": int(e_task_id),
                    "node": int(e_node), "start": int(round(e_start - now)),
                })
    for entries in (inc_transfers or {}).values():
        for e_job_id, e_node, e_start, e_end, e_dur in entries:
            if e_job_id == new_job.job_id:
                transfers_ws.append({
                    "job_index": job_index[new_job.job_id], "node": int(e_node),
                    "start": int(round(e_start - now)),
                })

    warm_start = {"job_placements": job_placements, "transfers": transfers_ws}
    with open(os.path.join(solver_model_dir(), "inputs", "warm_start.json"), "w") as f:
        json.dump(warm_start, f)


def run_online_warmstart_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs):
    """Same joint reconsideration as run_online_style, but the search is seeded with a warm start
    (see build_warm_start_for_reconsideration): Incremental's own decision for the new job, plus
    state A's own committed placement for every not-finished job's remaining tasks -- i.e. "as if
    nothing changed for existing jobs, and the new job landed wherever Incremental would put it".

    IMPORTANT: this is a value-ordering HEURISTIC (Choco's IntDomainLast tries the warm-start value
    first, falls back otherwise), not a hard incumbent -- propagation from jointly reconsidering
    ALL not-finished jobs together (unlike Incremental's single-job solve) can still force some
    values to deviate. There is NO mathematical guarantee this solve's result is >= Incremental's
    own quality; it is only a strong empirical nudge (see MainOnlineWarmStart.java's own comment
    and the docstring on build_warm_start_for_reconsideration). Compare this function's own result
    against run_incremental_style's on the same batch to check whether the guarantee actually held
    for a given scenario -- do not assume it."""
    build_warm_start_for_reconsideration(master, new_job, not_finished_jobs, nb_nodes, now, replicas_locations)

    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnlineWarmStart'
    master.objective_choice = 1
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
            cf = committed_finish_time(master, nb_nodes, j.job_id)
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

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
    }


def run_online_biobj_warmstart_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                                      epsilon_fraction, epsilon_phase1_fraction, epsilon_max_cap=None):
    """Same joint reconsideration + warm start as run_online_warmstart_style, but bi-objective:
    MainOnlineMultiObjWarmStart's epsilon-constraint mode minimizes max flow time first (phase 1,
    warm-started the same way as the mono-obj variant), then minimizes transfer energy subject to
    max flow time staying within epsilon_fraction of phase 1's own result (phase 2, warm-started
    from phase 1's own live best solution -- see MainOnlineMultiObjWarmStart.java's second
    solver.setSearch() call after reset()). No pre-processing/freezing here -- every not-finished
    job is jointly reconsidered; that selective freezing is hybrid's own escalation step, not this
    approach's."""
    build_warm_start_for_reconsideration(master, new_job, not_finished_jobs, nb_nodes, now, replicas_locations)

    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnlineMultiObjWarmStart'
    master.multi_objective = 2
    master.epsilon_fraction = epsilon_fraction
    master.epsilon_phase1_fraction = epsilon_phase1_fraction
    master.epsilon_max_cap = epsilon_max_cap
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
            cf = committed_finish_time(master, nb_nodes, j.job_id)
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

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
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
    for j in master.jobs + [new_job]:
        if j.job_id == new_job.job_id:
            if new_job_finish is not None:
                flow_by_job[j.job_id] = new_job_finish - j.arriving_time
        else:
            cf = committed_finish_time(master, nb_nodes, j.job_id)
            if cf is not None:
                flow_by_job[j.job_id] = cf - j.arriving_time
    flow_times = list(flow_by_job.values())
    isolated_flow_times = [ft for jid, ft in flow_by_job.items() if jid in isolated_ids]

    wait_time = (new_job_start - new_job.arriving_time) if new_job_start is not None else None
    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (new_job_finish - new_job.arriving_time) if new_job_finish is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
    }


def run_epsilon_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                       epsilon_fraction, epsilon_phase1_fraction, epsilon_max_cap=None):
    """Online-style joint replan (same jobs_to_reschedule as run_online_style), but via
    MainOnlineMultiObj.java's epsilon-constraint mode: phase 1 minimizes max flow time (identical
    objective/search to run_online_style's MainOnline.java), phase 2 then minimizes transfer
    energy subject to max flow time staying within epsilon_fraction of phase 1's own result --
    warm-started from phase 1's own solution (verified in MainOnlineMultiObj.java itself), not
    from scratch. Found empirically to score MUCH better than a raw Pareto-front search within
    the same budget (which gets stuck far from single-objective quality) while still trading a
    controlled, bounded amount of flow-time for a large transfer-energy reduction."""
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = master.nodesFreeTime(master.ongoing_transfers, master.ongoing_works)

    master.java_main_class = 'MainOnlineMultiObj'
    master.multi_objective = 2
    master.epsilon_fraction = epsilon_fraction
    master.epsilon_phase1_fraction = epsilon_phase1_fraction
    master.epsilon_max_cap = epsilon_max_cap
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
            cf = committed_finish_time(master, nb_nodes, j.job_id)
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

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
    }


def run_hybrid_style(master, new_job, isolated_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                      hybrid_alpha, hybrid_max_budget, hybrid_incremental_time_limit,
                      epsilon_fraction, epsilon_phase1_fraction):
    """Mirrors SchedulingUsingCSPAdaptiveJoint.schedulingNewJob()'s escalation branch EXACTLY --
    same code, not a reimplementation. Replaces this function's 2025-era single-warmstart-solve
    version (which predates the live class's 4-way parallel escalation, and used the old
    freeze_large_jobs_threshold/freeze_remaining_time_threshold/freeze_jobs_with_ongoing_transfer
    pre-processing -- both since superseded by the median-split freeze_below_median/
    freeze_above_median/nofreeze/warm_nofreeze design; see _timedParallelEscalation's own
    docstring in master_node_with_heterogeneous_nodes_csp.py for why). Probes Incremental for F1
    (bounded by hybrid_incremental_time_limit), then runs the real 4-way parallel escalation
    budgeted at min(hybrid_alpha*F1, hybrid_max_budget) seconds. Reached by shallow-copying master
    and reassigning its class to SchedulingUsingCSPAdaptiveJoint -- state A itself is always built
    the plain Online/MainOnline way (see build_state_a), so this only ever affects how the new
    job's own placement gets solved. charge_thinking_time is off: run_tier's own run_captured
    already measures this whole call's wall-clock time (see finalize_run's scheduling_time_s), and
    there is no live SimPy env.run() loop here for env.timeout() to advance anyway."""
    hybrid_master = copy.copy(master)
    hybrid_master.__class__ = SchedulingUsingCSPAdaptiveJoint
    hybrid_master._config = dict(master._config)
    hybrid_master._config['charge_thinking_time'] = False
    hybrid_master._config['epsilon_fraction'] = epsilon_fraction
    hybrid_master._config['epsilon_phase1_fraction'] = epsilon_phase1_fraction
    hybrid_master._config['parallel_warm_cold_escalation'] = True
    hybrid_master._config['incremental_time_limit_s'] = hybrid_incremental_time_limit
    hybrid_master._config['adaptive_alpha'] = hybrid_alpha
    hybrid_master._config['adaptive_max_budget_s'] = hybrid_max_budget

    def drive(gen):
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        raise RuntimeError("hybrid: generator unexpectedly yielded -- charge_thinking_time should be False")

    inc_transfers, inc_works, inc_deletions, f1 = drive(hybrid_master._placeSingleJobIncremental(new_job))
    if f1 is None:
        return None

    inc_flow_by_job = {}
    for j in master.jobs + [new_job]:
        if j.job_id == new_job.job_id:
            inc_flow_by_job[j.job_id] = f1
        else:
            cf = committed_finish_time(master, nb_nodes, j.job_id)
            if cf is not None:
                inc_flow_by_job[j.job_id] = cf - j.arriving_time
    inc_flow_times = list(inc_flow_by_job.values())
    inc_isolated_flow_times = [ft for jid, ft in inc_flow_by_job.items() if jid in isolated_ids]
    fallback_nodes_free_time = SchedulingUsingCSPIncremental.nodesFreeTimeIncremental(
        master, master.ongoing_transfers, master.ongoing_works)
    incremental_result = {
        "wait_time_new_job": None,
        "flow_time_new_job": f1,
        "mean_flow_time_all": sum(inc_flow_times) / len(inc_flow_times) if inc_flow_times else None,
        "max_flow_time_all": max(inc_flow_times) if inc_flow_times else None,
        "mean_flow_time_isolated": sum(inc_isolated_flow_times) / len(inc_isolated_flow_times) if inc_isolated_flow_times else None,
        "max_flow_time_isolated": max(inc_isolated_flow_times) if inc_isolated_flow_times else None,
        "n_jobs_isolated": len(inc_isolated_flow_times),
        "jobs_to_reschedule": [new_job.job_id],
        "transfer_energy_total": compute_transfer_energy(inc_transfers, master),
        "hybrid_f1": f1,
        "hybrid_escalation_budget": None,
        "_plan": {"transfers": inc_transfers, "works": inc_works, "deletions": inc_deletions,
                  "nodes_free_time": dict(fallback_nodes_free_time), "flow_by_job": inc_flow_by_job},
    }

    budget = min(hybrid_alpha * f1, hybrid_max_budget)
    jobs_to_reschedule = [new_job] + not_finished_jobs
    nodes_free_time = hybrid_master.nodesFreeTime(hybrid_master.ongoing_transfers, hybrid_master.ongoing_works)

    transfers_, works_, deletions_ = drive(hybrid_master._timedParallelEscalation(
        jobs_to_reschedule, not_finished_jobs, replicas_locations, nodes_free_time, now, budget))

    if not transfers_ or not works_:
        # 4-way escalation found no solution -- fall back to the Incremental placement already
        # computed above, untouched (mirrors the live schedulingNewJob's own fallback).
        return incremental_result

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
            cf = committed_finish_time(master, nb_nodes, j.job_id)
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

    return {
        "wait_time_new_job": wait_time,
        "flow_time_new_job": (finish.get(new_job.job_id) - new_job.arriving_time) if finish.get(new_job.job_id) is not None else None,
        "mean_flow_time_all": sum(flow_times) / len(flow_times) if flow_times else None,
        "max_flow_time_all": max(flow_times) if flow_times else None,
        "mean_flow_time_isolated": sum(isolated_flow_times) / len(isolated_flow_times) if isolated_flow_times else None,
        "max_flow_time_isolated": max(isolated_flow_times) if isolated_flow_times else None,
        "n_jobs_isolated": len(isolated_flow_times),
        "jobs_to_reschedule": [j.job_id for j in jobs_to_reschedule],
        "transfer_energy_total": compute_transfer_energy(transfers_, master),
        "hybrid_f1": f1,
        "hybrid_escalation_budget": budget,
        "_plan": {"transfers": transfers_, "works": works_, "deletions": deletions_,
                  "nodes_free_time": dict(nodes_free_time), "flow_by_job": flow_by_job},
    }


def run_tier(config, args, results_dir, tier_name, size_range, tier_index):
    """Builds state A ONCE for this tier, then runs --repeats independent trials from that same
    frozen state, each drawing a fresh dataset_size uniformly from size_range -- heterogeneity
    within the tier instead of a single fixed value. Safe to reuse the one state-A build across
    all repeats and all approaches: schedulingUsingJavaCSP only reads master's state and returns
    a proposed solution, it never mutates works/ongoing_works/replicas_locations.

    Every solve's complete record is written to <results_dir>/runs/ the moment it ends (see the
    full-export notes above finalize_state_a); the returned dict additionally carries the same
    detail under underscore-prefixed keys, which main() uses for the consolidated CSVs and strips
    before writing results.json."""
    master, new_job_raw, not_finished_jobs = build_state_a(config, args, results_dir, tier_name, size_range, tier_index)
    now = master.env.now
    replicas_locations = master.replicas_locations
    not_finished_ids = {j.job_id for j in not_finished_jobs}
    runs_dir = os.path.join(results_dir, "runs")

    state_a_detail = finalize_state_a(master, tier_name, runs_dir, master._state_a_log, master._state_a_wall,
                                      args, not_finished_ids)
    write_rows(os.path.join(results_dir, f"nodes_config_{tier_name}.csv"),
               ["node_id", "bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"],
               [{"node_id": i, **{k: cfg.get(k) for k in ("bandwidth", "computation_nodes", "energy_consumption", "storage_capacity")}}
                for i, cfg in enumerate(master._nodes_config)])

    rng = random.Random(args.seed * 1000 + tier_index)
    low, high = size_range

    trials, details = [], {}
    for r in range(args.repeats):
        dataset_size = rng.randint(low, high)
        isolated_ids = not_finished_ids | {new_job_raw["job_id"]}

        results_by_approach = {}

        def solve(approach, label, time_limit, runner):
            # A fresh Job object per approach: keeps every approach's solve fully decoupled.
            new_job = make_new_job(new_job_raw, dataset_size, now)
            master.tracker.register_job(new_job.job_id, now)
            master._config["solver_time_limit_s"] = time_limit
            print(f"\n### [{tier_name} #{r}] Solving new job's (dataset_size={dataset_size}) placement -- "
                  f"{label}, {time_limit}s budget ###", flush=True)
            res, log_text, wall_s = run_captured(runner, new_job)
            run_dir = os.path.join(runs_dir, f"{tier_name}_r{r}_{approach}")
            res, detail = finalize_run(master, new_job, res, log_text, wall_s, run_dir, tier_name, r,
                                       dataset_size, approach, time_limit, isolated_ids, args)
            results_by_approach[approach] = res
            if detail is not None:
                details[(r, approach)] = detail
            print(f"[{tier_name} #{r}] {approach} result:", res, flush=True)

        if "online" in args.approaches:
            solve("online", "ONLINE style", args.solver_time_limit,
                  lambda nj: run_online_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations, not_finished_jobs))
        if "online_warmstart" in args.approaches:
            solve("online_warmstart", "ONLINE style (warm-started from Incremental + state A)", args.solver_time_limit,
                  lambda nj: run_online_warmstart_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations, not_finished_jobs))
        if "incremental" in args.approaches:
            solve("incremental", "INCREMENTAL style", args.incremental_time_limit,
                  lambda nj: run_incremental_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations))
        if "epsilon" in args.approaches:
            solve("epsilon", f"EPSILON-CONSTRAINT style ({args.epsilon_fraction * 100:.0f}% max-flow slack)",
                  args.epsilon_time_limit,
                  lambda nj: run_epsilon_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations,
                                               not_finished_jobs, args.epsilon_fraction, args.epsilon_phase1_fraction,
                                               args.epsilon_max_cap))
        if "online_biobj_warmstart" in args.approaches:
            solve("online_biobj_warmstart",
                  f"ONLINE-WARMSTART BI-OBJ style ({args.epsilon_fraction * 100:.0f}% max-flow slack)",
                  args.epsilon_time_limit if args.epsilon_time_limit is not None else args.solver_time_limit,
                  lambda nj: run_online_biobj_warmstart_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations,
                                                              not_finished_jobs, args.epsilon_fraction,
                                                              args.epsilon_phase1_fraction, args.epsilon_max_cap))
        if "hybrid" in args.approaches:
            hybrid_incremental_limit = args.hybrid_incremental_time_limit
            if hybrid_incremental_limit is None:
                hybrid_incremental_limit = args.incremental_time_limit
            solve("hybrid", "HYBRID style (F1 probe + 4-way parallel escalation)", args.hybrid_max_budget,
                  lambda nj: run_hybrid_style(master, nj, isolated_ids, args.nb_nodes, now, replicas_locations,
                                              not_finished_jobs, args.hybrid_alpha, args.hybrid_max_budget,
                                              hybrid_incremental_limit, args.epsilon_fraction,
                                              args.epsilon_phase1_fraction))

        trials.append({
            "repeat": r,
            "dataset_size": dataset_size,
            "n_jobs_isolated": len(isolated_ids),
            **results_by_approach,
        })

    return {
        "tier": tier_name,
        "range": [low, high],
        "trials": trials,
        "_details": details,
        "_state_a": state_a_detail,
    }


def write_consolidated_tables(results_dir, all_results):
    """<table>_detail_by_run.csv: every run's records stacked, prefixed with tier/repeat/dataset_size/approach."""
    prefix = ["tier", "repeat", "dataset_size", "approach"]
    for table, fields in DETAIL_TABLES.items():
        path = os.path.join(results_dir, f"{table}_detail_by_run.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=prefix + fields, extrasaction="ignore")
            writer.writeheader()
            for tr in all_results:
                for row in tr["_state_a"].get(table, []):
                    writer.writerow({"tier": tr["tier"], "repeat": -1, "dataset_size": None, "approach": "state_A", **row})
                for (repeat, approach), detail in tr["_details"].items():
                    for row in detail.get(table, []):
                        writer.writerow({"tier": tr["tier"], "repeat": repeat,
                                         "dataset_size": detail["meta"]["dataset_size"], "approach": approach, **row})
        print(f"### {table} detail written to {path} ###", flush=True)


def main():
    args = parse_args()
    configure_logging(logging.WARNING)

    results_dir = args.results_dir or os.path.join(
        SIMULATOR_DIR, "results-grid5000",
        f"dataset_size_sweep_{args.n_existing}j-{args.nb_nodes}n_{args.solver_time_limit}s_{date.today().isoformat()}",
    )
    os.makedirs(results_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Full-run log + parameters. archive=False: this script already saves every solve in full under
    # runs/ (state A and each approach), so the generic per-solve archive would only duplicate it.
    start_recording(results_dir, params=args, config=config, archive=False)

    all_tiers = [
        ("small", tuple(args.small_range)),
        ("medium", tuple(args.medium_range)),
        ("large", tuple(args.large_range)),
        ("full", tuple(args.full_range)),
    ]
    # tier_index must stay anchored to each tier's position in the CANONICAL small/medium/large/full
    # order (0/1/2/3), not its position within the --tiers-filtered subset -- job_size_rng and the
    # new job's own size draw are both seeded from `args.seed * K + tier_index`, so e.g. running
    # --tiers medium alone must still use tier_index=1 (medium's canonical slot), or every draw
    # for that tier silently becomes a DIFFERENT scenario than a prior run that included all
    # three tiers, defeating the whole point of running one approach in isolation for a
    # job-for-job-identical comparison against results already collected elsewhere.
    tiers = [(name, rng, i) for i, (name, rng) in enumerate(all_tiers) if name in args.tiers]

    all_results = [run_tier(config, args, results_dir, tier_name, size_range, tier_index)
                   for tier_name, size_range, tier_index in tiers]

    def fmt(v):
        return ('%.2f' % v) if v is not None else 'N/A'

    print("\n" + "=" * 110)
    print(f"{'Tier':<8}{'#':>3}{'size':>8}{'Approach':>13}{'Wait':>10}{'Flow':>10}"
          f"{'Mean(all)':>12}{'Max(all)':>10}{'Mean(iso)':>12}{'Max(iso)':>10}")
    print("=" * 110)
    for tier_result in all_results:
        for trial in tier_result["trials"]:
            for approach in args.approaches:
                res = trial.get(approach)
                print(f"{tier_result['tier']:<8}{trial['repeat']:>3}{trial['dataset_size']:>8}{approach:>13}"
                      f"{fmt(res.get('wait_time_new_job') if res else None):>10}"
                      f"{fmt(res.get('flow_time_new_job') if res else None):>10}"
                      f"{fmt(res.get('mean_flow_time_all') if res else None):>12}"
                      f"{fmt(res.get('max_flow_time_all') if res else None):>10}"
                      f"{fmt(res.get('mean_flow_time_isolated') if res else None):>12}"
                      f"{fmt(res.get('max_flow_time_isolated') if res else None):>10}")
    print("=" * 110)

    def mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None
        m = sum(values) / len(values)
        var = sum((v - m) ** 2 for v in values) / len(values)
        return m, var ** 0.5

    col_width = 20
    header = f"{'Tier':<10}{'range':>13}{'n':>4}"
    for approach in args.approaches:
        header += f"{approach + ' mean-FT':>{col_width}}{approach + ' energy':>{col_width}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    aggregates = []
    for tier_result in all_results:
        per_approach = {}
        for approach in args.approaches:
            means = [t[approach].get("mean_flow_time_isolated") if t.get(approach) else None for t in tier_result["trials"]]
            energies = [t[approach].get("transfer_energy_total") if t.get(approach) else None for t in tier_result["trials"]]
            avg, std = mean_std(means)
            energy_avg, energy_std = mean_std(energies)
            per_approach[approach] = {"mean": avg, "std": std, "energy_mean": energy_avg, "energy_std": energy_std}
        aggregates.append({
            "tier": tier_result["tier"], "range": tier_result["range"], "n": len(tier_result["trials"]),
            **{f"{approach}_mean_flow_time": v["mean"] for approach, v in per_approach.items()},
            **{f"{approach}_energy": v["energy_mean"] for approach, v in per_approach.items()},
        })
        range_str = f"{tier_result['range'][0]}-{tier_result['range'][1]}"
        row = f"{tier_result['tier']:<10}{range_str:>13}{len(tier_result['trials']):>4}"
        for approach in args.approaches:
            v = per_approach[approach]
            row += f"{fmt(v['mean']):>{col_width}}{fmt(v['energy_mean']):>{col_width}}"
        print(row)
    print("=" * len(header))
    if "online" in args.approaches and "incremental" in args.approaches:
        print("(mean-FT = mean isolated flow time across --repeats trials per tier; cost of Online's")
        print("reconsideration = its mean-FT minus Incremental's, as a function of dataset size.)")
    else:
        print("(mean-FT = mean isolated flow time across --repeats trials per tier.)")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "n_existing": args.n_existing,
            "new_job_index": args.new_job_index if args.new_job_index is not None else args.n_existing,
            "nb_nodes": args.nb_nodes,
            "instance_dir": args.instance_dir,
            "approaches": args.approaches,
            "solver_time_limit_s": args.solver_time_limit,
            "epsilon_time_limit_s": args.epsilon_time_limit,
            "incremental_time_limit_s": args.incremental_time_limit,
            "state_a_time_limit_s": args.state_a_time_limit,
            "epsilon_fraction": args.epsilon_fraction,
            "epsilon_phase1_fraction": args.epsilon_phase1_fraction,
            "epsilon_max_cap": args.epsilon_max_cap,
            "repeats": args.repeats,
            "seed": args.seed,
            "tiers": [{k: v for k, v in tr.items() if not k.startswith("_")} for tr in all_results],
            "aggregates": aggregates,
        }, f, indent=2)
    print(f"\n### Results written to {results_path} ###", flush=True)

    # Flat per-run table (one row per tier x repeat x approach), indexed by the ACTUAL sampled
    # dataset_size -- for plotting/analysis of how each stat scales with data size, rather than
    # digging through the nested JSON.
    csv_path = os.path.join(results_dir, "runs_by_dataset_size.csv")
    fieldnames = ["tier", "repeat", "dataset_size", "approach", "n_jobs_isolated",
                  "wait_time_new_job", "flow_time_new_job", "mean_flow_time_all", "max_flow_time_all",
                  "mean_flow_time_isolated", "max_flow_time_isolated", "transfer_energy_total"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tier_result in all_results:
            for trial in tier_result["trials"]:
                for approach in args.approaches:
                    res = trial.get(approach) or {}
                    writer.writerow({
                        "tier": tier_result["tier"],
                        "repeat": trial["repeat"],
                        "dataset_size": trial["dataset_size"],
                        "approach": approach,
                        "n_jobs_isolated": trial["n_jobs_isolated"],
                        **{k: res.get(k) for k in fieldnames if k in (
                            "wait_time_new_job", "flow_time_new_job", "mean_flow_time_all",
                            "max_flow_time_all", "mean_flow_time_isolated", "max_flow_time_isolated",
                            "transfer_energy_total")},
                    })
    print(f"### Per-run stats by dataset size written to {csv_path} ###", flush=True)

    write_consolidated_tables(results_dir, all_results)


if __name__ == "__main__":
    main()
