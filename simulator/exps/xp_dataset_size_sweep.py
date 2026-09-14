"""
Controlled single-decision-point test, swept over the NEW job's dataset size (small / medium /
large), to show how the cost of Online's "reconsideration" -- jointly re-planning every existing
job that still has unstarted tasks, instead of just placing the new job alone like Incremental
does -- scales with the size of the data being introduced.

For each size tier: build state A (the same --n-existing jobs, unaffected by the tier) as ONE
direct joint CSP solve -- node free times at 0, no pre-existing replicas, a genuine one-shot
"offline" placement, not a replay of N sequential live-simulation arrivals -- ONCE, then solve
the SAME new job's placement (with that tier's dataset_size) BOTH Online-style and
Incremental-style from that one state. This is safe to do from a single build (not one build per
approach, and not one per repeat either): schedulingUsingJavaCSP never mutates the master's
persistent state (works/ongoing_works/replicas_locations) -- it only reads it and returns a
proposed solution -- so calling it many times in a row for the same snapshot doesn't let one
solve contaminate another's starting point.

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
from master_node_with_heterogeneous_nodes_csp import SchedulingUsingCSPOnline, SchedulingUsingCSPIncremental
from utils.modelCSP import schedulingUsingJavaCSP
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
    parser.add_argument("--repeats", type=int, default=5,
                         help="Number of trials per tier, each drawing a fresh dataset_size uniformly "
                              "from that tier's range -- for heterogeneity within a tier instead of a "
                              "single fixed value (default: 5). State A is built ONCE per tier and "
                              "reused across its repeats (safe: solving never mutates it), so this "
                              "only adds solves, not state-A rebuilds.")
    parser.add_argument("--solver-time-limit", type=int, default=30,
                         help="CSP solver time budget for the new job's placement solve, in seconds -- "
                              "applies to both Online and Incremental at every tier (default: 30).")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def build_state_a(config, args, results_dir, tier_name):
    """Builds state A as ONE direct joint CSP solve over all --n-existing jobs at once (their
    real dataset_size/nb_tasks/task_duration/arriving_time), with node free times at 0 and no
    pre-existing replicas -- a genuine one-shot "offline" placement, not a replay of N sequential
    live-simulation arrivals. Rebuilt fresh per tier so each tier's comparison starts from an
    identical, uncontaminated state A. env.now is fast-forwarded directly to the moment the new
    job would arrive via env.run(until=...) with NO processes registered (SimPy just advances the
    clock)."""
    with open(os.path.join(args.instance_dir, "jobs.json")) as f:
        all_jobs_raw = json.load(f)
    existing_jobs_raw = all_jobs_raw[:args.n_existing]
    new_job_index = args.new_job_index if args.new_job_index is not None else args.n_existing
    new_job_raw = all_jobs_raw[new_job_index]
    freeze_at = new_job_raw["arriving_time"]

    with open(os.path.join(results_dir, f"state_a_jobs_{tier_name}.json"), "w") as f:
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
    print(f"### [{tier_name}] Building state A: ONE joint solve over jobs 0..{args.n_existing - 1} "
          f"(solver_time_limit={args.state_a_time_limit}s, node free times=0, no pre-existing "
          f"replicas) ###", flush=True)
    transfers_, works_, deletions_ = schedulingUsingJavaCSP(master, existing_jobs, {}, nodes_free_time, freeze_at)

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
    print(f"### [{tier_name}] State A built. env.now={env.now:.2f}, "
          f"jobs placed={len(state_a_finish)}/{args.n_existing} ###", flush=True)

    return master, new_job_raw


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
    }


def run_tier(config, args, results_dir, tier_name, size_range, tier_index):
    """Builds state A ONCE for this tier, then runs --repeats independent trials from that same
    frozen state, each drawing a fresh dataset_size uniformly from size_range -- heterogeneity
    within the tier instead of a single fixed value. Safe to reuse the one state-A build across
    all repeats and both approaches: schedulingUsingJavaCSP only reads master's state and returns
    a proposed solution, it never mutates works/ongoing_works/replicas_locations."""
    master, new_job_raw = build_state_a(config, args, results_dir, tier_name)
    now = master.env.now
    replicas_locations = master.replicas_locations
    not_finished_ids = {j.job_id for j in master.jobs if j.status != "Finished"}

    rng = random.Random(args.seed * 1000 + tier_index)
    low, high = size_range
    master._config["solver_time_limit_s"] = args.solver_time_limit

    trials = []
    for r in range(args.repeats):
        dataset_size = rng.randint(low, high)
        isolated_ids = not_finished_ids | {new_job_raw["job_id"]}

        new_job_online = make_new_job(new_job_raw, dataset_size, now)
        master.tracker.register_job(new_job_online.job_id, now)
        print(f"\n### [{tier_name} #{r}] Solving new job's (dataset_size={dataset_size}) placement -- "
              f"ONLINE style, {args.solver_time_limit}s budget ###", flush=True)
        online_result = run_online_style(master, new_job_online, isolated_ids, args.nb_nodes, now, replicas_locations)
        print(f"[{tier_name} #{r}] Online result:", online_result, flush=True)

        # A second, independently-constructed Job object for the Incremental solve:
        # schedulingUsingJavaCSP doesn't mutate new_job_online, so reusing it would also be
        # correct, but building a fresh one keeps the two solves fully decoupled.
        new_job_incremental = make_new_job(new_job_raw, dataset_size, now)
        print(f"\n### [{tier_name} #{r}] Solving new job's (dataset_size={dataset_size}) placement -- "
              f"INCREMENTAL style, {args.solver_time_limit}s budget ###", flush=True)
        incremental_result = run_incremental_style(master, new_job_incremental, isolated_ids, args.nb_nodes, now, replicas_locations)
        print(f"[{tier_name} #{r}] Incremental result:", incremental_result, flush=True)

        trials.append({
            "repeat": r,
            "dataset_size": dataset_size,
            "n_jobs_isolated": len(isolated_ids),
            "online": online_result,
            "incremental": incremental_result,
        })

    return {
        "tier": tier_name,
        "range": [low, high],
        "trials": trials,
    }


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

    tiers = [
        ("small", tuple(args.small_range)),
        ("medium", tuple(args.medium_range)),
        ("large", tuple(args.large_range)),
    ]

    all_results = [run_tier(config, args, results_dir, tier_name, size_range, i)
                   for i, (tier_name, size_range) in enumerate(tiers)]

    def fmt(v):
        return ('%.2f' % v) if v is not None else 'N/A'

    print("\n" + "=" * 110)
    print(f"{'Tier':<8}{'#':>3}{'size':>8}{'Approach':>13}{'Wait':>10}{'Flow':>10}"
          f"{'Mean(all)':>12}{'Max(all)':>10}{'Mean(iso)':>12}{'Max(iso)':>10}")
    print("=" * 110)
    for tier_result in all_results:
        for trial in tier_result["trials"]:
            for approach in ("online", "incremental"):
                res = trial[approach]
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

    print("\n" + "=" * 90)
    print(f"{'Tier':<10}{'range':>13}{'n':>4}{'Online iso mean':>20}{'Incr iso mean':>20}{'Cost (mean)':>13}{'Cost %':>10}")
    print("=" * 90)
    aggregates = []
    for tier_result in all_results:
        online_means = [t["online"].get("mean_flow_time_isolated") if t["online"] else None for t in tier_result["trials"]]
        incr_means = [t["incremental"].get("mean_flow_time_isolated") if t["incremental"] else None for t in tier_result["trials"]]
        online_avg, online_std = mean_std(online_means)
        incr_avg, incr_std = mean_std(incr_means)
        cost = (online_avg - incr_avg) if (online_avg is not None and incr_avg is not None) else None
        cost_pct = (100 * cost / incr_avg) if (cost is not None and incr_avg) else None
        aggregates.append({
            "tier": tier_result["tier"], "range": tier_result["range"], "n": len(tier_result["trials"]),
            "online_mean": online_avg, "online_std": online_std,
            "incremental_mean": incr_avg, "incremental_std": incr_std,
            "cost": cost, "cost_pct": cost_pct,
        })
        range_str = f"{tier_result['range'][0]}-{tier_result['range'][1]}"
        print(f"{tier_result['tier']:<10}{range_str:>13}{len(tier_result['trials']):>4}"
              f"{fmt(online_avg):>13} (+/-{fmt(online_std)})"
              f"{fmt(incr_avg):>13} (+/-{fmt(incr_std)})"
              f"{fmt(cost):>13}"
              f"{(f'{cost_pct:.1f}%' if cost_pct is not None else 'N/A'):>10}")
    print("=" * 90)
    print("Cost = Online's isolated mean flow time minus Incremental's, averaged over --repeats trials")
    print("per tier (each drawing dataset_size uniformly from that tier's range) -- the price paid for")
    print("letting Online disturb every not-yet-finished existing job to accommodate the new one, as a")
    print("function of how large the new job's data is.")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "n_existing": args.n_existing,
            "new_job_index": args.new_job_index if args.new_job_index is not None else args.n_existing,
            "nb_nodes": args.nb_nodes,
            "instance_dir": args.instance_dir,
            "solver_time_limit_s": args.solver_time_limit,
            "state_a_time_limit_s": args.state_a_time_limit,
            "repeats": args.repeats,
            "tiers": all_results,
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
                for approach in ("online", "incremental"):
                    res = trial[approach] or {}
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


if __name__ == "__main__":
    main()
