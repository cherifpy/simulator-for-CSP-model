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
    SchedulingUsingCSPOnlineWarmStart,
    SchedulingUsingCSPOnlineMultiObjWarmStart,
    SchedulingUsingCSPIncremental,
    SchedulingUsingCSPIncrementalFreeNodesOnly,
    SchedulingUsingCSPAdaptive,
    SchedulingUsingCSPAdaptiveJoint,
    SchedulingUsingCSPOnlineNewJob,
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
    # Same full-replan Online approach (mono-objective, max flow time), but each replan seeds the
    # Choco search with a warm start instead of starting cold: this scheduler's own last-decided
    # plan for jobs it already knew about, plus Incremental's decision for the brand-new job(s).
    # See SchedulingUsingCSPOnlineWarmStart in master_node_with_heterogeneous_nodes_csp.py.
    "online_warmstart": SchedulingUsingCSPOnlineWarmStart,
    # Combines online_biobj's epsilon-constraint bi-objective solve with online_warmstart's warm
    # start (this scheduler's own last-decided plan for known jobs + a throwaway Incremental
    # solve for the brand-new job(s)). See SchedulingUsingCSPOnlineMultiObjWarmStart's docstring
    # and MainOnlineMultiObjWarmStart.java for exactly how phase 1 vs phase 2 are each seeded.
    "online_biobj_warmstart": SchedulingUsingCSPOnlineMultiObjWarmStart,
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
    # Same Incremental-first gate as "adaptive", but escalation is the FULL joint replan (new job
    # + every already-running job) via online_biobj_warmstart, with pre-processing (freeze_*)
    # applied to keep that joint solve's search space down -- and a fallback to Incremental for
    # just the new job if the joint solve finds no solution. See
    # SchedulingUsingCSPAdaptiveJoint's docstring for the full flow.
    "hybrid": SchedulingUsingCSPAdaptiveJoint,
    # Live, continuous version of "online_newjob" (2026-10-07): for each arriving job, a SINGLE
    # joint solve over it + every currently-running job, under objective_choice=2 (minimize ONLY
    # the new job's own flow time) plus --adaptive-degradation-cap-pct -- no F1 probe, no
    # escalation variants, no gate. See SchedulingUsingCSPOnlineNewJob's own docstring.
    "online_newjob": SchedulingUsingCSPOnlineNewJob,
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
    parser.add_argument("--epsilon-phase1-fraction", type=float, default=0.75,
                         help="online_biobj only: fraction of --solver-time-limit given to phase "
                              "1 (max flow time); the rest goes to phase 2 (energy) (default: 0.75).")
    parser.add_argument("--epsilon-max-cap", type=float, default=None,
                         help="online_biobj only: absolute ceiling on the max-flow-time cap phase "
                              "2 is allowed to accept, regardless of epsilon_fraction (default: none).")
    parser.add_argument("--adaptive-alpha", type=float, default=0.2,
                         help="adaptive only: escalation search budget as a fraction of "
                              "Incremental's own predicted flow time F1, and the minimum relative "
                              "gain over F1 required to keep the escalated plan -- same knob for "
                              "both (default: 0.2 = 20%%).")
    parser.add_argument("--adaptive-max-budget", type=float, default=1200,
                         help="adaptive/hybrid only: hard ceiling on the escalation search "
                              "budget (adaptive_alpha * F1), in seconds -- keeps a rare very-large-"
                              "F1 job from running unboundedly long (default: 1200 = 20min).")
    parser.add_argument("--adaptive-f1-threshold", type=float, default=None,
                         help="hybrid only: F1 must exceed this to bother escalating at "
                              "all. Default: unset (always try escalating -- the budget already "
                              "scales with F1, so a small job gets a proportionally cheap attempt "
                              "rather than being blocked from one entirely).")
    parser.add_argument("--adaptive-f1-relative-margin", type=float, default=None,
                         help="hybrid only: alternative to --adaptive-f1-threshold's fixed cutoff "
                              "-- escalate only if F1 exceeds the MEAN estimated flow time of "
                              "currently-running jobs by more than this fraction (e.g. 0.25 = F1 "
                              "must be at least 25%% worse than the running jobs' own average). "
                              "Context-sensitive: the same F1 triggers escalation when the system "
                              "is lightly loaded but not when it's already congested. Takes "
                              "precedence over --adaptive-f1-threshold when both are set. Default: "
                              "unset (disabled).")
    parser.add_argument("--adaptive-f1-dynamic-margin", action="store_true",
                         help="hybrid only: when set, --adaptive-f1-relative-margin is a BASE "
                              "margin that SHRINKS as F1 already exceeds the running mean "
                              "(effective_margin = base / (1 + max(0, f1/mean - 1))) instead of "
                              "one fixed margin applied uniformly -- makes escalation "
                              "progressively easier to trigger as congestion grows, rather than "
                              "a bar that's too strict once the mean itself is already inflated "
                              "by congestion (observed with a fixed 25%% margin on the generated-"
                              "scenarios state-A tests).")
    parser.add_argument("--adaptive-f1-stability-cv-threshold", type=float, default=None,
                         help="hybrid only: vetoes escalation (regardless of what the margin "
                              "check says) when the running jobs' own committed/estimated flow "
                              "times are already 'stable' -- coefficient of variation (std/mean) "
                              "below this threshold. Default: unset (no veto).")
    parser.add_argument("--adaptive-new-job-objective", action="store_true",
                         help="hybrid only ('hybrid-n_j' design, 2026-10-05): every one of the 6 "
                              "parallel escalation variants solves to minimize ONLY the new "
                              "job's own flow time instead of the whole batch's max/sum, with "
                              "--adaptive-degradation-cap-pct as the only thing stopping that "
                              "from coming at an existing job's expense.")
    parser.add_argument("--adaptive-degradation-cap-pct", type=float, default=None,
                         help="hybrid only, with --adaptive-new-job-objective: hard cap on an "
                              "already-running job's flow time, as a fraction ABOVE its own "
                              "pre-escalation committed flow time (e.g. 0.20 = may grow by at "
                              "most 20%%).")
    parser.add_argument("--no-adaptive-bi-objective", action="store_true",
                         help="hybrid only: the 'hybrid-n_j' design (2026-10-05) runs every "
                              "escalation variant single-objective (phase 1 only, no transfer-"
                              "energy phase 2) instead of the class default's bi-objective "
                              "epsilon-constraint search. Pass this to match that design; omit "
                              "it to keep the class default (bi-objective, "
                              "adaptive_bi_objective=True).")
    parser.add_argument("--adaptive-gate-metric", choices=["new_job", "max", "mean"], default="new_job",
                         help="hybrid only: metric the post-escalation quality gate uses to decide "
                              "whether to keep the escalation or fall back to Incremental's own "
                              "placement. 'new_job' (default): only the new arrival's own flow "
                              "time -- can accept an escalation that helps the new job while "
                              "quietly making an already-running job worse. 'max'/'mean': flow "
                              "time across the WHOLE batch instead, reconstructing Incremental's "
                              "side as F1 for the new job + every other job's currently committed "
                              "flow time (since Incremental never touches them).")
    parser.add_argument("--adaptive-selection-metric", choices=["new_job", "max"], default="max",
                         help="hybrid only: metric _timedParallelEscalation uses to pick the best "
                              "of its 6 concurrent variants -- separate from --adaptive-gate-"
                              "metric, which only judges the ALREADY-PICKED plan against "
                              "Incremental. 'max' (default): batch-wide max flow time, the "
                              "validated-safe choice. 'new_job': the new arrival's own flow time "
                              "only -- deliberately reintroduces a known blind spot (a variant can "
                              "win by helping the new job while hurting an already-running job) "
                              "to measure it alongside --adaptive-f1-relative-margin and "
                              "--adaptive-gate-metric new_job.")
    parser.add_argument("--no-charge-thinking-time", action="store_true",
                         help="ALL approaches: DON'T charge each CSP solve's own real wall-clock "
                              "time as simulated wait (yield env.timeout(elapsed)) against the "
                              "flow time of whatever job(s) it just decided -- see "
                              "SchedulingUsingCSPOnline.charge_thinking_time's own comment. "
                              "Applies uniformly to online/online_biobj/online_warmstart/"
                              "online_biobj_warmstart/incremental/hybrid, so every approach pays "
                              "the same way for its own actual decision time. Default: off "
                              "(charge it -- the fair default; before this existed, only hybrid's "
                              "escalation charged anything at all, and only an ESTIMATE, not its "
                              "real solve time).")
    parser.add_argument("--parallel-warm-cold-escalation", action="store_true",
                         help="hybrid only: run the escalation as TWO concurrent solves -- warm-"
                              "started (the usual escalation_java_main_class, seeded via "
                              "_writeWarmStart) and cold (MainOnlineMultiObj, Choco's own default "
                              "search) -- and keep whichever finds the lower max flow time "
                              "(phase 1's own objective; energy is the tie-breaker). A warm-"
                              "started time-limited search can converge to a MUCH worse optimum "
                              "than a cold one on a bigger joint problem (confirmed: identical "
                              "15s phase-1 budget, warm-started result 2592 vs cold 1228 on the "
                              "same 4-job batch) -- this hedges against that at the cost of "
                              "roughly 2x the CPU (though not 2x the wall-clock time charged, "
                              "since both run concurrently and only the slower one's real elapsed "
                              "time is charged). Default: off (single warm-started solve only, "
                              "identical to before this existed).")
    parser.add_argument("--hybrid-incremental-time-limit", type=float, default=None,
                         help="hybrid only: solver time budget (seconds) for its internal "
                              "Incremental calls (the F1 probe and the fallback-on-failure "
                              "placement), independent of --solver-time-limit -- Incremental's "
                              "own decision is meant to be cheap, so it shouldn't have to share "
                              "the (often much larger) budget used for the joint escalation. "
                              "Default: unset (falls back to --solver-time-limit).")
    parser.add_argument("--freeze-large-jobs-threshold", type=float, default=None,
                         help="online/online_biobj/adaptive: a not-finished job already resident "
                              "somewhere whose dataset_size (MB) is at or above this threshold is "
                              "frozen for the solve -- no new replica, no move, kept exactly where "
                              "it is (see SchedulingUsingCSPOnline docs / frozen_jobs.txt). "
                              "Default: unset (no job frozen, identical to before this existed).")
    parser.add_argument("--freeze-remaining-time-threshold", type=float, default=None,
                         help="online/online_biobj/adaptive: a not-finished job already resident "
                              "somewhere whose own remaining work (nb_tasks_not_started * "
                              "task_duration) is at or below this threshold is frozen for the "
                              "solve -- close enough to finishing that reconsidering it has little "
                              "left to gain. Independent of --freeze-large-jobs-threshold; a job "
                              "frozen by either criterion is frozen. Default: unset.")
    parser.add_argument("--freeze-jobs-with-ongoing-transfer", action="store_true",
                         help="online/online_biobj/adaptive: freezes any not-finished job that "
                              "has at least one transfer currently IN FLIGHT -- it can't be "
                              "cancelled anyway, so this stops the solver from 'changing its mind' "
                              "about that job's placement mid-transfer, which otherwise leaves an "
                              "orphaned, never-cleaned-up replica once the abandoned transfer "
                              "lands (a real leak confirmed via events_history.json). Independent "
                              "of the other two --freeze-* flags. Default: off.")
    parser.add_argument("--freeze-blocks-node-until-done", action="store_true",
                         help="hybrid only: a frozen job's resident node(s) are reserved for it "
                              "until its OWN last task ends -- no other job's task may start "
                              "there any earlier. Without this, a frozen job's node is fixed but "
                              "its own task TIMING stays free, so the solver can (and does) push "
                              "the frozen job's own tasks later to make room for other jobs "
                              "sharing that node, inflating the frozen job's own flow time as "
                              "collateral damage (confirmed root cause of freeze's own quality "
                              "regression vs not freezing at all). Default: off (identical to "
                              "before this existed).")
    parser.add_argument("--reschedule-top-fraction", type=float, default=None,
                         help="online/online_biobj/adaptive: confines every already-running "
                              "job's reconsideration to the top fraction (0-1) of nodes ranked by "
                              "bandwidth/compute_capacity (a brand-new arrival is never "
                              "restricted, and a job falls back to the full storage-eligible node "
                              "set if the powerful subset can't fit its dataset). "
                              "Default: unset (no restriction, identical to before this existed).")
    return parser.parse_args()


def run(args):
    master_class = APPROACHES[args.approach]
    # Applies to every approach uniformly -- see charge_thinking_time's own comment on the
    # SchedulingUsingCSPOnline base class (inherited by all of them).
    master_class.charge_thinking_time = not args.no_charge_thinking_time
    if args.approach in ("online_biobj", "online_biobj_warmstart", "hybrid"):
        master_class.epsilon_fraction = args.epsilon_fraction
        master_class.epsilon_phase1_fraction = args.epsilon_phase1_fraction
        master_class.epsilon_max_cap = args.epsilon_max_cap
    if args.approach in ("adaptive", "hybrid"):
        master_class.adaptive_alpha = args.adaptive_alpha
        master_class.adaptive_max_budget_s = args.adaptive_max_budget
    if args.approach == "hybrid":
        master_class.adaptive_f1_threshold = args.adaptive_f1_threshold
        master_class.adaptive_f1_relative_margin = args.adaptive_f1_relative_margin
        master_class.adaptive_f1_dynamic_margin = args.adaptive_f1_dynamic_margin
        master_class.adaptive_f1_stability_cv_threshold = args.adaptive_f1_stability_cv_threshold
        master_class.adaptive_new_job_objective = args.adaptive_new_job_objective
        master_class.adaptive_degradation_cap_pct = args.adaptive_degradation_cap_pct
        master_class.adaptive_gate_metric = args.adaptive_gate_metric
        master_class.adaptive_selection_metric = args.adaptive_selection_metric
        master_class.incremental_time_limit_s = args.hybrid_incremental_time_limit
        master_class.parallel_warm_cold_escalation = args.parallel_warm_cold_escalation
        master_class.adaptive_bi_objective = not args.no_adaptive_bi_objective

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["total_nb_jobs"] = args.nb_jobs
    config["total_nb_compute_nodes"] = args.nb_nodes
    config["jobs_file_path"] = os.path.join(args.instance_dir, "jobs.json")
    config["solver_time_limit_s"] = args.solver_time_limit
    config["lambda_rate"] = args.lambda_rate
    config["adaptive_alpha"] = args.adaptive_alpha
    config["adaptive_max_budget_s"] = args.adaptive_max_budget
    config["adaptive_f1_threshold"] = args.adaptive_f1_threshold
    config["adaptive_f1_relative_margin"] = args.adaptive_f1_relative_margin
    config["adaptive_f1_dynamic_margin"] = args.adaptive_f1_dynamic_margin
    config["adaptive_f1_stability_cv_threshold"] = args.adaptive_f1_stability_cv_threshold
    config["adaptive_new_job_objective"] = args.adaptive_new_job_objective
    config["adaptive_degradation_cap_pct"] = args.adaptive_degradation_cap_pct
    config["adaptive_gate_metric"] = args.adaptive_gate_metric
    config["adaptive_selection_metric"] = args.adaptive_selection_metric
    config["adaptive_bi_objective"] = not args.no_adaptive_bi_objective
    config["parallel_warm_cold_escalation"] = args.parallel_warm_cold_escalation
    config["charge_thinking_time"] = not args.no_charge_thinking_time
    config["incremental_time_limit_s"] = args.hybrid_incremental_time_limit
    config["freeze_large_jobs_threshold_mb"] = args.freeze_large_jobs_threshold
    config["freeze_remaining_time_threshold"] = args.freeze_remaining_time_threshold
    config["freeze_jobs_with_ongoing_transfer"] = args.freeze_jobs_with_ongoing_transfer
    config["freeze_blocks_node_until_done"] = args.freeze_blocks_node_until_done
    config["reschedule_top_fraction"] = args.reschedule_top_fraction

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
                "adaptive_f1_threshold": getattr(master_class, "adaptive_f1_threshold", None),
                "parallel_warm_cold_escalation": getattr(master_class, "parallel_warm_cold_escalation", None),
                "charge_thinking_time": getattr(master_class, "charge_thinking_time", None),
                "incremental_time_limit_s": getattr(master_class, "incremental_time_limit_s", None),
                "escalation_java_main_class": getattr(master_class, "escalation_java_main_class", None),
                "freeze_large_jobs_threshold_mb": args.freeze_large_jobs_threshold,
                "freeze_remaining_time_threshold": args.freeze_remaining_time_threshold,
                "freeze_jobs_with_ongoing_transfer": args.freeze_jobs_with_ongoing_transfer,
                "reschedule_top_fraction": args.reschedule_top_fraction},
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
