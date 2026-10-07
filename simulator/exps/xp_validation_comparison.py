"""Configurable, self-contained runner for the 2026-10-07 validation plan
(plan_experimental_validation_2026-10-07.md): compares incremental / online / hybrid on 16
single-decision-point scenarios (Phase A: 3 size tiers at x2 + the mixte tier at x0.5/x2/x4/x6 /
Phase B: n_existing in {2,5,10,15} / Phase C: 5 fully random scenarios), generating its own
instances on first use (fixed seeds, deterministic -- re-running reproduces the exact same
scenarios every time).

The one thing this file makes configurable that every earlier script in this investigation
hardcoded: WHICH OBJECTIVE online/hybrid optimize --

  --objective new_job (default): minimize ONLY the new job's own flow time, with a hard cap
      stopping any already-running job from degrading by more than --degradation-cap-pct beyond
      its own currently committed flow time (the "hybrid-n_j" design validated all week).
  --objective max: minimize the WHOLE BATCH's max flow time instead (the original Online/hybrid
      design, objective_choice=1) -- no degradation cap needed, since nothing is singled out.

Incremental is unaffected by --objective (a single-job solve has no "other jobs" to weigh against
and no batch to take a max over).

Usage:
    python3 exps/xp_validation_comparison.py --objective new_job
    python3 exps/xp_validation_comparison.py --objective max
    python3 exps/xp_validation_comparison.py --objective max --scenarios phaseA,phaseB
    python3 exps/xp_validation_comparison.py --objective max --output-dir /tmp/try_it_first

Writes <output-dir>/phaseABC_full_metrics_<objective>.csv (one row per scenario x approach), and
prints a one-line summary per (scenario, approach) as it goes.
"""
import argparse
import contextlib
import copy
import csv
import io
import json
import logging
import os
import random
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(SCRIPT_DIR)
if SIMULATOR_DIR not in sys.path:
    sys.path.insert(0, SIMULATOR_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from simulator import configure_logging
configure_logging(logging.WARNING)

from xp_simultaneous_sweep import generate_existing_jobs, draw_new_job_sizes, make_new_job_raw, write_instance, NB_TASKS_RANGE
from xp_dataset_size_sweep import (build_state_a, run_incremental_style, run_online_style,
                                    run_online_newjob_style, run_hybrid_style, make_new_job)

DEFAULT_OUTPUT_DIR = os.path.join(SIMULATOR_DIR, "results-validation-2026-10-07")
DEFAULT_INSTANCES_DIR = os.path.join(SIMULATOR_DIR, "workloads", "validation_phaseABC")

STATE_A_TIME_LIMIT = 30
SOLVER_TIME_LIMIT = 30         # incremental's own budget, and online's (both objectives)
HYBRID_F1_TIME_LIMIT = 15
HYBRID_MAX_BUDGET = 30
STORAGE_CEILING_FACTOR = 1.3

CONFIG = {
    "homogeneous": False, "overlap": True, "threshold": 1, "use_minizinc_model": False,
    "compute_node_bw_MBps": 100, "compute_node_latency_ms": 1,
}

TIERS = {
    "petit": {"task_duration_range": (100, 150), "dataset_size_range": (2048, 20480)},
    "grand": {"task_duration_range": (250, 300), "dataset_size_range": (102400, 204800)},
    "mixte": {"task_duration_range": (100, 300), "dataset_size_range": (2048, 204800)},
}


def scaled(base, scale):
    return (int(base[0] * scale), int(base[1] * scale))


def build_scenarios():
    """Returns the full 16-scenario list: (phase, group, tag, n_existing, nb_nodes, seed,
    task_duration_range, dataset_size_range). Matches plan_experimental_validation_2026-10-07.md
    exactly -- same seeds, same ranges, so results from any --objective are directly comparable
    to the already-documented new_job-objective numbers."""
    scenarios = []
    for i, tier_name in enumerate(["petit", "grand", "mixte"]):
        seed = 200 + i
        tdr = scaled(TIERS[tier_name]["task_duration_range"], 2)
        dsr = scaled(TIERS[tier_name]["dataset_size_range"], 2)
        scenarios.append(("A", "palier", tier_name, 5, 50, seed, tdr, dsr))

    mixte_scale_seeds = {0.5: 300, 2: 202, 4: 301, 6: 302}
    for scale, seed in mixte_scale_seeds.items():
        tag = f"mixte_x{scale}".rstrip("0").rstrip(".") if scale != int(scale) else f"mixte_x{int(scale)}"
        tdr = scaled(TIERS["mixte"]["task_duration_range"], scale)
        dsr = scaled(TIERS["mixte"]["dataset_size_range"], scale)
        scenarios.append(("A", "echelle", tag, 5, 50, seed, tdr, dsr))

    mixte_x2_tdr = scaled(TIERS["mixte"]["task_duration_range"], 2)
    mixte_x2_dsr = scaled(TIERS["mixte"]["dataset_size_range"], 2)
    for n_existing, seed in [(2, 400), (5, 202), (10, 401), (15, 402)]:
        scenarios.append(("B", "n_existing", f"n{n_existing}", n_existing, 50, seed, mixte_x2_tdr, mixte_x2_dsr))

    random_specs = [
        (0, 24, 10, (198, 367), (17777, 189369)),
        (1, 30, 6, (210, 350), (16813, 189397)),
        (2, 26, 7, (93, 304), (24083, 220491)),
        (3, 15, 4, (140, 374), (13615, 117573)),
        (4, 24, 15, (196, 339), (1834, 57039)),
    ]
    for i, nb_nodes, n_existing, tdr, dsr in random_specs:
        scenarios.append(("C", "aleatoire", f"s{i}", n_existing, nb_nodes, 500 + i, tdr, dsr))

    return scenarios


def ensure_instance(instance_dir, n_existing, nb_nodes, seed, task_duration_range, dataset_size_range,
                     phase, new_job_floor=None):
    """Builds the instance deterministically from (seed, ranges) if not already on disk. Phase C's
    random scenarios enforce a floor on the new job's own size/duration (never small) -- see
    plan_experimental_validation_2026-10-07.md's Phase C section."""
    if os.path.exists(os.path.join(instance_dir, "jobs.json")):
        return
    os.makedirs(instance_dir, exist_ok=True)
    storage_ceiling = int(dataset_size_range[1] * STORAGE_CEILING_FACTOR)
    rng = random.Random(seed if phase != "C" else seed * 7 + 1)
    existing_jobs_raw = generate_existing_jobs(rng, n_existing, task_duration_range, dataset_size_range)
    new_nb_tasks, new_task_duration, new_dataset_size = draw_new_job_sizes(
        rng, NB_TASKS_RANGE, task_duration_range, dataset_size_range, mode="upper-quarter")
    if new_job_floor is not None:
        new_task_duration = max(new_task_duration, new_job_floor["task_duration"])
        new_dataset_size = max(new_dataset_size, new_job_floor["dataset_size"])
    placeholder_new_job = make_new_job_raw(n_existing, new_nb_tasks, new_task_duration, new_dataset_size, arriving_time=1.0)
    write_instance(instance_dir, existing_jobs_raw, placeholder_new_job, rng, nb_nodes, storage_ceiling)


def ghost_entries_for(master, batch_job_ids):
    entries = []
    for jid, node_ids in master.replicas_locations.items():
        if jid in batch_job_ids:
            continue
        size = master.jobs[jid].dataset_size
        for node_id in node_ids:
            deletion_time = None
            for pending_jid, pending_time in master.deletions.get(f'node_{node_id}', []):
                if pending_jid == jid:
                    deletion_time = pending_time
                    break
            entries.append((node_id, size, deletion_time))
    return entries


def verify_plan_storage(plan, job_size, node_capacity, ghost_entries):
    events_per_node = {}
    for node_id, size, deletion_time in ghost_entries:
        events_per_node.setdefault(node_id, []).append((-1.0, size, "+ghost"))
        if deletion_time is not None:
            events_per_node[node_id].append((deletion_time, -size, "-ghost"))
    arrival = {}
    for key, entries in plan.get("transfers", {}).items():
        node = int(key.split('_')[1])
        for job_id, node_index, start_abs, end_abs, duration in entries:
            k = (node, job_id)
            if k not in arrival or end_abs < arrival[k]:
                arrival[k] = end_abs
    for (node, job_id), t_in in arrival.items():
        size = job_size.get(job_id)
        if size is not None:
            events_per_node.setdefault(node, []).append((t_in, size, f"+job{job_id}"))
    for key, entries in plan.get("deletions", {}).items():
        node = int(key.split('_')[1])
        for job_id, deletion_time in entries:
            size = job_size.get(job_id)
            if size is not None:
                events_per_node.setdefault(node, []).append((deletion_time, -size, f"-job{job_id}"))
    violations = 0
    for node, evts in events_per_node.items():
        evts.sort(key=lambda e: e[0])
        occupied = 0.0
        cap = node_capacity.get(node, float("inf"))
        for t, delta, label in evts:
            occupied += delta
            if occupied > cap + 1e-6:
                violations += 1
    return violations


def plan_metrics(plan, new_job_id):
    nb_transfers_total = sum(len(v) for v in plan.get("transfers", {}).values())
    nb_transfers_new_job = 0
    nodes_new_job, nodes_all = set(), set()
    for key, entries in plan.get("transfers", {}).items():
        node = int(key.split('_')[1])
        for job_id, node_index, start_abs, end_abs, duration in entries:
            nodes_all.add(node)
            if job_id == new_job_id:
                nb_transfers_new_job += 1
                nodes_new_job.add(node)
    for key, entries in plan.get("works", {}).items():
        node = int(key.split('_')[1])
        for job_id, node_index, task_index, start_abs, end_abs, duration in entries:
            nodes_all.add(node)
            if job_id == new_job_id:
                nodes_new_job.add(node)
    return {
        "nb_transfers_total": nb_transfers_total, "nb_transfers_new_job": nb_transfers_new_job,
        "nb_nodes_used_new_job": len(nodes_new_job), "nb_nodes_used_total": len(nodes_all),
    }


def run_scenario(phase, group, tag, n_existing, nb_nodes, seed, task_duration_range, dataset_size_range,
                  objective, degradation_cap_pct, instances_dir, results_dir, rows):
    full_tag = f"{phase}_{group}_{tag}"
    print(f"\n{'='*70}\n### {full_tag}  (objective={objective}) ###\n{'='*70}", flush=True)

    instance_dir = os.path.join(instances_dir, full_tag)
    new_job_floor = {"task_duration": 200, "dataset_size": 51200} if phase == "C" else None
    ensure_instance(instance_dir, n_existing, nb_nodes, seed, task_duration_range, dataset_size_range,
                    phase, new_job_floor)

    args = argparse.Namespace(
        instance_dir=instance_dir, n_existing=n_existing, new_job_index=n_existing,
        arrival_lambda=None, seed=seed, nb_nodes=nb_nodes,
        state_a_time_limit=STATE_A_TIME_LIMIT, lambda_rate=100,
    )
    run_dir = os.path.join(results_dir, ".state_a_scratch", full_tag)
    os.makedirs(run_dir, exist_ok=True)

    master_probe, _, _ = build_state_a(CONFIG, args, run_dir, "full", dataset_size_range, 0)
    max_flow = max(master_probe._state_a_finish.values())
    new_arrival = max_flow / 2.0 + 200.0
    with open(os.path.join(instance_dir, "jobs.json")) as f:
        jobs = json.load(f)
    jobs[n_existing]["arriving_time"] = new_arrival
    with open(os.path.join(instance_dir, "jobs.json"), "w") as f:
        json.dump(jobs, f)

    master, new_job_raw, not_finished_jobs = build_state_a(CONFIG, args, run_dir, "full", dataset_size_range, 0)
    not_finished_ids = {j.job_id for j in not_finished_jobs} | {new_job_raw["job_id"]}
    now = new_job_raw["arriving_time"]
    replicas_locations = master.replicas_locations
    node_capacity = {i: cn.storage_capacity for i, cn in enumerate(master.compute_nodes)}
    job_size = {j.job_id: j.dataset_size for j in master.jobs}
    job_size[new_job_raw["job_id"]] = new_job_raw["dataset_size"]

    approaches = ["incremental", "online", "hybrid"]
    for approach in approaches:
        m = copy.copy(master)
        m._config = dict(getattr(master, "_config", {}))
        nj = make_new_job(new_job_raw, new_job_raw["dataset_size"], now)

        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            if approach == "incremental":
                res = run_incremental_style(m, nj, not_finished_ids, nb_nodes, now, replicas_locations,
                                             not_finished_jobs=not_finished_jobs)
                batch_job_ids = {nj.job_id}
            elif approach == "online":
                m._config['solver_time_limit_s'] = SOLVER_TIME_LIMIT
                if objective == "max":
                    res = run_online_style(m, nj, not_finished_ids, nb_nodes, now, replicas_locations, not_finished_jobs)
                else:
                    res = run_online_newjob_style(m, nj, not_finished_ids, nb_nodes, now, replicas_locations,
                                                   not_finished_jobs, degradation_cap_pct=degradation_cap_pct,
                                                   solver_time_limit=SOLVER_TIME_LIMIT)
                batch_job_ids = {nj.job_id} | {j.job_id for j in not_finished_jobs}
            else:
                if objective == "max":
                    res = run_hybrid_style(m, nj, not_finished_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                                            hybrid_alpha=0.25, hybrid_max_budget=HYBRID_MAX_BUDGET,
                                            hybrid_incremental_time_limit=HYBRID_F1_TIME_LIMIT,
                                            epsilon_fraction=0.15, epsilon_phase1_fraction=0.5,
                                            adaptive_gate_metric='max', adaptive_selection_metric='max',
                                            adaptive_new_job_objective=False, adaptive_degradation_cap_pct=None,
                                            adaptive_bi_objective=False)
                else:
                    res = run_hybrid_style(m, nj, not_finished_ids, nb_nodes, now, replicas_locations, not_finished_jobs,
                                            hybrid_alpha=0.25, hybrid_max_budget=HYBRID_MAX_BUDGET,
                                            hybrid_incremental_time_limit=HYBRID_F1_TIME_LIMIT,
                                            epsilon_fraction=0.15, epsilon_phase1_fraction=0.5,
                                            adaptive_gate_metric='new_job', adaptive_selection_metric='new_job',
                                            adaptive_new_job_objective=True, adaptive_degradation_cap_pct=degradation_cap_pct,
                                            adaptive_bi_objective=False)
                batch_job_ids = {nj.job_id} | {j.job_id for j in not_finished_jobs}
        elapsed = time.time() - t0
        captured = buf.getvalue()

        if res is None:
            print(f"  [{approach}] NO SOLUTION", flush=True)
            rows.append({"phase": phase, "group": group, "tag": tag, "objective": objective,
                         "n_existing": n_existing, "nb_nodes": nb_nodes, "approach": approach, "no_solution": True})
            continue

        plan = res.pop("_plan")
        pm = plan_metrics(plan, nj.job_id)
        ghosts = ghost_entries_for(master, batch_job_ids)
        violations = verify_plan_storage(plan, job_size, node_capacity, ghosts)

        winning_variant = None
        hybrid_accepted = res.get("hybrid_escalation_budget") is not None
        if approach == "hybrid":
            m_ = re.search(r"PARALLEL ESCALATION WINNER: (\S+)", captured)
            if m_:
                winning_variant = m_.group(1)

        row = {
            "phase": phase, "group": group, "tag": tag, "objective": objective,
            "n_existing": n_existing, "nb_nodes": nb_nodes,
            "new_job_dataset_size": new_job_raw["dataset_size"], "new_job_nb_tasks": new_job_raw["nb_tasks"],
            "new_job_task_duration": new_job_raw["task_duration"],
            "dataset_size_range_lo": dataset_size_range[0], "dataset_size_range_hi": dataset_size_range[1],
            "approach": approach,
            "flow_time_new_job": res["flow_time_new_job"],
            "mean_flow_time_all": res.get("mean_flow_time_all"),
            "max_flow_time_all": res.get("max_flow_time_all"),
            "scheduling_time_s": elapsed,
            "transfer_energy_total": res["transfer_energy_total"],
            "batch_size": len(res.get("jobs_to_reschedule", [])),
            "nb_transfers_total": pm["nb_transfers_total"],
            "nb_transfers_new_job": pm["nb_transfers_new_job"],
            "nb_nodes_used_new_job": pm["nb_nodes_used_new_job"],
            "nb_nodes_used_total": pm["nb_nodes_used_total"],
            "data_transferred_new_job_mb": pm["nb_transfers_new_job"] * new_job_raw["dataset_size"],
            "storage_violations": violations,
            "hybrid_f1": res.get("hybrid_f1"),
            "hybrid_escalation_budget": res.get("hybrid_escalation_budget"),
            "hybrid_accepted": hybrid_accepted if approach == "hybrid" else None,
            "hybrid_winning_variant": winning_variant,
            "no_solution": False,
        }
        rows.append(row)
        print(f"  [{approach}] flow_time_new_job={row['flow_time_new_job']} max_flow_time_all={row['max_flow_time_all']} "
              f"sched_time={elapsed:.1f}s storage_violations={violations} winning_variant={winning_variant}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["new_job", "max"], default="new_job",
                   help="'new_job' (default): online/hybrid minimize only the new job's own flow "
                        "time under --degradation-cap-pct. 'max': minimize the whole batch's max "
                        "flow time instead (no degradation cap).")
    p.add_argument("--degradation-cap-pct", type=float, default=0.25,
                   help="Only used with --objective new_job (default: 0.25 = 25%%).")
    p.add_argument("--scenarios", default="phaseA,phaseB,phaseC",
                   help="Comma-separated subset of {phaseA,phaseB,phaseC} to run (default: all).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write the result CSV (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--instances-dir", default=DEFAULT_INSTANCES_DIR,
                   help=f"Where generated instances live/are cached (default: {DEFAULT_INSTANCES_DIR}).")
    args = p.parse_args()

    wanted_phases = {s.strip().replace("phase", "").upper() for s in args.scenarios.split(",")}
    scenarios = [s for s in build_scenarios() if s[0] in wanted_phases]

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for phase, group, tag, n_existing, nb_nodes, seed, tdr, dsr in scenarios:
        run_scenario(phase, group, tag, n_existing, nb_nodes, seed, tdr, dsr,
                     args.objective, args.degradation_cap_pct, args.instances_dir, args.output_dir, rows)

    out_csv = os.path.join(args.output_dir, f"phaseABC_full_metrics_{args.objective}.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, None)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n### written {len(rows)} rows to {out_csv} ###", flush=True)


if __name__ == "__main__":
    main()
