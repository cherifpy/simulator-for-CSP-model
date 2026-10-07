#!/usr/bin/env python3
"""Reads run_plan.csv and submits one oarsub job per row to Grid5000, via
exps/xp_online_grid5000.py -- the same per-approach, per-instance launch convention used by
submit_online_vs_biobj_workload_grid5000.sh, generalized to a CSV so a whole batch of runs (any
mix of instances/approaches/params) can be edited in one place and launched in one call.

Run this ON the Grid5000 frontend, from inside the simulator/ directory, AFTER sim.zip has been
deployed and setup_grid5000.sh has already run at least once (venv + compiled Java model in
place) -- same prerequisites as the existing submit_*.sh scripts.

DRY RUN BY DEFAULT: prints exactly what would be submitted (the resolved command for every row)
without calling oarsub. Pass --submit to actually launch. This mirrors the project's own
convention of negotiating parameters before launching anything real.

Usage:
    python3 launch_from_plan.py run_plan.csv                 # dry run -- just show commands
    python3 launch_from_plan.py run_plan.csv --submit         # actually submit to oarsub
    python3 launch_from_plan.py run_plan.csv --submit --rows 1,3   # only submit rows 1 and 3 (1-indexed, header excluded)

CSV columns (see run_plan.csv for a filled-in example):
    label                          free text, your own note -- not passed to the simulator
    approach                       incremental | online_biobj | hybrid | online |
                                    online_warmstart | online_biobj_warmstart | adaptive |
                                    incremental_free_nodes_only
    instance_name                  e.g. inst-20J-50N (must exist under
                                    workloads/workloads-100-for_storage_constraintes/)
    nb_jobs, nb_nodes              must match the instance's own jobs.json / infrastructure.csv
    lambda_rate                    Poisson inter-arrival mean
    solver_time_limit              CSP solver budget per replan, seconds
    adaptive_max_budget            hybrid/adaptive only -- escalation budget ceiling, seconds.
                                    Leave blank for other approaches.
    adaptive_alpha                 hybrid/adaptive only -- escalation budget fraction of F1 and
                                    acceptance-gain bar. Leave blank to use the CLI default (0.2).
    adaptive_f1_threshold          hybrid only -- F1 must exceed this to bother escalating at
                                    all. Leave blank to always escalate.
    hybrid_incremental_time_limit  hybrid only -- budget for its internal Incremental calls.
                                    Leave blank for other approaches.
    parallel_warm_cold_escalation  hybrid only -- TRUE to run the 4-way concurrent escalation
                                    (freeze_below_mean/freeze_above_mean/nofreeze/warm_nofreeze),
                                    FALSE/blank for the single warm-started solve.
    freeze_jobs_with_ongoing_transfer  TRUE/FALSE -- freeze any not-finished job with a transfer
                                    currently in flight for the solve.
    no_charge_thinking_time        TRUE/FALSE -- whether to exempt each solve's own real
                                    wall-clock cost from being charged against simulated time.
    epsilon_fraction               online_biobj/hybrid only, default 0.1 if blank.
    epsilon_phase1_fraction        online_biobj/hybrid only, default 0.5 if blank.
    seed                           default 42 if blank.
    walltime                       oarsub walltime for this one job, HH:MM:SS.
    results_subdir                 results land at results-grid5000/<results_subdir>/<approach>/
                                    (and the private run dir at .run_cwd/<results_subdir>/<approach>/)
"""
import argparse
import csv
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../simulator-for-CSP-model/simulator
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, "env", "bin", "python3")

BOOL_TRUE = {"true", "1", "yes", "y"}


def as_bool(s):
    return (s or "").strip().lower() in BOOL_TRUE


def build_command(row):
    approach = row["approach"].strip()
    instance_name = row["instance_name"].strip()
    instance_dir = os.path.join(SCRIPT_DIR, "workloads", "workloads-100-for_storage_constraintes", instance_name)
    results_subdir = row["results_subdir"].strip()
    results_dir = os.path.join(SCRIPT_DIR, "results-grid5000", results_subdir, approach)
    run_cwd = os.path.join(SCRIPT_DIR, ".run_cwd", results_subdir, approach)

    seed = row.get("seed", "").strip() or "42"

    cmd = (
        f"SIMULATOR_RUN_CWD={run_cwd} {VENV_PYTHON} {SCRIPT_DIR}/exps/xp_online_grid5000.py"
        f" --approach {approach} --instance-dir {instance_dir}"
        f" --nb-jobs {row['nb_jobs'].strip()} --nb-nodes {row['nb_nodes'].strip()}"
        f" --solver-time-limit {row['solver_time_limit'].strip()}"
        f" --lambda-rate {row['lambda_rate'].strip()} --seed {seed}"
        f" --results-dir {results_dir}"
    )

    if approach in ("online_biobj", "online_biobj_warmstart", "hybrid"):
        eps = row.get("epsilon_fraction", "").strip() or "0.1"
        eps_p1 = row.get("epsilon_phase1_fraction", "").strip() or "0.5"
        cmd += f" --epsilon-fraction {eps} --epsilon-phase1-fraction {eps_p1}"

    if approach in ("adaptive", "hybrid"):
        budget = row.get("adaptive_max_budget", "").strip()
        if budget:
            cmd += f" --adaptive-max-budget {budget}"
        alpha = row.get("adaptive_alpha", "").strip()
        if alpha:
            cmd += f" --adaptive-alpha {alpha}"

    if approach == "hybrid":
        hybrid_inc = row.get("hybrid_incremental_time_limit", "").strip()
        if hybrid_inc:
            cmd += f" --hybrid-incremental-time-limit {hybrid_inc}"
        if as_bool(row.get("parallel_warm_cold_escalation")):
            cmd += " --parallel-warm-cold-escalation"
        f1_threshold = row.get("adaptive_f1_threshold", "").strip()
        if f1_threshold:
            cmd += f" --adaptive-f1-threshold {f1_threshold}"
        f1_margin = row.get("adaptive_f1_relative_margin", "").strip()
        if f1_margin:
            cmd += f" --adaptive-f1-relative-margin {f1_margin}"
        if as_bool(row.get("adaptive_f1_dynamic_margin")):
            cmd += " --adaptive-f1-dynamic-margin"
        if as_bool(row.get("freeze_blocks_node_until_done")):
            cmd += " --freeze-blocks-node-until-done"
        gate_metric = row.get("adaptive_gate_metric", "").strip()
        if gate_metric:
            cmd += f" --adaptive-gate-metric {gate_metric}"
        selection_metric = row.get("adaptive_selection_metric", "").strip()
        if selection_metric:
            cmd += f" --adaptive-selection-metric {selection_metric}"

    if as_bool(row.get("freeze_jobs_with_ongoing_transfer")):
        cmd += " --freeze-jobs-with-ongoing-transfer"

    if as_bool(row.get("no_charge_thinking_time")):
        cmd += " --no-charge-thinking-time"

    return cmd, results_dir, run_cwd, instance_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the run plan CSV (see run_plan.csv for the format).")
    parser.add_argument("--submit", action="store_true",
                         help="Actually call oarsub. Without this flag, only prints what would be submitted.")
    parser.add_argument("--rows", default=None,
                         help="Comma-separated 1-indexed row numbers (header excluded) to act on. "
                              "Default: every row in the CSV.")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    selected = set(int(x) for x in args.rows.split(",")) if args.rows else None

    print(f"### Run plan: {args.csv_path} ({len(rows)} row(s))")
    print(f"### Mode: {'SUBMIT' if args.submit else 'DRY RUN (pass --submit to actually launch)'}")
    print()

    for i, row in enumerate(rows, start=1):
        if selected is not None and i not in selected:
            continue

        label = row.get("label", "").strip() or f"row {i}"
        if not row.get("instance_name", "").strip():
            print(f"[{i}] SKIPPED ({label}): empty row")
            continue

        instance_dir_check = os.path.join(
            SCRIPT_DIR, "workloads", "workloads-100-for_storage_constraintes", row["instance_name"].strip()
        )
        if not os.path.isfile(os.path.join(instance_dir_check, "jobs.json")):
            print(f"[{i}] ERROR ({label}): instance not found at {instance_dir_check} (no jobs.json)", file=sys.stderr)
            continue

        cmd, results_dir, run_cwd, instance_dir = build_command(row)
        walltime = row.get("walltime", "").strip() or "01:00:00"

        print(f"[{i}] {label}")
        print(f"    approach={row['approach']}  instance={row['instance_name']}  walltime={walltime}")
        print(f"    results -> {results_dir}")
        print(f"    cmd: {cmd}")

        if args.submit:
            os.makedirs(results_dir, exist_ok=True)
            if os.path.isdir(run_cwd):
                subprocess.run(["rm", "-rf", run_cwd], check=True)
            os.makedirs(os.path.join(run_cwd, "utils", "model", "inputs"), exist_ok=True)
            os.makedirs(os.path.join(run_cwd, "utils", "model", "outputs"), exist_ok=True)
            os.makedirs(os.path.join(run_cwd, "utils", "model", "bin"), exist_ok=True)
            subprocess.run(["ln", "-sfn", os.path.join(SCRIPT_DIR, "utils", "model", "lib"),
                             os.path.join(run_cwd, "utils", "model", "lib")], check=True)
            subprocess.run(["ln", "-sfn", os.path.join(SCRIPT_DIR, "utils", "model", "src"),
                             os.path.join(run_cwd, "utils", "model", "src")], check=True)

            result = subprocess.run(["oarsub", "-l", f"host=1,walltime={walltime}", cmd],
                                     capture_output=True, text=True)
            print(f"    oarsub stdout: {result.stdout.strip()}")
            if result.returncode != 0:
                print(f"    oarsub stderr: {result.stderr.strip()}", file=sys.stderr)
        print()

    if not args.submit:
        print("### Nothing submitted (dry run). Re-run with --submit once this looks right.")
    else:
        print("### Submission done. Check status with: oarstat -u $(whoami)")


if __name__ == "__main__":
    main()
