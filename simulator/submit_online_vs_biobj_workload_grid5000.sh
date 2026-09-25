#!/usr/bin/env bash
# Submits a full live-simulation workload (lambda_rate=100, 20 jobs, 50 nodes, 10min solver
# budget per replan -- same setup used before for online-vs-incremental comparisons, via
# exps/xp_online_grid5000.py) as separate oarsub jobs running in parallel, one per approach:
#   - "online":       mono-objective Online, max flow time (MainOnline.java)
#   - "online_biobj": bi-objective Online, epsilon-constraint phase1=max flow time /
#                      phase2=transfer energy (MainOnlineMultiObj.java), same total solver
#                      budget as "online", split 50/50 between the two phases by default.
#   - "adaptive":     Incremental-first escalation (SchedulingUsingCSPAdaptive): each job is
#                      placed by Incremental first (F1), then optionally escalated to
#                      online_biobj's MainOnlineMultiObj for that same job, budgeted at
#                      adaptive_alpha * F1 and kept only if it beats F1 by more than
#                      adaptive_alpha. Uses --solver-time-limit only as the Incremental probe's
#                      own budget (Incremental converges fast; the real solver-time knob for
#                      this approach is --adaptive-alpha, since the escalation budget scales
#                      with each job's own F1, not with a fixed wall-clock limit).
#   - "online_biobj_warmstart": online_biobj's epsilon-constraint bi-objective solve, but warm-
#                      started (this scheduler's own last-decided plan for known jobs + a
#                      throwaway Incremental solve for the brand-new job(s)) -- see
#                      SchedulingUsingCSPOnlineMultiObjWarmStart's docstring. This is the ONLY
#                      approach --pre-process turns pre-processing (job freezing) on for: a job
#                      already resident somewhere gets frozen (confined to its current nodes, no
#                      new replica) if its dataset is large, if it's close to finishing, or if it
#                      has a transfer currently in flight (which can't be cancelled anyway -- see
#                      --freeze-jobs-with-ongoing-transfer's docstring in xp_online_grid5000.py
#                      for the real leak this prevents). Trims the search space so the solve
#                      itself is faster, at some cost in solution quality -- see PRE_PROCESS
#                      below for the exact thresholds used.
#   - "hybrid":        SchedulingUsingCSPAdaptiveJoint: Incremental-first gate (F1), and only when
#                      F1 looks high does it escalate to a FULL joint replan (new job + every
#                      already-running job) via online_biobj_warmstart -- pre-processing (the same
#                      freeze_* settings as online_biobj_warmstart's --pre-process) is ALWAYS
#                      applied here, since it's core to the design, not an opt-in comparison knob.
#                      Falls back to Incremental for just the new job if the joint replan finds no
#                      solution -- confirmed necessary: pre-processing can make that joint solve
#                      genuinely infeasible even with no shared node between the frozen jobs.
#
# Same isolation rationale as submit_nexisting_sweep.sh / submit_dataset_size_sweep_grid5000.sh:
# schedulingUsingJavaCSP exchanges data with the Java solver through FIXED file paths under
# utils/model/{inputs,outputs}, and compiles into utils/model/bin -- there is no per-call
# uniqueness. Running "online" and "online_biobj" at the same time from the same checkout (even
# across different physical Grid5000 nodes, since $HOME is usually NFS-shared site-wide) would
# clobber each other's exchange files mid-solve. The fix: give each approach its own private
# utils/model/{inputs,outputs,bin} (real, writable) plus symlinks back to the canonical
# utils/model/{lib,src} (read-only, safe to share), pointed at via SIMULATOR_RUN_CWD. Done
# automatically below -- nothing to set up by hand.
#
# IMPORTANT -- this is the FULL LIVE SIMULATION methodology (continuous SimPy run with Poisson
# job arrivals via jobsInjectorBasedOnLambdaPoisson, one CSP replan per new arrival), NOT the
# frozen-state-A single-decision-point protocol used by xp_dataset_size_sweep.py /
# submit_dataset_size_sweep_grid5000.sh. Results land as the classic infos_on_jobs.csv /
# infos_on_tasks.csv / infos_on_transfers_energy.csv / events_history.json / gantt.png set
# (see exps/xp_online_grid5000.py's save_results_to_csv call), one full set per approach.
#
# Usage (run from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --walltime 05:00:00
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --epsilon-fraction 0.15
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --approaches online_biobj   # relaunch just one
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --approaches "online online_biobj adaptive incremental"
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --approaches adaptive --adaptive-alpha 0.2
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --solver-time-limit 120 \
#       --approaches "online_biobj_warmstart online_biobj incremental" --pre-process
#   ./simulator/submit_online_vs_biobj_workload_grid5000.sh --solver-time-limit 120 \
#       --approaches "hybrid online_biobj incremental"
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place),
# and inst-20J-50N present under workloads/.

set -euo pipefail

WALLTIME="04:00:00"
SOLVER_TIME_LIMIT=600   # 10 min per replan, shared total budget for both approaches
NB_JOBS=20
NB_NODES=50
LAMBDA_RATE=100
INSTANCE_NAME="inst-20J-50N"
SEED=42
EPSILON_FRACTION=0.1
EPSILON_PHASE1_FRACTION=0.5
EPSILON_MAX_CAP=""
ADAPTIVE_ALPHA=0.2
ADAPTIVE_MAX_BUDGET=300   # ceiling on the escalation search budget (adaptive_alpha * F1)
HYBRID_INCREMENTAL_TIME_LIMIT=30   # hybrid only: budget for its internal Incremental calls (F1 probe + fallback)
# Pre-processing (job freezing), applied ONLY to online_biobj_warmstart when --pre-process is
# passed. Defaults calibrated to inst-20J-50N's own dataset_size distribution (1024-10240 MB):
# 7000 freezes only its two largest sizes (7168, 10240), not everything.
PRE_PROCESS=0
FREEZE_LARGE_JOBS_THRESHOLD=7000
FREEZE_REMAINING_TIME_THRESHOLD=300
APPROACHES="online online_biobj"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --solver-time-limit) SOLVER_TIME_LIMIT="$2"; shift 2 ;;
        --nb-jobs) NB_JOBS="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --lambda-rate) LAMBDA_RATE="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --epsilon-fraction) EPSILON_FRACTION="$2"; shift 2 ;;
        --epsilon-phase1-fraction) EPSILON_PHASE1_FRACTION="$2"; shift 2 ;;
        --epsilon-max-cap) EPSILON_MAX_CAP="$2"; shift 2 ;;
        --adaptive-alpha) ADAPTIVE_ALPHA="$2"; shift 2 ;;
        --adaptive-max-budget) ADAPTIVE_MAX_BUDGET="$2"; shift 2 ;;
        --hybrid-incremental-time-limit) HYBRID_INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --pre-process) PRE_PROCESS=1; shift 1 ;;
        --freeze-large-jobs-threshold) FREEZE_LARGE_JOBS_THRESHOLD="$2"; shift 2 ;;
        --freeze-remaining-time-threshold) FREEZE_REMAINING_TIME_THRESHOLD="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                      # .../simulator-for-CSP-model
VENV_PYTHON="$PROJECT_ROOT/env/bin/python3"
INSTANCE_DIR="$SCRIPT_DIR/workloads/workloads-100-for_storage_constraintes/$INSTANCE_NAME"
RESULTS_ROOT="$SCRIPT_DIR/results-grid5000/online_vs_biobj_workload"
RUN_CWD_ROOT="$SCRIPT_DIR/.run_cwd/online_vs_biobj_workload"   # private per-run utils/model trees

if [[ ! -f "$INSTANCE_DIR/jobs.json" ]]; then
    echo "ERROR: instance not found at $INSTANCE_DIR (expected jobs.json there)." >&2
    exit 1
fi

# Unlike submit_dataset_size_sweep_grid5000.sh / submit_nexisting_sweep.sh, this script calls
# xp_online_grid5000.py DIRECTLY rather than through launcher.py -- launcher.py has its own
# auto-reexec logic that always switches to the venv's interpreter, even if invoked with a bare
# `python3` from a broken/absent shell rc file. This script has no such fallback, so it needs the
# venv's own python3 explicitly (relying on plain `python3` from PATH would silently pick up the
# system interpreter -- lacking simpy/pandas/etc -- whenever the venv isn't already active).
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VENV_PYTHON. Run setup_grid5000.sh (or launcher.py " >&2
    echo "once without --skip-setup) first to create it." >&2
    exit 1
fi

echo "### Submitting approaches: $APPROACHES"
echo "### Instance: $INSTANCE_DIR ($NB_JOBS jobs / $NB_NODES nodes, lambda_rate=$LAMBDA_RATE)"
echo "### Solver time budget: ${SOLVER_TIME_LIMIT}s per replan (online_biobj splits it "
echo "###   ${EPSILON_PHASE1_FRACTION} / $(python3 -c "print(1 - $EPSILON_PHASE1_FRACTION)") between phase1/phase2)"
echo "### epsilon_fraction=$EPSILON_FRACTION  epsilon_max_cap=${EPSILON_MAX_CAP:-<none>}"
echo "### adaptive_alpha=$ADAPTIVE_ALPHA (budget and acceptance-gain fraction, adaptive only)"
echo "### adaptive_max_budget=${ADAPTIVE_MAX_BUDGET}s (ceiling on the escalation budget, adaptive/hybrid)"
echo "### hybrid_incremental_time_limit=${HYBRID_INCREMENTAL_TIME_LIMIT}s (hybrid's internal Incremental budget)"
if [[ "$PRE_PROCESS" == "1" ]]; then
    echo "### pre-processing ON for online_biobj_warmstart: freeze_large_jobs_threshold=${FREEZE_LARGE_JOBS_THRESHOLD}MB"
    echo "###   freeze_remaining_time_threshold=${FREEZE_REMAINING_TIME_THRESHOLD}  freeze_jobs_with_ongoing_transfer=on"
else
    echo "### pre-processing OFF (pass --pre-process to freeze jobs for online_biobj_warmstart)"
fi
echo "### Walltime per job: $WALLTIME"
echo ""

for APPROACH in $APPROACHES; do
    RESULTS_DIR="$RESULTS_ROOT/$APPROACH"
    RUN_CWD="$RUN_CWD_ROOT/$APPROACH"
    mkdir -p "$RESULTS_DIR"

    rm -rf "$RUN_CWD"
    mkdir -p "$RUN_CWD/utils/model/inputs" "$RUN_CWD/utils/model/outputs" "$RUN_CWD/utils/model/bin"
    ln -sfn "$SCRIPT_DIR/utils/model/lib" "$RUN_CWD/utils/model/lib"
    ln -sfn "$SCRIPT_DIR/utils/model/src" "$RUN_CWD/utils/model/src"

    CMD="SIMULATOR_RUN_CWD=$RUN_CWD $VENV_PYTHON $SCRIPT_DIR/exps/xp_online_grid5000.py --approach $APPROACH"
    CMD+=" --instance-dir $INSTANCE_DIR --nb-jobs $NB_JOBS --nb-nodes $NB_NODES"
    CMD+=" --solver-time-limit $SOLVER_TIME_LIMIT --lambda-rate $LAMBDA_RATE --seed $SEED"
    CMD+=" --results-dir $RESULTS_DIR"
    if [[ "$APPROACH" == "online_biobj" || "$APPROACH" == "online_biobj_warmstart" || "$APPROACH" == "hybrid" ]]; then
        CMD+=" --epsilon-fraction $EPSILON_FRACTION --epsilon-phase1-fraction $EPSILON_PHASE1_FRACTION"
        if [[ -n "$EPSILON_MAX_CAP" ]]; then
            CMD+=" --epsilon-max-cap $EPSILON_MAX_CAP"
        fi
    fi
    if [[ "$APPROACH" == "adaptive" || "$APPROACH" == "hybrid" ]]; then
        CMD+=" --adaptive-alpha $ADAPTIVE_ALPHA --adaptive-max-budget $ADAPTIVE_MAX_BUDGET"
    fi
    if [[ "$APPROACH" == "hybrid" ]]; then
        CMD+=" --hybrid-incremental-time-limit $HYBRID_INCREMENTAL_TIME_LIMIT"
    fi
    if [[ "$APPROACH" == "online_biobj_warmstart" && "$PRE_PROCESS" == "1" ]] || [[ "$APPROACH" == "hybrid" ]]; then
        CMD+=" --freeze-large-jobs-threshold $FREEZE_LARGE_JOBS_THRESHOLD"
        CMD+=" --freeze-remaining-time-threshold $FREEZE_REMAINING_TIME_THRESHOLD"
        CMD+=" --freeze-jobs-with-ongoing-transfer"
    fi

    echo "### approach=$APPROACH -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/<approach>/  (one dir per approach in \$APPROACHES)"
echo "### Each dir will contain infos_on_jobs.csv, infos_on_tasks.csv, infos_on_replicas.csv,"
echo "### infos_on_transfers_energy.csv, events_history.json, and gantt.png."
