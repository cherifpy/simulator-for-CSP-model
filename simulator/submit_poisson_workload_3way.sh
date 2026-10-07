#!/usr/bin/env bash
# Full live-simulation workload (lambda_rate=300, 20 jobs, 50 nodes, mixte-x2 sizing: task
# duration [200,600]s, dataset_size [4096,409600]MB) comparing the 3 approaches this week's
# single-decision-point investigation settled on, via exps/xp_online_grid5000.py:
#   - "incremental":    SchedulingUsingCSPIncremental -- each arrival placed alone, every
#                        already-running job untouched.
#   - "online_newjob":  SchedulingUsingCSPOnlineNewJob (new class, 2026-10-07) -- each arrival
#                        jointly reconsidered with every running job, objective_choice=2
#                        (minimize ONLY the new job's own flow time) + a per-running-job
#                        degradation cap, single solve, no escalation, no gate.
#   - "hybrid":          SchedulingUsingCSPAdaptiveJoint ("hybrid-n_j" design) -- Incremental
#                        probe (F1) first, escalates to the same objective_choice=2 + degradation
#                        cap mechanism via an 11-variant parallel race, kept only if it beats F1.
#
# IMPORTANT -- this is the FULL LIVE SIMULATION methodology (continuous SimPy run with Poisson
# job arrivals via jobsInjectorBasedOnLambdaPoisson, one CSP replan per new arrival), NOT the
# frozen-state-A single-decision-point protocol used by xp_dataset_size_sweep.py. See
# submit_online_vs_biobj_workload_grid5000.sh's own header for why each approach gets its own
# private utils/model/{inputs,outputs,bin} via SIMULATOR_RUN_CWD (concurrent runs from the same
# checkout would otherwise clobber each other's Java exchange files).
#
# Usage (run from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_poisson_workload_3way.sh
#   ./simulator/submit_poisson_workload_3way.sh --walltime 02:00:00
#   ./simulator/submit_poisson_workload_3way.sh --approaches "incremental online_newjob"
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place),
# and workloads/poisson_mixte_x2_20j_50n/{jobs.json,infrastructure.csv} present.

set -euo pipefail

WALLTIME="01:00:00"
SOLVER_TIME_LIMIT=30
NB_JOBS=20
NB_NODES=50
LAMBDA_RATE=300
INSTANCE_NAME="poisson_mixte_x2_20j_50n"
SEED=600
ADAPTIVE_ALPHA=0.25
ADAPTIVE_MAX_BUDGET=30
HYBRID_INCREMENTAL_TIME_LIMIT=15
ADAPTIVE_DEGRADATION_CAP_PCT=0.25
APPROACHES="incremental online_newjob hybrid"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --solver-time-limit) SOLVER_TIME_LIMIT="$2"; shift 2 ;;
        --nb-jobs) NB_JOBS="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --lambda-rate) LAMBDA_RATE="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --adaptive-alpha) ADAPTIVE_ALPHA="$2"; shift 2 ;;
        --adaptive-max-budget) ADAPTIVE_MAX_BUDGET="$2"; shift 2 ;;
        --hybrid-incremental-time-limit) HYBRID_INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --adaptive-degradation-cap-pct) ADAPTIVE_DEGRADATION_CAP_PCT="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                      # .../simulator-for-CSP-model
VENV_PYTHON="$PROJECT_ROOT/env/bin/python3"
INSTANCE_DIR="$SCRIPT_DIR/workloads/$INSTANCE_NAME"
RESULTS_ROOT="$SCRIPT_DIR/results-grid5000/poisson_workload_3way_$INSTANCE_NAME"
RUN_CWD_ROOT="$SCRIPT_DIR/.run_cwd/poisson_workload_3way_$INSTANCE_NAME"

if [[ ! -f "$INSTANCE_DIR/jobs.json" ]]; then
    echo "ERROR: instance not found at $INSTANCE_DIR (expected jobs.json there)." >&2
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VENV_PYTHON. Run setup_grid5000.sh first." >&2
    exit 1
fi

echo "### Submitting approaches: $APPROACHES"
echo "### Instance: $INSTANCE_DIR ($NB_JOBS jobs / $NB_NODES nodes, lambda_rate=$LAMBDA_RATE)"
echo "### Solver time budget: ${SOLVER_TIME_LIMIT}s per replan"
echo "### adaptive_alpha=$ADAPTIVE_ALPHA adaptive_max_budget=${ADAPTIVE_MAX_BUDGET}s"
echo "### hybrid_incremental_time_limit=${HYBRID_INCREMENTAL_TIME_LIMIT}s"
echo "### adaptive_degradation_cap_pct=$ADAPTIVE_DEGRADATION_CAP_PCT"
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
    if [[ "$APPROACH" == "online_newjob" ]]; then
        CMD+=" --adaptive-degradation-cap-pct $ADAPTIVE_DEGRADATION_CAP_PCT"
    fi
    if [[ "$APPROACH" == "hybrid" ]]; then
        CMD+=" --adaptive-alpha $ADAPTIVE_ALPHA --adaptive-max-budget $ADAPTIVE_MAX_BUDGET"
        CMD+=" --hybrid-incremental-time-limit $HYBRID_INCREMENTAL_TIME_LIMIT"
        CMD+=" --adaptive-new-job-objective --adaptive-degradation-cap-pct $ADAPTIVE_DEGRADATION_CAP_PCT"
        CMD+=" --adaptive-gate-metric new_job --adaptive-selection-metric new_job"
        CMD+=" --parallel-warm-cold-escalation --no-adaptive-bi-objective"
        CMD+=" --epsilon-fraction 0.15 --epsilon-phase1-fraction 0.5"
    fi

    echo "### approach=$APPROACH -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/<approach>/"
echo "### Each dir will contain infos_on_jobs.csv, infos_on_tasks.csv, infos_on_replicas.csv,"
echo "### infos_on_transfers_energy.csv, events_history.json, and gantt.png."
