#!/usr/bin/env bash
# Submits a full live-simulation workload (lambda_rate=100, 20 jobs, 50 nodes, 10min solver
# budget per replan -- same setup used before for online-vs-incremental comparisons, via
# exps/xp_online_grid5000.py) as TWO separate oarsub jobs running in parallel:
#   - "online":       mono-objective Online, max flow time (MainOnline.java)
#   - "online_biobj": bi-objective Online, epsilon-constraint phase1=max flow time /
#                      phase2=transfer energy (MainOnlineMultiObj.java), same total solver
#                      budget as "online", split 50/50 between the two phases by default.
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
    if [[ "$APPROACH" == "online_biobj" ]]; then
        CMD+=" --epsilon-fraction $EPSILON_FRACTION --epsilon-phase1-fraction $EPSILON_PHASE1_FRACTION"
        if [[ -n "$EPSILON_MAX_CAP" ]]; then
            CMD+=" --epsilon-max-cap $EPSILON_MAX_CAP"
        fi
    fi

    echo "### approach=$APPROACH -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/{online,online_biobj}/"
echo "### Each dir will contain infos_on_jobs.csv, infos_on_tasks.csv, infos_on_replicas.csv,"
echo "### infos_on_transfers_energy.csv, events_history.json, and gantt.png."
