#!/usr/bin/env bash
# Exp1, sweep 2/2 -- infra-load sweep: n_existing in {5,10,20}, medium dataset-size tier, 3
# approaches (incremental / online_biobj [Java class MainOnlineMultiObj, called "epsilon" in
# xp_dataset_size_sweep.py's own --approaches vocabulary] / hybrid [F1 probe + 4-way parallel
# escalation]). Submits ONE oarsub job per n_existing value, each with its own private
# utils/model/{inputs,outputs,bin} via SIMULATOR_RUN_CWD -- see submit_nexisting_sweep.sh's own
# comment for why this matters: concurrent jobs from the same checkout share Grid5000's NFS home
# and WILL clobber each other's Java exchange files if they aren't isolated (confirmed the hard
# way in this project's own history: an IndexError from a job's inputs being overwritten mid-solve
# by another job running at the same time).
#
# Budgets match the live workload experiment (xp_online_grid5000.py) for apples-to-apples
# comparison: incremental 180s, online_biobj 180s, hybrid F1-probe<=30s + escalation<=180s@20%.
#
# Usage (from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_nexisting_sweep_3way.sh
#   ./simulator/submit_nexisting_sweep_3way.sh --walltime 02:00:00
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place).

set -euo pipefail

INCREMENTAL_TIME_LIMIT=180
ONLINE_BIOBJ_TIME_LIMIT=180
HYBRID_INCREMENTAL_TIME_LIMIT=30
HYBRID_ALPHA=0.2
HYBRID_MAX_BUDGET=180
EPSILON_FRACTION=0.05
EPSILON_PHASE1_FRACTION=0.75
STATE_A_TIME_LIMIT=30
NB_NODES=50
INSTANCE_NAME="inst-20J-50N"
N_EXISTING_VALUES=(5 10 20)
SEED=42
WALLTIME="01:30:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --incremental-time-limit) INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --online-biobj-time-limit) ONLINE_BIOBJ_TIME_LIMIT="$2"; shift 2 ;;
        --hybrid-incremental-time-limit) HYBRID_INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --hybrid-alpha) HYBRID_ALPHA="$2"; shift 2 ;;
        --hybrid-max-budget) HYBRID_MAX_BUDGET="$2"; shift 2 ;;
        --state-a-time-limit) STATE_A_TIME_LIMIT="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --n-existing-values) IFS=' ' read -r -a N_EXISTING_VALUES <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTANCE_DIR="$SCRIPT_DIR/workloads/workloads-100-for_storage_constraintes/$INSTANCE_NAME"
RESULTS_ROOT="$SCRIPT_DIR/results-grid5000/nexisting_sweep_3way"
RUN_CWD_ROOT="$SCRIPT_DIR/.run_cwd/nexisting_sweep_3way"

if [[ ! -f "$INSTANCE_DIR/jobs.json" ]]; then
    echo "ERROR: instance not found at $INSTANCE_DIR (expected jobs.json there)." >&2
    exit 1
fi

VENV_PYTHON="$PROJECT_ROOT/env/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -c "" >/dev/null 2>&1; then
    echo "ERROR: venv python missing or unusable at $VENV_PYTHON." >&2
    echo "Run ./simulator/setup_grid5000.sh on this site first." >&2
    exit 1
fi

echo "### Submitting ${#N_EXISTING_VALUES[@]} job(s): n_existing=${N_EXISTING_VALUES[*]}"
echo "### Approaches: incremental online_biobj hybrid (seed $SEED)"
echo "### Budgets: incremental=${INCREMENTAL_TIME_LIMIT}s online_biobj=${ONLINE_BIOBJ_TIME_LIMIT}s hybrid(F1<=${HYBRID_INCREMENTAL_TIME_LIMIT}s + escalation<=${HYBRID_MAX_BUDGET}s@${HYBRID_ALPHA})"
echo "### Instance: $INSTANCE_DIR ($NB_NODES nodes)"
echo "### Walltime per job: $WALLTIME"
echo ""

for N in "${N_EXISTING_VALUES[@]}"; do
    RESULTS_DIR="$RESULTS_ROOT/n${N}"
    RUN_CWD="$RUN_CWD_ROOT/n${N}"
    mkdir -p "$RESULTS_DIR"

    rm -rf "$RUN_CWD"
    mkdir -p "$RUN_CWD/utils/model/inputs" "$RUN_CWD/utils/model/outputs" "$RUN_CWD/utils/model/bin"
    ln -sfn "$SCRIPT_DIR/utils/model/lib" "$RUN_CWD/utils/model/lib"
    ln -sfn "$SCRIPT_DIR/utils/model/src" "$RUN_CWD/utils/model/src"

    CMD="SIMULATOR_RUN_CWD=$RUN_CWD PYTHONUNBUFFERED=1 $VENV_PYTHON $SCRIPT_DIR/exps/xp_dataset_size_sweep.py"
    CMD+=" --instance-dir $INSTANCE_DIR --nb-nodes $NB_NODES --n-existing $N --tiers medium --repeats 1 --seed $SEED"
    CMD+=" --approaches incremental epsilon hybrid"
    CMD+=" --incremental-time-limit $INCREMENTAL_TIME_LIMIT --epsilon-time-limit $ONLINE_BIOBJ_TIME_LIMIT --solver-time-limit $ONLINE_BIOBJ_TIME_LIMIT"
    CMD+=" --epsilon-fraction $EPSILON_FRACTION --epsilon-phase1-fraction $EPSILON_PHASE1_FRACTION"
    CMD+=" --hybrid-incremental-time-limit $HYBRID_INCREMENTAL_TIME_LIMIT --hybrid-alpha $HYBRID_ALPHA --hybrid-max-budget $HYBRID_MAX_BUDGET"
    CMD+=" --state-a-time-limit $STATE_A_TIME_LIMIT"
    CMD+=" --results-dir $RESULTS_DIR"

    echo "### n_existing=$N -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/n{5,10,20}/results.json"
