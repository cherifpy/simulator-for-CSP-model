#!/usr/bin/env bash
# Submits the n-existing sweep (5 / 10 / 20 existing jobs, medium dataset-size tier) as THREE
# separate oarsub jobs, one per n-existing value.
#
# Each job runs xp_dataset_size_sweep.py once (--tiers medium only) with Online given a 2h
# budget and Incremental a 10min budget.
#
# IMPORTANT -- why each job gets its own private utils/model/{inputs,outputs,bin}:
# schedulingUsingJavaCSP (utils/modelCSP.py) exchanges data with the Java solver through FIXED
# file paths under utils/model/inputs and utils/model/outputs, and compiles into utils/model/bin
# -- there is no per-call uniqueness. Two solves in flight AT THE SAME TIME, from the SAME
# checkout, clobber each other's inputs mid-solve (one process's jobs_data gets overwritten by
# another's before it's read back), producing corrupted results or outright crashes
# (IndexError, or Choco "wrong domain: lower bound > upper bound" from a mismatched job count).
# `oarsub -l host=1` giving each job its own dedicated compute node does NOT by itself prevent
# this: Grid5000 home directories are typically NFS-shared across every node at a site, so
# multiple jobs -- even on different physical machines -- can still all be reading and writing
# the exact same files if they run from the same checkout path.
#
# The real fix: give each n-existing value its own private copy of utils/model/{inputs,outputs,
# bin} (real, writable directories) plus symlinks back to the canonical utils/model/{lib,src}
# (read-only, safe to share), then point that run at it via the SIMULATOR_RUN_CWD environment
# variable, which utils/modelCSP.py's schedulingUsingJavaCSP checks before falling back to the
# canonical SIMULATOR_DIR. This is done automatically by this script for each n-existing value
# below -- nothing to set up by hand.
#
# Usage (run from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_nexisting_sweep.sh
#   ./simulator/submit_nexisting_sweep.sh --walltime 04:00:00      # default depends on --approaches
#   ./simulator/submit_nexisting_sweep.sh --online-time-limit 3600 --incremental-time-limit 300
#   ./simulator/submit_nexisting_sweep.sh --approaches epsilon      # add the bi-objective on the same scenarios
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place),
# and inst-50J-50N present under workloads/ -- inst-20J-50N only has 20 jobs total, so
# --n-existing 20 there would leave no job left over to act as the "new" arrival.

set -euo pipefail

ONLINE_TIME_LIMIT=7200
INCREMENTAL_TIME_LIMIT=600
STATE_A_TIME_LIMIT=60
NB_NODES=50
INSTANCE_NAME="inst-50J-50N"
N_EXISTING_VALUES=(5 10 20)
SEED=42
# Approaches to run per n_existing value. The original sweep ran "online incremental"; to add the
# bi-objective (epsilon-constraint) approach on the SAME scenarios, run `--approaches epsilon` --
# same seed/instance/tier => the same state A and new job as the earlier online/incremental runs.
APPROACHES="online incremental"
# epsilon only: phase 1 gets the same 2h Online got (2h per phase = 4h total), so its 10% slack is
# measured against a converged phase-1 result (a 1h/1h split under-converged on the hardest tier,
# see the dataset-size notebook). Each n_existing value also gets an absolute cap on the slack =
# the Incremental baseline's max flow time for that scenario (from the earlier sweep), so epsilon
# can never end up worse than the trivial baseline.
EPSILON_TIME_LIMIT=14400
USE_EPSILON_CAP=true
epsilon_cap_for() {   # incremental max_flow_time_all, medium tier, inst-50J-50N, seed 42
    case "$1" in
        5) echo "1649.04" ;;
        10) echo "6144.30" ;;
        20) echo "18436.2" ;;
        *) echo "" ;;
    esac
}
WALLTIME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --online-time-limit) ONLINE_TIME_LIMIT="$2"; shift 2 ;;
        --incremental-time-limit) INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --state-a-time-limit) STATE_A_TIME_LIMIT="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        --epsilon-time-limit) EPSILON_TIME_LIMIT="$2"; shift 2 ;;
        --no-epsilon-cap) USE_EPSILON_CAP=false; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --n-existing-values) IFS=' ' read -r -a N_EXISTING_VALUES <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$WALLTIME" ]]; then
    if [[ " $APPROACHES " == *" epsilon "* && " $APPROACHES " == *" online "* ]]; then WALLTIME="06:30:00"
    elif [[ " $APPROACHES " == *" epsilon "* ]]; then WALLTIME="04:30:00"
    else WALLTIME="03:30:00"; fi
fi
RESULTS_SUFFIX=""
[[ "$APPROACHES" == "epsilon" ]] && RESULTS_SUFFIX="_epsilon"   # keep apart from the online/incremental results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"                      # .../simulator-for-CSP-model
INSTANCE_DIR="$SCRIPT_DIR/workloads/workloads-100-for_storage_constraintes/$INSTANCE_NAME"
RESULTS_ROOT="$SCRIPT_DIR/results-grid5000/nexisting_sweep_medium${RESULTS_SUFFIX}"
RUN_CWD_ROOT="$SCRIPT_DIR/.run_cwd/nexisting_sweep_medium${RESULTS_SUFFIX}"   # private per-run utils/model trees

if [[ ! -f "$INSTANCE_DIR/jobs.json" ]]; then
    echo "ERROR: instance not found at $INSTANCE_DIR (expected jobs.json there)." >&2
    exit 1
fi

# With --skip-setup, launcher.py re-execs into the venv's interpreter without creating it -- if
# env/bin/python3 is missing or broken on this site (venvs aren't portable across machines/sites),
# every submitted job dies within seconds with FileNotFoundError. Fail here instead.
VENV_PYTHON="$(dirname "$SCRIPT_DIR")/env/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -c "" >/dev/null 2>&1; then
    echo "ERROR: venv python missing or unusable at $VENV_PYTHON." >&2
    echo "Run ./simulator/setup_grid5000.sh on this site first (it recreates a broken venv)." >&2
    exit 1
fi

echo "### Submitting ${#N_EXISTING_VALUES[@]} job(s): n_existing=${N_EXISTING_VALUES[*]}"
echo "### Approaches: $APPROACHES (seed $SEED)"
echo "### Online budget: ${ONLINE_TIME_LIMIT}s, Incremental budget: ${INCREMENTAL_TIME_LIMIT}s, epsilon budget: ${EPSILON_TIME_LIMIT}s total, state-A budget: ${STATE_A_TIME_LIMIT}s"
echo "### Instance: $INSTANCE_DIR ($NB_NODES nodes)"
echo "### Walltime per job: $WALLTIME"
echo ""

for N in "${N_EXISTING_VALUES[@]}"; do
    RESULTS_DIR="$RESULTS_ROOT/n${N}"
    RUN_CWD="$RUN_CWD_ROOT/n${N}"
    mkdir -p "$RESULTS_DIR"

    # Fresh private inputs/outputs/bin for this run (wiped and recreated so no stale exchange
    # file from a previous attempt can be misread); lib/src are symlinked back to the canonical
    # copy since javac/java only ever READ those (the jars and .java sources never change
    # per-run), so there's no need to duplicate them.
    rm -rf "$RUN_CWD"
    mkdir -p "$RUN_CWD/utils/model/inputs" "$RUN_CWD/utils/model/outputs" "$RUN_CWD/utils/model/bin"
    ln -sfn "$SCRIPT_DIR/utils/model/lib" "$RUN_CWD/utils/model/lib"
    ln -sfn "$SCRIPT_DIR/utils/model/src" "$RUN_CWD/utils/model/src"

    CMD="SIMULATOR_RUN_CWD=$RUN_CWD python3 $SCRIPT_DIR/launcher.py --experiment dataset_size_sweep --skip-setup"
    CMD+=" --instance-dir $INSTANCE_DIR --nb-nodes $NB_NODES --n-existing $N --tiers medium --repeats 1 --seed $SEED"
    CMD+=" --approaches $APPROACHES"
    if [[ " $APPROACHES " == *" epsilon "* ]]; then
        CMD+=" --epsilon-time-limit $EPSILON_TIME_LIMIT"
        CAP="$(epsilon_cap_for "$N")"
        if [[ "$USE_EPSILON_CAP" == true && -n "$CAP" ]]; then CMD+=" --epsilon-max-cap $CAP"; fi
    fi
    CMD+=" --state-a-time-limit $STATE_A_TIME_LIMIT --solver-time-limit $ONLINE_TIME_LIMIT --incremental-time-limit $INCREMENTAL_TIME_LIMIT"
    CMD+=" --results-dir $RESULTS_DIR"

    echo "### n_existing=$N -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/n{5,10,20}/"
