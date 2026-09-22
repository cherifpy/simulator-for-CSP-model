#!/usr/bin/env bash
# Submits the dataset-size sweep (small / medium / large tiers, n_existing=10, 50 nodes) as
# THREE separate oarsub jobs, one per tier, so they run in parallel on Grid5000 -- same pattern
# as submit_nexisting_sweep.sh used for the infrastructure-load sweep.
#
# Each job runs xp_dataset_size_sweep.py once (--tiers <tier> only), producing both the usual
# results.json / runs_by_dataset_size.csv AND the new per-job jobs_detail_by_run.csv (job_id,
# arriving_time, finishing_time, flow_time for every job in that tier's run), which is what lets
# us compute per-job flow-time variance afterwards without re-running anything.
#
# Same isolation rationale as submit_nexisting_sweep.sh: schedulingUsingJavaCSP exchanges data
# with the Java solver through FIXED file paths under utils/model/{inputs,outputs}, and compiles
# into utils/model/bin -- there is no per-call uniqueness. Running the 3 tiers at the same time
# from the same checkout (even across different physical Grid5000 nodes, since $HOME is usually
# NFS-shared site-wide) would clobber each other's exchange files mid-solve. The fix: give each
# tier its own private utils/model/{inputs,outputs,bin} (real, writable) plus symlinks back to
# the canonical utils/model/{lib,src} (read-only, safe to share), pointed at via SIMULATOR_RUN_CWD.
# Done automatically below -- nothing to set up by hand.
#
# IMPORTANT -- --skip-setup: launcher.py runs `pip install --upgrade pip` + reinstalls
# requirements.txt into the SHARED venv (and recompiles Java into the SHARED canonical
# utils/model/bin) on every invocation unless told not to -- SIMULATOR_RUN_CWD does NOT cover
# this (it only isolates modelCSP.py's own per-solve exchange files/compile, not launcher.py's
# global setup step). Three parallel launcher.py calls without --skip-setup collide on that
# shared venv (observed: pip failing after a few seconds on 2 of 3 concurrent tiers) -- this
# script always passes --skip-setup, so the venv + compiled Java from setup_grid5000.sh (or one
# prior launcher.py run without --skip-setup) must already be in place before running this.
#
# Usage (run from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_dataset_size_sweep_grid5000.sh
#   ./simulator/submit_dataset_size_sweep_grid5000.sh --walltime 03:00:00
#   ./simulator/submit_dataset_size_sweep_grid5000.sh --approaches "online incremental"   # skip bi-obj
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place),
# and inst-20J-50N present under workloads/ (n_existing=10 needs only 11 of its 20 jobs).

set -euo pipefail

# Defaults reproduce the runs already in the results notebook (seed 42, all 3 approaches):
# Online 2h, Incremental 60s, Online bi-obj (epsilon) 2h total = 1h/1h -- except the large tier,
# where epsilon gets 4h total (2h per phase) plus an absolute cap on its max-flow-time slack
# (the Incremental baseline's max flow for that scenario) so phase 1 converges as far as Online's.
WALLTIME="04:30:00"
LARGE_WALLTIME="07:30:00"
ONLINE_TIME_LIMIT=7200
INCREMENTAL_TIME_LIMIT=60
EPSILON_TIME_LIMIT=7200
LARGE_EPSILON_TIME_LIMIT=14400
LARGE_EPSILON_MAX_CAP="6053.76"   # incremental max_flow_time_all, large tier, seed 42
STATE_A_TIME_LIMIT=60
NB_NODES=50
N_EXISTING=10
INSTANCE_NAME="inst-20J-50N"
TIERS=(small medium large)
APPROACHES="online incremental epsilon"
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --large-walltime) LARGE_WALLTIME="$2"; shift 2 ;;
        --epsilon-time-limit) EPSILON_TIME_LIMIT="$2"; shift 2 ;;
        --large-epsilon-time-limit) LARGE_EPSILON_TIME_LIMIT="$2"; shift 2 ;;
        --large-epsilon-max-cap) LARGE_EPSILON_MAX_CAP="$2"; shift 2 ;;
        --online-time-limit) ONLINE_TIME_LIMIT="$2"; shift 2 ;;
        --incremental-time-limit) INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --state-a-time-limit) STATE_A_TIME_LIMIT="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --n-existing) N_EXISTING="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --tiers) IFS=' ' read -r -a TIERS <<< "$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
INSTANCE_DIR="$SCRIPT_DIR/workloads/workloads-100-for_storage_constraintes/$INSTANCE_NAME"
RESULTS_ROOT="$SCRIPT_DIR/results-grid5000/dataset_size_sweep_per_job_seed42"
RUN_CWD_ROOT="$SCRIPT_DIR/.run_cwd/dataset_size_sweep_per_job_seed42"   # private per-run utils/model trees

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

echo "### Submitting ${#TIERS[@]} job(s): tiers=${TIERS[*]}"
echo "### Approaches: $APPROACHES"
echo "### Online budget: ${ONLINE_TIME_LIMIT}s, Incremental budget: ${INCREMENTAL_TIME_LIMIT}s, state-A budget: ${STATE_A_TIME_LIMIT}s"
echo "### Instance: $INSTANCE_DIR ($NB_NODES nodes, n_existing=$N_EXISTING)"
echo "### Walltime: $WALLTIME (large: $LARGE_WALLTIME), seed=$SEED"
echo ""

for TIER in "${TIERS[@]}"; do
    RESULTS_DIR="$RESULTS_ROOT/$TIER"
    RUN_CWD="$RUN_CWD_ROOT/$TIER"
    mkdir -p "$RESULTS_DIR"

    rm -rf "$RUN_CWD"
    mkdir -p "$RUN_CWD/utils/model/inputs" "$RUN_CWD/utils/model/outputs" "$RUN_CWD/utils/model/bin"
    ln -sfn "$SCRIPT_DIR/utils/model/lib" "$RUN_CWD/utils/model/lib"
    ln -sfn "$SCRIPT_DIR/utils/model/src" "$RUN_CWD/utils/model/src"

    CMD="SIMULATOR_RUN_CWD=$RUN_CWD python3 $SCRIPT_DIR/launcher.py --experiment dataset_size_sweep --skip-setup"
    CMD+=" --instance-dir $INSTANCE_DIR --nb-nodes $NB_NODES --n-existing $N_EXISTING --tiers $TIER --repeats 1 --seed $SEED"
    CMD+=" --approaches $APPROACHES"
    TIER_WALLTIME="$WALLTIME"
    if [[ "$TIER" == "large" ]]; then
        CMD+=" --epsilon-time-limit $LARGE_EPSILON_TIME_LIMIT --epsilon-max-cap $LARGE_EPSILON_MAX_CAP"
        TIER_WALLTIME="$LARGE_WALLTIME"
    else
        CMD+=" --epsilon-time-limit $EPSILON_TIME_LIMIT"
    fi
    CMD+=" --state-a-time-limit $STATE_A_TIME_LIMIT --solver-time-limit $ONLINE_TIME_LIMIT --incremental-time-limit $INCREMENTAL_TIME_LIMIT"
    CMD+=" --results-dir $RESULTS_DIR"

    echo "### tier=$TIER -> $RESULTS_DIR (private run dir: $RUN_CWD)"
    oarsub -l "host=1,walltime=$TIER_WALLTIME" "$CMD"
    echo ""
done

echo "### All jobs submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_ROOT/{small,medium,large}/"
echo "### Each tier's dir will contain results.json, runs_by_dataset_size.csv, and jobs_detail_by_run.csv"
echo "### (jobs_detail_by_run.csv has per-job arriving_time/finishing_time/flow_time -- use it to compute"
echo "### per-job flow-time variance/std within a workload without re-running anything.)"
