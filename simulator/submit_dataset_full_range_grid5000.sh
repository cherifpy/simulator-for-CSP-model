#!/usr/bin/env bash
# One-off experiment: dataset size drawn from a SINGLE 10-100GB range (not split into small/
# medium/large tiers), state A built from 15 jobs whose arrival times are a Poisson process with a
# small mean gap (not all arriving at once), the new job arriving strictly after the last one.
#
# IMPORTANT -- why this is ONE oarsub job, not 3 parallel ones like the other submit_*.sh scripts:
# the whole point of this run is that Online mono-obj / Incremental / Online bi-obj share the
# EXACT SAME state A -- not just the same job sizes/arrival times (already guaranteed by --seed),
# but the same SOLVED starting schedule. State A's own build is a time-limited CSP solve, which is
# NOT guaranteed deterministic across separate process runs even with the same seed (confirmed
# empirically: a re-run of the dataset-size sweep on a different Grid5000 site gave a visibly
# different state A despite identical inputs). The only way to guarantee a truly shared state A is
# to build it ONCE and run the 3 approaches one after another IN THE SAME PROCESS -- which rules
# out submitting them as 3 separate (parallel) oarsub jobs. See feedback_state_reuse memory note.
#
# Sequential budget: state A 1h + Online mono-obj 2h + Online bi-obj 4h (2h/2h) + Incremental 1h
# = 8h total in one job. Each approach's own solver time (wall-clock + the solver's own reported
# search time) is recorded in runs/full_r0_<approach>/summary.json.
#
# Usage (run from the Grid5000 frontend, inside the simulator-for-CSP-model checkout):
#   ./simulator/submit_dataset_full_range_grid5000.sh
#   ./simulator/submit_dataset_full_range_grid5000.sh --walltime 09:30:00
#   ./simulator/submit_dataset_full_range_grid5000.sh --arrival-lambda 10   # pack the 15 jobs closer
#
# Requires: setup_grid5000.sh already run at least once (venv + compiled Java model in place),
# and inst-20J-50N present under workloads/ (n_existing=15 needs 16 of its 20 jobs).

set -euo pipefail

WALLTIME="08:30:00"
STATE_A_TIME_LIMIT=3600      # 1h
ONLINE_TIME_LIMIT=7200       # 2h
INCREMENTAL_TIME_LIMIT=3600  # 1h
EPSILON_TIME_LIMIT=14400     # 4h total = 2h/2h (default --epsilon-phase1-fraction 0.5)
EPSILON_PHASE1_FRACTION=0.5
N_EXISTING=15
NB_NODES=50
INSTANCE_NAME="inst-20J-50N"
FULL_RANGE_LOW=10240          # 10GB
FULL_RANGE_HIGH=102400        # 100GB
ARRIVAL_LAMBDA=20             # mean inter-arrival gap (seconds) for the 15 state-A jobs -- small
                               # on purpose, so they arrive close together rather than spread out.
SEED=42
APPROACHES="online incremental epsilon"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --walltime) WALLTIME="$2"; shift 2 ;;
        --state-a-time-limit) STATE_A_TIME_LIMIT="$2"; shift 2 ;;
        --online-time-limit) ONLINE_TIME_LIMIT="$2"; shift 2 ;;
        --incremental-time-limit) INCREMENTAL_TIME_LIMIT="$2"; shift 2 ;;
        --epsilon-time-limit) EPSILON_TIME_LIMIT="$2"; shift 2 ;;
        --epsilon-phase1-fraction) EPSILON_PHASE1_FRACTION="$2"; shift 2 ;;
        --n-existing) N_EXISTING="$2"; shift 2 ;;
        --nb-nodes) NB_NODES="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --full-range) FULL_RANGE_LOW="$2"; FULL_RANGE_HIGH="$3"; shift 3 ;;
        --arrival-lambda) ARRIVAL_LAMBDA="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulator-for-CSP-model/simulator
INSTANCE_DIR="$SCRIPT_DIR/workloads/workloads-100-for_storage_constraintes/$INSTANCE_NAME"
RESULTS_DIR="$SCRIPT_DIR/results-grid5000/dataset_full_range_10_100GB"
RUN_CWD="$SCRIPT_DIR/.run_cwd/dataset_full_range_10_100GB"   # private utils/model tree for this run

if [[ ! -f "$INSTANCE_DIR/jobs.json" ]]; then
    echo "ERROR: instance not found at $INSTANCE_DIR (expected jobs.json there)." >&2
    exit 1
fi

# Same venv preflight as the other submit_*.sh scripts -- launcher.py's --skip-setup re-execs into
# the venv's interpreter without creating it; a missing/broken venv otherwise kills the job within
# seconds with FileNotFoundError, hours into an 8h+ walltime reservation.
VENV_PYTHON="$(dirname "$SCRIPT_DIR")/env/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -c "" >/dev/null 2>&1; then
    echo "ERROR: venv python missing or unusable at $VENV_PYTHON." >&2
    echo "Run ./simulator/setup_grid5000.sh on this site first (it recreates a broken venv)." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
rm -rf "$RUN_CWD"
mkdir -p "$RUN_CWD/utils/model/inputs" "$RUN_CWD/utils/model/outputs" "$RUN_CWD/utils/model/bin"
ln -sfn "$SCRIPT_DIR/utils/model/lib" "$RUN_CWD/utils/model/lib"
ln -sfn "$SCRIPT_DIR/utils/model/src" "$RUN_CWD/utils/model/src"

CMD="SIMULATOR_RUN_CWD=$RUN_CWD python3 $SCRIPT_DIR/launcher.py --experiment dataset_size_sweep --skip-setup"
CMD+=" --instance-dir $INSTANCE_DIR --nb-nodes $NB_NODES --n-existing $N_EXISTING"
CMD+=" --tiers full --full-range $FULL_RANGE_LOW $FULL_RANGE_HIGH --arrival-lambda $ARRIVAL_LAMBDA"
CMD+=" --repeats 1 --seed $SEED --approaches $APPROACHES"
CMD+=" --state-a-time-limit $STATE_A_TIME_LIMIT --solver-time-limit $ONLINE_TIME_LIMIT"
CMD+=" --incremental-time-limit $INCREMENTAL_TIME_LIMIT"
CMD+=" --epsilon-time-limit $EPSILON_TIME_LIMIT --epsilon-phase1-fraction $EPSILON_PHASE1_FRACTION"
CMD+=" --results-dir $RESULTS_DIR"

echo "### Dataset-size range: ${FULL_RANGE_LOW}-${FULL_RANGE_HIGH}MB, n_existing=$N_EXISTING jobs,"
echo "###   arrival_lambda=${ARRIVAL_LAMBDA}s (Poisson, small gaps), seed=$SEED"
echo "### Sequential budgets: state-A ${STATE_A_TIME_LIMIT}s, Online mono-obj ${ONLINE_TIME_LIMIT}s,"
echo "###   Incremental ${INCREMENTAL_TIME_LIMIT}s, Online bi-obj ${EPSILON_TIME_LIMIT}s total"
echo "###   (phase1 fraction=$EPSILON_PHASE1_FRACTION)"
echo "### ONE job (not parallelized across approaches) -- see the header comment for why: this is"
echo "###   what guarantees the 3 approaches share the exact same solved state A."
echo "### Results -> $RESULTS_DIR (private run dir: $RUN_CWD)"
echo "### Walltime: $WALLTIME"
echo ""

oarsub -l "host=1,walltime=$WALLTIME" "$CMD"

echo ""
echo "### Job submitted. Check status with: oarstat -u \$(whoami)"
echo "### Results will land under: $RESULTS_DIR/"
echo "### runs/full_state_A/ + runs/full_r0_<approach>/ each have solver.log, trajectory.csv,"
echo "### tasks/transfers/replicas/deletions.csv, raw_io/, and summary.json (wall_time_s + the"
echo "### solver's own search_time_s/solution_count/objective_optimal -- see 'solver' in summary.json)."
