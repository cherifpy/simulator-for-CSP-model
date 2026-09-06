import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: gantt charts are saved to file, not shown interactively

import csv
import json
import logging
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator import (
    simulatorForOptimalPerfsUsingCSPOnline,
    generateHeterogeneousInfrastructureEquilibre,
    save_results_to_csv,
    configure_logging,
)
from master_node_with_heterogeneous_nodes_csp import SchedulingUsingCSPOnline, SchedulingUsingCSPIncremental, SchedulingUsingCSPIncrementalFreeNodesOnly
from utils.plots import plot_gantt_chart

logger = logging.getLogger(__name__)

NB_JOBS, NB_NODES = 20, 50
INSTANCE_DIR = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/workloads-100-for_storage_constraintes/inst-20J-50N"
RESULTS_BASE = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results-online-vs-incremental"

APPROACHES = {
    "online": SchedulingUsingCSPOnline,
    "incremental": SchedulingUsingCSPIncremental,
    # free-nodes-only variant kept in the codebase (SchedulingUsingCSPIncrementalFreeNodesOnly)
    # but set aside for now -- back to validating online vs incremental with storage.
}

SOLVER_TIME_LIMIT_S = {
    "online": 60,
    "incremental": 60,
    "incremental_free_nodes_only": 60,
}


def run_one(name, master_class, config_template):
    config = dict(config_template)
    config['total_nb_jobs'] = NB_JOBS
    config['total_nb_compute_nodes'] = NB_NODES
    config['jobs_file_path'] = f"{INSTANCE_DIR}/jobs.json"
    config['solver_time_limit_s'] = SOLVER_TIME_LIMIT_S[name]

    results_destination = f"{RESULTS_BASE}/{name}"
    os.makedirs(results_destination, exist_ok=True)

    logger.info("=== Running approach '%s' ===", name)

    random.seed(42)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(config, path=f"{INSTANCE_DIR}/infrastructure.csv")

    random.seed(42)
    results, _ = simulatorForOptimalPerfsUsingCSPOnline(
        config=config, jobs=[], overlap=True, poisson=True, varying_load=False,
        nodes_config=nodes_config, master_class=master_class,
    )

    save_results_to_csv(logger, results, results_destination, "")

    # Ground truth for storage-occupancy verification: real transfer completions and real
    # deletions as they actually executed in the simulation (not the CSP's raw per-solve
    # decisions, which for Online get discarded/overwritten by later replans without always
    # being enacted).
    with open(f"{results_destination}/events_history.json", "w") as f:
        json.dump(results.events_history, f)

    gantt_path = f"{results_destination}/gantt.png"
    plot_gantt_chart(results.events_history, NB_NODES, title=f"inst-20J-50N ({name})", save_path=gantt_path)

    print(f"### {name}: total wall time = {results.total_wall_time}")
    return results_destination


def analyze(name, results_destination):
    jobs_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_jobs.csv")))
    task_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_tasks.csv")))
    replica_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_replicas.csv")))

    n = len(jobs_rows)
    flow_times = [float(r['finishing_time']) - float(r['arriving_time']) for r in jobs_rows]
    wait_times = [float(r['starting_time']) - float(r['arriving_time']) for r in jobs_rows]

    avg_flow = sum(flow_times) / n
    max_flow = max(flow_times)
    avg_wait = sum(wait_times) / n

    # data movements
    transfers = [r for r in task_rows if int(r['task']) == -1]
    tasks_only = [r for r in task_rows if int(r['task']) != -1]

    used_pairs = {(int(r['job']), int(r['node'])) for r in tasks_only}
    transfer_pairs = {(int(r['job']), int(r['node'])) for r in transfers}
    orphan_pairs = transfer_pairs - used_pairs

    job_size = {int(r['job_id']): float(r['dataset size']) for r in jobs_rows}

    total_data_moved = sum(job_size[int(r['job'])] for r in transfers)
    unique_data = sum(job_size.values())
    wasted_data = sum(job_size[j] for (j, node) in orphan_pairs)

    return {
        "name": name,
        "nb_jobs": n,
        "avg_flow_time": avg_flow,
        "max_flow_time": max_flow,
        "avg_wait_time": avg_wait,
        "nb_transfers": len(transfers),
        "nb_orphan_transfers": len(orphan_pairs),
        "pct_orphan_transfers": 100.0 * len(orphan_pairs) / len(transfers) if transfers else 0.0,
        "total_data_moved_mb": total_data_moved,
        "unique_data_mb": unique_data,
        "wasted_data_mb": wasted_data,
        "pct_wasted_data": 100.0 * wasted_data / total_data_moved if total_data_moved else 0.0,
        "replication_factor": total_data_moved / unique_data if unique_data else 0.0,
    }


def main():
    with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/config.json", "r", encoding="utf-8") as f:
        config_template = json.load(f)
    config_template['lambda_rate'] = 60

    configure_logging(logging.WARNING)

    stats = []
    for name, master_class in APPROACHES.items():
        results_destination = run_one(name, master_class, config_template)
        stats.append(analyze(name, results_destination))

    print()
    print("=" * 100)
    print(f"{'Approach':<14}{'Flow avg':>10}{'Flow max':>10}{'Wait avg':>10}{'Transfers':>11}{'Orphans':>9}{'%Orphan':>9}{'Data moved(MB)':>16}{'%Wasted':>9}{'Repl.factor':>12}")
    for s in stats:
        print(f"{s['name']:<14}{s['avg_flow_time']:>10.1f}{s['max_flow_time']:>10.1f}{s['avg_wait_time']:>10.1f}"
              f"{s['nb_transfers']:>11}{s['nb_orphan_transfers']:>9}{s['pct_orphan_transfers']:>8.1f}%"
              f"{s['total_data_moved_mb']:>16.0f}{s['pct_wasted_data']:>8.1f}%{s['replication_factor']:>12.2f}")
    print("=" * 100)

    with open(f"{RESULTS_BASE}/comparison_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"### Full summary saved to {RESULTS_BASE}/comparison_summary.json")


if __name__ == "__main__":
    main()
