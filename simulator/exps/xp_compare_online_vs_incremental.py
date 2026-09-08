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

RESULTS_BASE = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results-online-vs-incremental"

INSTANCES = {
    "20J-50N": (20, 50, "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/workloads-100-for_storage_constraintes/inst-20J-50N"),
    "50J-50N": (50, 50, "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/workloads-100-for_storage_constraintes/inst-50J-50N"),
}

APPROACHES = {
    "online": SchedulingUsingCSPOnline,
    # Only-empty-nodes variant: the CSP may only place a job on nodes that are completely idle
    # right now (nothing ongoing, nothing queued) -- see restrict_to_free_nodes. When no idle
    # node fits, schedulingNewJob now waits for one to free up instead of re-solving every tick.
    "incremental_free_nodes_only": SchedulingUsingCSPIncrementalFreeNodesOnly,
}

SOLVER_TIME_LIMIT_S = {
    "online": 30,
    "incremental": 30,
    "incremental_free_nodes_only": 30,
}


def run_one(name, master_class, config_template, nb_jobs, nb_nodes, instance_dir, results_base):
    config = dict(config_template)
    config['total_nb_jobs'] = nb_jobs
    config['total_nb_compute_nodes'] = nb_nodes
    config['jobs_file_path'] = f"{instance_dir}/jobs.json"
    config['solver_time_limit_s'] = SOLVER_TIME_LIMIT_S[name]

    results_destination = f"{results_base}/{name}"
    os.makedirs(results_destination, exist_ok=True)

    logger.info("=== Running approach '%s' ===", name)

    random.seed(42)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(config, path=f"{instance_dir}/infrastructure.csv")

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
    plot_gantt_chart(results.events_history, nb_nodes, title=f"{name}", save_path=gantt_path)

    print(f"### {name}: total wall time = {results.total_wall_time}")
    return results_destination


def verify_storage(results_destination, instance_dir):
    """Ground truth for storage-occupancy: rebuild real per-node occupancy from actual transfer
    completions and deletions as they executed in the simulation, and flag any instant where it
    exceeds the node's capacity. Returns the list of violations (empty if none)."""
    with open(f"{instance_dir}/jobs.json") as f:
        jobs = json.load(f)
    job_size = {j['job_id']: j['dataset_size'] for j in jobs}

    node_capacity = {}
    with open(f"{instance_dir}/infrastructure.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            node_capacity[i] = float(row['storage_capacity']) if 'storage_capacity' in row else float('inf')

    with open(f"{results_destination}/events_history.json") as f:
        events = json.load(f)

    arrival = {}
    for e in events:
        if e['type'] == 'transfer':
            key = (e['node_id'], e['job_id'])
            if key not in arrival or e['end'] < arrival[key]:
                arrival[key] = e['end']

    departures = {}
    for e in events:
        if e['type'] == 'deletion':
            departures.setdefault((e['node_id'], e['job_id']), []).append(e['time'])

    events_per_node = {}
    for (node, job), t_in in arrival.items():
        size = job_size[job]
        events_per_node.setdefault(node, []).append((t_in, size, f"+job{job}"))
        if (node, job) in departures:
            events_per_node[node].append((min(departures[(node, job)]), -size, f"-job{job}"))

    violations = []
    for node, evts in events_per_node.items():
        evts.sort(key=lambda e: e[0])
        occupied = 0.0
        cap = node_capacity.get(node, float('inf'))
        for t, delta, label in evts:
            occupied += delta
            if occupied > cap + 1e-6:
                violations.append((node, t, occupied, cap, label))
    return violations


def analyze(name, results_destination, instance_dir):
    jobs_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_jobs.csv")))
    task_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_tasks.csv")))
    replica_rows = list(csv.DictReader(open(f"{results_destination}/infos_on_replicas.csv")))

    violations = verify_storage(results_destination, instance_dir)

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
        "nb_storage_violations": len(violations),
        "storage_violations": violations[:10],  # sample, avoid huge dumps
    }


def main():
    with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/config.json", "r", encoding="utf-8") as f:
        config_template = json.load(f)
    config_template['lambda_rate'] = 60

    configure_logging(logging.WARNING)

    all_stats = {}
    for instance_name, (nb_jobs, nb_nodes, instance_dir) in INSTANCES.items():
        results_base = f"{RESULTS_BASE}/{instance_name}"
        stats = []
        for name, master_class in APPROACHES.items():
            print(f"\n### Running '{name}' on instance {instance_name} ({nb_jobs} jobs, {nb_nodes} nodes) ###")
            results_destination = run_one(name, master_class, config_template, nb_jobs, nb_nodes, instance_dir, results_base)
            stats.append(analyze(name, results_destination, instance_dir))
        all_stats[instance_name] = stats

        print()
        print("=" * 110)
        print(f"Instance {instance_name}")
        print(f"{'Approach':<26}{'Flow avg':>10}{'Flow max':>10}{'Wait avg':>10}{'Transfers':>11}{'Orphans':>9}{'%Orphan':>9}{'Data moved(MB)':>16}{'%Wasted':>9}{'Repl.factor':>12}{'StorageViol':>12}")
        for s in stats:
            print(f"{s['name']:<26}{s['avg_flow_time']:>10.1f}{s['max_flow_time']:>10.1f}{s['avg_wait_time']:>10.1f}"
                  f"{s['nb_transfers']:>11}{s['nb_orphan_transfers']:>9}{s['pct_orphan_transfers']:>8.1f}%"
                  f"{s['total_data_moved_mb']:>16.0f}{s['pct_wasted_data']:>8.1f}%{s['replication_factor']:>12.2f}{s['nb_storage_violations']:>12}")
        print("=" * 110)

    os.makedirs(RESULTS_BASE, exist_ok=True)
    with open(f"{RESULTS_BASE}/comparison_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n### Full summary saved to {RESULTS_BASE}/comparison_summary.json")


if __name__ == "__main__":
    main()
