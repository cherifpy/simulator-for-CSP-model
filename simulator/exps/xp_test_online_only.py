import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: gantt chart is saved to file, not shown interactively

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
from master_node_with_heterogeneous_nodes_csp import SchedulingUsingCSPOnline
from utils.plots import plot_gantt_chart

logger = logging.getLogger(__name__)

NB_JOBS, NB_NODES = 10, 20
LAMBDA_RATE = 60
SOLVER_TIME_LIMIT_S = 30

INSTANCE_DIR = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/workloads-100-for_storage_constraintes/inst-10J-20N"
RESULTS_DIR = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results-online-only"


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configure_logging(logging.WARNING)

    with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    config['total_nb_jobs'] = NB_JOBS
    config['total_nb_compute_nodes'] = NB_NODES
    config['jobs_file_path'] = f"{INSTANCE_DIR}/jobs.json"
    config['lambda_rate'] = LAMBDA_RATE
    config['solver_time_limit_s'] = SOLVER_TIME_LIMIT_S

    random.seed(42)
    nodes_config = generateHeterogeneousInfrastructureEquilibre(config, path=f"{INSTANCE_DIR}/infrastructure.csv")

    random.seed(42)
    results, _ = simulatorForOptimalPerfsUsingCSPOnline(
        config=config, jobs=[], overlap=True, poisson=True, varying_load=False,
        nodes_config=nodes_config, master_class=SchedulingUsingCSPOnline,
    )

    save_results_to_csv(logger, results, RESULTS_DIR, "")

    with open(f"{RESULTS_DIR}/events_history.json", "w") as f:
        json.dump(results.events_history, f)

    gantt_path = f"{RESULTS_DIR}/gantt.png"
    plot_gantt_chart(results.events_history, NB_NODES, title="inst-10J-20N (online only)", save_path=gantt_path)

    print(f"### DONE: total wall time = {results.total_wall_time}")
    print(f"### GANTT_SAVED: {gantt_path}")
    return RESULTS_DIR


def verify_storage(results_dir):
    """Rebuild real per-node storage occupancy from the actual simulated events (transfer
    completions + deletions), and flag any instant where it exceeds the node's capacity.
    Uses ground truth, not the CSP's own per-solve decisions, so it also catches violations
    caused by non-sticky replanning across solves, not just modeling gaps."""
    with open(f"{INSTANCE_DIR}/jobs.json") as f:
        jobs = json.load(f)
    job_size = {j['job_id']: j['dataset_size'] for j in jobs}

    import csv
    node_capacity = {}
    with open(f"{INSTANCE_DIR}/infrastructure.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            node_capacity[i] = float(row['storage_capacity']) if 'storage_capacity' in row else float('inf')

    with open(f"{results_dir}/events_history.json") as f:
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

    print()
    print("=" * 70)
    if not violations:
        print("STORAGE CHECK: OK -- no violation at any instant, on any node.")
    else:
        print(f"STORAGE CHECK: {len(violations)} violation(s) found:")
        for node, t, occupied, cap, label in violations:
            print(f"  node_{node} at t={t:.2f}: occupied={occupied:.1f} > capacity={cap:.1f} ({label})")
    print("=" * 70)


if __name__ == "__main__":
    results_dir = run()
    verify_storage(results_dir)
