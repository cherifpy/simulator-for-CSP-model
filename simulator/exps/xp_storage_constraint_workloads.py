import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: gantt charts are saved to file, not shown interactively

import json
import logging
import random
import sys
import multiprocessing
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator import (
    simulatorForOptimalPerfsUsingCSPOnline,
    generateHeterogeneousInfrastructureEquilibre,
    save_results_to_csv,
    configure_logging,
)

from utils.parser import ArgumentParser
from utils.plots import plot_gantt_chart

logger = logging.getLogger(__name__)


def main():
    arg_parser = ArgumentParser()
    args = arg_parser.parse()

    configure_logging(args.log_level)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config['lambda_rate'] = 100

    processes = []

    random.seed(42)
    # (nb_jobs, nb_nodes) matching the inst-{J}J-{N}N folders in workloads-100-for_storage_constraintes.
    # "pools" instances are intentionally skipped for now.
    for nb_jobs, nb_nodes in [(20, 50), (50, 50), (50, 100), (100, 100), (200, 100)]:
        instance_dir = f"/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/workloads-100-for_storage_constraintes/inst-{nb_jobs}J-{nb_nodes}N"
        results_destination = f"/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results-with-storage-contrainte-100/inst1-{nb_jobs}j-{nb_nodes}Nodes"

        os.makedirs(results_destination, exist_ok=True)

        config['total_nb_jobs'] = nb_jobs
        config['total_nb_compute_nodes'] = nb_nodes
        config['jobs_file_path'] = f"{instance_dir}/jobs.json"

        logger.info("Simulation begins for inst-%sJ-%sN with config: %s", nb_jobs, nb_nodes, str(config))

        nodes_config = generateHeterogeneousInfrastructureEquilibre(config, path=f"{instance_dir}/infrastructure.csv")

        random.seed(42)
        results, nodes_config_ = simulatorForOptimalPerfsUsingCSPOnline(
            config=config, jobs=[], overlap=True, poisson=True, varying_load=False, nodes_config=nodes_config
        )
        save_results_to_csv(logger, results, results_destination, "")

        pd.DataFrame(nodes_config).to_csv(f"{results_destination}/nodes_config.csv", index=False)

        gantt_path = f"{results_destination}/gantt.png"
        process = multiprocessing.Process(
            target=plot_gantt_chart,
            args=(results.events_history, config['total_nb_compute_nodes'], f'inst-{nb_jobs}J-{nb_nodes}N'),
            kwargs={'save_path': gantt_path},
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
