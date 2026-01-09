import json
import logging
import random
import sys
import os
import argparse
import multiprocessing
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator import (
    simulatorForOptimalPerfsUsingCSPOnline,
    generateHeterogeneousInfrastructure,
    generateHeterogeneousInfrastructureEquilibre,
    save_results_to_csv,
    configure_logging,
)

from utils.parser import ArgumentParser


from utils.plots import plot_line, plot_gantt_chart

logger = logging.getLogger(__name__)

def generateJobs(config):
    jobs= []
    for i in range(config['total_nb_jobs']):
        nb_tasks = random.randint(config['min_nb_tasks_per_job'], config['max_nb_tasks_per_job'])
        task_duration = random.uniform(config['min_task_duration_sec'], config['max_task_duration_sec'])
        dataset_size = random.uniform(config['min_dataset_size_MB'], config['max_dataset_size_MB'])
        jobs.append((nb_tasks,task_duration,dataset_size))
    return jobs


def main():

    arg_parser = ArgumentParser()
    args = arg_parser.parse()

    # Initialize logging
    configure_logging(args.log_level)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Run the simulation
    logger.info("Simulation begins with config: %s" ,str(config))
    processes = []
    
    """ Heterogeneous version """    
    random.seed(42)
    
    
    for nb_jobs, nb_nodes in [(50,100)]:# ('s','s'),('s','b'),('b','b'),('b','s') (5,10),(10,50),(20,50),(20,100),
        config['jobs_file_path'] = f"/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/GeneratedJobs/instances-100/inst1-{nb_jobs}j-{nb_nodes}Nodes/jobs.json"
        results_destination = f"/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results/results-on-instances-100/inst1-{nb_jobs}j-{nb_nodes}Nodes"
        exp_name = ""

        config['total_nb_jobs'] = nb_jobs
        config['total_nb_compute_nodes'] = nb_nodes

        nodes_config = generateHeterogeneousInfrastructureEquilibre(config, path=f"/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/workloads/GeneratedJobs/inst1-{nb_jobs}j-{nb_nodes}Nodes/infrastructure.csv")
                    
        random.seed(42)
        results, nodes_config_ = simulatorForOptimalPerfsUsingCSPOnline(config=config, jobs=[], overlap=True, poisson=True, varying_load=False,nodes_config=nodes_config)
        save_results_to_csv(logger, results, results_destination, exp_name)
        
        nodes_config_save = pd.DataFrame(nodes_config)
        nodes_config_save.to_csv(f"/{results_destination}/nodes_config.csv", index=False)

    if args.plot_gantt:
        process = multiprocessing.Process(target=plot_gantt_chart, args=(results.events_history,config['total_nb_compute_nodes'],f'Order {0}'))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
    
if __name__ == "__main__":
    main()
