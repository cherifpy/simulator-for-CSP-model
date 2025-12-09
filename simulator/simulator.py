import argparse
import json
import logging
import random
import simpy
import pandas as pd
from compute_node import ComputeNode
from classes.job import Job
from master_node_with_heterogeneous_nodes_csp import SchedulingUsingCSPOnline
from utils.plots import plot_gantt_chart
from classes.tracker import Tracker


logger = logging.getLogger(__name__)

def generate_jobs(env, master, config,jobs=[]):

    """Generate jobs, each with multiple tasks with the same duration and a related dataset
    The number of tasks in each job, the tasks duration, and the dataset size are all selected
    from a uniform distribution between a Min and Max values."""
    print(f"generate jobs {len(jobs)}")
    for i in range(config['total_nb_jobs']):
        job_id = i
        if len(jobs)>0:
            nb_tasks = jobs[i][0]
            task_duration= jobs[i][1]
            dataset_size=jobs[i][2]
        else:
            nb_tasks = random.randint(config['min_nb_tasks_per_job'], config['max_nb_tasks_per_job'])
            task_duration = random.uniform(config['min_task_duration_sec'], config['max_task_duration_sec'])
            dataset_size = random.uniform(config['min_dataset_size_MB'], config['max_dataset_size_MB'])
        yield master.queue.put(Job(job_id, task_duration, nb_tasks, dataset_size))
        inter_arrival_time = random.expovariate(config['lambda_rat'])
        yield env.timeout(inter_arrival_time)  # Simulate job inter-arrivals


def jobsInjectorBasedOnLambdaPoisson(env, master, job_file_path = None, lambda_rate=10):

    if job_file_path !=None:
        jobs = []
        with open(job_file_path, "r", encoding="utf-8") as file:
            jobs = json.load(file)
        
        nb_injected_jobs = 0
        while nb_injected_jobs < len(jobs):
            inter_arrival_time = random.expovariate(lambd=1/lambda_rate)
            yield env.timeout(inter_arrival_time)  # Simulate job inter-arrivals # Wait for a short time to allow the environment to process events
            
            job = jobs[nb_injected_jobs]
            
            yield master.queue.put(Job(job['job_id'], job['task_duration'], job['nb_tasks'], job['dataset_size']))
            nb_injected_jobs += 1
    else:
        master.logger.error("No job file path provided. Please provide a valid path to the job file.")  
        return

def jobsInjectorWithVariyingLoad(env, master, job_file_path = None):

    if job_file_path !=None:
        jobs = []
        with open(job_file_path, "r", encoding="utf-8") as file:
            jobs = json.load(file)
        
        nb_injected_jobs = 0
        lambda_rate = master._config['lambda_rate']

        while nb_injected_jobs < len(jobs):
            if nb_injected_jobs % 200 == 0 and nb_injected_jobs != 0:
                # Simulate a varying load by changing the inter-arrival time
                while lambda_rate == master._config['lambda_rate']:
                    lambda_rate = random.choice([3, 5, 7, 10])
                    print("lambda changed to ", lambda_rate)
                master._config['lambda_rate'] = lambda_rate
            inter_arrival_time = random.expovariate(lambd=1/lambda_rate)
            
            yield env.timeout(inter_arrival_time)  # Simulate job inter-arrivals # Wait for a short time to allow the environment to process events
            
            job = jobs[nb_injected_jobs]
            
            yield master.queue.put(Job(job['job_id'], job['task_duration'], job['nb_tasks'], job['dataset_size']))
            nb_injected_jobs += 1
    else:
        
        return

def jobs_injector(env, master, jobs = [], job_file_path = None, config= None):

    """
        if job_file_path is not None, we read the jobs from the file
        else we generate the jobs
    """
    
    if job_file_path !=None:
        jobs = []
        with open(job_file_path, "r", encoding="utf-8") as file:
            jobs = json.load(file)
        config['total_nb_jobs'] = len(jobs)
        current_timeout = 0
        for job in jobs:
            next_arriving_time = job['arriving_time'] 
            waiting_time = next_arriving_time - current_timeout
            yield env.timeout(waiting_time)
            current_timeout = next_arriving_time
            yield master.queue.put(Job(job['job_id'], job['task_duration'], job['nb_tasks'], job['dataset_size']))
    else:
        for i in range(config['total_nb_jobs']):
            job_id = i
            if len(jobs)>0:
                nb_tasks = jobs[i][0]
                task_duration= jobs[i][1]
                dataset_size=jobs[i][2]
            else:
                nb_tasks = random.randint(config['min_nb_tasks_per_job'], config['max_nb_tasks_per_job'])
                task_duration = random.uniform(config['min_task_duration_sec'], config['max_task_duration_sec'])
                dataset_size = random.uniform(config['min_dataset_size_MB'], config['max_dataset_size_MB'])
            yield master.queue.put(Job(job_id, task_duration, nb_tasks, dataset_size))
            inter_arrival_time = random.expovariate(config['jobs_inter_arrival_expovariate'])# np.random.poisson(lam=15) #
            yield env.timeout(inter_arrival_time)  # Simulate job inter-arrivals
    
def configure_logging(log_level):
    """Configure the logger format."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )




def simulatorForOptimalPerfsUsingCSPOnline(config,jobs=[], overlap = False, threshold = 1, poisson = False, varying_load = False, nodes_config=[]):
    env = simpy.Environment()
    tracker = Tracker(env)

    if len(nodes_config) == 0:
        logging.error("No nodes configuration provided for heterogeneous nodes.")

    master = SchedulingUsingCSPOnline(env, [], tracker, config, overlap=overlap)
    
    
    compute_nodes = [ComputeNode(env, i, master,  
                                bandwidth=nodes_config[i]['bandwidth'],
                                compute_capacity=nodes_config[i]['computation_nodes'], 
                                energy_consumption=nodes_config[i]['energy_consumption']) for i, node in enumerate(nodes_config)] 

    env.process(master.receiveJobs())
    env.process(master.scheduling())
    env.process(master.checkOnJobs())
    
    master.compute_nodes = compute_nodes
    master.nb_nodes = len(compute_nodes)
    
    for node in compute_nodes:
        env.process(node.processTasks())

    if poisson:   
        env.process(jobsInjectorBasedOnLambdaPoisson(env, master, job_file_path=config['jobs_file_path'], lambda_rate=config['lambda_rate']))        
    else:
        env.process(jobs_injector(env, master,jobs=jobs,job_file_path=config['jobs_file_path'], config=config))
    
    env.run()
    
    saveReplicaState(master)

    return tracker, nodes_config



def generateHeterogeneousInfrastructure(config, node_homogeneous = True):
    """Generate a heterogeneous infrastructure with random bandwidth for each compute node."""
    bandwidth = [random.randint(config['min_compute_node_bw_MBps'], config['max_compute_node_bw_MBps']) for i in range(config['total_nb_compute_nodes'])]
    computation_nodes = [1 for i in range(config['total_nb_compute_nodes'])] if config['homogeneous'] else [random.uniform(0.1,2.1) for i in range(config['total_nb_compute_nodes'])]  # Random computation power for each node
    energy_consumption = [1 for i in range(config['total_nb_compute_nodes'])] if config['homogeneous'] else  [random.uniform(0.1, 2.1) for i in range(config['total_nb_compute_nodes'])]  # Random energy consumption for each node
    nodes_config = [{'bandwidth':bandwidth[i], 'computation_nodes':computation_nodes[i], 'energy_consumption':energy_consumption[i]} for i in range(config['total_nb_compute_nodes'])]
    
    return nodes_config



def save_results_to_csv(logger, results, file_name, exp_name, prefex=""):
    logger.info("total wall time: %s", str(results.total_wall_time))
    logger.info("total transferred bytes: %s", str(results.total_nb_transferred_bytes))
    mean_job_processing_time = get_jobs_mean_processing_time(results)
    logger.info("mean job processing time (end_time - submission_time): %s", str(mean_job_processing_time))
    df = pd.DataFrame(results.stats_on_tasks).to_csv(f"{file_name}/{exp_name}/infos_on_tasks{prefex}.csv")
    df = pd.DataFrame(results.stats_on_jobs).to_csv(f"{file_name}/{exp_name}/infos_on_jobs{prefex}.csv")
    df = pd.DataFrame(results.stats_on_replicas).to_csv(f"{file_name}/{exp_name}/infos_on_replicas{prefex}.csv")
    if len(results.threshold_history) > 0: df = pd.DataFrame(results.threshold_history).to_csv(f"{file_name}/{exp_name}/threshold_history{prefex}.csv")

# Example on how to extract basic statistics about simulation results
def get_jobs_mean_processing_time(results):
    total_job_duration = 0
    for job in results.tasks_duration_per_job.values():
        job_duration = job['end_time'] - job['start_time']
        total_job_duration += job_duration
    return total_job_duration/len(results.tasks_duration_per_job)


def saveReplicaState(self):
    infos = []
    for key, item in self.replicas_stats.items():
        infos.append({
            "job_id":key[0],
            "node_id":key[1],
            "nb_tasks":item.nb_tasks,
            "transfert_time":item.transfer_time,
            "time_of_use":item.nb_tasks*item.task_execution_time if item.nb_tasks > 0 else 0,
            "data_size":item.data_size
        })

    self.tracker.stats_on_replicas = infos
