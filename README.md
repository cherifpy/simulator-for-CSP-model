# Replicas policies simulator

# Description
This implements a simple discrete-event simulator of a distributed computing platform where clients can submit *Jobs* to a MasterNode.
These Jobs will be scheduled to be executed on a set of ComputeNodes.
A Job is composed of M *Tasks* that all have the same execution time and that share the same *Dataset*.
Tasks among the same Job can be scheduled on different ComputeNodes according to any scheduling policy.
The processing of a Task can only start when the related dataset has been totally transferred on the chosen ComputeNode.
A *Tacker* monitors what's happening and log the transfer and processing durations for post-processing.

Note that this only simulates the processing and transfer times, i.e. no real application are actually executed and no data are actually transferred.

# Usage

## Create a virtual environment and install required packages
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

## Configure the simulation parameters. 
There is a config.json in the repo contain all the hyperparametre needed: 

    
    "total_nb_jobs"
    
    "total_nb_compute_nodes"
    "homogeneous"

    "min_nb_tasks_per_job"
    "max_nb_tasks_per_job"
    
    "min_task_duration_sec"
    "max_task_duration_sec"
    
    "min_dataset_size_MB"
    "max_dataset_size_MB"
    
    "replication_factor_defaults"
    
    "compute_node_bw_MBps"
    "compute_node_latency_ms"
    "min_compute_node_bw_MBps"
    "max_compute_node_bw_MBps"
    
    "lambda_rate"
    
    "max_replica_peer_job"
    "max_replica_with_rescheduling"
    "jobs_file_path"
    "changing_priority"

    "threshold"
    # Replicas policies simulator (CSP model)

    A small discrete-event simulator that models job scheduling and dataset
    replication across compute nodes. This repository simulates task execution
    and dataset transfers (no real computation or network I/O is performed).

    Key ideas
    - A Job is composed of several Tasks that share the same Dataset.
    - Tasks can be scheduled on different ComputeNodes, but a Task can start
        only after its Dataset is available on the ComputeNode.
    - A MasterNode schedules tasks and a Tracker logs transfer and processing
        events for post-processing and visualization.

    Prerequisites
    - Python 3.10+ (virtual environment recommended)
    - See `requirements.txt` for Python package dependencies.

    Quick start

    1. Create and activate a virtual environment, then install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    2. Configure the simulation (optional). There is a `simulator/config.json`
         file that contains the main hyper-parameters. Important keys include:

    - total_nb_jobs
    - total_nb_compute_nodes
    - homogeneous
    - min_nb_tasks_per_job / max_nb_tasks_per_job
    - min_task_duration_sec / max_task_duration_sec
    - min_dataset_size_MB / max_dataset_size_MB
    - replication_factor_defaults
    - compute_node_bw_MBps / compute_node_latency_ms
    - lambda_rate
    - jobs_file_path
    - (See `simulator/config.json` for the full list and default values)

    3. Run an example experiment from the repository root:

    ```bash
    # run from repo root (example)
    python3 simulator/exps/xp_on_heterogeneous_nodes_csp.py --config simulator/config.json --log-level INFO

    # or, change into the simulator folder and run:
    cd simulator
    python3 exps/xp_on_heterogeneous_nodes_csp.py --config config.json --log-level INFO
    ```

    Notes on visualization
    - The tracker can produce a Gantt-style chart showing transfer and
        processing timelines per node. This is useful for small toy examples;
        with many jobs/nodes the plot becomes cluttered.

    Assumptions and limitations
    - Jobs are generated with exponential inter-arrival times (see `lambda_rate`) or uploaded from the workload folder a set of jobs
    - Tasks of a Job share the same dataset and have equal execution time in the
        current model.
    - Transfers are serialized on a link — bandwidth sharing is not modeled in
        the first version.
    - MasterNode bandwidth is effectively infinite in the current model; the
        compute nodes are the bottleneck.
    - Many features are simple by design (e.g., scheduling policy is "least
        loaded queue"); the code is intended to be a starting point for
        experimentation and extensions.

    Project layout (important files)

    - `simulator/` — simulation modules and scripts
        - `simulator/compute_node.py` — compute node abstraction and task loop
        - `simulator/master_node_with_heterogeneous_nodes_csp.py` — master/scheduler
        - `simulator/classes/` — `job.py`, `tracker.py` (models and logging)
        - `simulator/utils/` — helpers: parsers, plotting, CSP model utilities
        - `simulator/exps/` — example experiment scripts

    - update the `requirements.txt` if you want pinned versions for reproducible runs.