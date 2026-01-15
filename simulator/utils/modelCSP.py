import re
from pychoco import *
import pandas as pd
import random as rnd
import math
import numpy as np
import copy

CPU_UNIT = 1  # defines one unit of work per second
#rnd.seed(42)
def getTransferTime(job, node_id, node_bandwidth, replicas_locations):
    """Calcule le temps de transfert en fonction des réplicas déjà disponibles"""
    if job.job_id not in replicas_locations.keys():
        return job.dataset_size / node_bandwidth
    elif node_id in replicas_locations[job.job_id]:
        return 0
    else:
        return job.dataset_size / node_bandwidth

def sortSolution(transfers, works):

    for key, item in transfers.items():
        transfers[key] = sorted(item, key= lambda x: x[2])

    for key, item in works.items():
        works[key] = sorted(item, key= lambda x: x[3])
    
    return transfers, works

def onLineSchedulingUsingCSP(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list):
    """
    Planifie dynamiquement les transferts et les exécutions de tâches sur les nœuds de calcul
    en utilisant le solveur Choco (via pychoco).

    master_node : contient la liste des compute_nodes accessibles
    jobs : liste des jobs (avec .job_id, .dataset_size, .tasks)
    replicas_locations : {job_id: [node_ids où les données sont déjà présentes]}
    nodes_free_time : liste des instants à partir desquels chaque nœud est disponible
    """

    # --------------------------
    # DATA
    # --------------------------
    nb_data = len(jobs)
    data_sizes = [job.dataset_size for job in jobs]

    # liste des durées de travaux restants à exécuter pour chaque job
    works = []
    for job in jobs:
        not_executed_tasks = [task for task in job.tasks if task.status == "NotStarted"]
        works.append([task.duration for task in not_executed_tasks])

    possible_nodes = copy.copy(master_node.compute_nodes)
    nb_nodes = len(possible_nodes)

    # --------------------------
    # NODES
    # --------------------------
    bandwidths = [node.bandwidth for node in possible_nodes]
    cpus = [node.compute_capacity * CPU_UNIT for node in possible_nodes]

    def getTransferTime(job, node_id, node_bandwidth, replicas_locations):
        """Calcule le temps de transfert en fonction des réplicas déjà disponibles"""
        if job.job_id not in replicas_locations.keys():
            return job.dataset_size / node_bandwidth
        elif node_id in replicas_locations[job.job_id]:
            return 0
        else:
            return job.dataset_size / node_bandwidth

    makespan = sum(data_sizes) // min(bandwidths)
    makespan += sum(sum(w) for w in works) * CPU_UNIT * max(cpus)
    makespan *= 2
    makespan = int(makespan)

    model = Model("Bag of Tasks Scheduling")

    transfer_tasks = []
    heights = []
    for j in range(nb_nodes):
        transfer_tasks.append([])
        heights.append([])
        for i in range(nb_data):
            s = model.intvar(int(nodes_free_time[j]) + 1, makespan, name=f"start_transfer_d{i}_n{j}")
            transfer_time = getTransferTime(jobs[i], possible_nodes[j].node_id, bandwidths[j], replicas_locations)
            d = math.ceil(transfer_time)
            t = model.task(s, d)
            h = model.intvar(0, 1, name=f"height_transfer_d{i}_n{j}")
            transfer_tasks[j].append(t)
            heights[j].append(h)

    work_tasks = []
    for j in range(nb_nodes):
        for i in range(nb_data):
            for k, w in enumerate(works[i]):
                s = model.intvar(int(nodes_free_time[j]) + 1, makespan, name=f"start_work_d{i}_w{k}_n{j}")
                d = math.ceil(w * cpus[j])  # ✅ durée corrigée
                t = model.task(s, d)
                h = model.intvar(0, 1, name=f"height_work_d{i}_w{k}_n{j}")
                work_tasks.append((t, i, j, k, h))

    # --------------------------
    # CONTRAINTES
    # --------------------------
    # Limite de bande passante sur chaque nœud
    for j in range(nb_nodes):
        model.cumulative(transfer_tasks[j], heights[j], model.intvar(1)).post()

    # Chaque donnée doit être transférée au moins une fois
    for i in range(nb_data):
        model.sum([heights[j][i] for j in range(nb_nodes)], ">=", 1).post()

    # Un travail ne peut commencer que si les données sont transférées
    for (t, i, j, k, h) in work_tasks:
        model.arithm(t.start, ">=", transfer_tasks[j][i].end).post()
        model.arithm(h, "<=", heights[j][i]).post()

    # Si un transfert a lieu sur (i,j), il doit y avoir au moins un travail sur ce nœud
    for j in range(nb_nodes):
        for i in range(nb_data):
            works_heights = [hh for (_, ii, jj, _, hh) in work_tasks if ii == i and jj == j]
            if works_heights:
                model.sum(works_heights, ">=", heights[j][i]).post()
            else:
                model.arithm(heights[j][i], "=", 0).post()

    # Capacité CPU sur chaque nœud
    for j in range(nb_nodes):
        node_work_tasks = [t for (t, _, jj, _, _) in work_tasks if jj == j]
        node_heights = [h for (_, _, jj, _, h) in work_tasks if jj == j]
        model.cumulative(node_work_tasks, node_heights, model.intvar(1)).post()

    # Chaque travail doit être exécuté une seule fois
    for i in range(nb_data):
        for k in range(len(works[i])):
            model.sum([h for (_, ii, _, kk, h) in work_tasks if ii == i and kk == k], "=", 1).post()

    makespan_var = model.intvar(0, makespan, name="makespan")
    
    """avg_flow = model.intvar(0, makespan, name="avg_flow")

    for i in range(nb_data):
        tmp = model.intvar(0, makespan)
        model.sum([heights[j][i] for j in range(nb_nodes)], "=", tmp).post()"""

    ends_tmp = []
    for (t, _, _, _, h) in work_tasks:
        tmp = model.intvar(0, makespan)
        model.times(t.end, h, tmp).post()
        ends_tmp.append(tmp)

    model.max(makespan_var, ends_tmp).post()
    model.set_objective(makespan_var, False)

    # --------------------------
    # SOLVER
    # --------------------------
    solver = model.get_solver()
    solver.show_short_statistics()
    best_transfers = {}
    best_works = {}
    best_avg_utility = -1
    transfers = {}
    works_exec = {}
    solver.limit_time("10s")
    while solver.solve():
        transfers = {}
        works_exec = {}

        for j in range(nb_nodes):
            
            

            works_exec[f"node_{j}"] = []
            for (t, ii, jj, kk, h) in work_tasks:
                if jj == j and h.get_value() == 1:
                    works_exec[f"node_{j}"].append((
                        jobs[ii].job_id,possible_nodes[j].node_id,kk,t.start.get_value(),t.end.get_value(),t.end.get_value() - t.start.get_value()
                    ))
            transfers[f"node_{j}"] = []
            for i in range(nb_data):
                if heights[j][i].get_value() == 1 and len(works_exec[f"node_{j}"]) > 0:
                    transfers[f"node_{j}"].append((
                        jobs[i].job_id,possible_nodes[j].node_id,transfer_tasks[j][i].start.get_value(),transfer_tasks[j][i].end.get_value(),transfer_tasks[j][i].end.get_value() - transfer_tasks[j][i].start.get_value()
                    ))
        #current_avg_utility = evaluateUtility(master_node, jobs, transfers, works_exec)
        #if current_avg_utility <= 1 :#master_node.threshold:
        #best_avg_utility = current_avg_utility
        best_transfers = copy.copy(transfers)
        best_works = copy.copy(works_exec)
    else:
        print("No solution found")

    return sortSolution(best_transfers, best_works)

def evaluateUtility(master_node, jobs,transfers:dict, works:dict):
    total_transfer_time = 0
    total_work_time = 0

    data_uses = {}

    for job in jobs:
        data_uses[job.job_id] = {}

    for node_key, transfer_list in transfers.items():
        for (job_id, node_id, start_time, end_time, duration) in transfer_list:
            total_transfer_time = jobs[job_id].dataset_size / master_node.compute_nodes[node_id].bandwidth
            data_uses[job_id][node_id] = (node_id, total_transfer_time, 0)
    
    for node_key, work_list in works.items():
        
        for (job_id, node_id, work_index, start_time, end_time, duration) in work_list:
            data_uses[job_id][node_id] = (node_id, data_uses[job_id][node_id][1], data_uses[job_id][node_id][2] + duration)
            
    
    utility_per_data = {}
    avg_utility = 0

    to_ckeck = [key for key in data_uses.keys() if len(data_uses[key])>1]


    for key in to_ckeck:
        job_id, uses = key , data_uses[key] 
        for node_id, values in uses.items():
            utility_per_data[(job_id, node_id)] = values[1] / values[2] if values[2] > 0 else 100
    if len(utility_per_data) == 0:
        return 0
    avg_utility = np.mean(list(utility_per_data.values()))

    return avg_utility


def schedulingUsingJavaCSP(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list):
    """
    Wrapper to call the Java CSP solver via command line.
    """
    import json

    matrix = []
    print("replicas_locations:", replicas_locations)
    print('jobs:', [job.job_id for job in jobs])
    
    for job in jobs:
        if job in replicas_locations.keys():
            matrix.append(replicas_locations[job.job_id])
        else:
            matrix.append([])

    with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/replicas_locations.json", "w") as f:
        json.dump(matrix, f)

    jobs_data = []
    for job in jobs:
        if len([task.duration for task in job.tasks if task.status == "NotStarted"])> 0:
            jobs_data.append({
                "job_id": job.job_id,
                "dataset_size": job.dataset_size,
                "nb_tasks": len([task.duration for task in job.tasks if task.status == "NotStarted"]),
                "task_duration": job.tasks[0].duration 
            })
    jobs_data = sorted(jobs_data, key=lambda x: x['job_id'])
    

    pd.DataFrame(jobs_data).to_json("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/jobs.json", orient="records", indent=4)

    nodes_list = []
    for node_id, node in enumerate(master_node.compute_nodes):
        nodes_list.append({
            "node_id": node_id,
            "bandwidth": node.bandwidth,
            "compute_capacity": node.compute_capacity,
            "free_time": nodes_free_time[node_id]
        })
    pd.DataFrame(nodes_list).to_json("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/nodes.json", orient="records", indent=4)

    import subprocess

    """nodes = [{"bandwidth":n.bandwidth,"cpu": n.compute_capacity, "free_time": nodes_free_time[i]} for i, n in enumerate(master_node.compute_nodes)]
    pd.DataFrame(nodes).to_csv("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/inputs/nodes.json", index=False)"""
    
   

    # Run
    #javac   -d bin src/main/Main.java
    result = subprocess.run(
        [
            "javac", 
            "-cp",
            "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/lib/*",
            "-d",
            "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/bin",
            "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/src/main/Main.java"
        ],
        capture_output=True,
        text=True
    )  
    
    print(str(result.stderr))

    print('Start looking for a solution')
    #java -cp "" main.Main
    result = subprocess.run(
        [
            "java", 
            "-cp",
            "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/bin:/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/lib/*",    
            "main.Main"
        ],
        capture_output=True,
        text=True
    )  

    print("results")

    

    transfers = {}
    works = {}

    model_output_path = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/model/outputs"

    #df_transfers = pd.read_csv(f"{model_output_path}/transfers.csv")
    #df_works = pd.read_csv(f"{model_output_path}/works.csv")
    job_ids = [job['job_id'] for job in jobs_data]
    works = toDict(f"{model_output_path}/works.csv", job_list=job_ids)
    transfers = toDict(f"{model_output_path}/transfers.csv", job_list=job_ids)
    
    
    return sortSolution(transfers, works) # Implementation would go here


def toDict(path_to_csv, nb_nodes=None, job_list=None):
    import csv
    # Création du dictionnaire
    dict_info = {}

    for node_id in range(nb_nodes if nb_nodes is not None else 100):
        dict_info[f"node_{node_id}"] = []

    # Lecture du CSV
    with open(path_to_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'task_index' in row.keys():
                task_index = int(row["task_index"])
            job_index = int(row["job_index"])
            start_time = int(row["start_time"])
            end_time = int(row["end_time"])
            node_index = int(row["node_index"])
            
            # On remplit la structure works_exec
            
            if 'task_index' in row.keys():
                dict_info[f"node_{node_index}"].append((job_list[job_index], node_index, task_index, start_time, end_time, end_time - start_time ))
            else:
                dict_info[f"node_{node_index}"].append((job_list[job_index], node_index, start_time, end_time, end_time - start_time ))

    return dict_info


def startMinizincModel(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list):

    transfers_time = []
    for node in master_node.compute_nodes:
        transfer_time_for_node = []
        for job in jobs:
            transfer_time = getTransferTime(job, node.node_id, node.bandwidth, replicas_locations)
            transfer_time_for_node.append(int(transfer_time))
        transfers_time.append(transfer_time_for_node)
    params = {
        "nb_nodes": len(master_node.compute_nodes),
        "nb_data": len(jobs),
        "makespan": 221506,
        "data_sizes": [job.dataset_size for job in jobs],
        "work_duration": [jobs[i].tasks[0].duration for i in range(len(jobs))],
        "bandwidths": [node.bandwidth for node in master_node.compute_nodes],
        "cpus": [node.compute_capacity * CPU_UNIT for node in master_node.compute_nodes],
        "transfers_time": transfers_time,
        "nb_works": [len([task for task in job.tasks if task.status == "NotStarted"]) for job in jobs],
        "node_free_timespan": [t for t in nodes_free_time.values()],
    }

        # ---- Convert to DZN ----
    dzn_path = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/minizincModel/inputs/params.dzn"
    with open(dzn_path, "w") as d:
        for key, value in params.items():
            if key == "transfers_time":
                d.write("transfers_time = [")
                for row in value:
                    d.write("|" + ", ".join(map(str, row)) + ",")
                d.write("|];\n")
            elif isinstance(value, list):
                d.write(f"{key} = {value};\n")
            else:
                d.write(f"{key} = {value};\n")

    import subprocess

    command = [
        "minizinc",
        "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/minizincModel/scheduler.mzn",
        dzn_path,
        "--solver", "CP-SAT",
        "--output-mode", "json",
        "-p", "8",
        "-i", 
        "-t","600000",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("MiniZinc ERROR:")
        print(result.stderr)
    else:
        raw = result.stdout

        # isoler tous les blocs {...}
        json_blocks = re.findall(r'\{[\s\S]*?\}', raw)

        if not json_blocks:
            print("Aucun bloc JSON trouvé.")
        else:
            # prendre la dernière solution
            last_json = json_blocks[-1]

            # écrire proprement dans le fichier
            with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/minizincModel/outputs/sortie.json", "w") as f:
                f.write(last_json)

    if result.returncode == 0:
        transfers, works = getResults(
            jobs, master_node, params["nb_data"],
            params["nb_nodes"], params["nb_works"],
            "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/minizincModel/outputs/sortie.json"
        )

        # reset du fichier
        with open("/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/utils/minizincModel/outputs/sortie.json", "w") as f:
            f.write('{}')

        return sortSolution(transfers, works)

    return {}, {}


def getResults(jobs, master_node, nb_data, nb_nodes, nb_works, output_path: str):
    import json

    with open(output_path, "r") as f:
        data = json.load(f)
    if len(data) == 0:
        return {}, {}
    transfers = {f"node_{j}": [] for j in range(nb_nodes)}
    works = {f"node_{j}": [] for j in range(nb_nodes)}

    for j in range(nb_nodes):
        for i in range(nb_data):
            for k in range(nb_works[i]):
                if data["work_height"][j][i][k] == 1:
                    works[f"node_{j}"].append((jobs[i].job_id, master_node.compute_nodes[j].node_id, k, data["work_start"][j][i][k], data["work_end"][j][i][k], data["work_end"][j][i][k] - data["work_start"][j][i][k]))
    
    for j in range(nb_nodes):
        for i in range(nb_data):
            #dict_info[f"node_{node_index}"].append((job_index, node_index, task_index, start_time, end_time, end_time - start_time ))
            if data["transfer_height"][j][i] == 1:
                transfers[f"node_{j}"].append((jobs[i].job_id, master_node.compute_nodes[j].node_id, data["transfer_start"][j][i], data["transfer_end"][j][i], data["transfer_end"][j][i] - data["transfer_start"][j][i]))

    
    return transfers, works
