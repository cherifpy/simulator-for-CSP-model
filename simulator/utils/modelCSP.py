from pychoco import *
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

def choiceNodesOnlineVersion2(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list):
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
    solver.limit_time("30s")
    while solver.solve():
        transfers = {}
        works_exec = {}

        for j in range(nb_nodes):
            transfers[f"node_{j}"] = []
            for i in range(nb_data):
                if heights[j][i].get_value() == 1:
                    transfers[f"node_{j}"].append((
                        jobs[i].job_id,possible_nodes[j].node_id,transfer_tasks[j][i].start.get_value(),transfer_tasks[j][i].end.get_value(),transfer_tasks[j][i].end.get_value() - transfer_tasks[j][i].start.get_value()
                    ))

            works_exec[f"node_{j}"] = []
            for (t, ii, jj, kk, h) in work_tasks:
                if jj == j and h.get_value() == 1:
                    works_exec[f"node_{j}"].append((
                        jobs[ii].job_id,possible_nodes[j].node_id,kk,t.start.get_value(),t.end.get_value(),t.end.get_value() - t.start.get_value()
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