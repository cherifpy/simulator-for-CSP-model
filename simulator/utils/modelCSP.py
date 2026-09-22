import re
import os
from pychoco import *
import pandas as pd
import random as rnd
import math
import numpy as np
import copy

# Portable base paths: computed from this file's own location instead of hardcoded to any one
# machine's home directory, so the whole `simulator/` folder can be copied anywhere (e.g. to
# Grid5000, under a different username/home path) and still work unchanged.
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.dirname(UTILS_DIR)
MODEL_DIR = os.path.join(UTILS_DIR, "model")
MINIZINC_DIR = os.path.join(UTILS_DIR, "minizincModel")

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


def schedulingUsingJavaCSP(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list, scheduling_start_time=None):
    """Single entry point to the Java CSP model. Behaves exactly like _schedulingUsingJavaCSP_impl
    (below) unless a run recording is active (utils.run_export.start_recording sets
    SOLVER_ARCHIVE_DIR), in which case every solve is also archived -- decisions, solver log, raw
    Java input/output files -- so any statistic can be computed later without re-running."""
    archive_dir = os.environ.get("SOLVER_ARCHIVE_DIR")
    if not archive_dir:
        return _schedulingUsingJavaCSP_impl(master_node, jobs, replicas_locations, nodes_free_time, scheduling_start_time)
    from utils.run_export import archive_solve
    return archive_solve(_schedulingUsingJavaCSP_impl, master_node, jobs, replicas_locations, nodes_free_time,
                         scheduling_start_time, archive_dir)


def _schedulingUsingJavaCSP_impl(master_node, jobs: list, replicas_locations: dict, nodes_free_time: list, scheduling_start_time=None):
    """
    Wrapper to call the Java CSP solver via command line.
    """
    import json

    # All exchange files (utils/model/inputs/*, outputs/*) and the compiled .class output
    # (utils/model/bin) live at fixed paths under SIMULATOR_DIR by default -- fine for one
    # solve at a time, but TWO solves running concurrently on the SAME machine (e.g. two
    # separate n_existing scenarios launched in parallel) clobber each other's inputs mid-solve
    # and read back garbage or crash outright. Setting SIMULATOR_RUN_CWD in the environment
    # (e.g. from a launcher script that gave each concurrent run its own private copy of
    # utils/model/{inputs,outputs,bin}, with lib/src symlinked back to the canonical copy since
    # those are read-only) redirects both Python's own reads/writes AND Java's (which resolves
    # its paths from its own cwd, i.e. this same directory) to that private location instead.
    # Unset (the default): behavior is byte-identical to before this existed.
    run_cwd = os.environ.get("SIMULATOR_RUN_CWD", SIMULATOR_DIR)
    model_dir = os.path.join(run_cwd, "utils", "model") if run_cwd != SIMULATOR_DIR else MODEL_DIR

    matrix = []
    jobs = sorted(jobs, key=lambda x: x.job_id)
    print(f"\n### schedulingUsingJavaCSP: {len(jobs)} job(s) to (re)schedule ###")
    print(f"{'job_id':>8}{'nb_tasks':>10}{'task_duration':>15}{'dataset_size':>14}{'arriving_time':>15}")
    for job in jobs:
        task_duration = job.tasks[0].duration if job.tasks else None
        print(f"{job.job_id:>8}{job.nb_tasks:>10}{task_duration!s:>15}{job.dataset_size:>14}{job.arriving_time:>15.2f}")
    print("replicas_locations:")
    for job_id, nodes in sorted(replicas_locations.items()):
        print(f"  job {job_id}: nodes {nodes}")
    # A transfer that has started but not yet completed doesn't appear in replicas_locations
    # yet (that's only updated on completion), so without this a job's data mid-flight to a
    # node looks like a brand-new, freely re-plannable candidate to the CSP -- letting a later
    # solve treat that node's storage as available and place something else there too, even
    # though the in-flight transfer is physically unstoppable and will land regardless. Fold
    # in every node currently transferring this job's data so it's marked alreadyResident too.
    ongoing_nodes_by_job = {}
    for key, ongoing in master_node.ongoing_transfers.items():
        if ongoing is not None:
            ongoing_job_id, ongoing_node_id = ongoing[0], ongoing[1]
            ongoing_nodes_by_job.setdefault(ongoing_job_id, set()).add(ongoing_node_id)

    for job in jobs:
        resident_nodes = list(replicas_locations.get(job.job_id, []))
        for node_id in ongoing_nodes_by_job.get(job.job_id, ()):
            if node_id not in resident_nodes:
                resident_nodes.append(node_id)
        matrix.append(resident_nodes)
    print("matrix:", matrix)
    with open(os.path.join(model_dir, "inputs", "replicas_locations.json"), "w") as f:
        json.dump(matrix, f)

    # Optional hard node filter: only written when the caller opts in (restrict_to_free_nodes),
    # so Online/plain Incremental keep considering every (storage-eligible) node, unrestricted,
    # as before. A node counts as "free" if it has nothing ongoing RIGHT NOW (no active transfer,
    # no active task) -- already-queued-but-not-yet-started future work doesn't disqualify it,
    # since nodes_free_time (built from nodesFreeTimeIncremental, which sums that queued backlog's
    # duration on top of whatever's ongoing) already tells the CSP exactly when this node truly
    # becomes free, so it won't be double-booked against that backlog either way. Requiring the
    # backlog itself to be empty was much stricter than necessary: a job with many tasks chained
    # on one node can occupy it, as far as this filter is concerned, for the job's entire
    # remaining lifetime, so the pool of "free" nodes could shrink toward zero and never recover
    # under sustained load, starving whichever job is waiting.
    free_nodes_path = os.path.join(model_dir, "inputs", "free_nodes.txt")
    if getattr(master_node, 'restrict_to_free_nodes', False):
        free_node_ids = []
        for node_id in range(len(master_node.compute_nodes)):
            key = f'node_{node_id}'
            idle = (
                master_node.ongoing_transfers.get(key) is None
                and master_node.ongoing_works.get(key) is None
            )
            if idle:
                free_node_ids.append(node_id)
        print("free nodes:", free_node_ids)
        with open(free_nodes_path, "w") as f:
            f.write(",".join(str(n) for n in free_node_ids))
    else:
        with open(free_nodes_path, "w") as f:
            f.write("")

    jobs_data = []
    for job in jobs:
        if len([task.duration for task in job.tasks if task.status == "NotStarted"])> 0:
            jobs_data.append({
                "job_id": job.job_id,
                "dataset_size": job.dataset_size,
                "nb_tasks": len([task.duration for task in job.tasks if task.status == "NotStarted"]),
                "task_duration": job.tasks[0].duration ,
                "timelasped": int(master_node.env.now - job.arriving_time)+1,
                # LOCAL-frame lower bound on this job's first transfer start (0 = "now" for this
                # solve, same frame as every "start"/"end" the solver outputs). Always 0 for any
                # job that's genuinely already arrived (job.arriving_time <= env.now, the normal
                # live-replanning case) -- only nonzero when a batch jointly solves jobs with
                # staggered real arrival times relative to this solve's own env.now (e.g. state
                # A's one-shot joint solve at env.now=0 over jobs that "arrive" at different real
                # times), where it's exactly job.arriving_time itself. Without this, nothing in
                # the CSP stops a job's data transfer from starting before the job has arrived.
                "job_arriving_time": max(0, job.arriving_time - master_node.env.now),
            })
    jobs_data = sorted(jobs_data, key=lambda x: x['job_id'])
    

    pd.DataFrame(jobs_data).to_json(os.path.join(model_dir, "inputs", "jobs.json"), orient="records", indent=4)

    # Per-scheduler-class solver time budget (e.g. Online vs Incremental can be compared at
    # different budgets); Main.java falls back to 120s if this file is missing/unreadable.
    solver_time_limit_s = master_node._config.get('solver_time_limit_s', 120)
    with open(os.path.join(model_dir, "inputs", "solver_time_limit.txt"), "w") as f:
        f.write(str(int(solver_time_limit_s)))

    # Optional: which objectives[] entry MainOnline/MainOnlineWarmStart should actually optimize
    # (0=sum flow time, 1=max flow time, 2=one specific job's own flow time -- see those files'
    # own comments). Only written when a caller explicitly opts in via master_node.objective_choice
    # (e.g. xp_online_warmstart_test.py); absent otherwise, so every existing caller keeps its
    # current default untouched (MainOnline.java: 1: MainOnlineWarmStart.java: 0).
    objective_choice = getattr(master_node, 'objective_choice', None)
    objective_choice_path = os.path.join(model_dir, "inputs", "objective_choice.txt")
    if objective_choice is not None:
        with open(objective_choice_path, "w") as f:
            f.write(str(int(objective_choice)))
    else:
        with open(objective_choice_path, "w") as f:
            f.write("")

    # Java only ever works in a LOCAL frame (0 = "now" for this solve) -- it has no idea what
    # the simulator's absolute clock reads. Pass it along purely so debug prints can show
    # absolute times directly comparable to the final solution's "start:"/"end:" values.
    with open(os.path.join(model_dir, "inputs", "current_sim_time.txt"), "w") as f:
        f.write(str(master_node.env.now))

    # Ghost storage: jobs NOT part of this solve's batch (e.g. a job that already had every
    # task dispatched and dropped out of reconsideration, or -- for Incremental -- literally
    # every other job) but whose data is still physically resident somewhere. The CSP can't
    # reconsider them, but it still needs to know that space is taken -- and, crucially, WHEN
    # it frees up, using the same deletion time already decided for it if one exists, rather
    # than blindly shrinking capacity for the entire horizon.
    batch_job_ids = {job.job_id for job in jobs}
    ghost_lines = []
    for jid, node_ids in replicas_locations.items():
        if jid in batch_job_ids or jid >= len(master_node.jobs):
            continue
        size = master_node.jobs[jid].dataset_size
        for node_id in node_ids:
            deletion_time = -1
            for pending_jid, pending_time in master_node.deletions.get(f'node_{node_id}', []):
                if pending_jid == jid:
                    deletion_time = max(0, int(pending_time - master_node.env.now))
                    break
            ghost_lines.append(f"{node_id},{int(size)},{deletion_time}")
    with open(os.path.join(model_dir, "inputs", "ghost_storage.txt"), "w") as f:
        f.write("\n".join(ghost_lines))

    nodes_list = []
    for node_id, node in enumerate(master_node.compute_nodes):
        storage_capacity = getattr(node, 'storage_capacity', float('inf'))
        nodes_list.append({
            "node_id": node_id,
            "bandwidth": node.bandwidth,
            "compute_capacity": node.compute_capacity,
            "free_time": nodes_free_time[node_id],
            # JSON/Java have no "infinity": cap at a value the CSP treats as effectively unlimited.
            "storage_capacity": int(storage_capacity) if storage_capacity != float('inf') else 2**30,
            # Only consumed by MainOnlineMultiObj.java (energy as a genuine second objective,
            # not just a post-hoc Python-side computation) -- harmless additive field for every
            # other Java entry point, which never reads it.
            "energy_consumption": getattr(node, 'energy_consumption', 0.0),
        })
    pd.DataFrame(nodes_list).to_json(os.path.join(model_dir, "inputs", "nodes.json"), orient="records", indent=4)

    # Only consumed by MainOnlineMultiObj.java, to price each candidate transfer's energy
    # exactly the way Tracker.log_transfer / compute_transfer_energy already do on the Python
    # side. Written unconditionally (cheap) so it's always in sync with config.json; every other
    # Java entry point never reads this file.
    with open(os.path.join(model_dir, "inputs", "energy_config.txt"), "w") as f:
        f.write(f"{master_node._config.get('master_energy_consumption', 0.0)}\n")
        f.write(f"{master_node._config.get('network_energy_per_transfer', 0.0)}\n")

    # Optional: opt into MainOnlineMultiObj.java's multi-objective modes instead of a plain
    # single-objective findOptimalSolution. master_node.multi_objective: 1 (or True) = raw
    # Pareto-front search over {max flow time, energy} (found to perform far worse than
    # single-objective within the same budget on real scenarios -- kept for reference/further
    # investigation, not recommended); 2 = epsilon-constraint (recommended): phase 1 minimizes
    # max flow time via the SAME well-tuned single-objective search, phase 2 then minimizes
    # energy subject to max flow time staying within master_node.epsilon_fraction (default 10%,
    # see MainOnlineMultiObj.java) of phase 1's result. Absent/falsy -> ordinary single-objective
    # search, every existing caller unaffected.
    multi_objective = getattr(master_node, 'multi_objective', None)
    with open(os.path.join(model_dir, "inputs", "multi_objective.txt"), "w") as f:
        f.write(str(int(multi_objective)) if multi_objective else "")

    epsilon_fraction = getattr(master_node, 'epsilon_fraction', None)
    with open(os.path.join(model_dir, "inputs", "epsilon_fraction.txt"), "w") as f:
        f.write(str(epsilon_fraction) if epsilon_fraction is not None else "")

    epsilon_phase1_fraction = getattr(master_node, 'epsilon_phase1_fraction', None)
    with open(os.path.join(model_dir, "inputs", "epsilon_phase1_fraction.txt"), "w") as f:
        f.write(str(epsilon_phase1_fraction) if epsilon_phase1_fraction is not None else "")

    # Optional absolute ceiling on phase 2's cap (e.g. a baseline approach's own max flow time
    # from a prior run) -- never let epsilon_fraction's relative slack push phase 2 to accept
    # worse flow time than that baseline already achieves for free. Absent -> no ceiling.
    epsilon_max_cap = getattr(master_node, 'epsilon_max_cap', None)
    with open(os.path.join(model_dir, "inputs", "epsilon_max_cap.txt"), "w") as f:
        f.write(str(epsilon_max_cap) if epsilon_max_cap is not None else "")

    import subprocess

    """nodes = [{"bandwidth":n.bandwidth,"cpu": n.compute_capacity, "free_time": nodes_free_time[i]} for i, n in enumerate(master_node.compute_nodes)]
    pd.DataFrame(nodes).to_csv(os.path.join(MODEL_DIR, "inputs", "nodes.json"), index=False)"""
    
   

    # Run
    # Each scheduler class points at its own Java entry point (MainOnline / MainIncremental)
    # via a `java_main_class` attribute; falls back to the original shared "Main" for any
    # scheduler that doesn't set one (e.g. SemiOnline).
    java_main_class = getattr(master_node, 'java_main_class', 'Main')
    result = subprocess.run(
        [
            "javac",
            "-cp",
            os.path.join(model_dir, "lib", "*"),
            "-d",
            os.path.join(model_dir, "bin"),
            os.path.join(model_dir, "src", "main", f"{java_main_class}.java")
        ],
        capture_output=True,
        text=True,
        # The Java side resolves its input/output file paths relative to its own working
        # directory (System.getProperty("user.dir")), on the assumption that it's launched
        # with run_cwd (SIMULATOR_DIR unless SIMULATOR_RUN_CWD overrides it) as cwd. That's
        # only true by accident when a human runs `cd simulator && python3 ...` -- e.g. under
        # `oarsub "python3 ~/.../launcher.py ..."` the process inherits oarsub's own cwd (the
        # submitter's home dir) instead, and Java then looks for jobs.json etc. under the wrong
        # directory entirely. Pin it explicitly so it doesn't depend on how/where the caller
        # happened to be when this got invoked.
        cwd=run_cwd,
    )
    print("Compilation Error")
    print(str(result.stderr))
    if result.returncode != 0:
        raise RuntimeError(
            f"javac failed to compile {java_main_class}.java (exit code {result.returncode}).\n"
            f"--- stderr ---\n{result.stderr}"
        )
    print('Start looking for a solution')
    result = subprocess.run(
        [
            "java",
            # Some JDK builds (25+) enable the Graal-based JIT (JVMCI) by default, which fails
            # at startup with "graal_create_isolate error" in memory-constrained environments
            # (e.g. a tightly cgroup-limited Grid5000 job) -- forcing the standard JIT sidesteps
            # that isolate-allocation failure entirely, and Choco Solver needs nothing from Graal.
            # UnlockExperimentalVMOptions is required on JDKs where JVMCI is still gated as
            # experimental; it's a harmless no-op on newer ones where it no longer is.
            "-XX:+UnlockExperimentalVMOptions",
            "-XX:-UseJVMCICompiler",
            "-cp",
            os.path.join(model_dir, "bin") + ":" + os.path.join(model_dir, "lib", "*"),
            f"main.{java_main_class}"
        ],
        capture_output=True,
        text=True,
        cwd=run_cwd,  # see the javac call above -- Java resolves paths relative to this.
    )

    print("results")
    print(str(result.stdout))
    if result.returncode != 0:
        print("Java scheduler exited with code", result.returncode)
        print(str(result.stderr))
        # Don't fall through to toDict(): it would silently read whatever works.csv/
        # transfers.csv were left over from a PREVIOUS successful solve (or nothing at all),
        # producing a confusing downstream IndexError/KeyError instead of surfacing the real
        # cause. Fail loudly here, with the JVM's own stderr, right where it happened.
        raise RuntimeError(
            f"Java scheduler ({java_main_class}) exited with code {result.returncode}.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


    transfers = {}
    works = {}

    model_output_path = os.path.join(model_dir, "outputs")

    #df_transfers = pd.read_csv(f"{model_output_path}/transfers.csv")
    #df_works = pd.read_csv(f"{model_output_path}/works.csv")
    job_ids = [job['job_id'] for job in jobs_data]
    works = toDict(f"{model_output_path}/works.csv", job_list=job_ids, master_node=master_node )
    transfers = toDict(f"{model_output_path}/transfers.csv", job_list=job_ids, master_node=master_node)
    deletions = loadDeletions(f"{model_output_path}/deletions.csv", job_list=job_ids, master_node=master_node)

    # toDict()/loadDeletions() always pre-populate one key per node (even with an empty CSV), so
    # their dicts are never actually empty -- callers can't tell "no solution" apart from "solved"
    # by checking len(keys()) > 0, which is always true. A real solution always assigns every
    # NotStarted task in the batch a work slot, so an all-empty `works` is the real "solver found
    # nothing" signal; surface it as the {} sentinel callers already use for "nothing to solve".
    if not any(len(v) > 0 for v in works.values()):
        return {}, {}, {}

    transfers, works = sortSolution(transfers, works)
    return transfers, works, deletions


def toDict(path_to_csv, nb_nodes=None, job_list=None, time=None,master_node=None):
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
            now = master_node.env.now
            if 'task_index' in row.keys():
                dict_info[f"node_{node_index}"].append((job_list[job_index], node_index, task_index, now+start_time, now+end_time, end_time - start_time))
                print(f"node_{node_index} - job {job_list[job_index]} - task {task_index} - start: {now+start_time} - end: {now+end_time}")
            else:
                dict_info[f"node_{node_index}"].append((job_list[job_index], node_index, now+start_time, now+end_time, end_time - start_time))
                print(f"node_{node_index} - transfer {job_list[job_index]} - start: {now+start_time} - end: {now+end_time}")

    return dict_info


def loadDeletions(path_to_csv, nb_nodes=None, job_list=None, master_node=None):
    """
    Read the CSP's keep-vs-abandon decisions: for each (job, node) it chose to abandon
    within the horizon, when to actually free that node's storage (real deletion, not
    just the model dropping the pair from consideration).
    """
    import csv

    dict_info = {}
    for node_id in range(nb_nodes if nb_nodes is not None else 100):
        dict_info[f"node_{node_id}"] = []

    with open(path_to_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        now = master_node.env.now
        for row in reader:
            job_index = int(row["job_index"])
            node_index = int(row["node_index"])
            deletion_time = int(row["deletion_time"])
            dict_info[f"node_{node_index}"].append((job_list[job_index], now + deletion_time))
            print(f"node_{node_index} - deletion of job {job_list[job_index]} scheduled at {now + deletion_time}")

    for key, item in dict_info.items():
        dict_info[key] = sorted(item, key=lambda x: x[1])

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
    dzn_path = os.path.join(MINIZINC_DIR, "inputs", "params.dzn")
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
        os.path.join(MINIZINC_DIR, "scheduler.mzn"),
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
            with open(os.path.join(MINIZINC_DIR, "outputs", "sortie.json"), "w") as f:
                f.write(last_json)

    if result.returncode == 0:
        transfers, works = getResults(
            jobs, master_node, params["nb_data"],
            params["nb_nodes"], params["nb_works"],
            os.path.join(MINIZINC_DIR, "outputs", "sortie.json")
        )

        # reset du fichier
        with open(os.path.join(MINIZINC_DIR, "outputs", "sortie.json"), "w") as f:
            f.write('{}')

        transfers, works = sortSolution(transfers, works)
        return transfers, works, {}  # no storage constraint / deletions in the Minizinc model

    return {}, {}, {}


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
