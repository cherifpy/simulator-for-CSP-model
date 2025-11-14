from pychoco import *
import random as rnd
import math

seed = 42
rnd.seed(seed)

CPU_UNIT = 1  # defines one unit of work per second

# DATA
## Data size, in MB
nb_data = 5
data_sizes = [rnd.choice([2048, 40960]) for _ in range(nb_data)]
## for each data, works to do
nb_min_works = 1
nb_max_works = 20
max_duration_per_work = 100  # in units of work
works = [[rnd.choice([10, max_duration_per_work]) for _ in range(rnd.randint(nb_min_works, nb_max_works))] for _ in range(nb_data)]

# NODES
nb_nodes = 100
## bandwidth
bandwidths = [rnd.randint(12, 800) for _ in range(nb_nodes)]
cpus = [rnd.randint(1, 10) * CPU_UNIT for _ in range(nb_nodes)]

makespan = sum(data_sizes) // min(bandwidths)
makespan += sum(sum(w) for w in works) * CPU_UNIT // min(cpus)
makespan *= 2
print(makespan)

# print inputs
print("DATA")
for i in range(nb_data):
    print(f" Data {i}: size={data_sizes[i]} MB, works={works[i]}")
    for w in works[i]:
        print(f"   - work of {w} units")
print("NODES")
for j in range(nb_nodes):
    print(f" Node {j}: bandwidth={bandwidths[j]} MB/s, cpu={cpus[j]} units/s")

# MODEL
model = Model("Bag of Tasks Scheduling")

## TASKS
### First, create tasks corresponding to data transfers to nodes
transfer_tasks = []
heights = []
for j in range(nb_nodes):
    transfer_tasks.append([])
    heights.append([])
    for i in range(nb_data):
        s = model.intvar(0, makespan, name=f"start_transfer_d{i}_n{j}")
        d = math.ceil(data_sizes[i] / bandwidths[j])
        t = model.task(s, d)
        h = model.intvar(0, 1, name=f"height_transfer_d{i}_n{j}")
        transfer_tasks[j].append(t)
        heights[j].append(h)

### Then, create tasks corresponding to works on nodes
work_tasks = []
for j in range(nb_nodes):
    for i in range(nb_data):
        for k, w in enumerate(works[i]):
            s = model.intvar(0, makespan, name=f"start_work_d{i}_w{k}_n{j}")
            # --- MODIF : durée des travaux conforme à la version Java d'origine ---
            d = w * cpus[j]               # <-- correction : multiplication (Java used w * cpus[j])
            t = model.task(s, d)
            h = model.intvar(0, 1, name=f"height_work_d{i}_w{k}_n{j}")
            work_tasks.append((t, i, j, k, h))

## CONSTRAINTS
### Cumulative constraints on edges
for j in range(nb_nodes):
    model.cumulative(transfer_tasks[j], heights[j], model.intvar(1)).post()

### at least one transfer per data
for i in range(nb_data):
    model.sum([heights[j][i] for j in range(nb_nodes)], ">=", 1).post()

### A work can start only after the corresponding data transfer is done, 
### and only if the data has been transferred to the node
for (t, i, j, k, h) in work_tasks:
    model.arithm(t.start, ">=", transfer_tasks[j][i].end).post()
    model.arithm(h, "<=", heights[j][i]).post()    

### --- MODIF : si un transfert a lieu sur (j,i) alors au moins un travail sur (j,i) doit être fait ---
for j in range(nb_nodes):
    for i in range(nb_data):
        # collect heights of works for data i on node j
        work_heights = [hh for (_, ii, jj, _, hh) in work_tasks if ii == i and jj == j]
        if work_heights:
            model.sum(work_heights, ">=", heights[j][i]).post()
        else:
            # s'il n'y a aucun work pour (j,i) alors on impose que le transfert ne puisse pas avoir lieu
            model.arithm(heights[j][i], "=", 0).post()

### Cumulative constraint on nodes for works
for j in range(nb_nodes):
    node_work_tasks = [t for (t, _, jj, _, _) in work_tasks if jj == j]
    node_heights = [h for (_, _, jj, _, h) in work_tasks if jj == j]
    model.cumulative(node_work_tasks, node_heights, model.intvar(1)).post()

### Each work must be done exactly once
# for each work
for i in range(nb_data):
    for k in range(len(works[i])):
        model.sum([h for (_, ii, _, kk, h) in work_tasks if ii == i and kk == k], "=", 1).post()


## OBJECTIVE
makespan_var = model.intvar(0, makespan, name="makespan")

# --- MODIF : ne prendre en compte que les travaux activés dans le max du makespan ---
ends_tmp = []
for (t, _, _, _, h) in work_tasks:
    tmp = model.intvar(0, makespan)                # variable auxiliaire
    model.times(t.end, h, tmp).post()              # tmp = t.end * h -> 0 si travail non choisi
    ends_tmp.append(tmp)

model.max(makespan_var, ends_tmp).post()
model.set_objective(makespan_var, False)

# SOLVER
solver = model.get_solver()
solver.limit_time("30s")
solver.show_short_statistics()
while solver.solve():
    print("Solution found with makespan =", makespan_var.get_value())
    for j in range(nb_nodes):
        print(f"Node {j}:")
        for i in range(nb_data):
            if heights[j][i].get_value() == 1:
                print(f"  Data {i} transferred from {transfer_tasks[j][i].start.get_value()} to {transfer_tasks[j][i].end.get_value()}")
        for (t, ii, jj, kk, h) in work_tasks:
            if jj == j and h.get_value() == 1:
                print(f"  Work {kk} of Data {ii} processed from {t.start.get_value()} to {t.end.get_value()}")
    print()
else:
    print("No solution found")
