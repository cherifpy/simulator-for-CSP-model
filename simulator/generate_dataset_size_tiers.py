"""Generates 4 new 10J-20N instance variants (small/medium/large/mixed dataset-size tiers) for the
live SimPy workload comparison (xp_online_grid5000.py), reusing the existing inst-10J-20N job
"shape" (nb_tasks/task_duration/arriving_time per job) but redrawing dataset_size from each
tier's range -- same tier boundaries used by exps/xp_dataset_size_sweep.py (Exp1):
    small:  [1024, 10240]    (1-10GB)
    medium: [15360, 40960]   (15-40GB)
    large:  [51200, 102400]  (50-100GB)
    mixed:  [10240, 102400]  (10-100GB span -- "full" tier in xp_dataset_size_sweep.py)

A single new infrastructure.csv (same 20 nodes, same bandwidth/computation_nodes as the existing
inst-10J-20N, storage_capacity rescaled to comfortably fit the large tier) is shared by all 4
variants, so dataset size is the only thing that varies between them -- infra capacity is not a
confound. inst-10J-20N's own storage_capacity (max ~40946MB) can't fit a single large-tier job
(51200-102400MB) on any node, hence the rescale.

Run with the venv python. Writes to
workloads/workloads-100-for_storage_constraintes/inst-10J-20N-{small,medium,large,mixed}/.
"""
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_INSTANCE_DIR = os.path.join(SCRIPT_DIR, "workloads", "workloads-100-for_storage_constraintes", "inst-10J-20N")
OUT_ROOT = os.path.join(SCRIPT_DIR, "workloads", "workloads-100-for_storage_constraintes")

TIERS = {
    "small": (1024, 10240),
    "medium": (15360, 40960),
    "large": (51200, 102400),
    "mixed": (10240, 102400),
}

SEED = 42

with open(os.path.join(BASE_INSTANCE_DIR, "jobs.json")) as f:
    base_jobs = json.load(f)

# --- Shared infrastructure: same bandwidth/computation_nodes as inst-10J-20N, storage_capacity
# rescaled to inst-20J-50N's own range (2050-102400MB) so every tier (including large) is
# feasible on at least some nodes. Drawn with its own seeded RNG, independent of the per-tier job
# RNGs below, so regenerating a tier's jobs never perturbs the shared infra.
import csv

with open(os.path.join(BASE_INSTANCE_DIR, "infrastructure.csv")) as f:
    base_infra_rows = list(csv.DictReader(f))

infra_rng = random.Random(SEED)
new_infra_rows = []
for row in base_infra_rows:
    new_infra_rows.append({
        "bandwidth": row["bandwidth"],
        "computation_nodes": row["computation_nodes"],
        "energy_consumption": row["energy_consumption"],
        # Ceiling raised to 130000 (above the mixed/large tiers' own 102400 max) so every single
        # job drawn from any tier is guaranteed placeable on at least one node -- a first
        # comprehensive run with "no solution" on a random draw would muddy the comparison.
        "storage_capacity": infra_rng.randint(2050, 130000),
    })

for tier_name, size_range in TIERS.items():
    out_dir = os.path.join(OUT_ROOT, f"inst-10J-20N-{tier_name}")
    os.makedirs(out_dir, exist_ok=True)

    job_rng = random.Random(SEED)
    new_jobs = []
    for raw in base_jobs:
        job = dict(raw)
        job["dataset_size"] = job_rng.randint(*size_range)
        new_jobs.append(job)

    with open(os.path.join(out_dir, "jobs.json"), "w") as f:
        json.dump(new_jobs, f, indent=4)

    with open(os.path.join(out_dir, "infrastructure.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bandwidth", "computation_nodes", "energy_consumption", "storage_capacity"])
        writer.writeheader()
        writer.writerows(new_infra_rows)

    sizes = [j["dataset_size"] for j in new_jobs]
    print(f"{tier_name:7s} range={size_range}  sizes={sizes}")

print(f"\nShared infra storage_capacity: min={min(r['storage_capacity'] for r in new_infra_rows)} "
      f"max={max(r['storage_capacity'] for r in new_infra_rows)} (n={len(new_infra_rows)} nodes)")
print("\nDone -- 4 instance dirs written under workloads/workloads-100-for_storage_constraintes/")
