"""Builds results-grid5000/analysis/generated_scenarios_notebook.ipynb: complete analysis of the
10 randomly-generated scenarios (exps/xp_generated_scenarios.py) -- each iteration draws a FRESH
n_existing (uniform in [5,15]), fresh job characteristics (nb_tasks/task_duration/dataset_size,
mixed-tier range) and fresh infrastructure (bandwidth/compute_capacity/storage_capacity), then
runs incremental@30s / online_biobj@600s / hybrid (alpha=0.5, max_budget=600, gate=max) on that
same scenario for all 3 approaches.

For flow time / scheduling time / energy: mean/std/max/min + CDF. For data volume / number of
transfers: barplots. Plus the n_existing relationship, since that varies freely across the 10
scenarios instead of being fixed.

Run with the venv python, then execute via nbconvert.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# 10 scénarios générés aléatoirement -- analyse complète

Chaque itération tire, indépendamment des autres : un `n_existing` uniforme dans [5, 15], des
jobs frais (nb_tasks/durée/taille de dataset, plage "mixte" 10240-102400 MB) et une
infrastructure fraîche (bande passante/capacité de calcul/stockage) -- rien n'est rejoué depuis
un fichier figé. Les 3 approches (`incremental`, `online_biobj`, `hybrid` -- version finale :
6 variantes, split par moyenne, sélection batch-wide) tournent sur le **même** scénario à
l'intérieur d'une itération, donc directement comparables entre elles.

Pour flow time / temps de scheduling / énergie : mean/std/max/min + CDF. Pour volume de données /
nombre de transferts : barplots.""")

code("""
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

RESULTS_DIR = ".."  # notebook lives in results-grid5000/analysis/
SCENARIOS_DIR = f"{RESULTS_DIR}/generated_scenarios_10iter"
N_ITERATIONS = 10

APPROACH_COLORS = {"incremental": "#4C72A0", "online_biobj": "#DD8452", "hybrid": "#55A868"}
APPROACH_ORDER = ["incremental", "online_biobj", "hybrid"]
APPROACH_ALIAS = {"epsilon": "online_biobj"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#888888",
    "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})

def c(a):
    return APPROACH_COLORS.get(a, "#999999")

def plot_cdf(ax, values, color, label):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, linewidth=2.2, label=label)

def stats_row(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": np.nan, "std": np.nan, "max": np.nan, "min": np.nan, "n": 0}
    return {"mean": v.mean(), "std": v.std(), "max": v.max(), "min": v.min(), "n": len(v)}

_RESULT_LINE_RE = re.compile(r"\\[(\\w+) #(-?\\d+)\\] (\\w+) result: (\\{.*\\})")

def scheduling_times(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = _RESULT_LINE_RE.search(line)
            if m:
                tier, repeat, approach, d = m.groups()
                dd = ast.literal_eval(d)
                rows.append({"approach": APPROACH_ALIAS.get(approach, approach),
                             "scheduling_time_s": dd.get("scheduling_time_s")})
    return pd.DataFrame(rows)

jobs = pd.read_csv(f"{SCENARIOS_DIR}/combined_jobs_detail.csv")
jobs["approach"] = jobs["approach"].replace(APPROACH_ALIAS)
transfers = pd.read_csv(f"{SCENARIOS_DIR}/combined_transfers_detail.csv")
transfers["approach"] = transfers["approach"].replace(APPROACH_ALIAS)

jobs_cmp = jobs[jobs.approach.isin(APPROACH_ORDER)]
transfers_cmp = transfers[transfers.approach.isin(APPROACH_ORDER)]

n_existing_by_iter = {
    it: jobs[(jobs.iteration == it) & (jobs.approach == "state_A")]["job_id"].nunique()
    for it in range(N_ITERATIONS)
}

sched_frames = []
for it in range(N_ITERATIONS):
    s = scheduling_times(f"{SCENARIOS_DIR}/iter_{it}/solver_stdout.log")
    s["iteration"] = it
    sched_frames.append(s)
sched = pd.concat(sched_frames, ignore_index=True)

overview = pd.DataFrame([{"iteration": it, "n_existing": n_existing_by_iter[it]} for it in range(N_ITERATIONS)])
overview
""")

md("""## 1. Flow time du nouveau job -- mean/std/max/min + CDF""")

code("""
new_jobs = jobs_cmp[jobs_cmp.is_new == True]
rows = []
for a in APPROACH_ORDER:
    rows.append({"approach": a, **stats_row(new_jobs[new_jobs.approach == a]["flow_time"].values)})
pd.DataFrame(rows).set_index("approach")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for a in APPROACH_ORDER:
    plot_cdf(ax, new_jobs[new_jobs.approach == a]["flow_time"].values, c(a), a)
ax.set_xlabel("Flow time du nouveau job (s)")
ax.set_ylabel("Proportion cumulée")
ax.set_ylim(0, 1.02)
ax.set_title("CDF du flow time du nouveau job -- 10 scénarios générés")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("""## 2. Flow time batch (tous les jobs, état A + nouveau) -- mean/std/max/min + CDF""")

code("""
rows = []
for a in APPROACH_ORDER:
    rows.append({"approach": a, **stats_row(jobs_cmp[jobs_cmp.approach == a]["flow_time"].values)})
pd.DataFrame(rows).set_index("approach")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for a in APPROACH_ORDER:
    plot_cdf(ax, jobs_cmp[jobs_cmp.approach == a]["flow_time"].values, c(a), a)
ax.set_xlabel("Flow time (tous jobs, s)")
ax.set_ylabel("Proportion cumulée")
ax.set_ylim(0, 1.02)
ax.set_title("CDF du flow time batch -- 10 scénarios générés")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("""## 3. Temps de scheduling -- mean/std/max/min + CDF""")

code("""
rows = []
for a in APPROACH_ORDER:
    rows.append({"approach": a, **stats_row(sched[sched.approach == a]["scheduling_time_s"].values)})
pd.DataFrame(rows).set_index("approach")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for a in APPROACH_ORDER:
    plot_cdf(ax, sched[sched.approach == a]["scheduling_time_s"].values, c(a), a)
ax.set_xlabel("Temps de scheduling (s)")
ax.set_ylabel("Proportion cumulée")
ax.set_ylim(0, 1.02)
ax.set_title("CDF du temps de scheduling -- 10 scénarios générés")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("""## 4. Énergie par transfert -- mean/std/max/min + CDF""")

code("""
rows = []
for a in APPROACH_ORDER:
    rows.append({"approach": a, **stats_row(transfers_cmp[transfers_cmp.approach == a]["energy"].values)})
pd.DataFrame(rows).set_index("approach")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for a in APPROACH_ORDER:
    plot_cdf(ax, transfers_cmp[transfers_cmp.approach == a]["energy"].values, c(a), a)
ax.set_xlabel("Énergie par transfert")
ax.set_ylabel("Proportion cumulée")
ax.set_ylim(0, 1.02)
ax.set_title("CDF de l'énergie par transfert -- 10 scénarios générés")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("""## 5. Volume de données et nombre de transferts -- barplots""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
vals = [jobs_cmp[jobs_cmp.approach == a]["job_dataset_size"].sum() for a in APPROACH_ORDER]
ax.bar(range(len(APPROACH_ORDER)), vals, color=[c(a) for a in APPROACH_ORDER])
ax.set_xticks(range(len(APPROACH_ORDER))); ax.set_xticklabels(APPROACH_ORDER)
ax.set_ylabel("Volume de données total (MB)")
ax.set_title("Volume de données -- cumulé sur les 10 scénarios")

ax = axes[1]
vals = [len(transfers_cmp[transfers_cmp.approach == a]) for a in APPROACH_ORDER]
ax.bar(range(len(APPROACH_ORDER)), vals, color=[c(a) for a in APPROACH_ORDER])
ax.set_xticks(range(len(APPROACH_ORDER))); ax.set_xticklabels(APPROACH_ORDER)
ax.set_ylabel("Nombre de transferts")
ax.set_title("Nombre de transferts -- cumulé sur les 10 scénarios")

plt.tight_layout()
plt.show()
""")

md("""## 6. Effet de `n_existing` (chaque scénario a tiré sa propre valeur)""")

code("""
rows = []
for it in range(N_ITERATIONS):
    for a in APPROACH_ORDER:
        d = new_jobs[(new_jobs.iteration == it) & (new_jobs.approach == a)]
        s = sched[(sched.iteration == it) & (sched.approach == a)]
        rows.append({
            "iteration": it, "n_existing": n_existing_by_iter[it], "approach": a,
            "new_job_flow_time": d["flow_time"].values[0] if len(d) else np.nan,
            "scheduling_time_s": s["scheduling_time_s"].values[0] if len(s) else np.nan,
        })
n_existing_df = pd.DataFrame(rows).sort_values(["n_existing", "approach"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for a in APPROACH_ORDER:
    d = n_existing_df[n_existing_df.approach == a]
    ax.scatter(d["n_existing"], d["new_job_flow_time"], color=c(a), label=a, s=70, alpha=0.8, edgecolor="white")
ax.set_xlabel("n_existing (tiré par scénario)")
ax.set_ylabel("Flow time du nouveau job (s)")
ax.set_title("Flow time du nouveau job vs n_existing")
ax.legend(frameon=False)

ax = axes[1]
for a in APPROACH_ORDER:
    d = n_existing_df[n_existing_df.approach == a]
    ax.scatter(d["n_existing"], d["scheduling_time_s"], color=c(a), label=a, s=70, alpha=0.8, edgecolor="white")
ax.set_xlabel("n_existing (tiré par scénario)")
ax.set_ylabel("Temps de scheduling (s)")
ax.set_title("Temps de scheduling vs n_existing")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
n_existing_df
""")

md("""## 7. Détail par scénario (flow time du nouveau job)""")

code("""
pivot = n_existing_df.pivot_table(index=["n_existing", "iteration"], columns="approach", values="new_job_flow_time")
pivot
""")

md("""## Synthèse""")

code("""
summary_rows = []
for a in APPROACH_ORDER:
    summary_rows.append({
        "approach": a,
        "new_job_flow_mean": new_jobs[new_jobs.approach == a]["flow_time"].mean(),
        "new_job_flow_std": new_jobs[new_jobs.approach == a]["flow_time"].std(),
        "batch_flow_mean": jobs_cmp[jobs_cmp.approach == a]["flow_time"].mean(),
        "batch_flow_max_mean": jobs_cmp[jobs_cmp.approach == a].groupby("iteration")["flow_time"].max().mean(),
        "sched_time_mean": sched[sched.approach == a]["scheduling_time_s"].mean(),
        "energy_mean": transfers_cmp[transfers_cmp.approach == a]["energy"].mean(),
    })
pd.DataFrame(summary_rows).set_index("approach").round(1)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("results-grid5000/analysis/generated_scenarios_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
