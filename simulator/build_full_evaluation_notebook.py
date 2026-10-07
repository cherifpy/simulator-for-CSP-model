"""Builds results-grid5000/analysis/full_evaluation_notebook.ipynb: the complete evaluation of
incremental / online_biobj / hybrid (final version: 6-variant parallel escalation, mean-split,
batch-wide max-flow-time selection) across two experiment families:

  Part A -- Live workload (xp_online_grid5000.py), inst-10J-20N, 4 dataset-size tiers
            (small/medium/large/mixed), one full live run per (tier x approach) = 12 runs.
  Part B -- State-A single-decision-point protocol (xp_dataset_size_sweep.py):
            B.1 dataset-size sweep (n_existing=5, 4 tiers)
            B.2 infra-load sweep (mixed tier, n_existing in {5, 10, 15})

For flow time / scheduling time / energy: mean/std/max/min + CDF. For data volume / number of
transfers: barplots. Scheduling time per replan is now a REAL distribution for every approach
(not just hybrid) -- the "### SCHEDULING ELAPSED: Xs ###" print added this session to
_timedSchedulingUsingJavaCSP covers incremental/online_biobj/hybrid's F1 probe; hybrid's escalation
itself is still read from "### PARALLEL ESCALATION: ... (Xs), ... ###" (the slowest of the 6
concurrent variants, since that's what's actually charged against simulated time).

Run with the venv python, then execute via nbconvert.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Évaluation complète -- incremental / online_biobj / hybrid (version finale)

Hybrid "version finale" = escalade parallèle à 6 variantes (`freeze_below_mean`,
`freeze_above_mean`, `nofreeze`, `warm_nofreeze`, `restrict_powerful_nodes`,
`freeze_ongoing_transfer`), split par moyenne, sélection par flow time max du batch entier
(corrigé le 2026-10-02 après une régression trouvée avec un critère plus étroit).

**Partie A** -- workload live (`inst-10J-20N`), 4 paliers de taille de dataset, 12 runs complets.
**Partie B** -- protocole état-A (décision unique figée) : B.1 sweep taille de dataset
(`n_existing=5`), B.2 sweep charge infra (`n_existing` ∈ {5, 10, 15}, dataset mixte).

Pour flow time / temps de scheduling / énergie : mean/std/max/min + CDF. Pour volume de données /
nombre de transferts : barplots.""")

code("""
import ast
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

RESULTS_DIR = ".."  # notebook lives in results-grid5000/analysis/

APPROACH_COLORS = {"incremental": "#4C72A0", "online_biobj": "#DD8452", "hybrid": "#55A868"}
APPROACH_ORDER = ["incremental", "online_biobj", "hybrid"]
# exp1 (state-A) scripts call online_biobj "epsilon" -- normalized at load time, never displayed.
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

_SCHED_ELAPSED_RE = re.compile(r"### SCHEDULING ELAPSED: ([\\d.]+)s")
_PARALLEL_ESC_RE = re.compile(r"### PARALLEL ESCALATION: (.*) ###")
_VARIANT_TIME_RE = re.compile(r"\\(([\\d.]+)s\\)")
_WINNER_RE = re.compile(r"### PARALLEL ESCALATION WINNER: (\\S+)")

def live_replan_times(log_path, approach):
    \"\"\"Real wall-clock cost per replan/decision for a LIVE run. incremental/online_biobj: every
    '### SCHEDULING ELAPSED: Xs ###' line (one per replan, the whole decision). hybrid: every
    '### PARALLEL ESCALATION: ... (Xs), ... ###' line, max per line (the slowest of the 6
    concurrent variants -- what's actually charged); the cheap F1 probe (also logged via
    SCHEDULING ELAPSED) is excluded to stay comparable with the other two approaches' OWN full
    decision cost, not hybrid's two-stage breakdown.\"\"\"
    times = []
    if approach == "hybrid":
        with open(log_path) as f:
            for line in f:
                m = _PARALLEL_ESC_RE.search(line)
                if m:
                    vt = [float(v) for v in _VARIANT_TIME_RE.findall(m.group(1))]
                    if vt:
                        times.append(max(vt))
    else:
        with open(log_path) as f:
            for line in f:
                m = _SCHED_ELAPSED_RE.search(line)
                if m:
                    times.append(float(m.group(1)))
    return times

def winner_tally(log_path):
    counts = Counter()
    with open(log_path) as f:
        for line in f:
            m = _WINNER_RE.search(line)
            if m:
                counts[m.group(1)] += 1
    return counts

_RESULT_LINE_RE = re.compile(r"\\[(\\w+) #(-?\\d+)\\] (\\w+) result: (\\{.*\\})")

def state_a_scheduling_times(log_path):
    \"\"\"scheduling_time_s per (tier, repeat, approach) decision, parsed from the inline dict
    literal xp_dataset_size_sweep.py prints -- never exported to a CSV.\"\"\"
    rows = []
    with open(log_path) as f:
        for line in f:
            m = _RESULT_LINE_RE.search(line)
            if not m:
                continue
            tier, repeat, approach, dict_str = m.groups()
            d = ast.literal_eval(dict_str)
            rows.append({"tier": tier, "repeat": int(repeat),
                         "approach": APPROACH_ALIAS.get(approach, approach),
                         "scheduling_time_s": d.get("scheduling_time_s")})
    return pd.DataFrame(rows)
""")

# ===========================================================================
md("""## Partie A -- Workload live, `inst-10J-20N`, 4 paliers de taille

20 jobs... non, 10 jobs / 20 nœuds, lambda_rate=200. Chaque palier (small/medium/large/mixed)
redessine la taille de dataset de chacun des 10 jobs dans sa propre plage ; infrastructure
partagée entre les 4 paliers (stockage assez large pour que même le plus gros job de chaque
palier soit plaçable).""")

code("""
TIERS = ["small", "medium", "large", "mixed"]
TIER_DIRS = {t: f"dataset_size_tiers_inst-10J-20N-{t}" for t in TIERS}

def load_live_run(tier, approach):
    base = f"{RESULTS_DIR}/{TIER_DIRS[tier]}/{approach}"
    jobs = pd.read_csv(f"{base}/infos_on_jobs.csv", index_col=0)
    jobs["flow_time"] = jobs["finishing_time"] - jobs["arriving_time"]
    energy = pd.read_csv(f"{base}/infos_on_transfers_energy.csv", index_col=0)
    sched = live_replan_times(f"{base}/solver_stdout.log", approach)
    return jobs, energy, sched

live_data = {(t, a): load_live_run(t, a) for t in TIERS for a in APPROACH_ORDER}
print("Loaded", len(live_data), "(tier, approach) live runs")
""")

md("""### A.1 Flow time -- mean/std/max/min + CDF, par palier""")

code("""
rows = []
for (t, a), (jobs, energy, sched) in live_data.items():
    s = stats_row(jobs["flow_time"].values)
    rows.append({"tier": t, "approach": a, **s})
flow_time_stats = pd.DataFrame(rows).set_index(["tier", "approach"])
flow_time_stats
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, t in zip(axes, TIERS):
    for a in APPROACH_ORDER:
        jobs, _, _ = live_data[(t, a)]
        plot_cdf(ax, jobs["flow_time"].values, c(a), a)
    ax.set_title(f"tier={t}")
    ax.set_xlabel("Flow time (s)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("Partie A -- CDF du flow time par palier de taille", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""### A.2 Temps de scheduling par replan -- mean/std/max/min + CDF, par palier

Pour `incremental`/`online_biobj` : chaque replan = 1 décision = 1 appel Java, mesuré en entier.
Pour `hybrid` : coût de l'escalade (le plus lent des 6 variants concurrents, ce qui est
effectivement facturé) -- la sonde F1 (peu coûteuse, plafonnée à 30s) est exclue pour rester
comparable au coût de décision complet des deux autres approches.""")

code("""
rows = []
for (t, a), (jobs, energy, sched) in live_data.items():
    s = stats_row(sched)
    rows.append({"tier": t, "approach": a, **s})
sched_time_stats = pd.DataFrame(rows).set_index(["tier", "approach"])
sched_time_stats
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, t in zip(axes, TIERS):
    for a in APPROACH_ORDER:
        _, _, sched = live_data[(t, a)]
        plot_cdf(ax, sched, c(a), a)
    ax.set_title(f"tier={t}")
    ax.set_xlabel("Temps de scheduling par replan (s)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("Partie A -- CDF du temps de scheduling par replan, par palier", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""### A.3 Énergie par transfert -- mean/std/max/min + CDF, par palier""")

code("""
rows = []
for (t, a), (jobs, energy, sched) in live_data.items():
    s = stats_row(energy["total_energy"].values)
    rows.append({"tier": t, "approach": a, **s})
energy_stats = pd.DataFrame(rows).set_index(["tier", "approach"])
energy_stats
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, t in zip(axes, TIERS):
    for a in APPROACH_ORDER:
        _, energy, _ = live_data[(t, a)]
        plot_cdf(ax, energy["total_energy"].values, c(a), a)
    ax.set_title(f"tier={t}")
    ax.set_xlabel("Énergie par transfert")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("Partie A -- CDF de l'énergie par transfert, par palier", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""### A.4 Volume de données et nombre de transferts -- barplots""")

code("""
rows = []
for (t, a), (jobs, energy, sched) in live_data.items():
    rows.append({
        "tier": t, "approach": a,
        "data_volume_total_MB": jobs["dataset size"].sum(),
        "data_volume_mean_MB": jobs["dataset size"].mean(),
        "n_transfers": len(energy),
    })
volume_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(TIERS))
width = 0.25

ax = axes[0]
for i, a in enumerate(APPROACH_ORDER):
    vals = [volume_df[(volume_df.tier == t) & (volume_df.approach == a)]["data_volume_total_MB"].iloc[0] for t in TIERS]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(TIERS)
ax.set_ylabel("Volume de données total (MB)")
ax.set_title("Volume de données par palier")
ax.legend(frameon=False)

ax = axes[1]
for i, a in enumerate(APPROACH_ORDER):
    vals = [volume_df[(volume_df.tier == t) & (volume_df.approach == a)]["n_transfers"].iloc[0] for t in TIERS]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(TIERS)
ax.set_ylabel("Nombre de transferts")
ax.set_title("Nombre de transferts par palier")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""### A.5 Qui gagne l'escalade (hybrid), par palier""")

code("""
rows = []
for t in TIERS:
    base = f"{RESULTS_DIR}/{TIER_DIRS[t]}/hybrid"
    tally = winner_tally(f"{base}/solver_stdout.log")
    rows.append({"tier": t, **tally})
pd.DataFrame(rows).set_index("tier").fillna(0).astype(int)
""")

md("""### A.6 Synthèse Partie A""")

code("""
summary_a = pd.concat([
    flow_time_stats.add_prefix("flow_time_"),
    sched_time_stats.add_prefix("sched_time_"),
    energy_stats.add_prefix("energy_"),
], axis=1)
summary_a
""")

# ===========================================================================
md("""## Partie B -- Protocole état-A (décision unique figée)

État A construit une seule fois par cellule (tier, ou n_existing), puis les 3 approches
décident toutes sur ce même état A et le même instant T -- comparaison directe garantie.""")

md("""### B.1 Sweep taille de dataset (n_existing=5)""")

code("""
B1_DIR = f"{RESULTS_DIR}/exp1_final_dataset_size_sweep_n5"
b1_jobs = pd.read_csv(f"{B1_DIR}/jobs_detail_by_run.csv")
b1_jobs["approach"] = b1_jobs["approach"].replace(APPROACH_ALIAS)
b1_transfers = pd.read_csv(f"{B1_DIR}/transfers_detail_by_run.csv")
b1_transfers["approach"] = b1_transfers["approach"].replace(APPROACH_ALIAS)
b1_sched = state_a_scheduling_times(f"{B1_DIR}/solver_stdout.log")

# Exclude the frozen 'state_A' baseline rows -- we compare the 3 REAL approaches' own plans.
b1_jobs_cmp = b1_jobs[b1_jobs["approach"].isin(APPROACH_ORDER)]
b1_transfers_cmp = b1_transfers[b1_transfers["approach"].isin(APPROACH_ORDER)]
b1_jobs_cmp[["tier", "approach", "job_id", "is_new", "flow_time", "job_dataset_size"]].head(10)
""")

md("""#### B.1.1 Flow time (across all jobs in state A + the new job) -- mean/std/max/min + CDF""")

code("""
rows = []
for t in TIERS[:3] + ["full"]:
    for a in APPROACH_ORDER:
        d = b1_jobs_cmp[(b1_jobs_cmp.tier == t) & (b1_jobs_cmp.approach == a)]
        rows.append({"tier": t, "approach": a, **stats_row(d["flow_time"].values)})
pd.DataFrame(rows).set_index(["tier", "approach"])
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, t in zip(axes, ["small", "medium", "large", "full"]):
    for a in APPROACH_ORDER:
        d = b1_jobs_cmp[(b1_jobs_cmp.tier == t) & (b1_jobs_cmp.approach == a)]
        plot_cdf(ax, d["flow_time"].values, c(a), a)
    ax.set_title(f"tier={t}")
    ax.set_xlabel("Flow time (s)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("B.1 -- CDF du flow time (tous les jobs de l'état A + nouveau), par palier", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""#### B.1.2 Temps de scheduling par décision -- mean/std/max/min (une seule décision par cellule, pas de CDF significative)""")

code("""
b1_sched.pivot_table(index="tier", columns="approach", values="scheduling_time_s")
""")

md("""#### B.1.3 Énergie par transfert -- mean/std/max/min + CDF""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, t in zip(axes, ["small", "medium", "large", "full"]):
    for a in APPROACH_ORDER:
        d = b1_transfers_cmp[(b1_transfers_cmp.tier == t) & (b1_transfers_cmp.approach == a)]
        plot_cdf(ax, d["energy"].values, c(a), a)
    ax.set_title(f"tier={t}")
    ax.set_xlabel("Énergie par transfert")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("B.1 -- CDF de l'énergie par transfert, par palier", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""#### B.1.4 Volume de données et nombre de transferts -- barplots""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(4)
width = 0.25
tiers_b1 = ["small", "medium", "large", "full"]

ax = axes[0]
for i, a in enumerate(APPROACH_ORDER):
    vals = [b1_jobs_cmp[(b1_jobs_cmp.tier == t) & (b1_jobs_cmp.approach == a)]["job_dataset_size"].sum() for t in tiers_b1]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(tiers_b1)
ax.set_ylabel("Volume de données total (MB)")
ax.set_title("B.1 -- volume de données par palier")
ax.legend(frameon=False)

ax = axes[1]
for i, a in enumerate(APPROACH_ORDER):
    vals = [len(b1_transfers_cmp[(b1_transfers_cmp.tier == t) & (b1_transfers_cmp.approach == a)]) for t in tiers_b1]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(tiers_b1)
ax.set_ylabel("Nombre de transferts")
ax.set_title("B.1 -- nombre de transferts par palier")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""### B.2 Sweep charge infra (dataset mixte fixe, n_existing variable)""")

code("""
N_EXISTING_VALUES = [5, 10, 15]
B2_DIRS = {n: f"{RESULTS_DIR}/exp1_final_infra_load_sweep_n{n}" for n in N_EXISTING_VALUES}

b2_jobs_all, b2_transfers_all, b2_sched_all = [], [], []
for n, d in B2_DIRS.items():
    jobs = pd.read_csv(f"{d}/jobs_detail_by_run.csv")
    jobs["approach"] = jobs["approach"].replace(APPROACH_ALIAS)
    jobs["n_existing"] = n
    b2_jobs_all.append(jobs)

    transfers = pd.read_csv(f"{d}/transfers_detail_by_run.csv")
    transfers["approach"] = transfers["approach"].replace(APPROACH_ALIAS)
    transfers["n_existing"] = n
    b2_transfers_all.append(transfers)

    sched = state_a_scheduling_times(f"{d}/solver_stdout.log")
    sched["n_existing"] = n
    b2_sched_all.append(sched)

b2_jobs = pd.concat(b2_jobs_all, ignore_index=True)
b2_transfers = pd.concat(b2_transfers_all, ignore_index=True)
b2_sched = pd.concat(b2_sched_all, ignore_index=True)

b2_jobs_cmp = b2_jobs[b2_jobs["approach"].isin(APPROACH_ORDER)]
b2_transfers_cmp = b2_transfers[b2_transfers["approach"].isin(APPROACH_ORDER)]
b2_jobs_cmp[["n_existing", "approach", "job_id", "flow_time"]].head(10)
""")

md("""#### B.2.1 Flow time vs charge infra -- mean/std/max/min + CDF""")

code("""
rows = []
for n in N_EXISTING_VALUES:
    for a in APPROACH_ORDER:
        d = b2_jobs_cmp[(b2_jobs_cmp.n_existing == n) & (b2_jobs_cmp.approach == a)]
        rows.append({"n_existing": n, "approach": a, **stats_row(d["flow_time"].values)})
pd.DataFrame(rows).set_index(["n_existing", "approach"])
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, n in zip(axes, N_EXISTING_VALUES):
    for a in APPROACH_ORDER:
        d = b2_jobs_cmp[(b2_jobs_cmp.n_existing == n) & (b2_jobs_cmp.approach == a)]
        plot_cdf(ax, d["flow_time"].values, c(a), a)
    ax.set_title(f"n_existing={n}")
    ax.set_xlabel("Flow time (s)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("B.2 -- CDF du flow time vs charge infra (dataset mixte)", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""#### B.2.2 Temps de scheduling vs charge infra""")

code("""
b2_sched.pivot_table(index="n_existing", columns="approach", values="scheduling_time_s")
""")

code("""
fig, ax = plt.subplots(figsize=(7.5, 5.5))
piv = b2_sched.pivot_table(index="n_existing", columns="approach", values="scheduling_time_s")
for a in APPROACH_ORDER:
    if a in piv.columns:
        ax.plot(piv.index, piv[a], marker="o", markersize=7, linewidth=2.2, color=c(a), label=a)
ax.set_xticks(N_EXISTING_VALUES)
ax.set_xlabel("n_existing")
ax.set_ylabel("Temps de scheduling (s)")
ax.set_title("B.2 -- temps de scheduling vs charge infra")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("""#### B.2.3 Énergie par transfert vs charge infra -- mean/std/max/min + CDF""")

code("""
rows = []
for n in N_EXISTING_VALUES:
    for a in APPROACH_ORDER:
        d = b2_transfers_cmp[(b2_transfers_cmp.n_existing == n) & (b2_transfers_cmp.approach == a)]
        rows.append({"n_existing": n, "approach": a, **stats_row(d["energy"].values)})
pd.DataFrame(rows).set_index(["n_existing", "approach"])
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, n in zip(axes, N_EXISTING_VALUES):
    for a in APPROACH_ORDER:
        d = b2_transfers_cmp[(b2_transfers_cmp.n_existing == n) & (b2_transfers_cmp.approach == a)]
        plot_cdf(ax, d["energy"].values, c(a), a)
    ax.set_title(f"n_existing={n}")
    ax.set_xlabel("Énergie par transfert")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("B.2 -- CDF de l'énergie par transfert vs charge infra", y=1.03)
plt.tight_layout()
plt.show()
""")

md("""#### B.2.4 Volume de données et nombre de transferts vs charge infra -- barplots""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(N_EXISTING_VALUES))
width = 0.25

ax = axes[0]
for i, a in enumerate(APPROACH_ORDER):
    vals = [b2_jobs_cmp[(b2_jobs_cmp.n_existing == n) & (b2_jobs_cmp.approach == a)]["job_dataset_size"].sum() for n in N_EXISTING_VALUES]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(N_EXISTING_VALUES)
ax.set_xlabel("n_existing")
ax.set_ylabel("Volume de données total (MB)")
ax.set_title("B.2 -- volume de données vs charge infra")
ax.legend(frameon=False)

ax = axes[1]
for i, a in enumerate(APPROACH_ORDER):
    vals = [len(b2_transfers_cmp[(b2_transfers_cmp.n_existing == n) & (b2_transfers_cmp.approach == a)]) for n in N_EXISTING_VALUES]
    ax.bar(x + (i - 1) * width, vals, width, color=c(a), label=a)
ax.set_xticks(x); ax.set_xticklabels(N_EXISTING_VALUES)
ax.set_xlabel("n_existing")
ax.set_ylabel("Nombre de transferts")
ax.set_title("B.2 -- nombre de transferts vs charge infra")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""## Synthèse globale

Une ligne par (famille d'expérience × condition × approche), flow time et énergie moyens pour une
lecture rapide d'ensemble.""")

code("""
final_rows = []
for (t, a), (jobs, energy, sched) in live_data.items():
    final_rows.append({
        "famille": "A: live 10J-20N", "condition": t, "approche": a,
        "flow_time_moyen": jobs["flow_time"].mean(), "energie_moyenne": energy["total_energy"].mean(),
        "temps_scheduling_moyen_s": np.mean(sched) if sched else np.nan,
    })
for t in ["small", "medium", "large", "full"]:
    for a in APPROACH_ORDER:
        d = b1_jobs_cmp[(b1_jobs_cmp.tier == t) & (b1_jobs_cmp.approach == a)]
        dt = b1_transfers_cmp[(b1_transfers_cmp.tier == t) & (b1_transfers_cmp.approach == a)]
        st = b1_sched[(b1_sched.tier == t) & (b1_sched.approach == a)]["scheduling_time_s"]
        final_rows.append({
            "famille": "B.1: état-A taille dataset", "condition": t, "approche": a,
            "flow_time_moyen": d["flow_time"].mean(), "energie_moyenne": dt["energy"].mean(),
            "temps_scheduling_moyen_s": st.mean() if len(st) else np.nan,
        })
for n in N_EXISTING_VALUES:
    for a in APPROACH_ORDER:
        d = b2_jobs_cmp[(b2_jobs_cmp.n_existing == n) & (b2_jobs_cmp.approach == a)]
        dt = b2_transfers_cmp[(b2_transfers_cmp.n_existing == n) & (b2_transfers_cmp.approach == a)]
        st = b2_sched[(b2_sched.n_existing == n) & (b2_sched.approach == a)]["scheduling_time_s"]
        final_rows.append({
            "famille": "B.2: état-A charge infra", "condition": f"n_existing={n}", "approche": a,
            "flow_time_moyen": d["flow_time"].mean(), "energie_moyenne": dt["energy"].mean(),
            "temps_scheduling_moyen_s": st.mean() if len(st) else np.nan,
        })

final_summary = pd.DataFrame(final_rows)
final_summary
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("results-grid5000/analysis/full_evaluation_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
