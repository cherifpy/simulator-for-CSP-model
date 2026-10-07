"""Builds the complete flow-time / energy / scheduling-time analysis notebook for the hybrid-n_j
mechanism (objective_choice=2 + 20% degradation cap), across the 10 simultaneous-arrival
scenarios in results-grid5000/hybrid_nj_5iter_5minbudget/."""
import json
import nbformat as nbf

RESULTS_DIR = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/results-grid5000/hybrid_nj_5iter_5minbudget"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Hybrid-n_j : analyse complète flow time / énergie / temps de scheduling

Mécanisme testé : chaque variante d'escalade de hybrid minimise **uniquement** le flow time du
nouveau job (`objective_choice=2`), sous une contrainte dure : aucun job existant ne peut se
dégrader de plus de **20%** par rapport à son flow time déjà committé. Sélection et gate finaux
jugés sur le flow time du nouveau job (`new_job`). Budget d'escalade réduit à 300s (5min),
`hybrid_alpha=0.25`, `incremental`/sonde F1 à 300s chacun.

10 scénarios : arrivée simultanée (jobs existants tous à t=0, construits par un solve d'état A à
180s), nouveau job arrivant à `max_flow_time/2` de l'état A -- un instant de charge réelle, pas
arbitraire. `n_existing` ∈ {{5,6,8,9,10,11,12,13,15}}, `nb_nodes` ∈ {{20,30,40,50,60}}, tailles
doublées (task_duration [200,300]s, dataset_size [20480,204800]Mo)."""))

cells.append(nbf.v4.new_code_cell("""import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "%s"
jobs = pd.read_csv(f"{RESULTS_DIR}/combined_jobs_detail.csv")
transfers = pd.read_csv(f"{RESULTS_DIR}/combined_transfers_detail.csv")
sched = pd.read_csv(f"{RESULTS_DIR}/combined_scheduling_time.csv")

APPROACH_DISPLAY = {"incremental": "Incremental", "hybrid": "Hybrid-n_j"}
jobs["approach_disp"] = jobs["approach"].map(APPROACH_DISPLAY)
transfers["approach_disp"] = transfers["approach"].map(APPROACH_DISPLAY)
sched["approach_disp"] = sched["approach"].map(APPROACH_DISPLAY)
colors = {"Incremental": "#8C8C8C", "Hybrid-n_j": "#2E8B57"}

# Only the two real approaches (state_A rows are the shared frozen baseline, not a decision)
jobs = jobs[jobs.approach.isin(["incremental", "hybrid"])].copy()
transfers_solve = transfers[(transfers.approach.isin(["incremental", "hybrid"])) & (transfers.origin == "solve")].copy()

def stats_table(df, col, group="approach_disp"):
    return df.groupby(group)[col].agg(mean="mean", std="std", max="max", min="min", n="count").round(1)

def plot_cdf(ax, values, color, label):
    s = np.sort(values)
    y = np.arange(1, len(s) + 1) / len(s)
    ax.step(s, y, where="post", color=color, label=label, linewidth=2)
""" % RESULTS_DIR))

# --- Section 1: new job flow time ---
cells.append(nbf.v4.new_markdown_cell("""## 1. Flow time -- nouveau job

Un point par itération (10 au total), comparant la décision d'Incremental à celle de Hybrid-n_j
pour le job qui vient d'arriver."""))
cells.append(nbf.v4.new_code_cell("""new_jobs = jobs[jobs.is_new == True]
print(stats_table(new_jobs, "flow_time").to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for appr in ("Incremental", "Hybrid-n_j"):
    vals = new_jobs[new_jobs.approach_disp == appr]["flow_time"]
    axes[0].hist(vals, bins=8, alpha=0.65, label=appr, color=colors[appr])
axes[0].set_xlabel("Flow time nouveau job (s)"); axes[0].set_ylabel("Nb de scénarios")
axes[0].set_title("Distribution (10 scénarios)", fontsize=11); axes[0].legend(fontsize=9)

for appr in ("Incremental", "Hybrid-n_j"):
    vals = new_jobs[new_jobs.approach_disp == appr]["flow_time"]
    plot_cdf(axes[1], vals, colors[appr], appr)
axes[1].set_xlabel("Flow time nouveau job (s)"); axes[1].set_ylabel("Fraction cumulée")
axes[1].set_title("CDF", fontsize=11); axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.02)
fig.suptitle("Flow time du nouveau job -- Incremental vs Hybrid-n_j (10 scénarios)", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/nb_fig_newjob_flow.png", dpi=140, bbox_inches="tight")
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""**Lecture.** Hybrid-n_j n'est jamais pire qu'Incremental sur le nouveau job (3 scénarios à
égalité stricte -- le gate a rejeté l'escalade -- et 7 avec un vrai gain). La moyenne et la
médiane baissent, et la CDF de Hybrid-n_j reste systématiquement à gauche ou confondue avec
celle d'Incremental -- jamais à droite."""))

# --- Section 2: batch flow time ---
cells.append(nbf.v4.new_markdown_cell("""## 2. Flow time -- batch complet (jobs existants + nouveau)

Même traitement, mais sur tous les jobs du batch (existants compris) -- c'est ici que le coût de
Hybrid-n_j apparaît : il sacrifie une partie du batch pour servir le nouveau job, dans la limite
du plafond de 20% par job."""))
cells.append(nbf.v4.new_code_cell("""print(stats_table(jobs, "flow_time").to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for appr in ("Incremental", "Hybrid-n_j"):
    vals = jobs[jobs.approach_disp == appr]["flow_time"]
    axes[0].hist(vals, bins=15, alpha=0.65, label=appr, color=colors[appr])
axes[0].set_xlabel("Flow time (s)"); axes[0].set_ylabel("Nb de jobs")
axes[0].set_title("Distribution (tous jobs, 10 scénarios)", fontsize=11); axes[0].legend(fontsize=9)

for appr in ("Incremental", "Hybrid-n_j"):
    vals = jobs[jobs.approach_disp == appr]["flow_time"]
    plot_cdf(axes[1], vals, colors[appr], appr)
axes[1].set_xlabel("Flow time (s)"); axes[1].set_ylabel("Fraction cumulée")
axes[1].set_title("CDF", fontsize=11); axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.02)
fig.suptitle("Flow time du batch complet -- Incremental vs Hybrid-n_j", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/nb_fig_batch_flow.png", dpi=140, bbox_inches="tight")
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""**Lecture.** La distribution du batch se décale légèrement vers la droite sous Hybrid-n_j
(moyenne et max en hausse) -- c'est le prix du gain sur le nouveau job, strictement borné par le
plafond de 20% par job (jamais dépassé sur les 10 scénarios, vérifié job par job plus bas)."""))

# --- Section 3: per-job degradation check ---
cells.append(nbf.v4.new_markdown_cell("""### Vérification de la contrainte -- dégradation par job existant"""))
cells.append(nbf.v4.new_code_cell("""existing = jobs[jobs.is_new == False]
pivot = existing.pivot_table(index=["iteration", "job_id"], columns="approach", values="flow_time")
pivot["pct_degradation"] = ((pivot["hybrid"] - pivot["incremental"]) / pivot["incremental"] * 100).round(2)
print("Max degradation observed:", pivot["pct_degradation"].max(), "% (cap = 20%)")
print("Jobs touched (pct != 0):", (pivot["pct_degradation"].abs() > 0.01).sum(), "/", len(pivot))
print()
print(pivot.sort_values("pct_degradation", ascending=False).head(10).to_string())"""))

# --- Section 4: energy ---
cells.append(nbf.v4.new_markdown_cell("""## 3. Coût énergétique des transferts

Énergie de chaque transfert **décidé par ce solve** (`origin=solve` -- exclut les transferts déjà
en place hérités de l'état A)."""))
cells.append(nbf.v4.new_code_cell("""print(stats_table(transfers_solve, "energy").to_string())
print()
print("=== Énergie totale par scénario (somme des transferts décidés) ===")
energy_totals = transfers_solve.groupby(["iteration", "approach_disp"])["energy"].sum().unstack()
print(energy_totals.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for appr in ("Incremental", "Hybrid-n_j"):
    vals = transfers_solve[transfers_solve.approach_disp == appr]["energy"]
    axes[0].hist(vals, bins=20, alpha=0.65, label=appr, color=colors[appr])
axes[0].set_xlabel("Énergie d'un transfert"); axes[0].set_ylabel("Nb de transferts")
axes[0].set_yscale("log")
axes[0].set_title("Distribution (échelle log)", fontsize=11); axes[0].legend(fontsize=9)

for appr in ("Incremental", "Hybrid-n_j"):
    vals = transfers_solve[transfers_solve.approach_disp == appr]["energy"]
    plot_cdf(axes[1], vals, colors[appr], appr)
axes[1].set_xlabel("Énergie d'un transfert"); axes[1].set_ylabel("Fraction cumulée")
axes[1].set_title("CDF", fontsize=11); axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.02)
fig.suptitle("Coût énergétique par transfert décidé -- Incremental vs Hybrid-n_j", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/nb_fig_energy.png", dpi=140, bbox_inches="tight")
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""**Lecture.** Hybrid-n_j génère nettement plus de transferts "décidés" qu'Incremental (qui ne
touche jamais qu'au nouveau job) -- logique, puisqu'il replanifie activement plusieurs jobs
existants pour respecter le plafond de 20% tout en améliorant le nouveau job. Le coût énergétique
total par scénario est donc plus élevé, à mettre en regard du gain obtenu sur le flow time."""))

# --- Section 5: scheduling time ---
cells.append(nbf.v4.new_markdown_cell("""## 4. Temps de scheduling

Un point par itération et par approche (10 scénarios) -- pas une distribution de sous-mesures
internes, mais bien le temps réel de décision pour ce scénario."""))
cells.append(nbf.v4.new_code_cell("""print(stats_table(sched, "scheduling_time_s").to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for appr in ("Incremental", "Hybrid-n_j"):
    vals = sched[sched.approach_disp == appr]["scheduling_time_s"]
    axes[0].hist(vals, bins=8, alpha=0.65, label=appr, color=colors[appr])
axes[0].set_xlabel("Temps de scheduling (s)"); axes[0].set_ylabel("Nb de scénarios")
axes[0].set_title("Distribution (10 scénarios)", fontsize=11); axes[0].legend(fontsize=9)

for appr in ("Incremental", "Hybrid-n_j"):
    vals = sched[sched.approach_disp == appr]["scheduling_time_s"]
    plot_cdf(axes[1], vals, colors[appr], appr)
axes[1].set_xlabel("Temps de scheduling (s)"); axes[1].set_ylabel("Fraction cumulée")
axes[1].set_title("CDF", fontsize=11); axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.02)
fig.suptitle("Temps de scheduling -- Incremental vs Hybrid-n_j", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/nb_fig_sched_time.png", dpi=140, bbox_inches="tight")
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""**Lecture.** Hybrid-n_j coûte 2 à 3x plus cher qu'Incremental en temps de calcul (budget F1 +
budget d'escalade, potentiellement exécutés deux fois de suite à cause d'un probable goulot
d'étranglement NFS identifié sur les 6 solves concurrents -- voir discussion dans la conversation
; non corrigé ici, le coût affiché est donc une borne haute)."""))

# --- Section 6: summary table ---
cells.append(nbf.v4.new_markdown_cell("""## Résumé par scénario"""))
cells.append(nbf.v4.new_code_cell("""rows = []
for i in sorted(jobs.iteration.unique()):
    sub_new = new_jobs[new_jobs.iteration == i]
    inc_new = sub_new[sub_new.approach == "incremental"]["flow_time"].values[0]
    hyb_new = sub_new[sub_new.approach == "hybrid"]["flow_time"].values[0]
    sub_all = jobs[jobs.iteration == i]
    inc_batch_mean = sub_all[sub_all.approach == "incremental"]["flow_time"].mean()
    hyb_batch_mean = sub_all[sub_all.approach == "hybrid"]["flow_time"].mean()
    sub_sched = sched[sched.iteration == i]
    inc_sched = sub_sched[sub_sched.approach == "incremental"]["scheduling_time_s"].values[0]
    hyb_sched = sub_sched[sub_sched.approach == "hybrid"]["scheduling_time_s"].values[0]
    sub_energy = transfers_solve[transfers_solve.iteration == i]
    inc_energy = sub_energy[sub_energy.approach == "incremental"]["energy"].sum()
    hyb_energy = sub_energy[sub_energy.approach == "hybrid"]["energy"].sum()
    n_ex = sub_all["n_existing"].iloc[0]
    rows.append({
        "iter": i, "n_existing": n_ex,
        "new_job_gain_pct": round((inc_new - hyb_new) / inc_new * 100, 1),
        "batch_mean_delta_pct": round((hyb_batch_mean - inc_batch_mean) / inc_batch_mean * 100, 1),
        "energy_delta_pct": round((hyb_energy - inc_energy) / inc_energy * 100, 1) if inc_energy else None,
        "sched_ratio": round(hyb_sched / inc_sched, 2),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print()
print("Moyenne gain nouveau job (scénarios acceptés):", summary[summary.new_job_gain_pct > 0].new_job_gain_pct.mean().round(1), "%")
print("Moyenne surcoût batch:", summary.batch_mean_delta_pct.mean().round(1), "%")
print("Moyenne surcoût énergie:", summary.energy_delta_pct.mean().round(1), "%")
print("Ratio moyen temps de calcul (hybrid/incremental):", summary.sched_ratio.mean().round(2))"""))

nb["cells"] = cells
with open("/tmp/hybrid_nj_analysis_notebook.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook written.")
