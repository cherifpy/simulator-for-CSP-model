"""Builds the complete analysis notebook for the 2026-10-07 validation plan (Phase A: paliers +
echelles / Phase B: n_existing / Phase C: scenarios aleatoires), comparant Incremental /
Online_newjob / Hybrid-n_j sur toutes les metriques collectees par collect_phaseABC_full_metrics.py
(flow time nouveau job + reste du systeme, nombre de transferts, noeuds utilises, donnees
transferees, energie, temps de scheduling, variantes gagnantes d'Hybrid, violations de stockage,
evolution de la complexite du probleme)."""
import nbformat as nbf

# Relative to simulator/ (where this notebook lives and is meant to be opened/run from) -- the
# raw collection script still writes its working copy to the scratchpad; this is the in-repo,
# durable copy committed alongside the notebook.
CSV_PATH = "results-validation-2026-10-07/phaseABC_full_metrics.csv"
OUT_PATH = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/phaseABC_full_analysis.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Analyse complète — Incremental vs Online_newjob vs Hybrid-n_j (2026-10-07)

Protocole single-decision-point : un état A gelé (n_existing jobs déjà en place, construit via un
vrai dispatch SimPy live), puis un nouveau job arrive à mi-vie du batch (max_flow_time/2 + 200).
Les 3 approches décident de son placement à partir du même état A figé, sans contamination entre
elles (vérifié).

- **Phase A** : 3 paliers de taille (petit/grand/mixte) à échelle x2 fixe, puis le palier mixte
  seul balayé en échelle x0.5/x2/x4/x6 — `n_existing=5` fixe, 50 nœuds.
- **Phase B** : palier mixte x2 fixe, `n_existing` ∈ {2,5,10,15} — 50 nœuds.
- **Phase C** : 5 scénarios totalement aléatoires (nb_nodes∈[10,30], n_existing∈[2,15], plages de
  taille/durée aléatoires), avec un plancher forçant le nouveau job à ne jamais être petit.

Budgets solveur : état A=30s, Incremental=défaut, Online_newjob=30s, Hybrid F1=15s +
escalade=30s (11 variantes concurrentes), plafond de dégradation=25%, mono-objectif
(`adaptive_bi_objective=False`)."""))

cells.append(nbf.v4.new_code_cell("""import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("%s")
df = df[df.no_solution != True].copy()

APPROACH_DISPLAY = {"incremental": "Incremental", "online_newjob": "Online_newjob", "hybrid": "Hybrid-n_j"}
df["approach_disp"] = df["approach"].map(APPROACH_DISPLAY)
colors = {"Incremental": "#8C8C8C", "Online_newjob": "#4472C4", "Hybrid-n_j": "#2E8B57"}
order = ["Incremental", "Online_newjob", "Hybrid-n_j"]

PHASE_LABEL = {"A": "Phase A (palier/échelle)", "B": "Phase B (n_existing)", "C": "Phase C (aléatoire)"}
df["phase_label"] = df["phase"].map(PHASE_LABEL)

def scenario_label(row):
    if row.phase == "A":
        return f"{row.group}:{row.tag}"
    return row.tag

df["scenario"] = df.apply(scenario_label, axis=1)

def grouped_bar(ax, data, value_col, title, ylabel):
    scenarios = sorted(data["scenario"].unique(), key=lambda s: (len(s), s))
    x = np.arange(len(scenarios))
    width = 0.25
    for i, appr in enumerate(order):
        sub = data[data.approach_disp == appr].set_index("scenario").reindex(scenarios)[value_col]
        ax.bar(x + (i - 1) * width, sub.values, width, label=appr, color=colors[appr])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

print(f"{len(df)} lignes chargées, phases: {sorted(df.phase.unique())}")
df.head()""" % CSV_PATH))

# --- Section 1: flow time ---
cells.append(nbf.v4.new_markdown_cell("""## 1. Flow time — nouveau job et reste du système

`flow_time_new_job` : la métrique principale (ce qu'on optimise).
`mean_flow_time_all` / `max_flow_time_all` : impact sur TOUS les jobs du batch reconsidéré —
Incremental ne touche jamais aux jobs existants, Online_newjob/Hybrid le peuvent (sous plafond de
dégradation de 25%)."""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    grouped_bar(axes[0], sub, "flow_time_new_job", f"{PHASE_LABEL[phase]} — flow time nouveau job", "secondes")
    grouped_bar(axes[1], sub, "mean_flow_time_all", f"{PHASE_LABEL[phase]} — flow time moyen (tout le batch)", "secondes")
    grouped_bar(axes[2], sub, "max_flow_time_all", f"{PHASE_LABEL[phase]} — flow time max (tout le batch)", "secondes")
    plt.tight_layout()
    plt.show()"""))
cells.append(nbf.v4.new_code_cell("""print("Gain Hybrid vs Incremental sur flow_time_new_job, par scénario :")
pivot = df.pivot_table(index=["phase", "scenario"], columns="approach", values="flow_time_new_job")
pivot["gain_hybrid_pct"] = (pivot["incremental"] - pivot["hybrid"]) / pivot["incremental"] * 100
pivot["gain_online_newjob_pct"] = (pivot["incremental"] - pivot["online_newjob"]) / pivot["incremental"] * 100
pivot.round(1)"""))

# --- Section 2: transfers ---
cells.append(nbf.v4.new_markdown_cell("""## 2. Nombre de transferts

`nb_transfers_new_job` : combien de répliques (nœuds différents) le nouveau job reçoit — un
chiffre élevé signifie que l'approche étale ses tâches sur plus de nœuds en parallèle.
`nb_transfers_total` : tous les transferts du plan proposé (nouveau job + jobs reconsidérés)."""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    grouped_bar(axes[0], sub, "nb_transfers_new_job", f"{PHASE_LABEL[phase]} — transferts du nouveau job", "nb transferts")
    grouped_bar(axes[1], sub, "nb_transfers_total", f"{PHASE_LABEL[phase]} — transferts totaux du plan", "nb transferts")
    plt.tight_layout()
    plt.show()"""))

# --- Section 3: nodes used ---
cells.append(nbf.v4.new_markdown_cell("""## 3. Nombre de nœuds utilisés

`nb_nodes_used_new_job` : sur combien de nœuds distincts le nouveau job est étalé.
`nb_nodes_used_total` : nœuds actifs dans tout le plan proposé (reflète combien de nœuds
Online_newjob/Hybrid retouchent quand ils reconsidèrent les jobs existants)."""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    grouped_bar(axes[0], sub, "nb_nodes_used_new_job", f"{PHASE_LABEL[phase]} — nœuds du nouveau job", "nb nœuds")
    grouped_bar(axes[1], sub, "nb_nodes_used_total", f"{PHASE_LABEL[phase]} — nœuds actifs (tout le plan)", "nb nœuds")
    plt.tight_layout()
    plt.show()"""))

# --- Section 4: data transferred ---
cells.append(nbf.v4.new_markdown_cell("""## 4. Quantité de données transférées (nouveau job)

`data_transferred_new_job_mb` = nb_transfers_new_job × taille du dataset du nouveau job — le
volume réseau que chaque approche génère pour PLACER le nouveau job (pas le reste du batch)."""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bar(ax, sub, "data_transferred_new_job_mb", f"{PHASE_LABEL[phase]} — Mo transférés pour le nouveau job", "Mo")
    plt.tight_layout()
    plt.show()"""))

# --- Section 5: energy ---
cells.append(nbf.v4.new_markdown_cell("""## 5. Énergie de transfert"""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bar(ax, sub, "transfer_energy_total", f"{PHASE_LABEL[phase]} — énergie de transfert totale", "unités d'énergie")
    plt.tight_layout()
    plt.show()"""))

# --- Section 6: scheduling time ---
cells.append(nbf.v4.new_markdown_cell("""## 6. Temps de scheduling (mur)

Temps réel passé à résoudre (pas de temps simulé ici, protocole figé). Hybrid inclut la sonde F1
+ l'escalade (jusqu'à 11 variantes concurrentes)."""))
cells.append(nbf.v4.new_code_cell("""for phase in ["A", "B", "C"]:
    sub = df[df.phase == phase]
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bar(ax, sub, "scheduling_time_s", f"{PHASE_LABEL[phase]} — temps de scheduling", "secondes (mur)")
    plt.tight_layout()
    plt.show()"""))

# --- Section 7: hybrid winning variants ---
cells.append(nbf.v4.new_markdown_cell("""## 7. Variantes gagnantes d'Hybrid-n_j

Quelle variante d'escalade (sur les 11 concurrentes) a été retenue à chaque fois, et taux
d'acceptation de l'escalade (vs rejet/fallback sur F1=Incremental)."""))
cells.append(nbf.v4.new_code_cell("""hyb = df[df.approach == "hybrid"].copy()
print("Taux d'acceptation de l'escalade :", f"{hyb.hybrid_accepted.mean()*100:.0f}%", f"({hyb.hybrid_accepted.sum()}/{len(hyb)})")
print()
print("Distribution des variantes gagnantes (quand acceptée) :")
hyb[hyb.hybrid_accepted == True].hybrid_winning_variant.value_counts()"""))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 4))
counts = hyb[hyb.hybrid_accepted == True].hybrid_winning_variant.value_counts()
ax.bar(counts.index, counts.values, color="#2E8B57")
ax.set_xticklabels(counts.index, rotation=30, ha="right")
ax.set_title("Variantes gagnantes d'Hybrid-n_j (tous scénarios confondus, escalade acceptée)")
ax.set_ylabel("nb de fois gagnante")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""hyb[["phase", "scenario", "hybrid_accepted", "hybrid_winning_variant", "flow_time_new_job", "hybrid_f1"]]"""))

# --- Section 8: storage ---
cells.append(nbf.v4.new_markdown_cell("""## 8. Violations de stockage (vérification du plan proposé)

Contrairement au protocole live, le plan proposé ici n'est jamais réellement dispatché -- cette
vérification reconstruit l'occupation (stockage fantôme des autres jobs + transferts/suppressions
du plan) et compare à la capacité réelle de chaque nœud, exactement comme `verify_storage()` le
fait pour la simulation live."""))
cells.append(nbf.v4.new_code_cell("""print("Violations de stockage par approche (somme sur tous les scénarios) :")
df.groupby("approach_disp").storage_violations.sum()"""))

# --- Section 9: problem complexity ---
cells.append(nbf.v4.new_markdown_cell("""## 9. Évolution de la complexité du problème

`batch_size` = nombre de jobs reconsidérés dans ce solve (1 pour Incremental, toujours ; nouveau
job + jobs existants non finis pour Online_newjob/Hybrid). On regarde comment le temps de
scheduling et le gain évoluent avec cette taille de batch et avec `n_existing`/l'échelle."""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for appr in order:
    sub = df[df.approach_disp == appr]
    axes[0].scatter(sub.batch_size, sub.scheduling_time_s, label=appr, color=colors[appr], alpha=0.7)
axes[0].set_xlabel("taille du batch reconsidéré (nb jobs)")
axes[0].set_ylabel("temps de scheduling (s)")
axes[0].set_title("Temps de scheduling vs taille du batch")
axes[0].legend()
axes[0].grid(alpha=0.3)

b = df[df.phase == "B"].pivot_table(index="n_existing", columns="approach", values="flow_time_new_job")
b.plot(ax=axes[1], marker="o", color=[colors[APPROACH_DISPLAY[c]] for c in b.columns])
axes[1].set_xlabel("n_existing (Phase B)")
axes[1].set_ylabel("flow time nouveau job")
axes[1].set_title("Flow time vs n_existing (Phase B)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""scale_order = ["mixte_x0.5", "mixte_x2", "mixte_x4", "mixte_x6"]
a = df[(df.phase == "A") & (df.group == "echelle")].copy()
a["tag"] = pd.Categorical(a.tag, categories=scale_order, ordered=True)
a = a.sort_values("tag")
fig, ax = plt.subplots(figsize=(8, 4.5))
for appr in order:
    sub = a[a.approach_disp == appr]
    ax.plot(sub.tag.astype(str), sub.flow_time_new_job, marker="o", label=appr, color=colors[appr])
ax.set_title("Flow time vs échelle (palier mixte, Phase A)")
ax.set_ylabel("flow time nouveau job")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

# --- Section 10: synthesis ---
cells.append(nbf.v4.new_markdown_cell("""## 10. Synthèse générale par phase"""))
cells.append(nbf.v4.new_code_cell("""summary = df.groupby(["phase", "approach_disp"]).agg(
    flow_time_new_job_mean=("flow_time_new_job", "mean"),
    scheduling_time_s_mean=("scheduling_time_s", "mean"),
    transfer_energy_mean=("transfer_energy_total", "mean"),
    nb_transfers_new_job_mean=("nb_transfers_new_job", "mean"),
    nb_nodes_used_new_job_mean=("nb_nodes_used_new_job", "mean"),
    storage_violations_sum=("storage_violations", "sum"),
).round(1)
summary"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- **Hybrid-n_j** égale ou bat Incremental sur `flow_time_new_job` dans la quasi-totalité des
  scénarios (jamais pire, grâce à son filet de sécurité F1), au prix d'un temps de scheduling et
  d'une énergie plus élevés.
- **Online_newjob** seul peut être pire qu'Incremental (observé en Phase A à grande échelle et en
  Phase C) — il n'a pas de filet de sécurité, contrairement à Hybrid.
- Aucune violation de stockage détectée pour aucune des 3 approches dans ce protocole figé
  (single-decision-point) — à l'inverse du protocole live (simulation continue Poisson), où Hybrid
  seul en a montré (voir note séparée sur l'expérience Poisson du 2026-10-07)."""))

nb["cells"] = cells
with open(OUT_PATH, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {OUT_PATH}")
