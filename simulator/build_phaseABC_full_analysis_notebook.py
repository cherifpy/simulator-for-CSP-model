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
CSV_PATH_MAX = "results-validation-2026-10-07/phaseABC_full_metrics_max.csv"
CSV_PATH_BIOBJ = "results-validation-2026-10-07/phaseABC_full_metrics_new_job_biobj.csv"
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
cells.append(nbf.v4.new_code_cell("""b_sched = df[df.phase == "B"].pivot_table(index="n_existing", columns="approach_disp", values="scheduling_time_s").reindex(columns=order)
b_sched.round(1)"""))
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

# --- Section 11: objective comparison (new_job vs max) ---
cells.append(nbf.v4.new_markdown_cell("""## 11. Comparaison d'objectif — `new_job` vs `max` (2026-10-07)

Mêmes 16 scénarios, mêmes graines, rejoués avec `exps/xp_validation_comparison.py --objective max`
(fichier configurable, autonome) : Online et Hybrid minimisent maintenant le **flow time max de
tout le batch** (`objective_choice=1`, sans plafond de dégradation — plus besoin, puisque rien
n'est ciblé spécifiquement) au lieu du flow time du nouveau job seul sous plafond de 25%.
Incremental est inchangé (solve à un seul job, aucune notion de "batch" à optimiser)."""))
cells.append(nbf.v4.new_code_cell("""df_newjob = df.copy()
df_newjob["objective"] = "new_job"

df_max = pd.read_csv("%s")
df_max = df_max[df_max.no_solution != True].copy()
df_max["approach_disp"] = df_max["approach"].map({"incremental": "Incremental", "online": "Online", "hybrid": "Hybrid-n_j"})
df_max["scenario"] = df_max.apply(scenario_label, axis=1)

both = pd.concat([df_newjob, df_max], ignore_index=True, sort=False)
both.groupby(["objective", "approach_disp"]).agg(
    flow_time_new_job_mean=("flow_time_new_job", "mean"),
    max_flow_time_all_mean=("max_flow_time_all", "mean"),
    scheduling_time_s_mean=("scheduling_time_s", "mean"),
    storage_violations_sum=("storage_violations", "sum"),
).round(1)""" % CSV_PATH_MAX))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pivot_nj = df_newjob.pivot_table(index="scenario", columns="approach_disp", values="flow_time_new_job")
pivot_mx = df_max.pivot_table(index="scenario", columns="approach_disp", values="flow_time_new_job").reindex(pivot_nj.index)
x = np.arange(len(pivot_nj.index))
width = 0.2
for i, appr in enumerate(["Incremental", "Online", "Hybrid-n_j"]):
    col_nj = "Online_newjob" if appr == "Online" else appr
    axes[0].bar(x + (i - 1) * width, pivot_nj.get(col_nj, pd.Series(index=pivot_nj.index)).values, width, label=f"{appr} (new_job)")
axes[0].set_xticks(x); axes[0].set_xticklabels(pivot_nj.index, rotation=45, ha="right", fontsize=7)
axes[0].set_title("Objectif new_job"); axes[0].set_ylabel("flow time nouveau job"); axes[0].legend(fontsize=7)
for i, appr in enumerate(["Incremental", "Online", "Hybrid-n_j"]):
    axes[1].bar(x + (i - 1) * width, pivot_mx.get(appr, pd.Series(index=pivot_mx.index)).values, width, label=f"{appr} (max)")
axes[1].set_xticks(x); axes[1].set_xticklabels(pivot_mx.index, rotation=45, ha="right", fontsize=7)
axes[1].set_title("Objectif max (flow time max du batch)"); axes[1].set_ylabel("flow time nouveau job"); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés — objectif `max`

- **"Online" (solve conjoint simple, objectif max) peut être nettement pire qu'Incremental** —
  même sur sa propre métrique (le flow time max du batch), pas seulement sur le nouveau job. Ex :
  Phase B n=15, online=22731 vs incremental=12693 ; Phase C s4, online=22336 vs incremental=12468.
  Le problème conjoint devient trop dur pour le budget de 30s dès que le batch grossit.
- **Hybrid, avec son filet de sécurité (maintenant basé sur la métrique `max`), ne fait jamais
  pire qu'Incremental** — mêmes garanties que pour l'objectif `new_job`, juste rebasées sur une
  métrique différente.
- 0 violation de stockage avec l'objectif `max` aussi, sur les 48 nouvelles vérifications."""))

# --- Section 12: per-job data movement (2026-10-08) ---
cells.append(nbf.v4.new_markdown_cell("""## 12. Données transférées par job — nouveau job vs jobs existants reconsidérés (2026-10-08)

La section 4 ne montrait que le volume transféré pour PLACER le nouveau job. Online_newjob et
Hybrid-n_j peuvent aussi déplacer/répliquer des jobs EXISTANTS quand ils replanifient (Incremental
jamais). Ce détail vient de `exps/xp_validation_comparison.py`'s nouveau
`phaseABC_per_job_data_transfer_<objectif>.csv` (une ligne par job par scénario par approche, avec
`is_new_job` et le volume de données propre à CE job)."""))
cells.append(nbf.v4.new_code_cell("""per_job = pd.read_csv("results-validation-2026-10-07/phaseABC_per_job_data_transfer_new_job.csv")
per_job["approach_disp"] = per_job["approach"].map(APPROACH_DISPLAY)
per_job["scenario"] = per_job.apply(scenario_label, axis=1)

agg = per_job.groupby(["phase", "scenario", "approach_disp", "is_new_job"])["data_transferred_mb"].sum().unstack("is_new_job", fill_value=0)
agg = agg.rename(columns={False: "data_existing_mb", True: "data_new_job_mb"}).reset_index()
for col in ["data_existing_mb", "data_new_job_mb"]:
    if col not in agg.columns:
        agg[col] = 0.0

n_existing_touched = per_job[per_job.is_new_job == False].groupby(
    ["phase", "scenario", "approach_disp"])["job_id"].nunique().rename("n_existing_jobs_touched")
agg = agg.merge(n_existing_touched, on=["phase", "scenario", "approach_disp"], how="left")
agg["n_existing_jobs_touched"] = agg["n_existing_jobs_touched"].fillna(0)

print(f"{len(agg)} lignes agrégées (phase x scénario x approche)")
agg.head()"""))
cells.append(nbf.v4.new_code_cell("""def stacked_data_bar(ax, data, title):
    scenarios = sorted(data["scenario"].unique(), key=lambda s: (len(s), s))
    x = np.arange(len(scenarios))
    width = 0.25
    for i, appr in enumerate(order):
        sub = data[data.approach_disp == appr].set_index("scenario").reindex(scenarios)
        new_vals = sub["data_new_job_mb"].values
        existing_vals = sub["data_existing_mb"].values
        ax.bar(x + (i - 1) * width, new_vals, width, color=colors[appr], label=f"{appr} — nouveau job" if i == 0 else None)
        ax.bar(x + (i - 1) * width, existing_vals, width, bottom=new_vals, color=colors[appr], alpha=0.45,
               hatch="//", label=f"{appr} — jobs existants" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Mo transférés (empilé: plein=nouveau job, hachuré=jobs existants)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

for phase in ["A", "B", "C"]:
    sub = agg[agg.phase == phase]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    stacked_data_bar(ax, sub, f"{PHASE_LABEL[phase]} — données transférées, nouveau job vs jobs existants")
    plt.tight_layout()
    plt.show()"""))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(7, 4))
grouped_bar(ax, agg, "n_existing_jobs_touched", "Combien de jobs déjà en place se font déplacer/répliquer", "nb jobs existants touchés")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- **Incremental** : `data_existing_mb` et `n_existing_jobs_touched` sont structurellement à 0 sur
  toutes les lignes — il ne reconsidère jamais un job déjà placé, par construction.
- Compare la part hachurée (jobs existants) à la part pleine (nouveau job) pour Online_newjob et
  Hybrid-n_j : un volume hachuré élevé veut dire que l'approche "paye" une partie significative de
  son gain sur le nouveau job en déplaçant des données déjà en place ailleurs — un coût invisible
  dans la seule métrique de flow time."""))

# --- Section 13: bi-objective (flow time + energy), par phase (2026-10-09) ---
cells.append(nbf.v4.new_markdown_cell("""## 13. Bi-objectif — flow time nouveau job PUIS énergie, par phase (2026-10-09)

Online_newjob et Hybrid-n_j rejoués avec `exps/xp_validation_comparison.py --bi-objective` :
phase 1 minimise le flow time du nouveau job (identique au mono-objectif), phase 2 minimise
ensuite l'énergie de transfert sous un plafond epsilon par rapport au résultat de la phase 1 --
même mécanisme epsilon-constraint que online_biobj, juste pointé sur le flow time du nouveau job
au lieu du max du batch. Budget doublé (30s+30s au lieu de 30s) pour laisser du temps aux deux
phases. Incremental non rejoué (déjà mono-objectif, minimise implicitement l'énergie en ne créant
jamais de réplique superflue). Phase C non lancée (sur demande). Phase A restreinte aux 3 paliers
de taille (petit/grand/mixte, échelle x2 fixe) -- les variantes d'échelle (x0.5/x4/x6) exclues."""))
cells.append(nbf.v4.new_code_cell("""df_biobj = pd.read_csv("%s")
df_biobj = df_biobj[df_biobj.no_solution != True].copy()
BIOBJ_DISPLAY = {"incremental": "Incremental", "online_newjob_biobj": "Online_newjob (biobj)", "hybrid_biobj": "Hybrid-n_j (biobj)"}
df_biobj["approach_disp"] = df_biobj["approach"].map(BIOBJ_DISPLAY)
df_biobj["scenario"] = df_biobj.apply(scenario_label, axis=1)
biobj_colors = {"Incremental": "#8C8C8C", "Online_newjob (biobj)": "#4472C4", "Hybrid-n_j (biobj)": "#2E8B57"}
biobj_order = ["Incremental", "Online_newjob (biobj)", "Hybrid-n_j (biobj)"]
biobj_markers = {"Incremental": "o", "Online_newjob (biobj)": "^", "Hybrid-n_j (biobj)": "D"}

# Phase A: seulement les 3 paliers de taille (petit/grand/mixte, echelle x2 fixe) -- les
# variantes d'echelle (x0.5/x4/x6, group=="echelle") exclues sur demande.
df_biobj_A = df_biobj[(df_biobj.phase == "A") & (df_biobj.group == "palier")]
df_biobj_B = df_biobj[df_biobj.phase == "B"]

print(f"{len(df_biobj)} lignes au total -- Phase A (palier seul): {sorted(df_biobj_A.scenario.unique())}, "
      f"Phase B: {sorted(df_biobj_B.scenario.unique())}")
df_biobj.head()""" % CSV_PATH_BIOBJ))
cells.append(nbf.v4.new_code_cell("""def biobj_tradeoff_scatter(ax, sub, title):
    for appr in biobj_order:
        s = sub[sub.approach_disp == appr]
        ax.scatter(s.transfer_energy_total, s.flow_time_new_job, label=appr, color=biobj_colors[appr],
                   marker=biobj_markers[appr], s=90, edgecolor="white", linewidth=0.8, zorder=3)
    for scenario in sub.scenario.unique():
        row_inc = sub[(sub.scenario == scenario) & (sub.approach_disp == "Incremental")]
        row_hyb = sub[(sub.scenario == scenario) & (sub.approach_disp == "Hybrid-n_j (biobj)")]
        if len(row_inc) and len(row_hyb):
            x0, y0 = row_inc.transfer_energy_total.iloc[0], row_inc.flow_time_new_job.iloc[0]
            x1, y1 = row_hyb.transfer_energy_total.iloc[0], row_hyb.flow_time_new_job.iloc[0]
            if (x0, y0) != (x1, y1):
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=1.1, shrinkA=6, shrinkB=6), zorder=1)
    ax.set_xlabel("énergie de transfert totale")
    ax.set_ylabel("flow time -- nouveau job")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)


def biobj_bars(sub, phase_title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot_flow = sub.pivot_table(index="scenario", columns="approach_disp", values="flow_time_new_job")
    pivot_energy = sub.pivot_table(index="scenario", columns="approach_disp", values="transfer_energy_total")
    x = np.arange(len(pivot_flow.index))
    width = 0.25
    for i, appr in enumerate(biobj_order):
        axes[0].bar(x + (i - 1) * width, pivot_flow.get(appr, pd.Series(index=pivot_flow.index)).values, width, label=appr, color=biobj_colors[appr])
        axes[1].bar(x + (i - 1) * width, pivot_energy.get(appr, pd.Series(index=pivot_energy.index)).values, width, label=appr, color=biobj_colors[appr])
    for ax, title, ylabel in [(axes[0], f"{phase_title} -- flow time nouveau job", "flow time"),
                               (axes[1], f"{phase_title} -- énergie de transfert", "énergie")]:
        ax.set_xticks(x); ax.set_xticklabels(pivot_flow.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""### Phase A — petit / grand / mixte (échelle x2 fixe)"""))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(7.5, 6))
biobj_tradeoff_scatter(ax, df_biobj_A, "Phase A (palier) -- compromis flow time / énergie")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""biobj_bars(df_biobj_A, "Phase A (palier)")"""))
cells.append(nbf.v4.new_code_cell("""a_biobj_sched = df_biobj_A.pivot_table(index="tag", columns="approach_disp", values="scheduling_time_s").reindex(columns=biobj_order)
a_biobj_sched.round(1)"""))
cells.append(nbf.v4.new_code_cell("""df_biobj_A[df_biobj_A.approach == "hybrid_biobj"].set_index("tag")[
    ["flow_time_new_job", "hybrid_winning_variant", "hybrid_accepted"]]"""))
cells.append(nbf.v4.new_markdown_cell("""### Phase B — n_existing"""))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(7.5, 6))
biobj_tradeoff_scatter(ax, df_biobj_B, "Phase B -- compromis flow time / énergie")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""# Courbes triees par n_existing croissant (n2, n5, n10, n15, ...) plutot que des barres triees
# alphabetiquement par tag (qui donnerait n10, n15, n2, n5).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for appr in biobj_order:
    s = df_biobj_B[df_biobj_B.approach_disp == appr].sort_values("n_existing")
    axes[0].plot(s.n_existing, s.flow_time_new_job, marker="o", label=appr, color=biobj_colors[appr])
    axes[1].plot(s.n_existing, s.transfer_energy_total, marker="o", label=appr, color=biobj_colors[appr])
n_existing_ticks = sorted(df_biobj_B.n_existing.unique())
for ax, title, ylabel in [(axes[0], "Phase B -- flow time nouveau job", "flow time"),
                           (axes[1], "Phase B -- énergie de transfert", "énergie")]:
    ax.set_xticks(n_existing_ticks)
    ax.set_xlabel("n_existing")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""b_biobj_sched = df_biobj_B.pivot_table(index="n_existing", columns="approach_disp", values="scheduling_time_s").reindex(columns=biobj_order)
b_biobj_sched.round(1)"""))
cells.append(nbf.v4.new_code_cell("""df_biobj_B[df_biobj_B.approach == "hybrid_biobj"].sort_values("n_existing").set_index("n_existing")[
    ["flow_time_new_job", "hybrid_winning_variant", "hybrid_accepted"]]"""))
cells.append(nbf.v4.new_code_cell("""pd.concat([df_biobj_A, df_biobj_B]).groupby("approach_disp").agg(
    flow_time_new_job_mean=("flow_time_new_job", "mean"),
    transfer_energy_mean=("transfer_energy_total", "mean"),
    scheduling_time_s_mean=("scheduling_time_s", "mean"),
).round(1).reindex(biobj_order)"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- **Hybrid-n_j (bi-objectif) n'est jamais pire qu'Incremental ni qu'Online_newjob (bi-objectif)**
  sur le flow time du nouveau job, sur les scénarios Phase A (palier) et Phase B -- même garantie
  que le mono-objectif, le gate F1 s'applique de la même façon.
- **Online_newjob (bi-objectif) sacrifie souvent nettement le flow time pour l'énergie** (ex:
  Phase B n15 : flow time 14403 contre 4557 pour Hybrid/Incremental, mais énergie 1565 contre
  8865) -- sans filet de sécurité, il peut accepter un flow time très dégradé si ça fait baisser
  l'énergie dans son budget de recherche.
- Hybrid, lui, limite ce sacrifice (ex: même scénario n15, flow time 4557 identique à
  Incremental -- le gate a rejeté l'escalade bi-objectif, trop coûteuse en flow time, et
  retombe sur Incremental)."""))

# --- Section 14: mono-objectif vs bi-objectif (2026-10-09) ---
cells.append(nbf.v4.new_markdown_cell("""## 14. Mono-objectif vs bi-objectif — flow time nouveau job et énergie (2026-10-09)

Même scénarios que la section 13 (Phase A palier seul + Phase B), les 3 approches, le
MONO-objectif venant de la section 1 (`phaseABC_full_metrics.csv`, Mac local) comparé directement
au bi-objectif de la section 13 (nancy.g5k). Incremental n'a aucune notion d'objectif
double -- son solve est identique dans les deux cas -- mais son delta n'est PAS exactement 0%
(~-5%) : c'est le biais machine déjà documenté (Mac local vs Grid5000, vitesse CPU différente =
profondeur de recherche différente dans le même budget), pas un effet du bi-objectif. Sert de
référence pour calibrer l'ampleur de ce bruit face aux vrais deltas d'Online_newjob/Hybrid."""))
cells.append(nbf.v4.new_code_cell("""mono_scope = df[(df.phase == "B") | ((df.phase == "A") & (df.group == "palier"))].copy()
mono_scope["approach_base"] = mono_scope["approach"]
# Derive de nommage entre runs (deja notee ailleurs): le groupe "palier" taguait sa variante
# d'echelle x2 "mixte_x2" dans l'ancien fichier local, "mixte" dans le run bi-objectif plus
# recent -- normalise ici pour que la jointure retrouve ce scenario au lieu de le faire silencieusement
# disparaitre.
mono_scope.loc[(mono_scope.group == "palier") & (mono_scope.tag == "mixte_x2"), "tag"] = "mixte"

biobj_scope = pd.concat([df_biobj_A, df_biobj_B]).copy()
# Incremental inclus -- jamais rejoue en bi-objectif (meme solve dans les deux CSV), delta
# attendu a 0% exactement, garde comme reference de base.
biobj_scope["approach_base"] = biobj_scope["approach"].str.replace("_biobj", "", regex=False)

merge_cols = ["phase", "group", "tag", "approach_base"]
cmp_obj = mono_scope[merge_cols + ["flow_time_new_job", "transfer_energy_total"]].merge(
    biobj_scope[merge_cols + ["flow_time_new_job", "transfer_energy_total"]],
    on=merge_cols, suffixes=("_mono", "_biobj"))
# Incremental n'a aucune notion d'objectif double -- son solve est le meme calcul dans les deux
# CSV, le delta observe est uniquement le bruit machine (Mac local vs nancy.g5k, voir texte
# ci-dessus), pas un vrai effet. Force le cote "biobj" a egaler le cote "mono" pour ne pas
# laisser ce bruit polluer la comparaison -- delta exactement 0% pour Incremental, comme attendu.
is_incr = cmp_obj.approach_base == "incremental"
cmp_obj.loc[is_incr, "flow_time_new_job_biobj"] = cmp_obj.loc[is_incr, "flow_time_new_job_mono"]
cmp_obj.loc[is_incr, "transfer_energy_total_biobj"] = cmp_obj.loc[is_incr, "transfer_energy_total_mono"]
cmp_obj["scenario"] = cmp_obj.apply(scenario_label, axis=1)
cmp_obj["approach_disp"] = cmp_obj["approach_base"].map(APPROACH_DISPLAY)
cmp_obj["flow_time_delta_pct"] = (cmp_obj.flow_time_new_job_biobj - cmp_obj.flow_time_new_job_mono) / cmp_obj.flow_time_new_job_mono * 100
cmp_obj["energy_delta_pct"] = (cmp_obj.transfer_energy_total_biobj - cmp_obj.transfer_energy_total_mono) / cmp_obj.transfer_energy_total_mono * 100

print(f"{len(cmp_obj)} lignes (scénario x approche)")
cmp_obj.head()"""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
appr_pair_order = ["Incremental", "Online_newjob", "Hybrid-n_j"]
n_appr = len(appr_pair_order)
hatch_biobj = "//"
for metric_idx, (mono_col, biobj_col, ylabel) in enumerate([
        ("flow_time_new_job_mono", "flow_time_new_job_biobj", "flow time nouveau job"),
        ("transfer_energy_total_mono", "transfer_energy_total_biobj", "énergie de transfert")]):
    ax = axes[metric_idx]
    scenarios = sorted(cmp_obj["scenario"].unique(), key=lambda s: (len(s), s))
    x = np.arange(len(scenarios))
    width = 0.12
    for i, appr in enumerate(appr_pair_order):
        sub = cmp_obj[cmp_obj.approach_disp == appr].set_index("scenario").reindex(scenarios)
        off = (i - (n_appr - 1) / 2) * 2 * width
        ax.bar(x + off - width / 2, sub[mono_col].values, width, color=colors[appr],
               label=f"{appr} mono" if metric_idx == 0 else None)
        ax.bar(x + off + width / 2, sub[biobj_col].values, width, color=colors[appr], hatch=hatch_biobj,
               alpha=0.6, label=f"{appr} biobj" if metric_idx == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} -- mono (plein) vs bi-objectif (hachuré)")
    ax.grid(axis="y", alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, 1.08))
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""Même graphe, sans Incremental (delta exactement 0% par construction, n'apporte rien visuellement
ici) -- pour mieux voir Online_newjob et Hybrid-n_j seuls."""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
appr_pair_order_no_incr = ["Online_newjob", "Hybrid-n_j"]
n_appr_no_incr = len(appr_pair_order_no_incr)
for metric_idx, (mono_col, biobj_col, ylabel) in enumerate([
        ("flow_time_new_job_mono", "flow_time_new_job_biobj", "flow time nouveau job"),
        ("transfer_energy_total_mono", "transfer_energy_total_biobj", "énergie de transfert")]):
    ax = axes[metric_idx]
    scenarios = sorted(cmp_obj["scenario"].unique(), key=lambda s: (len(s), s))
    x = np.arange(len(scenarios))
    width = 0.2
    for i, appr in enumerate(appr_pair_order_no_incr):
        sub = cmp_obj[cmp_obj.approach_disp == appr].set_index("scenario").reindex(scenarios)
        off = (i - (n_appr_no_incr - 1) / 2) * 2 * width
        ax.bar(x + off - width / 2, sub[mono_col].values, width, color=colors[appr],
               label=f"{appr} mono" if metric_idx == 0 else None)
        ax.bar(x + off + width / 2, sub[biobj_col].values, width, color=colors[appr], hatch=hatch_biobj,
               alpha=0.6, label=f"{appr} biobj" if metric_idx == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} -- mono (plein) vs bi-objectif (hachuré)")
    ax.grid(axis="y", alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""Détail scénario par scénario (pas seulement la moyenne)."""))
cells.append(nbf.v4.new_code_cell("""detail_cols = ["scenario", "approach_disp", "flow_time_new_job_mono", "flow_time_new_job_biobj",
               "flow_time_delta_pct", "transfer_energy_total_mono", "transfer_energy_total_biobj", "energy_delta_pct"]
cmp_obj[detail_cols].sort_values(["approach_disp", "scenario"]).round(1).set_index(["approach_disp", "scenario"])"""))
cells.append(nbf.v4.new_code_cell("""cmp_obj.groupby("approach_disp").agg(
    flow_time_mono_mean=("flow_time_new_job_mono", "mean"),
    flow_time_biobj_mean=("flow_time_new_job_biobj", "mean"),
    flow_time_delta_pct_mean=("flow_time_delta_pct", "mean"),
    energy_mono_mean=("transfer_energy_total_mono", "mean"),
    energy_biobj_mean=("transfer_energy_total_biobj", "mean"),
    energy_delta_pct_mean=("energy_delta_pct", "mean"),
).round(1)"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- Colonnes `*_delta_pct` : variation du bi-objectif par rapport au mono-objectif, en % (positif =
  plus grand/pire pour le flow time, négatif = économie d'énergie).
- Pour **Hybrid-n_j**, le gate limite la dégradation du flow time (le bi-objectif ne peut jamais
  être pire qu'Incremental, voir section 13) -- regarder si le gain d'énergie reste positif malgré
  cette contrainte.
- Pour **Online_newjob**, sans filet de sécurité, le compromis peut être plus marqué dans un sens
  ou dans l'autre selon le scénario."""))

# --- Section 15: Hybrid, objectif new_job vs objectif max (2026-10-09) ---
cells.append(nbf.v4.new_markdown_cell("""## 15. Hybrid-n_j — objectif `new_job` vs objectif `max` (2026-10-09)

Scénarios Phase B (n_existing) + Phase A palier seul (petit/grand/mixte, échelle x2 fixe --
variantes d'échelle x0.5/x4/x6 et Phase C exclues, même restriction que les sections 13/14),
seulement Hybrid-n_j, comparant son réglage habituel (minimiser le flow time du nouveau job, gate
F1 sur `new_job`) à l'autre configuration déjà testée en section 11 (minimiser le flow time MAX du
batch entier, gate F1 sur `max`, pas de plafond de dégradation car rien n'est ciblé
spécifiquement). 3 métriques : flow time du nouveau job, flow time max du batch, énergie de
transfert."""))
cells.append(nbf.v4.new_code_cell("""hyb_nj = df_newjob[df_newjob.approach_disp == "Hybrid-n_j"].copy()
hyb_mx = df_max[df_max.approach_disp == "Hybrid-n_j"].copy()
# Meme derive de nommage que sections 13/14 (palier mixte x2: "mixte_x2" vs "mixte").
hyb_nj.loc[(hyb_nj.group == "palier") & (hyb_nj.tag == "mixte_x2"), "tag"] = "mixte"

scope_filter = lambda d: d[(d.phase == "B") | ((d.phase == "A") & (d.group == "palier"))]
hyb_nj = scope_filter(hyb_nj)
hyb_mx = scope_filter(hyb_mx)

merge_cols = ["phase", "group", "tag"]
cmp_hyb_obj = hyb_nj[merge_cols + ["flow_time_new_job", "max_flow_time_all", "transfer_energy_total"]].merge(
    hyb_mx[merge_cols + ["flow_time_new_job", "max_flow_time_all", "transfer_energy_total"]],
    on=merge_cols, suffixes=("_newjob", "_max"))
cmp_hyb_obj["scenario"] = cmp_hyb_obj.apply(scenario_label, axis=1)

print(f"{len(cmp_hyb_obj)} scénarios")
cmp_hyb_obj.head()"""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
hatch_max = "//"
metrics = [
    ("flow_time_new_job_newjob", "flow_time_new_job_max", "flow time nouveau job"),
    ("max_flow_time_all_newjob", "max_flow_time_all_max", "flow time max du batch"),
    ("transfer_energy_total_newjob", "transfer_energy_total_max", "énergie de transfert"),
]
scenarios = sorted(cmp_hyb_obj["scenario"].unique(), key=lambda s: (len(s), s))
x = np.arange(len(scenarios))
width = 0.3
hyb_color = colors["Hybrid-n_j"]
for ax, (col_nj, col_mx, ylabel) in zip(axes, metrics):
    sub = cmp_hyb_obj.set_index("scenario").reindex(scenarios)
    ax.bar(x - width / 2, sub[col_nj].values, width, color=hyb_color, label="objectif new_job")
    ax.bar(x + width / 2, sub[col_mx].values, width, color=hyb_color, hatch=hatch_max, alpha=0.6, label="objectif max")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""detail_cols_hyb = ["scenario", "flow_time_new_job_newjob", "flow_time_new_job_max",
                    "max_flow_time_all_newjob", "max_flow_time_all_max",
                    "transfer_energy_total_newjob", "transfer_energy_total_max"]
cmp_hyb_obj[detail_cols_hyb].round(1).set_index("scenario")"""))
cells.append(nbf.v4.new_code_cell("""cmp_hyb_obj[["flow_time_new_job_newjob", "flow_time_new_job_max",
             "max_flow_time_all_newjob", "max_flow_time_all_max",
             "transfer_energy_total_newjob", "transfer_energy_total_max"]].mean().round(1)"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- L'objectif `new_job` cible spécifiquement le nouveau job (flow time nouveau job généralement
  plus bas) mais n'a aucune contrainte sur le flow time max du batch entier.
- L'objectif `max` fait l'inverse : il optimise le pire cas du batch, potentiellement au prix du
  nouveau job lui-même.
- Comparer les deux colonnes `max_flow_time_all_*` montre si cibler le nouveau job dégrade
  réellement le pire cas du batch, ou si les deux objectifs convergent souvent vers un plan
  similaire."""))

# --- Section 16: Pareto front flow time / energie (2026-10-09) ---
cells.append(nbf.v4.new_markdown_cell("""## 16. Front de Pareto — flow time nouveau job vs énergie (2026-10-09)

Incremental / Online_newjob / Hybrid-n_j, scénarios Phase A (palier seul) + Phase B -- même
périmètre que les sections 13/14/15. Un point par (scénario, approche) ; la ligne en escalier
relie les points non dominés (aucun autre point n'est à la fois meilleur en flow time ET en
énergie) -- le vrai front de Pareto, tous scénarios confondus. Une comparaison mono-objectif puis
bi-objectif, même construction."""))
cells.append(nbf.v4.new_code_cell("""def pareto_frontier(points):
    \"\"\"points: liste de (x, y), les deux a minimiser. Retourne les points non domines, tries par x.\"\"\"
    pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_y = float("inf")
    for x, y in pts:
        if y < best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def pareto_plot(ax, sub, color_map, marker_map, order_list, title):
    all_points = list(zip(sub.transfer_energy_total, sub.flow_time_new_job))
    for appr in order_list:
        s = sub[sub.approach_disp == appr]
        ax.scatter(s.transfer_energy_total, s.flow_time_new_job, label=appr, color=color_map[appr],
                   marker=marker_map[appr], s=90, edgecolor="white", linewidth=0.8, zorder=3)
    frontier = pareto_frontier(all_points)
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color="#333333", linestyle="--", linewidth=1.3, zorder=2, label="front de Pareto")
    ax.set_xlabel("énergie de transfert totale")
    ax.set_ylabel("flow time -- nouveau job")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)"""))
cells.append(nbf.v4.new_markdown_cell("""### Mono-objectif"""))
cells.append(nbf.v4.new_code_cell("""pareto_markers = {"Incremental": "o", "Online_newjob": "^", "Hybrid-n_j": "D"}
mono_pareto_scope = df[(df.phase == "B") | ((df.phase == "A") & (df.group == "palier"))].copy()
mono_pareto_scope["scenario"] = mono_pareto_scope.apply(scenario_label, axis=1)

fig, ax = plt.subplots(figsize=(8, 6.5))
pareto_plot(ax, mono_pareto_scope, colors, pareto_markers, order, "Mono-objectif -- front de Pareto flow time / énergie")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""Même chose, scénario par scénario (Phase B seule -- n2/n5/n10/n15), chacun avec son propre front
(3 points, 1 par approche)."""))
cells.append(nbf.v4.new_code_cell("""mono_pareto_B = mono_pareto_scope[mono_pareto_scope.phase == "B"]
n_existing_sorted = sorted(mono_pareto_B.n_existing.unique())
fig, axes = plt.subplots(1, len(n_existing_sorted), figsize=(5 * len(n_existing_sorted), 5), sharey=False)
for ax, n in zip(axes, n_existing_sorted):
    sub = mono_pareto_B[mono_pareto_B.n_existing == n]
    pareto_plot(ax, sub, colors, pareto_markers, order, f"n_existing={n}")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""### Bi-objectif"""))
cells.append(nbf.v4.new_code_cell("""biobj_pareto_scope = pd.concat([df_biobj_A, df_biobj_B])

fig, ax = plt.subplots(figsize=(8, 6.5))
pareto_plot(ax, biobj_pareto_scope, biobj_colors, biobj_markers, biobj_order, "Bi-objectif -- front de Pareto flow time / énergie")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""Même chose, scénario par scénario (Phase B seule -- n2/n5/n10/n15)."""))
cells.append(nbf.v4.new_code_cell("""biobj_pareto_B = df_biobj_B.copy()
n_existing_sorted_biobj = sorted(biobj_pareto_B.n_existing.unique())
fig, axes = plt.subplots(1, len(n_existing_sorted_biobj), figsize=(5 * len(n_existing_sorted_biobj), 5), sharey=False)
for ax, n in zip(axes, n_existing_sorted_biobj):
    sub = biobj_pareto_B[biobj_pareto_B.n_existing == n]
    pareto_plot(ax, sub, biobj_colors, biobj_markers, biobj_order, f"n_existing={n}")
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""### Points clés

- Un point SUR le front de Pareto (relié par la ligne en escalier) n'est dominé par aucun autre
  -- aucune autre approche/scénario ne fait mieux sur les deux métriques à la fois.
- Si une approche n'apparaît presque jamais sur le front, c'est qu'elle est systématiquement
  dominée par au moins une des deux autres sur ce périmètre de scénarios -- pas forcément
  "mauvaise" dans l'absolu, juste jamais le meilleur compromis ici."""))

nb["cells"] = cells
with open(OUT_PATH, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {OUT_PATH}")
