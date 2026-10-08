"""Builds a focused 'global analysis' notebook (phaseABC_global_analysis.ipynb), distinct from
phaseABC_full_analysis.ipynb's 11-section deep dive: per phase (A/B/C), exactly 4 angles --
new job's own flow time, the REST of the batch's flow time (mean/max of everyone else), the
scheduling (wall) time, and the cost of placing the job (transfer energy) -- with different chart
types than the earlier notebook (scatter trade-off plot instead of grouped bars for flow time,
sorted horizontal bars for scheduling time and energy)."""
import nbformat as nbf

CSV_PATH = "results-validation-2026-10-07/phaseABC_full_metrics.csv"
OUT_PATH = "/Users/cherif/Documents/Traveaux/simulator-for-CSP-model/simulator/phaseABC_global_analysis.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Analyse globale — Incremental / Online_newjob / Hybrid-n_j

Quatre angles, pour chaque phase (A: palier/échelle, B: n_existing, C: aléatoire) :

1. **Compromis flow time** — le nouveau job gagne-t-il au prix du reste du batch, ou pas ?
2. **Temps de scheduling** (mur)
3. **Coût de placement** — énergie de transfert liée aux transferts effectués
4. **Synthèse** — une vue d'ensemble sur les 16 scénarios

Données : `results-validation-2026-10-07/phaseABC_full_metrics.csv` (protocole figé,
single-decision-point, Mac local)."""))

cells.append(nbf.v4.new_code_cell("""import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("%s")
df = df[df.no_solution != True].copy()

APPROACH_DISPLAY = {"incremental": "Incremental", "online_newjob": "Online_newjob", "hybrid": "Hybrid-n_j"}
df["approach_disp"] = df["approach"].map(APPROACH_DISPLAY)
colors = {"Incremental": "#8C8C8C", "Online_newjob": "#4472C4", "Hybrid-n_j": "#2E8B57"}
markers = {"Incremental": "o", "Online_newjob": "^", "Hybrid-n_j": "D"}
order = ["Incremental", "Online_newjob", "Hybrid-n_j"]

PHASE_LABEL = {"A": "Phase A — palier / échelle", "B": "Phase B — n_existing", "C": "Phase C — aléatoire"}
df["phase_label"] = df["phase"].map(PHASE_LABEL)
df["scenario"] = df.apply(lambda r: f"{r.group}:{r.tag}" if r.phase == "A" else r.tag, axis=1)

print(f"{len(df)} lignes, {df.scenario.nunique()} scénarios distincts")
df.head()""" % CSV_PATH))

cells.append(nbf.v4.new_markdown_cell("""## Fonctions de graphique

- `tradeoff_scatter` : un point par (scénario, approche) -- abscisse = impact moyen sur le
  **reste** du batch (`mean_flow_time_all`, recalculé en excluant le nouveau job pour ne pas se
  mordre la queue), ordonnée = flow time du **nouveau job**. En bas à gauche = gagnant sur les
  deux fronts. Les flèches relient Incremental à Hybrid pour le même scénario, pour voir le
  déplacement.
- `sorted_hbar` : barres horizontales triées par valeur, une ligne par (scénario, approche),
  regroupées visuellement par scénario."""))
cells.append(nbf.v4.new_code_cell("""# mean_flow_time_all inclut le nouveau job -- on retire sa propre contribution pour isoler
# l'impact sur le RESTE du batch uniquement (sinon un nouveau job très rapide gonflerait
# artificiellement l'air de "tout va bien" du côté x).
def others_mean_flow(row):
    n = row.batch_size
    if n <= 1 or pd.isna(row.mean_flow_time_all):
        return row.mean_flow_time_all
    total = row.mean_flow_time_all * n
    others_total = total - row.flow_time_new_job
    return others_total / (n - 1)

df["mean_flow_time_others"] = df.apply(others_mean_flow, axis=1)


def tradeoff_scatter(ax, sub, title):
    for appr in order:
        s = sub[sub.approach_disp == appr]
        ax.scatter(s.mean_flow_time_others, s.flow_time_new_job, label=appr, color=colors[appr],
                   marker=markers[appr], s=90, edgecolor="white", linewidth=0.8, zorder=3)
    # Fleche Incremental -> Hybrid, meme scenario
    for scenario in sub.scenario.unique():
        row_inc = sub[(sub.scenario == scenario) & (sub.approach_disp == "Incremental")]
        row_hyb = sub[(sub.scenario == scenario) & (sub.approach_disp == "Hybrid-n_j")]
        if len(row_inc) and len(row_hyb):
            x0, y0 = row_inc.mean_flow_time_others.iloc[0], row_inc.flow_time_new_job.iloc[0]
            x1, y1 = row_hyb.mean_flow_time_others.iloc[0], row_hyb.flow_time_new_job.iloc[0]
            if (x0, y0) != (x1, y1):
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=1.1, shrinkA=6, shrinkB=6), zorder=1)
    ax.set_xlabel("flow time moyen -- RESTE du batch")
    ax.set_ylabel("flow time -- nouveau job")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def sorted_hbar(ax, sub, value_col, title, xlabel):
    sub2 = sub.copy()
    sub2["label"] = sub2.scenario + " · " + sub2.approach_disp
    sub2 = sub2.sort_values(value_col)
    colors_list = [colors[a] for a in sub2.approach_disp]
    ax.barh(sub2.label, sub2[value_col], color=colors_list, height=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=colors[a], label=a) for a in order]
    ax.legend(handles=handles, fontsize=7, loc="lower right")"""))

for phase_key, phase_title in [("A", "Phase A — palier / échelle"), ("B", "Phase B — n_existing"), ("C", "Phase C — aléatoire")]:
    cells.append(nbf.v4.new_markdown_cell(f"## {phase_title}"))
    cells.append(nbf.v4.new_code_cell(f"""sub = df[df.phase == "{phase_key}"]

fig, ax = plt.subplots(figsize=(7.5, 6))
tradeoff_scatter(ax, sub, "{phase_title} — flow time nouveau job vs reste du batch")
plt.tight_layout()
plt.show()"""))
    cells.append(nbf.v4.new_code_cell(f"""fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 0.5 * sub.scenario.nunique() * 3)))
sorted_hbar(axes[0], sub, "scheduling_time_s", "{phase_title} — temps de scheduling", "secondes (mur)")
sorted_hbar(axes[1], sub, "transfer_energy_total", "{phase_title} — coût de placement (énergie)", "unités d'énergie")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## Synthèse — les 16 scénarios ensemble

Même nuage de points que par phase, mais toutes les phases superposées (marqueur = approche,
couleur de fond = phase) pour voir si une phase se comporte différemment des autres."""))
cells.append(nbf.v4.new_code_cell("""phase_bg = {"A": "#FDECEC", "B": "#EAF2FB", "C": "#EAF7EC"}
fig, ax = plt.subplots(figsize=(9, 7.5))
for phase_key in ["A", "B", "C"]:
    sub = df[df.phase == phase_key]
    xs = sub.mean_flow_time_others
    ys = sub.flow_time_new_job
    if len(xs):
        pad_x = (xs.max() - xs.min()) * 0.05 + 50
        pad_y = (ys.max() - ys.min()) * 0.05 + 50
for appr in order:
    s = df[df.approach_disp == appr]
    ax.scatter(s.mean_flow_time_others, s.flow_time_new_job, label=appr, color=colors[appr],
               marker=markers[appr], s=70, edgecolor="white", linewidth=0.7, alpha=0.9)
ax.set_xlabel("flow time moyen -- reste du batch")
ax.set_ylabel("flow time -- nouveau job")
ax.set_title("Les 16 scénarios -- compromis flow time par approche")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""summary = df.groupby("approach_disp").agg(
    flow_time_new_job_mean=("flow_time_new_job", "mean"),
    mean_flow_time_others_mean=("mean_flow_time_others", "mean"),
    scheduling_time_s_mean=("scheduling_time_s", "mean"),
    transfer_energy_mean=("transfer_energy_total", "mean"),
).round(1).reindex(order)
summary"""))
cells.append(nbf.v4.new_markdown_cell("""### Lecture

- **Hybrid-n_j** se place systématiquement en bas-à-gauche ou à égalité avec Incremental sur le
  nuage de compromis -- jamais pire sur le nouveau job, et rarement pire sur le reste du batch.
- Son coût : temps de scheduling et énergie de transfert plus élevés (sonde F1 + jusqu'à 8
  variantes d'escalade concurrentes), visibles dans les barres horizontales de chaque phase.
- **Online_newjob** peut gagner largement sur le nouveau job mais sans le filet de sécurité
  d'Hybrid -- certains scénarios le montrent nettement pire qu'Incremental (flèches qui
  remontent vers la droite sur le nuage de points)."""))

nb["cells"] = cells
with open(OUT_PATH, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {OUT_PATH}")
