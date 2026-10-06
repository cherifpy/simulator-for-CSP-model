"""Builds results-grid5000/analysis/flow_energy_volume_analysis.ipynb: flow time, data-volume
and transfer-energy analysis across Exp2 (live 20j/50n comparison) and Exp1 (dataset-size sweep +
infra-load / n_existing sweep). Run with the venv python, then execute via nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Flow time, volume de données et énergie de transfert

Analyse consolidée des runs Grid5000 du 2026-09-28/29 (après correctifs des 3 bugs de blocage + `--no-charge-thinking-time`) :

1. **Exp2** — comparaison live 20 jobs / 50 nœuds (`incremental` / `online_biobj` / `hybrid`)
2. **Exp1** — sweep taille de dataset (small/medium/large, 5 répétitions)
3. **Exp1** — sweep charge infra (`n_existing` ∈ {5, 10, 20})

Palette fixe par approche (ordre catégoriel constant sur tout le notebook) :
`incremental` (bleu-gris), `online_biobj` (orange), `hybrid` (vert).""")

code("""
import ast
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = ".."  # notebook lives in results-grid5000/analysis/

# Fixed categorical palette, assigned by identity (approach), never cycled/reused.
APPROACH_COLORS = {
    "incremental": "#4C72A0",
    "online_biobj": "#DD8452",
    "hybrid": "#55A868",
}
APPROACH_ORDER = ["incremental", "online_biobj", "hybrid"]
APPROACH_LABELS = {"incremental": "incremental", "online_biobj": "online_biobj", "hybrid": "hybrid"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

def approach_color(a):
    return APPROACH_COLORS.get(a, "#999999")

def plot_cdf(ax, values, color, label):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, linewidth=2.2, label=label)

# scheduling_time_s (real wall-clock cost of the CSP solve itself) isn't saved in any of the
# CSV/JSON summaries for Exp1's single-decision-point protocol -- only printed inline per run
# in solver_stdout.log as a Python dict literal ("[tier #repeat] approach result: {...}").
# Parsed here and merged back onto the per-run tables.
_RESULT_LINE_RE = re.compile(r"\\[(\\w+) #(-?\\d+)\\] (\\w+) result: (\\{.*\\})")

def parse_scheduling_times(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = _RESULT_LINE_RE.search(line)
            if not m:
                continue
            tier, repeat, approach, dict_str = m.groups()
            d = ast.literal_eval(dict_str)
            rows.append({
                "tier": tier,
                "repeat": int(repeat),
                "approach": approach,
                "scheduling_time_s": d.get("scheduling_time_s"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["approach"] = df["approach"].replace({"epsilon": "online_biobj"})
    return df
""")

# ---------------------------------------------------------------------------
md("""## 1. Exp2 — comparaison live 20j/50n

Chaque approche a traité les mêmes 20 jobs (mêmes tailles de dataset, mêmes arrivées). Les 3 runs ont
terminé avec succès (`STORAGE CHECK: OK`) le 2026-09-29, une première pour `online_biobj` sur ce
protocole.""")

code("""
# lambda_rate=100 run (archived after the lambda_rate=300 / incremental-60s-cap relaunch on
# 2026-09-29 -- see exp2_grid5000_parameters_2026-09-28.md for why).
EXP2_DIR = f"{RESULTS_DIR}/online_vs_biobj_workload_inst-20J-50N_lambda100_2026-09-29"

exp2_jobs = {}
exp2_energy = {}
exp2_wall_time = {}
for approach in APPROACH_ORDER:
    df = pd.read_csv(f"{EXP2_DIR}/{approach}/infos_on_jobs.csv", index_col=0)
    df["flow_time"] = df["finishing_time"] - df["arriving_time"]
    df["approach"] = approach
    exp2_jobs[approach] = df

    edf = pd.read_csv(f"{EXP2_DIR}/{approach}/infos_on_transfers_energy.csv", index_col=0)
    edf["approach"] = approach
    exp2_energy[approach] = edf

    # Total real wall-clock time for the whole live run (all replans combined), printed once at
    # the end by xp_online_grid5000.py -- the LAST occurrence matters if this results_dir was
    # reused across an earlier, superseded launch (solver_stdout.log then holds more than one
    # run's output, appended).
    wall_times = []
    with open(f"{EXP2_DIR}/{approach}/solver_stdout.log") as f:
        for line in f:
            m = re.search(r"total wall time = ([\\d.]+)", line)
            if m:
                wall_times.append(float(m.group(1)))
    exp2_wall_time[approach] = wall_times[-1] if wall_times else None

exp2_jobs_all = pd.concat(exp2_jobs.values(), ignore_index=True)
exp2_energy_all = pd.concat(exp2_energy.values(), ignore_index=True)
exp2_jobs_all[["job_id", "approach", "dataset size", "flow_time", "arriving_time", "finishing_time"]].head()
""")

md("### 1.1 Flow time par job, par approche")

code("""
fig, ax = plt.subplots(figsize=(11, 5))
for approach in APPROACH_ORDER:
    d = exp2_jobs_all[exp2_jobs_all["approach"] == approach].sort_values("job_id")
    ax.plot(d["job_id"], d["flow_time"], marker="o", markersize=4, linewidth=2,
            color=approach_color(approach), label=APPROACH_LABELS[approach])

ax.set_xlabel("job_id (ordre d'arrivée)")
ax.set_ylabel("Flow time (s)")
ax.set_title("Exp2 — flow time par job, 20j/50n live")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("### 1.2 CDF du flow time")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for approach in APPROACH_ORDER:
    d = exp2_jobs_all[exp2_jobs_all["approach"] == approach]
    plot_cdf(ax, d["flow_time"].values, approach_color(approach), APPROACH_LABELS[approach])

ax.set_xlabel("Flow time (s)")
ax.set_ylabel("Proportion cumulée des jobs")
ax.set_ylim(0, 1.02)
ax.set_title("Exp2 — CDF du flow time (20 jobs, 3 approches)")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("### 1.3 Volume de données (dataset_size) vs flow time")

code("""
fig, ax = plt.subplots(figsize=(8, 6))
for approach in APPROACH_ORDER:
    d = exp2_jobs_all[exp2_jobs_all["approach"] == approach]
    ax.scatter(d["dataset size"], d["flow_time"], s=60, alpha=0.75,
               color=approach_color(approach), edgecolor="white", linewidth=0.6,
               label=APPROACH_LABELS[approach])

ax.set_xlabel("Taille du dataset (MB)")
ax.set_ylabel("Flow time (s)")
ax.set_title("Exp2 — volume de données vs flow time")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

md("### 1.4 Énergie de transfert")

code("""
energy_by_approach = exp2_energy_all.groupby("approach")["total_energy"].sum().reindex(APPROACH_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bars = ax.bar(range(len(APPROACH_ORDER)), energy_by_approach.values,
              color=[approach_color(a) for a in APPROACH_ORDER])
ax.set_xticks(range(len(APPROACH_ORDER)))
ax.set_xticklabels([APPROACH_LABELS[a] for a in APPROACH_ORDER])
ax.set_ylabel("Énergie totale de transfert")
ax.set_title("Exp2 — énergie totale par approche")
for b, v in zip(bars, energy_by_approach.values):
    ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)

ax = axes[1]
energy_by_job = exp2_energy_all.groupby(["approach", "job"])["total_energy"].sum().reset_index()
for approach in APPROACH_ORDER:
    d = energy_by_job[energy_by_job["approach"] == approach].sort_values("job")
    ax.plot(d["job"], d["total_energy"], marker="o", markersize=4, linewidth=2,
            color=approach_color(approach), label=APPROACH_LABELS[approach])
ax.set_xlabel("job_id")
ax.set_ylabel("Énergie de transfert (par job)")
ax.set_title("Exp2 — énergie de transfert par job")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""### 1.5 Temps de scheduling (coût réel du run)

Temps réel total (toutes les tâches CSP confondues) pour dérouler le run live entier, tel
qu'imprimé par `xp_online_grid5000.py` à la fin (`### DONE: total wall time = ...`). Pour ce run
(`lambda_rate=100`), `incremental` et `online_biobj` partagent le même budget par solve (600s) ;
`hybrid` reste bien plus rapide grâce à son budget interne réduit (30s pour la majorité des jobs).""")

code("""
fig, ax = plt.subplots(figsize=(7, 5))
vals = [exp2_wall_time[a] for a in APPROACH_ORDER]
bars = ax.bar(range(len(APPROACH_ORDER)), vals, color=[approach_color(a) for a in APPROACH_ORDER])
ax.set_xticks(range(len(APPROACH_ORDER)))
ax.set_xticklabels([APPROACH_LABELS[a] for a in APPROACH_ORDER])
ax.set_ylabel("Temps de scheduling total (s, temps réel)")
ax.set_title("Exp2 — temps de scheduling total par approche")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:,.0f}s", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()
""")

md("### 1.6 Flow time moyen / max — vue synthèse")

code("""
summary_exp2 = exp2_jobs_all.groupby("approach")["flow_time"].agg(["mean", "max"]).reindex(APPROACH_ORDER)
summary_exp2["total_energy"] = energy_by_approach
summary_exp2["scheduling_time_total_s"] = pd.Series(exp2_wall_time).reindex(APPROACH_ORDER)
summary_exp2.columns = ["flow_time_moyen", "flow_time_max", "energie_totale", "temps_scheduling_total_s"]
summary_exp2
""")

# ---------------------------------------------------------------------------
md("""## 2. Exp1 — sweep taille de dataset

Tiers small / medium / large, 5 répétitions par tier, `n_existing=10` fixe. Budget solveur 600s
(mêmes ratios epsilon/hybrid qu'Exp2).""")

code("""
DSS_DIR = f"{RESULTS_DIR}/dataset_size_sweep_3way_10j-50n_2026-09-28"

with open(f"{DSS_DIR}/results.json") as f:
    dss_results = json.load(f)

dss_agg = pd.DataFrame(dss_results["aggregates"])
dss_runs = pd.read_csv(f"{DSS_DIR}/runs_by_dataset_size.csv")

# runs_by_dataset_size.csv uses 'epsilon' for online_biobj -- normalize to the same
# vocabulary used everywhere else in this notebook.
dss_runs["approach"] = dss_runs["approach"].replace({"epsilon": "online_biobj"})

# scheduling_time_s isn't in the CSV export -- parsed from the raw solver log and merged back on
# (tier, repeat, approach), the same key that identifies a row in dss_runs.
dss_sched = parse_scheduling_times(f"{DSS_DIR}/solver_stdout.log")
dss_runs = dss_runs.merge(dss_sched, on=["tier", "repeat", "approach"], how="left")

dss_agg
""")

md("### 2.1 Flow time moyen par tier et approche")

code("""
tiers = dss_agg["tier"].tolist()
x = np.arange(len(tiers))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i, approach in enumerate(APPROACH_ORDER):
    key = "epsilon_mean_flow_time" if approach == "online_biobj" else f"{approach}_mean_flow_time"
    ax.bar(x + (i - 1) * width, dss_agg[key], width, color=approach_color(approach),
           label=APPROACH_LABELS[approach])
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_xlabel("Tier (taille de dataset)")
ax.set_ylabel("Flow time moyen (s)")
ax.set_title("Exp1 — flow time moyen par tier")
ax.legend(frameon=False)

ax = axes[1]
for i, approach in enumerate(APPROACH_ORDER):
    key = "epsilon_energy" if approach == "online_biobj" else f"{approach}_energy"
    ax.bar(x + (i - 1) * width, dss_agg[key], width, color=approach_color(approach),
           label=APPROACH_LABELS[approach])
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_xlabel("Tier (taille de dataset)")
ax.set_ylabel("Énergie de transfert")
ax.set_title("Exp1 — énergie de transfert par tier")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""### 2.2 Temps de scheduling

Coût réel (temps solveur, pas temps simulé) de chaque décision — rappel de la découverte de cette
session : le solveur consomme quasi tout son budget alloué à chaque appel (il ne s'arrête jamais
tôt une fois la recherche de solutions améliorantes épuisée, faute de preuve d'optimalité).""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sched_by_tier = dss_runs.groupby(["tier", "approach"])["scheduling_time_s"].mean().unstack("approach")
sched_by_tier = sched_by_tier.reindex(tiers)[APPROACH_ORDER]
for i, approach in enumerate(APPROACH_ORDER):
    ax.bar(x + (i - 1) * width, sched_by_tier[approach], width, color=approach_color(approach),
           label=APPROACH_LABELS[approach])
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_xlabel("Tier (taille de dataset)")
ax.set_ylabel("Temps de scheduling moyen (s, temps réel)")
ax.set_title("Exp1 — temps de scheduling moyen par tier")
ax.legend(frameon=False)

ax = axes[1]
for approach in APPROACH_ORDER:
    d = dss_runs[dss_runs["approach"] == approach]
    ax.scatter(d["dataset_size"], d["scheduling_time_s"], s=55, alpha=0.75,
               color=approach_color(approach), edgecolor="white", linewidth=0.5,
               label=APPROACH_LABELS[approach])
ax.set_xlabel("Taille du dataset (MB)")
ax.set_ylabel("Temps de scheduling (s, temps réel)")
ax.set_title("Exp1 — volume de données vs temps de scheduling")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("### 2.3 Volume de données vs flow time / énergie (tous runs)")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for approach in APPROACH_ORDER:
    d = dss_runs[dss_runs["approach"] == approach]
    ax.scatter(d["dataset_size"], d["flow_time_new_job"], s=55, alpha=0.75,
               color=approach_color(approach), edgecolor="white", linewidth=0.5,
               label=APPROACH_LABELS[approach])
ax.set_xlabel("Taille du dataset (MB)")
ax.set_ylabel("Flow time du nouveau job (s)")
ax.set_title("Exp1 — volume de données vs flow time")
ax.legend(frameon=False)

ax = axes[1]
for approach in APPROACH_ORDER:
    d = dss_runs[dss_runs["approach"] == approach]
    ax.scatter(d["dataset_size"], d["transfer_energy_total"], s=55, alpha=0.75,
               color=approach_color(approach), edgecolor="white", linewidth=0.5,
               label=APPROACH_LABELS[approach])
ax.set_xlabel("Taille du dataset (MB)")
ax.set_ylabel("Énergie totale de transfert")
ax.set_title("Exp1 — volume de données vs énergie")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("### 2.4 Dispersion du flow time par approche (boxplot, tous tiers confondus)")

code("""
fig, ax = plt.subplots(figsize=(8, 5.5))
data_by_approach = [dss_runs[dss_runs["approach"] == a]["flow_time_new_job"].values for a in APPROACH_ORDER]
bp = ax.boxplot(data_by_approach, labels=[APPROACH_LABELS[a] for a in APPROACH_ORDER],
                 patch_artist=True, widths=0.5, medianprops=dict(color="black"))
for patch, approach in zip(bp["boxes"], APPROACH_ORDER):
    patch.set_facecolor(approach_color(approach))
    patch.set_alpha(0.75)
ax.set_ylabel("Flow time du nouveau job (s)")
ax.set_title("Exp1 — dispersion du flow time par approche")
plt.tight_layout()
plt.show()
""")

md("### 2.5 CDF du flow time — global et par tier")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)

ax = axes[0]
for approach in APPROACH_ORDER:
    d = dss_runs[dss_runs["approach"] == approach]
    plot_cdf(ax, d["flow_time_new_job"].values, approach_color(approach), APPROACH_LABELS[approach])
ax.set_title("Tous tiers confondus")
ax.set_xlabel("Flow time (s)")
ax.set_ylabel("Proportion cumulée")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False)

for ax, tier in zip(axes[1:], ["small", "medium", "large"]):
    for approach in APPROACH_ORDER:
        d = dss_runs[(dss_runs["approach"] == approach) & (dss_runs["tier"] == tier)]
        plot_cdf(ax, d["flow_time_new_job"].values, approach_color(approach), APPROACH_LABELS[approach])
    ax.set_title(f"Tier {tier}")
    ax.set_xlabel("Flow time (s)")
    ax.set_ylim(0, 1.02)

fig.suptitle("Exp1 — CDF du flow time (sweep taille de dataset)", y=1.03)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 3. Exp1 — sweep charge infra (n_existing)

`n_existing` ∈ {5, 10, 20} sur `inst-50J-50N`, tier medium fixe, 1 répétition par valeur. Mesure
l'impact de la charge déjà présente sur l'infrastructure au moment de placer un nouveau job.""")

code("""
NE_DIR = f"{RESULTS_DIR}/nexisting_sweep_3way"
ne_values = [5, 10, 20]

ne_rows = []
for n in ne_values:
    with open(f"{NE_DIR}/n{n}/results.json") as f:
        r = json.load(f)
    agg = r["aggregates"][0]  # single 'medium' tier point per n_existing value

    # Same story as the dataset-size sweep: scheduling_time_s only exists inline in the raw log.
    sched = parse_scheduling_times(f"{NE_DIR}/n{n}/solver_stdout.log")
    sched_by_approach = sched.set_index("approach")["scheduling_time_s"] if not sched.empty else {}

    for approach in APPROACH_ORDER:
        key_prefix = "epsilon" if approach == "online_biobj" else approach
        ne_rows.append({
            "n_existing": n,
            "approach": approach,
            "mean_flow_time": agg[f"{key_prefix}_mean_flow_time"],
            "energy": agg[f"{key_prefix}_energy"],
            "scheduling_time_s": sched_by_approach.get(approach) if len(sched_by_approach) else None,
        })

ne_df = pd.DataFrame(ne_rows)
ne_df.pivot(index="n_existing", columns="approach", values=["mean_flow_time", "energy", "scheduling_time_s"])
""")

md("### 3.1 Flow time moyen et énergie vs charge infra")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for approach in APPROACH_ORDER:
    d = ne_df[ne_df["approach"] == approach].sort_values("n_existing")
    ax.plot(d["n_existing"], d["mean_flow_time"], marker="o", markersize=7, linewidth=2.2,
            color=approach_color(approach), label=APPROACH_LABELS[approach])
ax.set_xticks(ne_values)
ax.set_xlabel("n_existing (jobs déjà présents)")
ax.set_ylabel("Flow time moyen (s)")
ax.set_title("Exp1 — flow time moyen vs charge infra")
ax.legend(frameon=False)

ax = axes[1]
for approach in APPROACH_ORDER:
    d = ne_df[ne_df["approach"] == approach].sort_values("n_existing")
    ax.plot(d["n_existing"], d["energy"], marker="o", markersize=7, linewidth=2.2,
            color=approach_color(approach), label=APPROACH_LABELS[approach])
ax.set_xticks(ne_values)
ax.set_xlabel("n_existing (jobs déjà présents)")
ax.set_ylabel("Énergie de transfert")
ax.set_title("Exp1 — énergie vs charge infra")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("### 3.2 Temps de scheduling vs charge infra")

code("""
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for approach in APPROACH_ORDER:
    d = ne_df[ne_df["approach"] == approach].sort_values("n_existing")
    ax.plot(d["n_existing"], d["scheduling_time_s"], marker="o", markersize=7, linewidth=2.2,
            color=approach_color(approach), label=APPROACH_LABELS[approach])
ax.set_xticks(ne_values)
ax.set_xlabel("n_existing (jobs déjà présents)")
ax.set_ylabel("Temps de scheduling (s, temps réel)")
ax.set_title("Exp1 — temps de scheduling vs charge infra")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 4. Synthèse

Récapitulatif des indicateurs clés (flow time moyen, énergie totale) sur les 3 jeux de résultats,
une ligne par (jeu de données × approche).""")

code("""
rows = []
for approach in APPROACH_ORDER:
    d = exp2_jobs_all[exp2_jobs_all["approach"] == approach]
    rows.append({
        "dataset": "Exp2 (live 20j/50n)",
        "approach": approach,
        "flow_time_moyen": d["flow_time"].mean(),
        "flow_time_max": d["flow_time"].max(),
        "energie_totale": energy_by_approach[approach],
        # Exp2's own number is the TOTAL for the whole run (all replans), not a per-run mean like
        # the other two datasets below -- different unit, kept in the same column for a quick
        # side-by-side read, not a like-for-like comparison.
        "temps_scheduling_s": exp2_wall_time[approach],
    })

dss_sched_mean = dss_runs.groupby(["tier", "approach"])["scheduling_time_s"].mean()
for _, r in dss_agg.iterrows():
    for approach in APPROACH_ORDER:
        key_prefix = "epsilon" if approach == "online_biobj" else approach
        rows.append({
            "dataset": f"Exp1 dataset-size ({r['tier']})",
            "approach": approach,
            "flow_time_moyen": r[f"{key_prefix}_mean_flow_time"],
            "flow_time_max": np.nan,
            "energie_totale": r[f"{key_prefix}_energy"],
            "temps_scheduling_s": dss_sched_mean.get((r["tier"], approach)),
        })

for n in ne_values:
    d = ne_df[ne_df["n_existing"] == n]
    for _, r in d.iterrows():
        rows.append({
            "dataset": f"Exp1 infra-load (n_existing={n})",
            "approach": r["approach"],
            "flow_time_moyen": r["mean_flow_time"],
            "flow_time_max": np.nan,
            "energie_totale": r["energy"],
            "temps_scheduling_s": r["scheduling_time_s"],
        })

summary = pd.DataFrame(rows)
summary
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("results-grid5000/analysis/flow_energy_volume_analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
