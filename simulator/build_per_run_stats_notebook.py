"""Builds results-grid5000/analysis/per_run_stats_notebook.ipynb: for every completed run
(5 Exp2 live comparisons + Exp1's 2 sweeps), a per-approach stats table (flow time and
scheduling time: mean/std/max/min), a flow-time CDF, and data-volume + energy figures.

Scheduling-time caveat (documented inline in the notebook too): a per-replan REAL wall-clock
duration was only ever logged for hybrid's 4-way parallel-escalation runs (the
"### PARALLEL ESCALATION: variant=(...) (Xs), ..." lines -- the charged cost is the slowest
variant, since all 4 run concurrently). incremental/online_biobj and hybrid's older
single-solver runs never printed a per-replan elapsed time (only Solver time limit and the
final total), so for those, std/max/min are NaN and only an implicit mean
(total wall time / nb_replans) is reported -- flagged, not fabricated.

Run with the venv python, then execute via nbconvert.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Stats par run — temps de scheduling, flow time, volume, énergie

Une section par run complété : temps de scheduling (mean/std/max/min), flow time (mean/std/max/min)
avec CDF, volume de données et énergie de transfert, pour chaque approche (`incremental`,
`online_biobj`, `hybrid`).

**Limite du temps de scheduling à connaître avant de lire les tableaux** : la durée réelle
(temps horloge) de *chaque* résolution CSP individuelle n'a été journalisée en clair que pour
les runs `hybrid` en escalade parallèle à 4 solveurs (les lignes `### PARALLEL ESCALATION: ... ###`
donnent les 4 durées, dont la plus lente est celle facturée). Pour `incremental`/`online_biobj`
et pour `hybrid` en mode 1-solveur (runs lambda=100 et lambda=300), seul le temps total du run
est disponible dans les logs — la moyenne par replan est donc dérivée (total ÷ nb de replans) et
l'écart-type/min/max sont indisponibles (`NaN`), plutôt que reconstruits arbitrairement.""")

code("""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = ".."  # notebook lives in results-grid5000/analysis/

APPROACH_COLORS = {
    "incremental": "#4C72A0",
    "online_biobj": "#DD8452",
    "hybrid": "#55A868",
}
APPROACH_ORDER = ["incremental", "online_biobj", "hybrid"]

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
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, linewidth=2.2, label=label)


_PARALLEL_ESC_RE = re.compile(r"### PARALLEL ESCALATION: (.*) ###")
_VARIANT_TIME_RE = re.compile(r"\\(([\\d.]+)s\\)")
_TOTAL_WALL_RE = re.compile(r"total wall time = ([\\d.]+)")
_REPLAN_MARKER_RE = re.compile(r"### schedulingUsingJavaCSP:")


def per_replan_scheduling_times(log_path):
    \"\"\"Real wall-clock time (s) per top-level replan decision, when recoverable.
    Only hybrid's parallel-escalation runs log this per-replan (max of the 4 concurrent
    variants = the charged cost). Returns an empty list otherwise.\"\"\"
    times = []
    with open(log_path) as f:
        for line in f:
            m = _PARALLEL_ESC_RE.search(line)
            if not m:
                continue
            variant_times = [float(v) for v in _VARIANT_TIME_RE.findall(m.group(1))]
            if variant_times:
                times.append(max(variant_times))
    return times


def run_summary_stats(log_path):
    \"\"\"Returns (nb_replans, total_wall_time_s, per_replan_times_or_None).\"\"\"
    nb_replans = 0
    total_wall = None
    with open(log_path) as f:
        for line in f:
            if _REPLAN_MARKER_RE.search(line):
                nb_replans += 1
            m = _TOTAL_WALL_RE.search(line)
            if m:
                total_wall = float(m.group(1))
    per_replan = per_replan_scheduling_times(log_path)
    return nb_replans, total_wall, (per_replan if per_replan else None)


def load_run(run_dir, approaches=APPROACH_ORDER):
    \"\"\"Loads per-job data, energy, and scheduling-time info for every approach of one run
    directory. Returns (jobs_by_approach: dict[str, DataFrame], sched_by_approach: dict[str, dict]).\"\"\"
    jobs_by_approach = {}
    sched_by_approach = {}
    for approach in approaches:
        base = f"{run_dir}/{approach}"
        jobs = pd.read_csv(f"{base}/infos_on_jobs.csv", index_col=0)
        jobs["flow_time"] = jobs["finishing_time"] - jobs["arriving_time"]
        jobs["approach"] = approach
        energy = pd.read_csv(f"{base}/infos_on_transfers_energy.csv", index_col=0)
        jobs_by_approach[approach] = jobs

        nb_replans, total_wall, per_replan = run_summary_stats(f"{base}/solver_stdout.log")
        sched_by_approach[approach] = {
            "nb_replans": nb_replans,
            "total_wall_time_s": total_wall,
            "per_replan_times": per_replan,
            "energy_total": energy["total_energy"].sum(),
        }
    return jobs_by_approach, sched_by_approach


def stats_table(jobs_by_approach, sched_by_approach, approaches=APPROACH_ORDER):
    rows = []
    for a in approaches:
        jobs = jobs_by_approach[a]
        sched = sched_by_approach[a]
        ft = jobs["flow_time"].values.astype(float)

        if sched["per_replan_times"] is not None:
            st = np.array(sched["per_replan_times"])
            st_mean, st_std, st_max, st_min = st.mean(), st.std(ddof=0), st.max(), st.min()
            st_note = "réel, par replan"
        else:
            st_mean = (sched["total_wall_time_s"] / sched["nb_replans"]
                       if sched["nb_replans"] else np.nan)
            st_std = st_max = st_min = np.nan
            st_note = "moyenne dérivée (total/replans) -- détail indisponible"

        rows.append({
            "approach": a,
            "flow_time_mean": ft.mean(), "flow_time_std": ft.std(ddof=0),
            "flow_time_max": ft.max(), "flow_time_min": ft.min(),
            "sched_time_mean_s": st_mean, "sched_time_std_s": st_std,
            "sched_time_max_s": st_max, "sched_time_min_s": st_min,
            "sched_time_note": st_note,
            "data_volume_mean_MB": jobs["dataset size"].mean(),
            "data_volume_total_MB": jobs["dataset size"].sum(),
            "energy_total": sched["energy_total"],
            "energy_mean_per_job": sched["energy_total"] / len(jobs),
        })
    return pd.DataFrame(rows).set_index("approach")


def plot_run_cdfs(jobs_by_approach, title, approaches=APPROACH_ORDER):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for a in approaches:
        plot_cdf(ax, jobs_by_approach[a]["flow_time"].values, approach_color(a), a)
    ax.set_xlabel("Flow time (s)")
    ax.set_ylabel("Proportion cumulée des jobs")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
""")

# ---------------------------------------------------------------------------
md("## 1. Exp2 — runs live")

EXP2_RUNS = [
    ("10J-20N (lambda=200, hybrid 4 solveurs)", "online_vs_biobj_workload_inst-10J-20N"),
    ("20J-50N (lambda=100, hybrid 1 solveur)", "online_vs_biobj_workload_inst-20J-50N_lambda100_2026-09-29"),
    ("20J-50N (lambda=300, hybrid 1 solveur)", "online_vs_biobj_workload_inst-20J-50N_lambda300_2026-09-29"),
    ("20J-50N (lambda=200, hybrid 4 solveurs)", "online_vs_biobj_workload_inst-20J-50N"),
    ("50J-50N (lambda=200, hybrid 4 solveurs)", "online_vs_biobj_workload_inst-50J-50N"),
]

all_run_stats = []

for label, dirname in EXP2_RUNS:
    md(f"### {label}")
    code(f"""
jobs_by_approach, sched_by_approach = load_run(f"{{RESULTS_DIR}}/{dirname}")
stats = stats_table(jobs_by_approach, sched_by_approach)
stats
""")
    code(f"""
plot_run_cdfs(jobs_by_approach, "CDF flow time -- {label}")
""")

md("""### Récapitulatif Exp2

Tableau compact : une ligne par (run × approche), colonnes essentielles seulement.""")

code(f"""
exp2_summary_rows = []
_exp2_runs = {[(label, dirname) for label, dirname in EXP2_RUNS]!r}
for label, dirname in _exp2_runs:
    jobs_by_approach, sched_by_approach = load_run(f"{{RESULTS_DIR}}/{{dirname}}")
    stats = stats_table(jobs_by_approach, sched_by_approach)
    for a in APPROACH_ORDER:
        r = stats.loc[a]
        exp2_summary_rows.append({{
            "run": label, "approach": a,
            "flow_time_mean": r["flow_time_mean"], "flow_time_std": r["flow_time_std"],
            "flow_time_max": r["flow_time_max"], "flow_time_min": r["flow_time_min"],
            "sched_time_mean_s": r["sched_time_mean_s"], "sched_time_std_s": r["sched_time_std_s"],
            "sched_time_max_s": r["sched_time_max_s"], "sched_time_min_s": r["sched_time_min_s"],
            "data_volume_total_MB": r["data_volume_total_MB"], "energy_total": r["energy_total"],
        }})
exp2_summary = pd.DataFrame(exp2_summary_rows)
exp2_summary
""")

# ---------------------------------------------------------------------------
md("""## 2. Exp1 — sweep taille de dataset

Tiers small/medium/large, `n_existing=10` fixe, 5 répétitions par tier -- assez de répétitions
pour un vrai écart-type sur le temps de scheduling (contrairement à Exp2 où chaque run live
n'offre qu'une seule trajectoire par approche).""")

code("""
import ast
import json

DSS_DIR = f"{RESULTS_DIR}/dataset_size_sweep_3way_10j-50n_2026-09-28"

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
                "tier": tier, "repeat": int(repeat), "approach": approach,
                "scheduling_time_s": d.get("scheduling_time_s"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["approach"] = df["approach"].replace({"epsilon": "online_biobj"})
    return df

dss_runs = pd.read_csv(f"{DSS_DIR}/runs_by_dataset_size.csv")
dss_runs["approach"] = dss_runs["approach"].replace({"epsilon": "online_biobj"})
dss_sched = parse_scheduling_times(f"{DSS_DIR}/solver_stdout.log")
dss_runs = dss_runs.merge(dss_sched, on=["tier", "repeat", "approach"], how="left")
dss_runs.head()
""")

md("### Stats par tier")

code("""
dss_stats_rows = []
for tier in ["small", "medium", "large"]:
    for a in APPROACH_ORDER:
        d = dss_runs[(dss_runs["tier"] == tier) & (dss_runs["approach"] == a)]
        ft = d["flow_time_new_job"].values.astype(float)
        st = d["scheduling_time_s"].dropna().values.astype(float)
        dss_stats_rows.append({
            "tier": tier, "approach": a,
            "flow_time_mean": ft.mean(), "flow_time_std": ft.std(ddof=0),
            "flow_time_max": ft.max(), "flow_time_min": ft.min(),
            "sched_time_mean_s": st.mean() if len(st) else np.nan,
            "sched_time_std_s": st.std(ddof=0) if len(st) else np.nan,
            "sched_time_max_s": st.max() if len(st) else np.nan,
            "sched_time_min_s": st.min() if len(st) else np.nan,
            "data_volume_mean_MB": d["dataset_size"].mean(),
            "energy_total": d["transfer_energy_total"].sum(),
            "n_repeats": len(d),
        })
dss_stats = pd.DataFrame(dss_stats_rows).set_index(["tier", "approach"])
dss_stats
""")

md("### CDF du flow time, par tier")

code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, tier in zip(axes, ["small", "medium", "large"]):
    for a in APPROACH_ORDER:
        d = dss_runs[(dss_runs["tier"] == tier) & (dss_runs["approach"] == a)]
        plot_cdf(ax, d["flow_time_new_job"].values, approach_color(a), a)
    ax.set_title(f"Tier {tier}")
    ax.set_xlabel("Flow time (s)")
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Proportion cumulée")
axes[0].legend(frameon=False)
fig.suptitle("Exp1 dataset-size sweep -- CDF du flow time par tier", y=1.03)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 3. Exp1 — sweep charge infra (n_existing)

`n_existing` ∈ {5, 10, 20}, tier medium fixe, **1 seule répétition par valeur** -- std/max/min
égalent donc la valeur unique elle-même (n=1), affichés pour la forme mais sans variance réelle
à en tirer.""")

code("""
NE_DIR = f"{RESULTS_DIR}/nexisting_sweep_3way"
ne_values = [5, 10, 20]

ne_stats_rows = []
ne_jobs_cache = {}
for n in ne_values:
    sched = parse_scheduling_times(f"{NE_DIR}/n{n}/solver_stdout.log")
    with open(f"{NE_DIR}/n{n}/results.json") as f:
        r = json.load(f)
    agg = r["aggregates"][0]
    for a in APPROACH_ORDER:
        key_prefix = "epsilon" if a == "online_biobj" else a
        st = sched[sched["approach"] == a]["scheduling_time_s"].dropna().values.astype(float)
        ne_stats_rows.append({
            "n_existing": n, "approach": a,
            "flow_time_mean": agg[f"{key_prefix}_mean_flow_time"],
            "sched_time_mean_s": st.mean() if len(st) else np.nan,
            "sched_time_std_s": st.std(ddof=0) if len(st) else np.nan,
            "sched_time_max_s": st.max() if len(st) else np.nan,
            "sched_time_min_s": st.min() if len(st) else np.nan,
            "energy_total": agg[f"{key_prefix}_energy"],
            "n_repeats": len(st),
        })
ne_stats = pd.DataFrame(ne_stats_rows).set_index(["n_existing", "approach"])
ne_stats
""")

# ---------------------------------------------------------------------------
md("""## 4. Hybrid — split par moyenne vs split par médiane

`_splitRunningJobsByMeanFlowTime` a remplacé `_splitRunningJobsByMedianFlowTime` (2026-10-01) pour
décider quels jobs en cours sont gelés dans les variantes `freeze_below_*`/`freeze_above_*` de
l'escalade parallèle à 4 solveurs. Contrairement à la médiane, la moyenne ne garantit pas un split
50/50 en nombre de jobs : sur une distribution de flow time asymétrique, elle gèle un groupe
déséquilibré. Les deux runs (mean-split) ont été relancés sur `10J-20N` et `20J-50N`, les résultats
`median-split` d'origine ont été archivés avant le relancement -- comparaison directe ci-dessous.
Les deux variantes utilisent l'escalade parallèle, donc le temps de scheduling réel par replan est
disponible pour les deux (pas de limite comme pour `incremental`/`online_biobj`).""")

code("""
VARIANT_COLORS = {"mean-split": "#55A868", "median-split": "#8172B2"}
VARIANTS = {"mean-split": "hybrid", "median-split": "hybrid_mediansplit_2026-10-01"}

split_comparison_runs = {
    "20J-50N": "online_vs_biobj_workload_inst-20J-50N",
    "10J-20N": "online_vs_biobj_workload_inst-10J-20N",
}
""")

for inst_label, run_dirname in [("20J-50N", "online_vs_biobj_workload_inst-20J-50N"),
                                  ("10J-20N", "online_vs_biobj_workload_inst-10J-20N")]:
    md(f"### {inst_label} — mean-split vs median-split")
    code(f"""
jobs_by_variant = {{}}
sched_by_variant = {{}}
for variant_label, subdir in VARIANTS.items():
    base = f"{{RESULTS_DIR}}/{run_dirname}/{{subdir}}"
    jobs = pd.read_csv(f"{{base}}/infos_on_jobs.csv", index_col=0)
    jobs["flow_time"] = jobs["finishing_time"] - jobs["arriving_time"]
    jobs_by_variant[variant_label] = jobs
    energy = pd.read_csv(f"{{base}}/infos_on_transfers_energy.csv", index_col=0)
    nb_replans, total_wall, per_replan = run_summary_stats(f"{{base}}/solver_stdout.log")
    sched_by_variant[variant_label] = {{
        "nb_replans": nb_replans, "total_wall_time_s": total_wall,
        "per_replan_times": per_replan, "energy_total": energy["total_energy"].sum(),
    }}

rows = []
for variant_label in VARIANTS:
    jobs = jobs_by_variant[variant_label]
    sched = sched_by_variant[variant_label]
    ft = jobs["flow_time"].values.astype(float)
    st = np.array(sched["per_replan_times"]) if sched["per_replan_times"] else np.array([])
    rows.append({{
        "variant": variant_label,
        "flow_time_mean": ft.mean(), "flow_time_std": ft.std(ddof=0),
        "flow_time_max": ft.max(), "flow_time_min": ft.min(),
        "sched_time_mean_s": st.mean() if len(st) else np.nan,
        "sched_time_std_s": st.std(ddof=0) if len(st) else np.nan,
        "sched_time_max_s": st.max() if len(st) else np.nan,
        "sched_time_min_s": st.min() if len(st) else np.nan,
        "data_volume_total_MB": jobs["dataset size"].sum(),
        "energy_total": sched["energy_total"],
        "total_wall_time_s": sched["total_wall_time_s"],
    }})
pd.DataFrame(rows).set_index("variant")
""")
    code(f"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

ax = axes[0]
for a, d in jobs_by_variant.items():
    d_sorted = d.sort_values("job_id")
    ax.plot(d_sorted["job_id"], d_sorted["flow_time"], marker="o", markersize=4, linewidth=2,
            color=VARIANT_COLORS[a], label=a)
ax.set_xlabel("job_id (ordre d'arrivee)")
ax.set_ylabel("Flow time (s)")
ax.set_title("{inst_label} -- flow time par job")
ax.legend(frameon=False)

ax = axes[1]
for a, d in jobs_by_variant.items():
    plot_cdf(ax, d["flow_time"].values, VARIANT_COLORS[a], a)
ax.set_xlabel("Flow time (s)")
ax.set_ylabel("Proportion cumulee")
ax.set_ylim(0, 1.02)
ax.set_title("{inst_label} -- CDF du flow time")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
""")

md("""### Synthèse mean vs median

Une ligne par (instance × variante), pour comparer d'un coup d'œil.""")

code("""
split_summary_rows = []
for inst_label, run_dirname in split_comparison_runs.items():
    for variant_label, subdir in VARIANTS.items():
        base = f"{RESULTS_DIR}/{run_dirname}/{subdir}"
        jobs = pd.read_csv(f"{base}/infos_on_jobs.csv", index_col=0)
        jobs["flow_time"] = jobs["finishing_time"] - jobs["arriving_time"]
        energy = pd.read_csv(f"{base}/infos_on_transfers_energy.csv", index_col=0)["total_energy"].sum()
        nb_replans, total_wall, per_replan = run_summary_stats(f"{base}/solver_stdout.log")
        st = np.array(per_replan) if per_replan else np.array([])
        split_summary_rows.append({
            "instance": inst_label, "variant": variant_label,
            "flow_time_mean": jobs["flow_time"].mean(), "flow_time_max": jobs["flow_time"].max(),
            "sched_time_mean_s": st.mean() if len(st) else np.nan,
            "sched_time_max_s": st.max() if len(st) else np.nan,
            "energy_total": energy, "total_wall_time_s": total_wall,
        })
split_summary = pd.DataFrame(split_summary_rows).set_index(["instance", "variant"])
split_summary
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("results-grid5000/analysis/per_run_stats_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
