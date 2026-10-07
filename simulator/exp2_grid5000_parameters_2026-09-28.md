# Exp2 — Paramètres de lancement Grid5000 (2026-09-28)

Comparaison live 20 jobs / 50 nœuds (`inst-20J-50N`) entre `incremental`, `online_biobj` et `hybrid`, via `submit_online_vs_biobj_workload_grid5000.sh` (paramètres personnalisés, script bypassé pour permettre les combinaisons ci-dessous).

## Contexte

Cette comparaison n'avait jamais abouti avant cette session : toutes les tâches finissaient bloquées au statut `Scheduled` indéfiniment. Trois bugs corrigés et validés localement (voir `master_node_with_heterogeneous_nodes_csp.py`, commentaires "Bug found 2026-09-28") avant ce lancement :

1. **Libération prématurée après transfert** — la libération de stockage se déclenchait dès qu'une tâche semblait "bloquée trop longtemps" (état normal en attente de transfert), effaçant les données jusqu'à 0.1s après leur arrivée. Déplacée pour ne s'exécuter qu'une fois la tâche réellement `Finished`.
2. **Mauvaise tâche dispatchée** — le dispatch prenait la première tâche `NotStarted` de la liste au lieu de respecter l'assignation exacte (`task_k`) décidée par le solveur CSP pour ce nœud.
3. **Suppressions programmées par le CSP non protégées** — `self.deletions` s'exécutait sans vérifier qu'une tâche encore en file sur ce nœud n'en avait pas besoin ; le CSP planifie sur un calendrier idéal, mais chaque replan (subprocess Java bloquant) retarde l'exécution réelle de 100+ unités de temps.

## Paramètres communs (3 approches)

| Paramètre | Valeur |
|---|---|
| Instance | `inst-20J-50N` (20 jobs, 50 nœuds) |
| `lambda_rate` | 100 |
| `seed` | 42 |
| `solver_time_limit` (base) | 600s |
| Walltime OAR | 04:00:00 par job |

## online_biobj

| Paramètre | Valeur | Détail |
|---|---|---|
| `epsilon_phase1_fraction` | 0.833333 | 500s phase 1 (flow time) / 100s phase 2 (énergie) |
| `epsilon_fraction` | 0.1 | Phase 2 minimise l'énergie avec une dégradation du flow time plafonnée à 10% du résultat de la phase 1 |

## hybrid

| Paramètre | Valeur | Détail |
|---|---|---|
| `epsilon_phase1_fraction` | 0.833333 | Même split 500s/100s que online_biobj |
| `epsilon_fraction` | 0.1 | Identique à online_biobj |
| `adaptive_alpha` | 0.5 | Budget d'escalade = 50% de F1 (probe Incremental) |
| `adaptive_max_budget` | 1200s (20 min) | Plafond du budget d'escalade |
| `adaptive_f1_threshold` | 30s | = `hybrid_incremental_time_limit` (budget du probe Incremental interne) ; en dessous, pas d'escalade |
| `hybrid_incremental_time_limit` | 30s | Budget du probe Incremental interne (F1 + fallback) |
| `freeze_large_jobs_threshold` | *(retiré)* | Critère désactivé |
| `freeze_remaining_time_threshold` | *(retiré)* | Critère désactivé |
| `freeze_jobs_with_ongoing_transfer` | actif | Seul critère de pré-traitement conservé |

## incremental

Aucun paramètre additionnel — uniquement les paramètres communs.

## Historique de cette décision

- Lancement initial avec les valeurs par défaut du script (`epsilon_phase1_fraction=0.5`, `adaptive_alpha=0.2`, `adaptive_max_budget=300s`, freeze avec les 2 seuils actifs) — tué avant complétion pour ajustement.
- 1er ajustement : split 500s/100s, `adaptive_alpha=0.5`, `adaptive_max_budget=600s`, `adaptive_f1_threshold=600s`, freeze réduit à `freeze_jobs_with_ongoing_transfer` seul — lancé (jobs OAR 4166295/96/97), puis arrêté avant complétion pour revalider les paramètres.
- 2e ajustement (final, ci-dessus) : `adaptive_max_budget` remonté à 1200s (20 min), `adaptive_f1_threshold` aligné sur `hybrid_incremental_time_limit` (30s) plutôt que sur `adaptive_max_budget`.

## Lancement final (17:26) — online_biobj bloqué indéfiniment

Soumis 2026-09-28 17:26 :

| Approche | OAR Job ID |
|---|---|
| incremental | 4166358 |
| online_biobj | 4166359 |
| hybrid | 4166360 |

`incremental` a terminé proprement (20/20 jobs). `hybrid` progressait sainement (job 15/20 avant coupure). `online_biobj` ne progressait plus après le job 10 — investigation en local (`simulator-3way-debug`) a révélé que `charge_thinking_time` (facturer le temps réel du solve CSP contre l'horloge simulée) provoque, pour `online_biobj` spécifiquement, un empilement de jobs arrivés pendant un solve bloquant de ~600s, que `self.waiting_jobs.clear()` efface ensuite sans qu'ils aient été traités — combiné à un crash `IndexError` distinct (task_k hors bornes) découvert au passage. Décision : désactiver `charge_thinking_time` via `--no-charge-thinking-time` (flag déjà existant) pour toutes les approches, en plus de deux nouveaux correctifs défensifs (garde-fou `k` hors bornes, clamp de `nodesFreeTime` à 0 — supprime aussi le warning "negative free time").

## Relancé avec `--no-charge-thinking-time` (22:48)

| Approche | OAR Job ID |
|---|---|
| incremental | 4167381 |
| online_biobj | 4167382 |
| hybrid | 4167383 |

Validé en local avant relance : sans `charge_thinking_time`, le dispatch d'un transfert planifié à `t_start=103.10` se fait maintenant à `now=103.20` (quasi immédiat) au lieu de `now=702.90` (600 unités de retard) observé avec `charge_thinking_time` actif.

Résultats sous `simulator/results-grid5000/online_vs_biobj_workload_inst-20J-50N/<approche>/` sur rennes.g5k.

## Exp1 relancé avec la même config (23:57)

Même protocole "state A" figé (`xp_dataset_size_sweep.py`, `charge_thinking_time=False` déjà en dur — le bug d'Exp2 ne s'applique pas ici), mêmes ratios/valeurs qu'Exp2 :

| Paramètre | Valeur |
|---|---|
| incremental/online_biobj budget | 600s (au lieu de 180s par défaut du script) |
| epsilon_fraction | 0.1 |
| epsilon_phase1_fraction | 0.833333 (500s/100s) |
| hybrid_incremental_time_limit | 30s |
| hybrid_alpha | 0.5 |
| hybrid_max_budget | 1200s (plafond) |
| state_a_time_limit | 30s (inchangé, pas d'équivalent Exp2) |

| Sweep | OAR Job ID | Walltime |
|---|---|---|
| Taille dataset (small/medium/large, 5 rép., n_existing=10) | 4167512 | 12h |
| Charge infra n_existing=5 | ~~4167513~~ → 4167591 | 5h |
| Charge infra n_existing=10 | ~~4167514~~ → 4167592 | 5h |
| Charge infra n_existing=20 | ~~4167515~~ → 4167593 | 5h |

**Correctif appliqué (00:19)** : le sweep charge infra utilisait `inst-20J-50N` (défaut du script), qui n'a que 20 jobs (indices 0-19) — `n_existing=20` a besoin d'un "nouveau job" à l'index 20, donc ça crash (`IndexError`, déjà rencontré et corrigé par le passé). Relancé les 3 valeurs (5/10/20) sur `inst-50J-50N` pour rester cohérent, comme établi précédemment.

Résultats sous `simulator/results-grid5000/dataset_size_sweep_3way_10j-50n_2026-09-28/results.json` et `simulator/results-grid5000/nexisting_sweep_3way/n{5,10,20}/results.json`.

## Bilan — 2026-09-29 02:10

**Exp2 (4167381/82/83) : les 3 approches ont terminé avec succès**, `STORAGE CHECK: OK` pour les 3, dans le même run — première fois que cette comparaison live 20j/50n va au bout intégralement.

| Approche | Wall time total | Statut |
|---|---|---|
| hybrid | 2141.8s | ✅ DONE (00:22) |
| online_biobj | 2684.0s | ✅ DONE (02:00) — première complétion jamais obtenue |
| incremental | 2205.4s | ✅ DONE (02:09) |

**Sweep infra n_existing (4167591/92/93)** : 3/3 terminés (relancés sur `inst-50J-50N` après le crash initial sur `inst-20J-50N`).

**Sweep taille dataset (4167512)** : ✅ terminé (07:12, `results.json` + tous les CSV de détail écrits, zéro erreur, 3 tiers × 5 répétitions × 3 approches).

## Clôture — 2026-09-29 07:12

**Les 7 jobs Grid5000 relancés cette session ont tous terminé avec succès, aucun crash.** Fin de l'investigation.

## Relance lambda_rate=300 — 2026-09-29 10:09

Résultats `lambda_rate=100` archivés (Grid5000 et local) sous `online_vs_biobj_workload_inst-20J-50N_lambda100_2026-09-29/`. Notebook d'analyse (`results-grid5000/analysis/flow_energy_volume_analysis.ipynb`) généré sur ces résultats archivés — pointe vers l'ancien chemin, à régénérer si besoin de comparer avec la nouvelle config.

Changements : `lambda_rate` 100 → **300**, `solver_time_limit` d'`incremental` 600s → **60s** (les autres approches inchangées — `online_biobj` reste à 600s, `hybrid` inchangé).

| Approche | OAR Job ID | solver_time_limit |
|---|---|---|
| incremental | 4169255 | 60s |
| online_biobj | 4169256 | 600s |
| hybrid | 4169257 | 600s |

`incremental` (4169255, DONE 10:30), `hybrid` (4169257, DONE ~11:19) et `online_biobj` (4169256, DONE 13:27, wall time 4189.6) ont tous terminé avec succès — **les 3 approches complètes** sur ce run.

## Clôture — 2026-09-29 18:14

**Les 9 jobs sur les 3 instances (10J-20N, 20J-50N, 50J-50N × incremental/online_biobj/hybrid) ont tous terminé avec succès, `STORAGE CHECK: OK`, aucun crash.** `hybrid` avec les 4 solveurs en parallèle (`--parallel-warm-cold-escalation`) confirmé fonctionnel sur les 3 instances.

## Lancement inst-10J-20N + inst-20J-50N, lambda_rate=200 — 2026-09-29 14:34

Même config que le run 50J-50N (300s incremental/online_biobj, hybrid `adaptive_max_budget=480s` + `--parallel-warm-cold-escalation`). Run lambda_rate=300 sur inst-20J-50N archivé sous `online_vs_biobj_workload_inst-20J-50N_lambda300_2026-09-29/`.

| Instance | Approche | OAR Job ID |
|---|---|---|
| inst-10J-20N | incremental | 4169985 |
| inst-10J-20N | online_biobj | 4169986 |
| inst-10J-20N | hybrid | 4169987 |
| inst-20J-50N | incremental | 4169988 |
| inst-20J-50N | online_biobj | 4169989 |
| inst-20J-50N | hybrid | 4169990 |

Stockage vérifié avant lancement : `inst-10J-20N` (20 nœuds, 1451-40946 MB) et `inst-20J-50N` (50 nœuds, 2050-102400 MB) ont bien des capacités de stockage finies définies par nœud.

## Relance sur inst-50J-50N, lambda_rate=200 — 2026-09-29 10:57

Même config que ci-dessus (`incremental` 60s/solve, `online_biobj`/`hybrid` 600s de base, mêmes ratios epsilon/hybrid, `--no-charge-thinking-time`), mais sur l'instance `inst-50J-50N` (50 jobs, 50 nœuds) avec `lambda_rate=200`.

| Approche | OAR Job ID | Walltime |
|---|---|---|
| incremental | ~~4169385~~ | 2h |
| online_biobj | ~~4169386~~ | 10h |
| hybrid | ~~4169387~~ | 8h |

**Annulé (11:20)** : 10h de walltime pour `online_biobj` jugé trop long. Relancé avec un budget uniforme de 5 min (300s) pour les 3 approches (`incremental`/`online_biobj` : `solver_time_limit=300s` ; `hybrid` : `adaptive_max_budget=300s`, `hybrid_incremental_time_limit=30s` inchangé).

| Approche | OAR Job ID | Détail |
|---|---|---|
| incremental | 4169456 | `solver_time_limit=300s` |
| online_biobj | 4169457 | `solver_time_limit=300s` |
| hybrid | ~~4169458~~ → ~~4169509~~ → 4169967 | `adaptive_max_budget=480s`, **`--parallel-warm-cold-escalation`** activé (4169509 arrêté à job40/50 pour repartir avec les 4 solveurs concurrents) |

**Découverte annexe** : chaque solve CSP consomme systématiquement ~600s (quasi pile `solver_time_limit`) même quand la solution optimale-candidate est trouvée en <1s — le solveur ne s'arrête jamais tôt, il tente (en vain) de prouver l'optimalité jusqu'à épuiser le budget. Explique pourquoi `incremental`/`online_biobj` (budget 600s/solve, ~20 solves) prennent bien plus longtemps en réel que `hybrid` (budget interne 30s/solve pour la majorité des jobs).
