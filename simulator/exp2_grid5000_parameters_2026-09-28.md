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

## Lancement final

Soumis 2026-09-28 17:26 :

| Approche | OAR Job ID |
|---|---|
| incremental | 4166358 |
| online_biobj | 4166359 |
| hybrid | 4166360 |

Résultats sous `simulator/results-grid5000/online_vs_biobj_workload_inst-20J-50N/<approche>/` sur rennes.g5k.
