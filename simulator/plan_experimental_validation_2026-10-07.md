# Plan de validation expérimentale — Incremental vs Online_newjob vs Hybrid-n_j

Objectif : sortir des essais à une graine (toutes les conclusions de cette semaine n'ont qu'une
graine par configuration) et produire une vraie comparaison statistique des 3 approches, sur deux
axes de variation séparés.

Toutes les expériences utilisent le pipeline corrigé du 2026-10-06 (dispatch SimPy live +
injection correcte de `master.deletions`), `nb_nodes=50` fixe partout.

## Protocole — validation méthodologique (2026-10-07)

Deux questions de fond vérifiées directement dans le code avant de lancer quoi que ce soit :

**1. Les 3 approches voient-elles exactement les mêmes conditions ?** Oui, confirmé. Chaque
approche reçoit `m = copy.copy(master)` (copie superficielle de l'état A gelé). Tracé
intégralement le chemin d'exécution de `run_incremental_style`, `run_online_newjob_style`,
`run_hybrid_style` et leurs fonctions internes (`_placeSingleJobIncremental`,
`_currentCommittedBatchFlowTimes`, `_timedParallelEscalation`) : **aucune ne mute en place**
`replicas_locations`, `transfers`, `works` ou `deletions` — elles ne font que les lire. Les
fonctions qui FERAIENT ces mutations (`_commitSingleJobPlan`, `_commitJointPlan`, utilisées par
la vraie boucle de production) ne sont **jamais appelées** par ce harnais de test. Donc pas de
contamination possible entre approches selon l'ordre d'exécution — chacune voit le même état A
intact.

**2. Est-ce que tout tourne "sur simulateur" (SimPy) ?** Partiellement, à nuancer :
- **État A** (les `n_existing` jobs déjà en place) : oui, dispatché via de vrais processus SimPy
  (`scheduling()` + `checkOnJobs()` + `processTasks()` par nœud, puis `env.run(until=freeze_at)`)
  depuis le correctif du §1 de `protocole_changements_2026-10-06.md`.
- **Décision de placement du nouveau job** (ce qu'on compare) : **non** — chaque approche appelle
  le solveur CSP (`schedulingUsingJavaCSP`) **directement, en un seul coup**, sans nouvelle
  dispatche SimPy pour ce job précis. `flow_time_new_job` est donc le plan **proposé** par le
  solveur, lu analytiquement sur sa sortie — pas le résultat d'une ré-exécution pas à pas qui
  vérifierait la contention réseau/stockage au fil du temps pour ce job spécifique.

**Limite méthodologique identifiée, pas encore comblée** : pour une validation complète
bout-en-bout, il faudrait un pas supplémentaire qui dispatche réellement le plan gagnant de chaque
approche via SimPy (nouveaux processus `env.process(...)` + `env.run(...)` après le point de gel),
pour vérifier qu'il s'exécute bien tel que prévu par le solveur. Pas fait à ce jour — toutes les
métriques de ce document viennent du plan proposé, pas d'une ré-exécution vérifiée.

## Catégories de taille (3 paliers, repris de `hybrid_mechanism_summary_2026-10-05.md` §10)

| palier | task_duration (s) | dataset_size (MB) |
| --- | --- | --- |
| **petit** | [100, 150] | [2048, 20480] |
| **grand** | [250, 300] | [102400, 204800] |
| **mixte** | [100, 300] | [2048, 204800] |

## Métriques collectées (par run, par approche)

- `flow_time_new_job` — métrique principale
- `scheduling_time_s` — temps de scheduling (mur)
- `transfer_energy_total` — énergie de transfert
- `mean_flow_time_all` / `max_flow_time_all` — impact sur l'ensemble des jobs (Incremental ne
  dégrade jamais les existants par construction, Online_newjob/Hybrid le peuvent)
- Pour Hybrid uniquement : `accepted` (bool), variante gagnante, `hybrid_f1`

## Agrégation par cellule (combinaison de paramètres)

Pour chaque cellule, sur N graines :
- moyenne/médiane ± écart-type de `flow_time_new_job` par approche
- taux de victoire (% de graines où chaque approche a le meilleur flow time)
- moyenne de l'énergie et du temps de scheduling
- pour Hybrid : taux d'acceptation de l'escalade, distribution des variantes gagnantes

---

## Phase A — 3 paliers à échelle x2 fixe, `n_existing=5` fixe

Simplifié (2026-10-07) : juste **3 sessions**, une par palier, toutes à l'échelle **x2**, une
seule graine chacune — pas de balayage d'échelle pour l'instant. Le stockage des nœuds
(`storage_ceiling`) suit automatiquement l'échelle du dataset_size, bande passante et vitesse de
calcul des nœuds restent fixes.

| palier | task_duration x2 (s) | dataset_size x2 (MB) |
| --- | --- | --- |
| petit | [200, 300] | [4096, 40960] |
| grand | [500, 600] | [204800, 409600] |
| mixte | [200, 600] | [4096, 409600] |

Chaque session : 3 approches (Incremental / Online_newjob / Hybrid-n_j) sur la même instance.

### Résultats Phase A (2026-10-07, une graine par palier : 200/201/202)

| palier | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? | variante gagnante |
| --- | --- | --- | --- | --- | --- |
| petit | 2106.0 | 2107.0 | 2106.0 | Non | `freeze_below_mean` (2107, pas mieux) |
| grand | 10799.0 | 10530.0 | **9482.0** | **Oui** | `hint_from_f1` |
| mixte | 6816.0 | 6817.0 | **6379.0** | **Oui** | `freeze_above_mean` |

Gain Hybrid vs Incremental : petit = 0% (quasi-identiques) ; grand = **+12.2%** ; mixte = **+6.4%**.

Impact sur les jobs existants (`mean_flow_time_all`) : Incremental reste toujours le plus doux
pour les existants (il ne les replanifie jamais) — grand : 8392 (Inc) vs 8919 (Hyb) ; mixte : 4536
(Inc) vs 4843 (Hyb). Coût en énergie côté Hybrid quand il gagne : +62% sur grand (21141 vs 13009),
+6.6% sur mixte (7885 vs 7399).

Fait notable : `hint_from_f1` a gagné sur "grand" sans planter (le crash `_writeWarmStart` du
§6.2 de `protocole_changements_2026-10-06.md` reste donc instance-dépendant, pas systématique).

### Extension Phase A — palier mixte seul, balayage x0.5/x2/x4/x6 (2026-10-07, seeds 300/202/301/302)

| échelle | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? | variante gagnante |
| --- | --- | --- | --- | --- | --- |
| x0.5 | 1351.0 | 1131.0 | **1131.0** | Oui | `freeze_below_mean` |
| x2 | 6816.0 | 6817.0 | **6379.0** | Oui | `freeze_above_mean` |
| x4 | 21414.0 | 22660.0 | **21285.0** | Oui | `freeze_above_mean` |
| x6 | 15812.0 | 16192.0 | **15581.0** | Oui | `freeze_below_mean` |

Gain Hybrid vs Incremental : x0.5 = +16.3% ; x2 = +6.4% ; x4 = +0.6% ; x6 = +1.5%.

**Fait important** : à x4 et x6, **Online_newjob est pire qu'Incremental** (22660 vs 21414, et
16192 vs 15812) — le filet de sécurité d'Hybrid (sonde Incremental comme F1, n'accepte l'escalade
que si elle fait mieux) le protège : Hybrid gagne ou égale Incremental sur les 4 échelles, jamais
pire, alors qu'Online_newjob seul perd deux fois sur quatre.

> Analyse globale (synthèse Phase A complète + Phase B) à faire plus tard, une fois toutes les
> données collectées.

## Phase B — Balayage `n_existing`, palier mixte fixe, échelle x2

Mis à jour (2026-10-07) : échelle **x2** (au lieu de x1 initialement prévu), pour rester cohérent
avec l'extension de la Phase A. Palier **mixte**, `n_existing` variable :

| n_existing | 2 | 5 | 10 | 15 |
| --- | --- | --- | --- | --- |
|  | ○ | ○ (déjà fait en Phase A, seed=202 : Inc=6816.0/ONJ=6817.0/Hyb=6379.0) | ○ | ○ |

> Note : le message original disait "0,5,10,15" — j'ai lu "0" comme probablement "2" (cohérent
> avec tous les balayages n_existing précédents de la session : 2/5/8/10/12/15). Si c'est vraiment
> n_existing=0 (aucun job existant, le nouveau job seul face à une infra vide) qui est voulu,
> dis-le et je l'ajoute à la place de — ou en plus de — 2.

Chaque cellule : N graines × 3 approches.

### Résultats Phase B (2026-10-07, une graine par n_existing : 400/202/401/402)

| n_existing | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? | variante gagnante |
| --- | --- | --- | --- | --- | --- |
| 2 | 4730.0 | **4633.0** | 4730.0 | Non | `freeze_below_mean` (4731, pas mieux) |
| 5 | 6816.0 | 6817.0 | **6379.0** | **Oui** | `freeze_above_mean` |
| 10 | 6099.0 | **5967.0** | 6099.0 | Non | `nofreeze` (6331, pas mieux) |
| 15 | 8050.0 | 7261.0 | **6953.0** | **Oui** | `hint_slow_to_fast` |

Gain Hybrid vs Incremental : n=2 = 0% ; n=5 = +6.4% ; n=10 = 0% ; n=15 = **+13.6%**.
Gain Online_newjob vs Incremental : n=2 = +2.1% ; n=5 ≈ 0% ; n=10 = +2.2% ; n=15 = **+9.8%**.

Tendance qui se dessine : plus `n_existing` augmente, plus il y a de marge de replanification pour
Online_newjob/Hybrid (meilleur gain des deux à n=15), alors qu'à faible/moyenne charge (n=2, n=10
ici) l'écart est faible ou nul — cohérent avec l'observation du §10 de
`hybrid_mechanism_summary_2026-10-05.md` (plus d'hétérogénéité/marge = plus de chances
d'accepter). Hybrid ne perd jamais contre Incremental sur les 4 valeurs (égalité au pire, à chaque
rejet), contrairement à Online_newjob seul qui pourrait en principe perdre (pas observé ici mais
vu ailleurs en Phase A à x4/x6).

---

## Paramètres encore à fixer avant de lancer

1. **Nombre de graines par cellule (N)** — proposé : 5 (compromis rapidité/significativité). Total
   runs : Phase A = 9×N, Phase B = 4×N (N=5 → 45 + 20 = 65 scénarios, chacun ~3 solves × ~30-300s).
2. **Local ou Grid5000** — un pilote local sur une grille réduite (ex. 2 cellules de chaque phase,
   3 graines) pour valider le pipeline de mesure avant la vraie campagne, ou direct sur Grid5000 ?
3. **Budgets solveur** par run : repris des sweeps précédents (état A=30s, Incremental=défaut,
   Online_newjob=30s, Hybrid F1=15s + budget d'escalade=30s) — sachant que le §7.1 du
   `protocole_changements_2026-10-06.md` a montré qu'un budget d'escalade plus long (300s) peut
   changer l'acceptation à grande échelle. Garde-t-on 30s partout pour rester comparable entre
   cellules, ou adapte-t-on le budget à l'échelle (x4 avec plus de temps) ?

---

## Phase C — 5 scénarios totalement aléatoires (2026-10-07)

Au lieu de faire varier un seul axe à la fois (Phases A/B), chaque scénario tire **tous les
paramètres structurels** indépendamment, pour couvrir des combinaisons que les phases A/B
n'atteignent jamais ensemble :

- `nb_nodes` : aléatoire dans [10, 30] (réduit depuis [20, 80] initial, jugé trop grand)
- `n_existing` : aléatoire dans [2, 15]
- `task_duration_range` (jobs existants) : borne basse aléatoire dans [50, 300], largeur
  aléatoire ajoutée dessus
- `dataset_size_range` (jobs existants) : borne basse aléatoire dans [1024, 50000], largeur
  aléatoire ajoutée dessus
- `storage_ceiling` : dérivé de `max(dataset_size_range[1], new_job_dataset_size) × 1.3` (doit
  couvrir le nouveau job même s'il dépasse la plage des existants, cf. point suivant)

**Contrainte explicite demandée : le nouveau job ne doit jamais être petit.** Tiré normalement
(`draw_new_job_sizes`, mode `upper-quarter`, donc déjà biaisé vers le haut de la plage du
scénario), puis plancher forcé après coup : `dataset_size >= 51200` MB et `task_duration >= 200`
s, quel que soit ce que la plage aléatoire du scénario aurait donné sinon.

5 scénarios, chacun comparé sur les 3 approches (Incremental / Online_newjob / Hybrid-n_j), une
seule graine chacun pour l'instant.

### Résultats Phase C — première passe (nb_nodes∈[20,80], seeds 500-504, abandonnée)

| scénario | nb_nodes | n_existing | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 71 | 9 | **4196.0** | 4619.0 (pire) | 4196.0 | Non |
| 1 | 60 | 6 | **2827.0** | 3061.0 (pire) | 2827.0 | Non |
| 2 | 52 | 7 | **4843.0** | 5545.0 (pire) | 4843.0 | Non |
| 3 | 30 | 4 | 3485.0 | 3178.0 | **3178.0** | **Oui** |
| 4 | 49 | 15 | 6878.0 | 6024.0 | **6024.0** | **Oui** |

Jugé trop de nœuds (20-80) — relancé avec nb_nodes∈[10,30] ci-dessous.

### Résultats Phase C — nb_nodes réduit à [10,30] (seeds 500-504)

| scénario | nb_nodes | n_existing | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? | variante gagnante |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24 | 10 | 9641.0 | **7081.0** | 7987.0 | **Oui** | `greedy_hints` |
| 1 | 30 | 6 | 4206.0 | 4204.0 | **4072.0** | **Oui** | `freeze_below_mean` |
| 2 | 26 | 7 | 7640.0 | 7931.0 (pire) | **7442.0** | **Oui** | `greedy_hints` |
| 3 | 15 | 4 | 4184.0 | 4009.0 | **4009.0** | **Oui** | `nofreeze` |
| 4 | 24 | 15 | 9773.0 | 8918.0 | **8998.0** | **Oui** | `hint_slow_to_fast` |

Gain Hybrid vs Incremental : +17.2% / +3.2% / +2.6% / +4.2% / +7.9% — **accepté sur les 5
scénarios** cette fois (contre 2/5 avec plus de nœuds). Moins de nœuds = plus de contention = plus
d'opportunités de replanification utile pour Hybrid/Online_newjob.

**Observation importante (scénario 0)** : Hybrid (7987.0) est *moins bon* qu'Online_newjob seul
(7081.0), alors qu'Hybrid "accepte" (bat son F1=9641.0). Ce n'est pas un bug : le filet de sécurité
d'Hybrid garantit seulement **"≥ Incremental"**, jamais **"≥ meilleure approche parmi les 3"** —
il ne connaît pas le résultat d'Online_newjob et ne l'inclut pas comme candidat. Cause probable :
le budget d'escalade (30s) est **partagé entre les ~11 variantes concurrentes** d'Hybrid (dont
`nofreeze`, conceptuellement équivalent à Online_newjob), donc chacune dispose de moins de calcul
réel que les 30s pleins qu'Online_newjob a pour lui seul (même motif déjà vu en Phase A x2 :
`nofreeze`=6530 vs Online_newjob dédié=6294).

**Idée en attente de décision** : ajouter Online_newjob comme candidat dans l'escalade d'Hybrid.
Deux options discutées :
- **A (rapide, test seulement)** : dans le harnais de test, prendre le meilleur de {F1, escalade
  Hybrid, Online_newjob dédié} comme résultat "Hybrid amélioré" — aucun changement au mécanisme
  réel.
- **B (changement réel)** : ajouter Online_newjob dans `master_node_with_heterogeneous_nodes_csp.py`
  comme candidat avec son **propre budget dédié**, séparé du pool des ~11 variantes concurrentes
  (sinon il subirait le même partage de ressources) — impacterait aussi les futurs runs Grid5000.
Pas encore tranché.
