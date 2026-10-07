# Changements au protocole expérimental — session du 2026-10-06

## 0. Recentrage

À partir d'aujourd'hui on ne compare plus que **3 approches** : `incremental`, `online_newjob`
(mono-objectif, `objective_choice=2` + plafond de dégradation, un seul solve, pas d'escalade) et
`hybrid-n_j` (sonde Incremental pour F1 puis escalade conditionnelle à N variantes concurrentes,
mono-objectif lui aussi via `adaptive_bi_objective=False`). `online` classique (objectif batch-wide
`max`/`sum`) n'est plus dans la boucle de comparaison — il ciblait le mauvais objectif pour cette
question.

## 1. Correctif majeur : `build_state_a` fait maintenant un vrai dispatch SimPy live

**Avant** : l'état A était construit par un seul solve CSP joint, puis le plan était **réfléchi
manuellement** sur les objets `Task`/`ComputeNode` (statuts et occupations recopiés depuis les
horaires idéalisés du solveur) — aucune simulation SimPy réelle n'avait lieu. Le docstring du
module le disait explicitement : *"a genuine one-shot offline placement, not a replay of
N sequential live-simulation arrivals"*.

**Conséquence découverte en cours de session** : `master.replicas_locations` (quel nœud détient
les données de quel job) et `job.replicas` (utilisé par `_estimateJobFlowTime`) ne sont renseignés
que par `startTransfer`/`transferData`, atteignables uniquement depuis un process `scheduling()`
réellement actif. Comme ce process n'était jamais lancé, **ces deux structures étaient vides tout
le long de la session**, et ce silencieusement :
- toutes les variantes d'escalade basées sur `_estimateJobFlowTime` (mean-split, `hint_slow_to_fast`)
  ne faisaient rien d'utile ;
- `ghost_storage.txt` (qui signale au CSP l'espace déjà occupé par des jobs hors-batch) était
  toujours vide, donc **jamais pris en compte** — le solveur voyait des nœuds libres qui ne
  l'étaient pas en réalité.

**Correctif** : `build_state_a` (`exps/xp_dataset_size_sweep.py`) injecte maintenant le plan dans
`master.transfers`/`master.works`, enregistre les **vrais** processes SimPy (`scheduling()`,
`checkOnJobs()`, `processTasks()` par nœud — exactement ceux que `simulator.py` lance en run live),
puis fait tourner `env.run(until=freeze_at)`. `not_finished_jobs` vient maintenant de
`job.status != "Finished"` réel, plus seulement du dict idéalisé `_state_a_finish` (gardé à part,
pour le reporting/plafond de dégradation uniquement).

**Effet de bord attendu** : le dispatch réel ne peut être que plus lent que le plan idéalisé
(contention de bande passante, granularité de polling à 0.1), donc plus de jobs peuvent apparaître
"non terminés" qu'avec l'ancienne version. **Les résultats d'avant ce correctif et d'après ne sont
pas directement comparables.**

## 2. Nouvelles variantes d'escalade Hybrid (11 au total désormais)

Ajoutées cette session : `freeze_random_half` (gèle aléatoirement 50% des jobs), `greedy_hints`
(réactivation du `hints()` pré-existant, dont le tri par taille était calculé mais jamais appliqué
— bug corrigé), `hint_from_f1` (réutilise la sortie de `_writeWarmStart` comme simple
`solver.addHint()` au lieu d'un seeding dur), `avoid_empty_slow_nodes` (exclut de la candidature
tout job les nœuds sans donnée résidente ET dans la moitié la plus lente en bande passante),
`hint_slow_to_fast` (associe le job à l'estimation de flow time la plus mauvaise au nœud le plus
rapide, via `_estimateJobFlowTime` — ne produit des hints utiles que depuis le correctif du §1).

## 3. Toggles de stratégie de recherche sur Incremental — investigation du "stuck" pattern

Observé à plusieurs reprises : Incremental converge vite (<1s) vers une solution puis n'améliore
plus rien pour le reste d'un budget pourtant long (ex. 300s=1800s, résultat identique). Deux
hypothèses testées via des toggles ajoutés à `MainIncremental.java` :
- `--sort-by-bandwidth` : ordre d'essai des nœuds du sélecteur glouton trié par bande passante
  plutôt que capacité de calcul — **aucun changement**.
- `--disable-lns` : désactive `setLNS(...)`, recherche arbre simple — **aucun changement**, testé
  aussi en isolant Online vs Incremental (4 combinaisons LNS on/off × Online/Incremental →
  résultat identique dans les 4 cas).

**Conclusion** : sur le cas investigué (seed=46, n_existing=12), ce n'était **pas** un problème de
qualité de recherche mais une **vraie contrainte de stockage** — seuls 2 des 50 nœuds avaient
réellement de la place une fois `ghost_storage` correctement pris en compte (cf. §1). Confirmé
par un second seed (50) où 8/50 nœuds avaient de la place et où le budget n'avait presque aucun
effet (5s suffisait déjà).

**Un vrai bug de qualité de recherche a été isolé séparément**, sur un scénario non contraint par
le stockage (un seul job sur une infra à 50 nœuds totalement vide) : le solveur choisit 6 nœuds à
bande passante moyenne alors que 10 nœuds complètement libres avec une meilleure bande passante ne
sont jamais essayés — motif identique (convergence rapide puis plateau total). LNS on/off et
Online vs Incremental donnent le même résultat sur ce cas aussi, ce qui pointe vers le
**sélecteur de valeur glouton partagé** (`IntValueSelector`, copié-collé dans les deux classes
Java) comme coupable. **Non corrigé** — reste la piste à suivre si on revient dessus.

## 4. Nouveau pipeline de test local (sans Grid5000)

Scripts qui importent directement les fonctions de production (`build_state_a`,
`run_incremental_style`, `run_online_newjob_style`, `run_hybrid_style`, `make_new_job`,
`generate_existing_jobs`, `draw_new_job_sizes`, `write_instance`) au lieu de passer par le CLI/OAR.
Un scénario complet (génération d'instance + état A sondé + état A réel + 3 approches) tourne en
quelques minutes en local contre 10–30+ minutes d'attente de queue sur Grid5000 — permet
d'itérer rapidement avant de lancer la version longue sur la grille.

## 5. Nouveau protocole de sweep sur `n_existing`

Pour chaque valeur de `n_existing` testée (2, 5, 10 cette session) :
1. Génère une instance fraîche (jobs existants + job entrant, via les mêmes tirages que
   `xp_simultaneous_sweep.py`).
2. **Sonde** l'état A seul (solve rapide) pour obtenir le max flow time des jobs existants.
3. Fixe l'arrivée du nouveau job à `max_flow/2 + 200` (place son arrivée à mi-vie du batch, pas à
   t=0 ni après la fin de tout).
4. **Reconstruit** l'état A réel (avec le vrai dispatch SimPy du §1) jusqu'à ce point d'arrivée.
5. Compare `incremental` / `online_newjob` / `hybrid-n_j` sur le même état A gelé : flow time du
   nouveau job, temps de scheduling (mur), énergie de transfert, et pour hybrid la variante
   d'escalade gagnante.

Premiers résultats (budgets courts, local) :

| n_existing | Incremental | Online_newjob | Hybrid-n_j | Variante gagnante |
| --- | --- | --- | --- | --- |
| 2 | 4269.0 | 2259.0 | 2240.0 | `warm_nofreeze` |
| 5 | 15605.0 | 2889.0 | 2889.0 | `freeze_below_mean` |
| 10 | 6542.0 | 2737.0 | 2737.0 | `freeze_above_mean` |

Hybrid égale ou bat systématiquement Online_newjob sur le flow time du nouveau job, et bat ou
égale sur l'énergie de transfert à chaque fois.

> **⚠️ Invalidé par le bug du §6.1 ci-dessous** — ce tableau (et le test du job injecté sur
> n_existing=12, Incremental=4880/Online=8136/Online_newjob=2399/Hybrid=2399) a tourné avec la
> version de `build_state_a` qui n'injectait pas `master.deletions`. Incremental y est
> artificiellement pénalisé à chaque fois qu'un job existant a des données encore résidentes. **À
> refaire** avec le correctif.

## 6. Bug connu, toujours ouvert

### 6.1 `build_state_a` n'injectait pas les suppressions planifiées de l'état A — CORRIGÉ 2026-10-06

**Symptôme** : sur n_existing=5 (seed=47), Incremental plaçait le nouveau job (15 tâches,
158761 Mo) entièrement sur un seul nœud (15), en série → flow time 15605, **identique** à 30s et
180s de budget (une seule solution trouvée en 0.075s, jamais amélioré). Semblait d'abord être soit
une vraie contrainte de stockage, soit (re)le bug de sélecteur glouton du §3.

**Vérification des paramètres transmis au solveur** (`nodes.json`, `ghost_storage.txt`) : capacités
nominales correctes, mais en soustrayant le stockage fantôme (`replicas_locations`), **un seul nœud
sur 50** avait réellement assez de place (251878 Mo) pour le job — tous les autres nœuds
storage-éligibles étaient déjà occupés par les données résidentes des 5 jobs existants. Jusque-là,
comportement cohérent, pas de bug apparent.

**Mais** (remarque de l'utilisateur) : pour Incremental, les jobs existants finissent leurs tâches
au cours du temps et leurs données sont censées être supprimées une fois non réutilisées — donc du
stockage doit se libérer avant l'horizon complet, pas seulement rester occupé indéfiniment.
Vérification : `build_state_a` injecte bien `master.transfers`/`master.works` depuis le plan de
l'état A avant de lancer le vrai dispatch SimPy, **mais jamais `master.deletions`** — contrairement
à la boucle de production réelle (`schedulingNewJob()`), qui le fait pour chaque job planifié
(`self.deletions[key].append(deletion)`). Résultat : `ghost_storage.txt` ne trouve jamais de
`deletion_time` connu pour aucune donnée résidente, et tombe sur `-1` à chaque fois — que
`MainIncremental.java`/`MainOnline.java` traitent comme *"occupe le nœud jusqu'à la fin de tout
l'horizon"* (`gDeletion < 0 ? makespan : gDeletion`, ligne ~580), au lieu du vrai moment de
libération déjà décidé par le solve de l'état A.

**Correctif** (`exps/xp_dataset_size_sweep.py`, `build_state_a`) : injecte maintenant aussi
`master.deletions[key] = list((deletions_ or {}).get(key, []))`, même boucle que
`transfers`/`works`. Le process `scheduling()` (déjà lancé depuis le §1) consomme cette liste tout
seul — rien d'autre à changer.

**Effet mesuré** (même instance, même budget 30s) : Incremental passe de **15605** (1 nœud) à
**3188** (8 nœuds : 1, 10, 12, 17, 33, 39, 43, 48) — un gain de 79.6%. L'écart avec Online_newjob/
Hybrid (2889) tombe de +440% à **+10%**. Gantt avant/après :
https://claude.ai/artifact/G2znZ3qKdmQCjiSeCQ7qEZ

**À refaire avec ce correctif** (tournées avec la version buguée, donc invalides) :
1. Test du job injecté sur n_existing=12 (seed=50) — 4 approches (Incremental/Online/
   Online_newjob/Hybrid-n_j).
2. Sweep local n_existing=2/5/10 (tableau du §5 ci-dessus) + le Gantt n=5.

### 6.2 Crash `_writeWarmStart`, toujours ouvert

`_writeWarmStart` lance un sous-solve `MainIncremental` jetable (pour le nouveau job seul) qui
plante avec `ArrayIndexOutOfBoundsException` à `ArrayUtils.flatten` — confirmé dans 34 logs
historiques, affecte `warm_nofreeze` et `hint_from_f1` de façon silencieuse (traité comme "pas de
solution"). Fait notable cette session : sur le scénario n_existing=2 du sweep ci-dessus,
`warm_nofreeze` a **gagné** sans planter — semble donc dépendre de l'instance plutôt que d'être
systématique. Cause racine (quel tableau est vide) toujours pas isolée.

## 7. Balayage d'échelle x1/x2/x4 à n_existing=10 fixe (avec le correctif §6.1)

Protocole du §5, mais avec `n_existing=10` fixe et `task_duration_range`/`dataset_size_range`
multipliés par 1/2/4 à chaque fois. Le plafond de tirage du stockage des nœuds
(`storage_ceiling = dataset_size_range[1] × 1.3`) suit automatiquement la même échelle — donc la
capacité de stockage augmente avec les tailles de données. Bande passante (tirée dans [12, 800]
MBps) et vitesse de calcul des nœuds restent fixes à toutes les échelles. Une seule graine par
échelle (même limite méthodologique que le §11 de `hybrid_mechanism_summary_2026-10-05.md` — à
confirmer avec plusieurs graines avant de conclure).

| échelle | Incremental | Online_newjob | Hybrid-n_j | Hybrid accepté ? | variante gagnante |
| --- | --- | --- | --- | --- | --- |
| x1 | **3584.0** | 4486.0 | 3584.0 | Non (fallback F1) | `freeze_random_half` (4126, insuffisant) |
| x2 | 7889.0 | **6294.0** | 6530.0 | **Oui** | `nofreeze` |
| x4 | 17957.0 | 17958.0 | 17957.0 | Non (fallback F1) | `hint_slow_to_fast` (19168, insuffisant) |

| échelle | sched Inc/ONJ/Hyb (s) | énergie Inc/ONJ/Hyb |
| --- | --- | --- |
| x1 | 30.7 / 30.8 / 65.6 | 2740 / 10061 / 2740 |
| x2 | 30.7 / 30.8 / 49.5 | 8704 / 17544 / 16737 |
| x4 | 30.7 / 30.8 / 49.4 | 18060 / 31422 / 18060 |

**Pas de tendance monotone** : à x1 Incremental bat Online_newjob (3584 vs 4486, l'inverse de tous
les résultats précédents de la session) ; à x2 c'est Online_newjob qui gagne, et l'escalade
d'Hybrid est acceptée mais trouve *moins bien* que le solve direct d'Online_newjob (6530 vs 6294 —
plausible : le budget d'escalade de 30s est partagé entre ~11 variantes concurrentes, donc chaque
variante dispose de moins de calcul réel que les 30s dédiés d'Online_newjob) ; à x4 les trois
convergent presque exactement au même point. L'énergie de transfert augmente nettement avec
l'échelle (bande passante fixe + données plus grosses = transferts plus longs) ; Incremental reste
le moins coûteux en énergie quand il gagne (x1, x4), car il ne replanifie rien d'autre.

### 7.1 x4 rejeté à 30s de budget d'escalade — ré-essai à 300s

Sur le scénario x4 ci-dessus (seed=104), rejeu d'Hybrid seul, même instance/état A, en augmentant
uniquement `hybrid_max_budget` (F1 reste à 15s) :

| budget max | flow time job 10 | accepté ? | variante gagnante | énergie |
| --- | --- | --- | --- | --- |
| 30s | 17957.0 (= F1, rejeté) | Non | `hint_slow_to_fast` (19168, insuffisant) | 18060 |
| **300s** | **16514.0** | **Oui** | `freeze_ongoing_transfer` | **40204** (+122%) |

Gain de **+8.0%** une fois accepté à 300s — modeste comparé aux gains vus ailleurs (47-82%), mais
le rejet à 30s n'était donc pas définitif : il manquait simplement de budget pour qu'une variante
qui replanifie plus largement (`freeze_ongoing_transfer`, différente de la gagnante à 30s) ait le
temps de converger. Coût : énergie de transfert plus que doublée (18060 → 40204).

**Piste à creuser** : peut-être jouer sur le **temps de scheduling** dans Hybrid (budget
d'escalade, et/ou répartition du budget entre les variantes concurrentes) plutôt que de le garder
fixe à 30s quelle que soit l'échelle — un budget proportionnel à la taille de l'instance pourrait
éviter ce genre de rejet qui n'est en fait qu'un manque de temps.
