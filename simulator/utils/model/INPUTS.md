# Contrat d'entrée du modèle CSP (Java/Choco)

Chaque appel `schedulingUsingJavaCSP` (Python, `utils/modelCSP.py`) écrit ces fichiers dans `utils/model/inputs/` (ou `$SIMULATOR_RUN_CWD/utils/model/inputs/` quand isolé) avant de lancer le solveur Java. Le Java les lit, résout, et écrit son plan dans `utils/model/outputs/`.

Tous les temps ("start", "end", `timelasped`, `job_arriving_time`) sont dans le référentiel **local** de ce solve : `0` = `env.now` au moment de l'appel. Le Java ne connaît jamais l'horloge absolue du simulateur.

## Fichiers toujours écrits

| Fichier | Format | Contenu |
|---|---|---|
| `jobs.json` | JSON, array d'objets | Un objet par job du batch ayant au moins une tâche `NotStarted` : `job_id`, `dataset_size`, `nb_tasks` (nb de tâches `NotStarted` seulement — renumérotées localement 0..nb_tasks-1, sans rapport avec le `task_id` réel), `task_duration`, `timelasped` (temps écoulé depuis l'arrivée du job), `job_arriving_time` (borne basse locale avant que le transfert puisse démarrer ; 0 pour un job déjà arrivé, le cas normal). Trié par `job_id`. |
| `nodes.json` | JSON, array d'objets | Un objet par nœud de calcul : `node_id`, `bandwidth`, `compute_capacity` (multiplicateur de durée — **plus petit = plus rapide**), `free_time` (quand ce nœud devient réellement libre, backlog inclus), `storage_capacity` (`2^30` si infinie, JSON n'a pas d'infini), `energy_consumption` (consommé seulement par `MainOnlineMultiObj.java`). |
| `replicas_locations.json` | JSON, matrice (liste de listes) | Une ligne par job de `jobs.json`, **même ordre** (indices alignés) : liste des `node_id` où la donnée du job est déjà résidente, transferts en cours inclus (sinon un nœud en cours de réception semblerait libre et un solve ultérieur pourrait y replacer autre chose). |
| `free_nodes.txt` | CSV d'entiers sur une ligne, ou vide | Filtre dur optionnel (`restrict_to_free_nodes`) : liste des `node_id` sans aucun transfert/tâche en cours ce tick précis. Vide = aucune restriction (comportement par défaut). |
| `frozen_jobs.txt` | CSV d'entiers sur une ligne, ou vide | **Indices locaux** (dans `jobs.json`, pas `job_id`) des jobs gelés pour ce solve : aucune nouvelle réplique, aucun déplacement, placement existant conservé tel quel. Un job est gelé si un des 3 critères ci-dessous est actif ET satisfait. |
| `freeze_blocks_node_until_done.txt` | `"1"` ou `"0"` | Si actif, le(s) nœud(s) d'un job gelé lui sont réservés jusqu'à la fin de sa dernière tâche (aucune autre tâche ne peut y démarrer avant). |
| `powerful_nodes.txt` | CSV d'entiers, ou vide | Sous-ensemble de nœuds (triés par `bandwidth / compute_capacity`) auquel confiner la reconsidération des jobs déjà en cours (opt-in `reschedule_top_fraction`). Vide = pas de restriction. |
| `power_restricted_jobs.txt` | CSV d'indices locaux, ou vide | Jobs (hors nouvelles arrivées) soumis à la restriction `powerful_nodes.txt`. |
| `solver_time_limit.txt` | Entier (secondes) | Budget solveur pour CE solve précis. Fallback Java à 120s si le fichier est absent/illisible. **Peut être temporairement substitué** par l'appelant Python (ex. budget d'escalade `hybrid`, probe Incremental interne) puis restauré après coup — voir la section Exp2 plus bas. |
| `objective_choice.txt` | `"0"`, `"1"`, `"2"` ou vide | `MainOnline`/`MainOnlineWarmStart` seulement : 0=somme des flow times, 1=max flow time, 2=flow time d'un job précis. Vide = défaut du binaire Java appelé. |
| `current_sim_time.txt` | Flottant | `env.now` absolu, pour affichage/debug Java uniquement — ne pilote aucune décision. |
| `ghost_storage.txt` | Une ligne par entrée : `node_id,taille,temps_avant_suppression` | Stockage occupé par des jobs **hors de ce batch** (déjà entièrement dispatchés, ou — pour Incremental — tout job autre que celui traité) : le CSP ne peut pas les reconsidérer mais doit savoir que l'espace est pris, et quand il se libère (`-1` si aucune suppression prévue). |
| `energy_config.txt` | 2 lignes : `master_energy_consumption`, `network_energy_per_transfer` | Consommé uniquement par `MainOnlineMultiObj.java`. Écrit systématiquement (coût négligeable), ignoré par tout autre binaire. |

## Fichiers liés au bi-objectif (epsilon-constraint)

Consommés uniquement par `MainOnlineMultiObj.java` (et ses variantes warm-start).

| Fichier | Format | Contenu |
|---|---|---|
| `multi_objective.txt` | `""`, `"1"`, `"2"` | Vide = recherche mono-objectif classique. `1` = front de Pareto brut {max flow time, énergie} (performe nettement moins bien à budget égal, gardé pour référence). `2` = **epsilon-constraint (recommandé)** : phase 1 minimise le max flow time, phase 2 minimise l'énergie sous contrainte que le flow time reste dans `epsilon_fraction` du résultat de phase 1. |
| `epsilon_fraction.txt` | Flottant ou vide | Dégradation relative du flow time tolérée en phase 2 (ex. `0.1` = 10%). |
| `epsilon_phase1_fraction.txt` | Flottant ou vide | Fraction de `solver_time_limit.txt` allouée à la phase 1 ; le reste va à la phase 2. |
| `epsilon_max_cap.txt` | Flottant ou vide | Plafond absolu optionnel sur le flow time accepté en phase 2 (ex. le résultat d'une approche de référence), indépendant de `epsilon_fraction`. |

## Warm-start

| Fichier | Format | Contenu |
|---|---|---|
| `warm_start.json` | JSON : `{"job_placements": [...], "transfers": [...]}` | Écrit uniquement par les schedulers warm-start (`hybrid` lors de son escalade). `job_placements[i]` = `{job_index, task_index, node, start}` ; `transfers[i]` = `{job_index, node, start}`. `job_index` = indice local dans **ce** solve (pas `job_id`). Construit à partir du dernier plan connu du scheduler (`self.works`/`self.transfers`) pour les jobs déjà en cours, plus un solve Incremental jetable pour les jobs tout juste arrivés. |

## Ce que reçoit chaque approche d'Exp2 (2026-09-28)

| Fichier | incremental | online_biobj | hybrid |
|---|---|---|---|
| `solver_time_limit.txt` | 600 | 600 | 600 en base ; **substitué** à 30 pour le probe Incremental interne, puis à `min(0.5×F1, 1200)` pour l'escalade |
| `multi_objective.txt` | vide | `2` | `2` (pour l'escalade) |
| `epsilon_fraction.txt` | vide | `0.1` | `0.1` |
| `epsilon_phase1_fraction.txt` | vide | `0.833333` (500s/100s) | `0.833333` |
| `frozen_jobs.txt` | vide | vide | job(s) avec transfert en cours (`freeze_jobs_with_ongoing_transfer` seul critère actif) |
| `freeze_blocks_node_until_done.txt` | `0` | `0` | `0` (non activé) |
| `warm_start.json` | non utilisé | non utilisé | rempli avant chaque escalade |
| `free_nodes.txt` / `powerful_nodes.txt` / `power_restricted_jobs.txt` | vides | vides | vides (`restrict_to_free_nodes` / `reschedule_top_fraction` non activés) |

Voir `simulator/exp2_grid5000_parameters_2026-09-28.md` pour le détail des paramètres de lancement (côté script Python) qui produisent ces valeurs.
