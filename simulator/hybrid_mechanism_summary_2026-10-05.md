# Résumé complet — mécanisme hybrid (session du 2026-10-04/05)

## Objectif général

Question testée : **hybrid** (sonde Incremental pour F1, puis escalade conditionnelle vers une
replanification conjointe à 6 variantes) bat-il **incremental** seul sur un vrai compromis
qualité/temps, et si oui dans quelles conditions ?

Protocole *single-decision-point* : un état A (jobs déjà en place) est construit une seule fois
par un solve CSP joint, un nouveau job arrive à un instant T figé, puis incremental/hybrid
décident de son placement à partir du **même état A**.

---

## 1. Premiers réglages — scénarios aléatoires (échelle normale)

| Variante | Changement | Résultat |
| --- | --- | --- |
| v3 | Sélection entre les 6 variantes sur `new_job` au lieu de `max` | Identique à v2 sur 9/10 — la marge F1 bloquait déjà l'escalade |
| v4 | Marge F1 **dynamique** (se réduit quand F1 dépasse déjà la moyenne) | Identique à v2/v3 sur 9/10 |
| v5 | **Escalade systématique** (pas de marge) | 10/10 tentées, **0/10 acceptées** — ~360s gaspillés à chaque fois |

**Diagnostic** : sur des scénarios tirés i.i.d. (existants et nouveau job confondus), le ratio
F1/moyenne-des-jobs-en-cours est structurellement bas — rien ne force le système à être sous
tension au moment précis où le nouveau job arrive.

## 2. Échelle doublée + gate=max

En doublant les tailles (task_duration [200,300]s, dataset_size [20480,204800]Mo) et en repassant
le gate sur `max` (jugement batch-wide), 3/5 scénarios acceptent enfin l'escalade, avec de vrais
gains batch (-22% à -64% sur le max) au prix d'une forte dégradation du nouveau job (+45% à
+329%). **C'est le gate=max qui débloque la capacité à "gagner", pas l'échelle doublée en
elle-même.**

## 3. Stress puis arrivée — effet de la qualité de l'état A

Jobs existants délibérément lourds, nouveau job ordinaire. Avec un état A à 30s de budget, 4/5
acceptent avec de bons ratios (3.0x à 14.3x). Mais en redonnant 10 minutes à l'état A sur les
**mêmes** scénarios, le tableau change radicalement :

- `n=11` : gain batch passe de 4610s à **0** (la congestion disparaît avec un meilleur état A)
- `n=14` : une opportunité apparaît qui était masquée avant (gain 0→267s)
- `n=20` : **inchangé**, seul cas où l'état A convergeait déjà en 30s

**Conclusion** : une grande partie du "bénéfice" observé à 30s était un artefact de méthode
(compensation d'un état A sous-optimisé), pas un vrai avantage structurel.

## 4. Arrivées simultanées

- **Cas simple** (nouveau job ordinaire, t=200s) : 0 gain, 603s gaspillées. Incremental résout en
  1.2s (état A déjà excellent).
- **Cas nouveau job maximal** (t=max_flow_time/2) : escalade acceptée mais ratio **0.11** (le pire
  de la session) — 2 jobs existants dégradés de +1300s chacun, non protégés par aucune des 6
  variantes.

C'est ce dernier cas qui a motivé la construction d'un **veto de stabilité**.

## 5. Veto de stabilité (CV)

Nouveau paramètre `adaptive_f1_stability_cv_threshold` : si le coefficient de variation
(écart-type/moyenne) des flow times déjà committés est sous ce seuil, l'escalade est vetoée avant
même d'être tentée. Calibré à 0.4 sur le cas problématique (CV=0.37).

Testé sur 5 puis 10 scénarios (arrivée simultanée) : **7/10 acceptent**, gain moyen **6.8%** sur le
nouveau job, dégradation max jamais au-dessus de 20% (le plafond posé séparément à l'époque).

---

## 6. Le mécanisme "hybrid-n_j" (nouvelle conception)

Changement de design demandé : au lieu de juger l'escalade sur le batch (`gate=max`), **l'objectif
du solveur lui-même** devient de minimiser le flow time du nouveau job, sous une **contrainte
dure** : aucun job existant ne peut se dégrader de plus de X% par rapport à son flow time déjà
committé.

### Implémentation

- **`objective_choice=2`** : objectif Java déjà existant (flow time d'UN job spécifique — toujours
  celui au plus grand `job_id`, convention déjà en place).
- **`flow_time_caps.txt`** : nouveau fichier d'échange Python→Java, un plafond par job (`-1` =
  pas de plafond), posé comme contrainte dure (`model.arithm(all_flow_time[i], "<=", cap)`)
  **avant** que la résolution ne commence — dans les 3 fichiers Java (`MainOnline.java`,
  `MainOnlineMultiObj.java`, `MainOnlineMultiObjWarmStart.java`).
- **`adaptive_selection_metric=new_job`** et **`adaptive_gate_metric=new_job`** : réutilisés tels
  quels (déjà existants).

### Erreur initiale et correction

Première implémentation : `multi_objective=0` pour forcer le chemin Java qui respecte
`objective_choice` — **mais ça désactivait complètement la phase énergie** (le but du bi-objectif).
Corrigé sur demande explicite de l'utilisateur :

- Fix Java : la phase 1 du mode bi-objectif (epsilon-constraint) utilisait `objectives[1]`
  (max flow time) **codé en dur**, ignorant totalement `objective_choice`. Changé en
  `objectives[objectiveChoice]` dans les 2 fichiers concernés.
- `multi_objective` repassé à sa valeur par défaut de classe (`2`, bi-objectif actif).
- Résultat : phase 1 minimise le nouveau job, phase 2 minimise l'énergie sous un plafond
  `epsilon_fraction` par job basé sur le résultat de la phase 1 — **les deux plafonds (le mien et
  celui d'epsilon) se cumulent**, sans dépasser la limite en pratique sur les tests effectués.

### Bug annexe découvert

`epsilon_fraction`, `epsilon_phase1_fraction` et `epsilon_max_cap` sont lus côté Java via
`getattr(master_node, ...)` **directement sur l'attribut**, jamais depuis le dict `_config`.
`run_hybrid_style` ne les écrivait que dans `_config` → ignorés silencieusement, retombant sur les
défauts de classe (`epsilon_phase1_fraction=0.5` au lieu du `0.833` demandé). Corrigé en les
réglant aussi comme attributs d'instance.

### Résultats — 10 scénarios, plafond 20%, budget 300s, split 83/17 (bugué → en réalité 50/50)

| iter | n_existing | gain nouveau job | pire dégradation existant |
| --- | --- | --- | --- |
| 0 | 15 | +14.1% | 18.9% |
| 1 | 5 | +7.7% | 17.9% |
| 2 | 11 | +2.8% | 19.1% |
| 5 | 10 | +8.1% | 19.2% |
| 6 | 13 | +7.7% | 20.0% (pile à la limite) |
| 7 | 6 | +3.7% | 19.5% |
| 8 | 12 | +3.4% | 19.3% |
| 3,4,9 | 9,6,8 | 0% (rejetées) | 0% |

**7/10 acceptées**, plafond jamais dépassé. Mais coût réel : **+114% d'énergie en moyenne**
(jusqu'à +303%), et **2.7x** le temps de calcul d'Incremental — ce dernier point en grande partie
imputable à un goulot d'étranglement identifié (voir section 8).

### Validation du bi-objectif (plafond resserré à 10%)

Sur le scénario `iter_0` (seul cas accepté du run suivant) : nouveau job 5869→5422 (**-7.6%**),
dégradation max 8.6-9.8% (sous le plafond de 10%), énergie +166% à +212% selon le split
phase1/phase2 testé.

### Run "final" à 10 scénarios (split 50/50, F1 probe réduite à 100s)

Chute du taux d'acceptation à **1/10** (contre 7/10 avant). Diagnostic : en réactivant le
bi-objectif, la phase 1 ne reçoit que **250s** (500s × 0.5) au lieu des **300s** complets de
l'ancienne version mono-objectif pure — suffisant pour expliquer la perte de qualité de
convergence sur un CSP de cette taille.

### Essais en cours au moment de la rédaction

- **Split 85/15** (phase1=425s, phase2=75s, budget=500s) : teste si redonner plus de temps à la
  phase 1 restaure le taux d'acceptation.
- **Budget relevé à 1000s** (même split 85/15, phase1=850s, phase2=150s) : teste en plus si la
  formule dynamique `budget=min(hybrid_alpha×F1, hybrid_max_budget)` peut enfin s'exprimer sous
  son plafond pour les scénarios à F1 plus faible.

---

## 7. La formule dynamique du budget

`budget = min(hybrid_alpha × F1, hybrid_max_budget)`. Avec `hybrid_alpha=0.25` et des F1 observés
entre 2146 et 12600s sur nos 10 scénarios, `0.25×F1` vaut systématiquement entre 536 et 3150 —
**toujours au-dessus du plafond de 500s testé**. Résultat : sur tous les runs à `max_budget=500`,
le budget réel a été une constante (500s), jamais la partie "dynamique" de la formule. Il faudrait
soit des F1 plus petits, soit un `hybrid_alpha` plus élevé, soit un plafond plus haut pour que la
formule s'exprime vraiment.

## 8. Goulot d'étranglement identifié (non corrigé, laissé de côté)

Sur les escalades à budget élevé, les 6 variantes ne démarrent pas toutes en même temps malgré un
pool de 6 threads : 3 démarrent immédiatement, les 3 autres attendent ~300s de plus avant même de
commencer leur propre résolution (chacune individuellement respecte bien son budget une fois
lancée). Cause la plus probable : contention réseau sur le `/home` monté en **NFS** sur
`lille.g5k` (confirmé via `mount`), 6 process concurrents lisant/compilant les mêmes jars Choco
sur ce montage partagé. Non corrigé — décision explicite de l'utilisateur de laisser ce surcoût de
côté tant que les résultats restent valides.

## 9. Mono-objectif vs bi-objectif — le vrai facteur derrière la chute d'acceptation

Après avoir éliminé le budget (500s→1000s, aucun effet) et le plafond de dégradation (10%→25%,
aucun effet sur le taux), le vrai coupable identifié : **la structure bi-objectif elle-même**
(mode epsilon-constraint à 2 phases, 4 objectifs trackés simultanément) fait chuter le taux
d'acceptation à ~1/10, quel que soit le réglage — un coût structurel de convergence, pas un
problème de réglage.

Nouveau flag `--adaptive-no-bi-objective` ajouté pour forcer `multi_objective=0` (phase 1 seule,
pas de phase énergie) tout en gardant `objective_choice=2` + le plafond de dégradation. Testé sur
4 scénarios (plafond 25%, F1=100s, budget=500s) : **4/4 acceptés**, confirmant que c'est bien le
bi-objectif qui écrasait le taux d'acceptation.

| iter | n_existing | Incremental (s) | Hybrid mono-objectif (s) | gain (s) | gain (%) | dégradation max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 15 | 5869.0 | 4447.0 | 1422.0 | 24.2% | 24.5% |
| 1 | 5 | 12600.0 | 11619.0 | 981.0 | 7.8% | 24.4% |
| 2 | 11 | 4604.0 | 4357.0 | 247.0 | 5.4% | 25.0% |
| 3 | 9 | 2146.0 | 2098.0 | 48.0 | 2.2% | 21.9% |

Sur `iter_0` spécifiquement (seul cas accepté en bi-objectif aussi), le résultat mono-objectif
est **identique bit à bit** au résultat bi-objectif (même flow time, même énergie) — la phase 2
n'avait en réalité rien trouvé à améliorer : la phase 1 convergeait déjà si près du plafond de
dégradation qu'il ne restait aucune marge pour la phase énergie. Ça suggère que le bi-objectif,
tel qu'implémenté ici, n'apporte quasiment aucun bénéfice énergétique réel sur ces scénarios tout
en coûtant cher en taux d'acceptation — un compromis net défavorable en l'état.

## 10. Effet du palier de taille de données (petit/grand/mixte) sur hybrid-n_j

Test avec `n_existing=5` fixe, `nb_nodes=50` fixe, le nouveau job suivant le même palier que les
jobs existants (`--new-job-mode same-tier`), même config mono-objectif validée (plafond 25%,
F1=100s, budget=500s). Trois paliers testés : **petit** (durée [100,150]s, données [2048,20480]MB),
**grand** (durée [250,300]s, données [102400,204800]MB), **mixte** (durée [100,300]s, données
[2048,204800]MB).

**Résultats à taille normale (x1)** :

| tier | new_job (inc → hyb, s) | gain | accepté | dégradation max | delta énergie |
| --- | --- | --- | --- | --- | --- |
| petit | 571.0 → 571.0 | 0% | Non (fallback) | 0% | 0% |
| grand | 2765.0 → 2765.0 | 0% | Non (fallback) | 0% | 0% |
| mixte | 342.0 → 235.0 | +31.3% | **Oui** (budget=85.5s) | 22.8% | +1577.3% |

Seul le palier **mixte** (hétérogène) est accepté. Petit et grand (homogènes) sont systématiquement
rejetés : l'escalade tourne (jusqu'à 707s de temps de calcul sur "grand") mais ne trouve jamais de
plan qui batte Incremental sous contrainte du plafond de dégradation.

**Résultats à 3x la taille/durée** (mêmes paramètres hybrid-n_j, seules les plages
dataset_size/task_duration sont multipliées par 3) :

| tier | new_job (inc → hyb, s) | gain | accepté | dégradation max | delta énergie | sched (s) |
| --- | --- | --- | --- | --- | --- | --- |
| petit x3 | 1299.0 → 751.0 | **+42.2%** | **Oui** (budget=324.75s) | 22.9% | +714.1% | 430.0 |
| grand x3 | 9482.0 → 9482.0 | 0% | Non (fallback) | 0.0% | 0% | 206.5 |
| mixte x3 | 7171.0 → 5969.0 | +16.8% | **Oui** (budget=500.0s, plafonné) | 24.2% | +297.9% | 703.8 |

Changement net : **petit bascule de rejeté (x1) à accepté (x3)**, avec le meilleur gain observé
toutes conditions confondues (+42.2%). **Grand reste rejeté dans les deux tailles**, même à 3x
l'échelle. Mixte reste accepté dans les deux cas mais avec un gain plus faible à x3 (16.8% vs 31.3%
à x1) — la dégradation des jobs existants cogne dans les deux cas très près du plafond de 25%
(22.8-24.2%), signe que l'escalade consomme systématiquement toute la marge autorisée quand elle
est acceptée.

**Interprétation** : ce n'est pas la taille absolue des données qui détermine l'acceptation, mais
l'hétérogénéité du pool de jobs. Un palier "grand" reste homogène même à 3x l'échelle (toutes les
durées/tailles restent proches les unes des autres en valeur relative), donc peu de marge de
replanification sous le plafond de dégradation. Un palier avec plus de variance relative (mixte
toujours, petit à partir d'une certaine échelle) laisse plus de marge au solveur pour replacer les
jobs existants sans les pénaliser au-delà du plafond, et donc plus de chances d'améliorer le
nouveau job. Dans tous les cas acceptés jusqu'ici, le coût en énergie de transfert explose
(+297% à +1577%), parce que l'escalade replanifie TOUS les jobs existants (pas seulement le
nouveau), ce qui multiplie les transferts réseau.

## 11. Balayage d'échelle x1/x3/x4/x5 — le pattern n'est pas monotone

Suite à la section 10, test de x3, x4 et x5 (mêmes plages de durée/taille multipliées, mêmes
paramètres hybrid-n_j, toujours `n_existing=5`, `nb_nodes=50`, une seule graine par combinaison
tier×échelle).

| échelle | petit | grand | mixte |
| --- | --- | --- | --- |
| x1 | Rejeté (0%) | Rejeté (0%) | Accepté (+31.3%, énergie +1577%, dégr 22.8%) |
| x3 | Accepté (+42.2%, énergie +714%, dégr 22.9%) | Rejeté (0%) | Accepté (+16.8%, énergie +298%, dégr 24.2%) |
| x4 | Accepté (+12.2%, énergie +48%, dégr 14.1%) | Accepté (+0.05%, énergie +60%, dégr **25.0%**, cas limite) | Accepté (+20.4%, énergie +676%, dégr 20.2%) |
| x5 | Rejeté (0%, escalade tourne 604.8s puis échoue) | Rejeté (0%) | Rejeté (0%) |

**Aucune tendance monotone** : "grand" passe rejeté→rejeté→accepté→rejeté sur la progression
x1→x3→x4→x5, ce qui exclut un effet d'échelle continu et propre (ni "plus gros = plus de marge",
ni "plus gros = plus dur"). Le cas "grand x4" accepté n'est d'ailleurs qu'un cas limite : gain
quasi nul (+0.05%) et dégradation pile au plafond (25.0%).

**Limite méthodologique importante** : chaque case du tableau est un **tirage aléatoire unique**
(une seule graine par tier×échelle) — la difficulté réelle d'une instance CSP dépend fortement du
tirage précis des durées/tailles, indépendamment du facteur d'échelle appliqué. Le zigzag observé
est donc plus probablement du bruit d'instance-à-instance que la preuve d'un vrai effet de taille.
Pour conclure proprement sur l'effet de l'échelle, il faudrait plusieurs graines par combinaison
et comparer un taux d'acceptation moyen, pas un seul run par case.

## 12. Pourquoi hybrid-n_j se déclenche peu souvent — explication structurelle

Synthèse de ce que confirment toutes les sections précédentes (10, 11, plus un test en cours sur
des seuils de ressources disponibles à l'arrivée, mixte x2, nb_nodes variable 10/20/30/40/50 :
1 accepté sur 4 testés jusqu'ici, sans lien avec le niveau de restriction).

**Le cœur du problème** : Incremental (F1) place le nouveau job **sans aucune contrainte** sur
les jobs existants — il optimise librement sur son propre objectif. L'escalade hybrid-n_j doit
optimiser le **même objectif** (flow time du nouveau job) **mais sous contrainte supplémentaire**
(plafond de dégradation de 25% sur chaque job existant).

Un problème **plus contraint** ne peut jamais trouver un optimum **meilleur** qu'un problème moins
contraint pour le même objectif — seulement égal ou pire. L'escalade ne peut donc battre F1 que
dans un cas précis : **quand la solution non contrainte de F1 respecte déjà le plafond par
hasard**, et qu'il reste encore de la marge pour améliorer le nouveau job sans le dépasser.

C'est une **coïncidence structurelle propre à chaque instance**, pas une propriété continue du
problème — ce qui explique directement l'absence de tendance observée partout dans ce document :
- Paliers de taille (section 10) : pas de tendance claire, grossièrement 1 cas sur 3 accepté.
- Balayage d'échelle x1→x5 (section 11) : zigzag total, pas de dégradation progressive.
- Seuils de ressources disponibles : 1 accepté sur 4 testés, sans lien avec le niveau de
  restriction.

Dans tous ces balayages, le taux d'acceptation tourne autour de **20-35%**, jamais 0% ni 100% —
cohérent avec "ça dépend du tirage précis de cette instance CSP précise", pas avec un levier
qu'on pourrait régler (taille, hétérogénéité, ressources disponibles).

**Pistes pour un déclenchement plus fiable** (non testées à ce stade) :
1. Desserrer le plafond de dégradation (plus de marge = plus de chances que F1 le respecte déjà).
2. Changer l'objectif de l'escalade pour viser autre chose que battre F1 sur le même critère
   exact (le critère actuel garantit presque par construction que F1 est difficile à battre).
3. Accepter que ce mécanisme reste un gain opportuniste ("coup de chance" par instance) plutôt
   qu'une amélioration systématique, et le présenter comme tel.

---

## Conclusions transversales

1. **Jugé sur le flow time du nouveau job seul, hybrid (ancien design gate=max) ne peut
   structurellement jamais gagner** — Incremental lui donne déjà un service quasi optimal.
2. **La détection du bon moment pour escalader est le vrai problème**, pas le mécanisme
   d'escalade lui-même. Marge F1 fixe → trop restrictive. Marge dynamique → même limite. Veto de
   stabilité CV → le plus fiable testé à ce jour.
3. **hybrid-n_j inverse le compromis** : il peut désormais réellement améliorer le nouveau job
   (jusqu'à +14%), avec une dégradation strictement bornée des jobs existants — mais au prix d'un
   vrai surcoût énergétique (+100 à +300%) qui n'avait jamais été quantifié avant.
4. **Toujours vérifier qu'un "gain" de hybrid survit à une baseline bien optimisée** (état A à
   budget généreux) avant de le créditer au mécanisme plutôt qu'à un artefact de méthode.
5. **Deux bugs de plomberie Python→Java** découverts et corrigés cette session (attribut vs.
   `_config` pour `objective_choice`/`multi_objective`/`epsilon_fraction`/`epsilon_phase1_fraction`)
   — à surveiller pour tout nouveau paramètre ajouté au pont Python/Java à l'avenir.

---

## Annexe — paramètres expérimentaux exacts (sections 9 à 12)

Config hybrid-n_j commune à TOUS les tests des sections 9-12 (mono-objectif, validée) :

| paramètre | valeur |
| --- | --- |
| `adaptive_new_job_objective` | activé (`objective_choice=2`, flow time du nouveau job seul) |
| `adaptive_degradation_cap_pct` | 0.25 (plafond 25% sur chaque job existant) |
| `adaptive_no_bi_objective` | activé (force `multi_objective=0`, pas de phase énergie) |
| `adaptive_selection_metric` | `new_job` |
| `adaptive_gate_metric` | `new_job` |
| `hybrid_incremental_time_limit` (budget sonde F1) | 100s |
| `hybrid_alpha` | 0.25 |
| `hybrid_max_budget` | 500s |
| `incremental_time_limit` (budget Incremental réel) | 300s |
| `state_a_time_limit` (budget construction état A) | 180s |
| `n_existing` | 5 (fixe sur toutes les sections 9-12) |

Script utilisé : `exps/xp_simultaneous_sweep.py` (`--new-job-mode same-tier`, nouveau job tiré de
la même plage que les jobs existants) sauf section 9 (`exps/xp_dataset_size_sweep.py` directement,
scénarios pré-existants `hybrid_nj_5iter_5minbudget`/`hybrid_nj_cap25_5iter`/etc., voir sections 2-8
pour leur propre contexte).

**Définitions des paliers de taille (section 10, x1)** :
| tier | task_duration (s) | dataset_size (MB) |
| --- | --- | --- |
| petit | [100, 150] | [2048, 20480] |
| grand | [250, 300] | [102400, 204800] |
| mixte | [100, 300] | [2048, 204800] |

**Facteurs d'échelle (section 11)** : mêmes paliers, plages multipliées par le facteur indiqué
(x3, x4, x5) — ex. petit x3 = durée [300,450]s, taille [6144,61440]MB. `nb_nodes=50` fixe sur
toute la section 10-11. Graines : petit=42, grand=43, mixte=44 (`--iteration` 0/1/2, `--seed`
identique à l'échelle de base pour chaque tier, non recombiné entre échelles).

**Test de seuil de ressources (section 12, mixte x2)** : task_duration [200,600]s, dataset_size
[4096,409600]MB. `n_existing=5`, contenu des jobs strictement identique sur les 5 niveaux (graine
effective forcée à 44 via `--seed=(44-iteration)` pour annuler le décalage `seed_i=seed+iteration`
du script). Seul `nb_nodes` varie :

| restriction | `nb_nodes` | résultat |
| --- | --- | --- |
| 80% | 10 | Rejeté (0%) |
| 60% | 20 | Accepté (+13.8%, énergie +29.7%, dégr 24.6%) |
| 40% | 30 | Rejeté (0%) |
| 20% | 40 | Accepté, cas limite (+0.43%, énergie +71.5%, dégr 21.9%) |
| 0% (infra complète) | 50 | Rejeté (0%) |

Bilan final : 2 acceptés sur 5 (dont un quasi nul, +0.43%), aucun lien visible avec le niveau de
restriction de ressources — même signature de bruit d'instance que les sections 10 et 11.

Tous les runs ci-dessus : `walltime=00:30:00`, un seul `host=1` par `oarsub`, cluster `lille.g5k`.
