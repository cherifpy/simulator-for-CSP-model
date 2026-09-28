## note sur avancement des exps

- le 11 septembre exp lancer sur 2 mais le probleme est que les resutlats sont les meme que avec 1h donc est ce que le model trouve vraiment la meilleure solution ?


- le 12 seprembre je relance avec le max pour valider aussi le probleme.
    - Resultats interessant net amelioration dans les perfs surtout le waiting time et le maxflow time 

le dossiers avec simulator-for-CSP-model-befor-variying-datasize continer les resultats so far

je relance avec une derniere version pour verifier

Faut faire les expspourq    

Apress correction ke model fait bien les choses  

data size entre 1 et 10 GO

| Métrique | online (1h = 1h30, identique) | incremental (30s) | avantage |
|---|---|---|---|
| Wait time nouveau job | 4.00 | 3.00 | incremental |
| Flow time nouveau job | 507.00 | 443.00 | incremental |
| Flow time moyen (tous) | 492.78 | 580.91 | online (-15%) |
| Flow time max (tous) | 520.23 | 634.04 | online (-18%) |
| Énergie de transfert | 721.35 | 260.16 | incremental (-64%) |  

Je relance avec datasize entre 20 et 50

###  Note du 17 Semptembre 
Le online montre des resultats interessant, mais n'est pas tout as faire bien validé 
ce qui rest donc a faire:
- Exp sur la charge de l'infra au moment de l'arrive du job 
- Exp sur la taille des données aussi 

Construire l'etat A dans un le model online, pour garantir l'optimal ensuite lancer le nouveau job 


=> Pour l'instant valider l'approche sur la variation des taille de dataset

#### Details exp:
=> On construit un état A avec N jobs occupant 100 % des M nœuds (aucune restriction de charge), on injecte un job N+1 qui arrive après, et on compare Online (objectif : minimiser le flow time max) à Incremental en faisant varier uniquement la taille des datasets — de tous les jobs, existants et nouveau — sur trois paliers (1-10 Go / 15-40 Go / 50-100 Go).

| Palier | Taille (MB) | Mean FT online | Mean FT incr | Max FT online | Max FT incr | Énergie online | Énergie incr |
|---|---|---|---|---|---|---|---|
| small  | 8126  | 984.55  | 2091.04 | 1041.33 | 3479.76 | 2785.66 | 62.05   |
| medium | 36647 | 2113.73 | 2505.50 | 2316.50 | 3284.30 | 6262.04 | 173.79  |
| large  | 94536 | 4345.46 | 4091.95 | 5396.02 | 6053.76 | 8234.62 | 1998.56 |

ici jai change le nombre de job du stat A
| n_existing | Approche    | Wait  | Flow (new) | Mean (tous) | Max (tous) | Énergie |
|---|---|---|---|---|---|---|
| 5  | online      | 1357.00  | 1521.00  | 1436.44 | 1603.28  | 3946.50  |
| 5  | incremental | 493.00   | 1593.00  | 1455.91 | 1601.04  | 667.00   |
| 10 | online      | 1510.00  | 2834.00  | 2428.14 | 2854.60  | 7832.40  |
| 10 | incremental | 42.00    | 321.00   | 3081.86 | 6144.30  | 201.96   |
| 20 | online      | 11338.00 | 12346.00 | 6714.21 | 12845.89 | (~13-14k)|
| 20 | incremental | 41.00    | 537.00   | 8231.31 | 18436.19 | 303.46   |


en variyant l'occupation de l'ijfra 
| n_existing | Approche    | Wait    | Flow (new) | Mean (tous) | Max (tous) | Énergie |
|---|---|---|---|---|---|---|
| 5  | online      | 1357.00 | 1521.00 | 1436.44 | 1603.28  | 3946.50 |
| 5  | incremental | 493.00  | 1593.00 | 1466.91 | 1649.04  | 667.00  |
| 10 | online      | 1681.00 | 2826.00 | 2479.32 | 2833.24  | 7988.58 |
| 10 | incremental | 42.00   | 321.00  | 3081.86 | 6144.30  | 201.96  |
| 20 | online      | 4597.00 | 5815.00 | 5242.31 | 8834.46  | 9895.93 |
| 20 | incremental | 41.00   | 537.00  | 8231.31 | 18436.19 | 303.46  |


| Métrique | mono (max flow time seul) | multi (Pareto) |
|---|---|---|
| Max flow time | 390.00 | 3202.00 – 3436.00 |
| Énergie | 525.77 | 444.00 – 530.04 |

J'ai ensuite essaie de faire un multi objectif avec fention

| Métrique | mono (max-flow seul) | epsilon-contrainte (10% marge) |
|---|---|---|
| Max flow time | 390.00 | 429.00 |
| Énergie | 525.77 | 132.34 |

## Exps 3
En testant une nouvelle approach 
| Palier | Taille | Métrique | online (2h) | incremental (1min) | epsilon (1h/1h) |
|---|---|---|---|---|---|
| small | 8126 | Flow time (new) | 1015.00 | 287.00 | 1010.00 |
| small | 8126 | Mean flow time | 984.55 | 2091.04 | 981.46 |
| small | 8126 | Max flow time | 1041.33 | 3479.76 | 1144.50 |
| small | 8126 | Énergie | 2785.66 | 62.05 | 1388.46 |
| medium | 36647 | Flow time (new) | 2309.00 | 1019.00 | 2087.00 |
| medium | 36647 | Mean flow time | 2113.73 | 2505.50 | 2079.28 |
| medium | 36647 | Max flow time | 2316.50 | 3284.30 | 2561.60 |
| medium | 36647 | Énergie | 6262.04 | 173.79 | 3902.97 |
| large | 94536 | Flow time (new) | 5247.00 | 4226.00 | 6353.00 |
| large | 94536 | Mean flow time | 4345.46 | 4091.95 | 4692.92 |
| large | 94536 | Max flow time | 5396.02 | 6053.76 | 6396.33 |
| large | 94536 | Énergie | 8234.62 | 1998.56 | 4286.51 |

j'ai refait pour large dataset 
| Métrique | online (2h) | incremental (1min) | epsilon (2h/2h, capé) |
|---|---|---|---|
| Flow time (new job) | 5247.00 | 4226.00 | 5892.00 |
| Mean flow time | 4345.46 | 4091.95 | 4112.28 |
| Max flow time | 5396.02 | 6053.76 | 5941.67 |
| Énergie | 8234.62 | 1998.56 | 6261.55 |


J'ai besoinde variance donc j'ai re relance les exps
Voici les deux commandes, une pour chaque script préparé :

1. Sweep taille de dataset (pour la variance par job) — 3 jobs en parallèle (small/medium/large) :

./simulator/submit_dataset_size_sweep_grid5000.sh
2. Workload complet online vs online bi-objectif — 2 jobs en parallèle :

./simulator/submit_online_vs_biobj_workload_grid5000.sh


## Exps sur workload complet

### le workload online 
J4ai lancé sur grid 5000 nancy une exp pour verifier sur un workload complet 
configu 
- appoach: online
- solving time: 30min /job
- nb job & node: 20 / 50

deja lancer sur 10 min ca a donner ca 
┌───────────────────────────────────┬────────────┬────────────┬──────────┐
│                run                │ flow moyen │ écart-type │ flow max │
├───────────────────────────────────┼────────────┼────────────┼──────────┤
│ Nancy online 10 min (nouveau)     │ 678        │ 266        │ 928      │
├───────────────────────────────────┼────────────┼────────────┼──────────┤
│ online_lambda100                  │ 693        │ 455        │ 1392     │
├───────────────────────────────────┼────────────┼────────────┼──────────┤
│ online_lambda100_warmstart_60s    │ 636        │ 460        │ 1517     │
├───────────────────────────────────┼────────────┼────────────┼──────────┤
│ incremental_lambda100             │ 490        │ 283        │ 993      │
├───────────────────────────────────┼────────────┼────────────┼──────────┤
│ online_lambda100_maxobj_fixed_30s │ 2171       │ 1997       │ 6107     │
└───────────────────────────────────┴────────────┴────────────┴──────────┘


### wokload online biobj 
lancer sur grenoble 

### Relance
1- sur lille j'ai relancé sur la taille des données pour avoir des stats plus detaillé


# Objectif actuell: accelere le model a travers des optimisations 

Bloquer des jobs et bloquer des noueds 

jusqu'a present on bloque que les jobs qui sont en fin de leurs flow time 

## exps en cours
- Warm start+ freazing

- PRoblmemes:
    - JObs qui n'on pas fini leurs transfers 
    - LEs jobs a qui rest peu de temps pour fenir 
    - Les jobs 


lancer actuellement le 26 sep une exp pour comparer 

Hypride, Incremental, OnlineBiobj

# Récap — session du jour

## Code modifié
**Fichier** : `exps/xp_dataset_size_sweep.py` (protocole state-A figé)

- Ajout de `run_hybrid_style` (1ère version), puis **réécrite** selon la spec :
  - `incremental` — inchangé
  - `online_warmstart` — inchangé (déjà mono-obj)
  - `online_biobj_warmstart` — **nouvelle fonction** (reconsidération jointe + warmstart, bi-objectif epsilon-constraint)
  - `hybrid` — réécrit :
    - F1 (probe Incremental, 15s par défaut) réutilisé directement comme warm-start → plus de double-solve redondant
    - budget d'escalade = 10% de F1 (`hybrid_alpha=0.1`, avant 0.2)
- `build_warm_start_for_reconsideration` : accepte un placement déjà calculé (évite un 2e solve)
- Synchronisé vers la copie de déploiement nested + compilé sans erreur

## Expés locales (smoke tests, instance 10J-10N)

| # | Test | Résultat |
|---|------|----------|
| 1 | incremental / epsilon / hybrid (1ère version) | OK |
| 2 | State-A sur **job17** (instance heterogeneous 20J-50N, 5 approches) | Crash initial (`arriving_time` manquant) → corrigé avec `--arrival-lambda` → temps de scheduling obtenus |
| 3 | 4 méthodes finales (incremental / online_warmstart / online_biobj_warmstart / hybrid) | Confirmé : hybrid passe de **152s → 31s** de wall time après le fix |

## Grid5000 (observation seule, pas de modif)

- Job `4157990` (hybrid eps=5%, protocole live) — toujours en cours, job18/19 restants
- Job17 traité : flow time ≈ **4041s** (vs 6183.5s à eps=10%) → amélioration confirmée
- ⚠️ **Anomalie non résolue** : le solve final de job17 ne montre pas le header bi-objectif attendu (possible fallback légitime vers Incremental, ou bug dans la classe live `SchedulingUsingCSPAdaptiveJoint`) — à investiguer si besoin

## État

- Rien n'a été commité (comme convenu)
- Rien n'a été poussé

#faut vraiment verifier ca Non, pas exactement comme tu le décris — j'ai vérifié dans le code, et voici précisément ce qui se passe, identique pour les 3 approches (incremental, online_biobj, hybrid) :

- nodes_free_time/replicas_locations sont capturés AVANT le solve, à l'instant T (self.env.now au moment de l'appel).
- Le solveur Java prend sa décision de placement sur cette photo de l'infra à T, comme si résoudre était instantané.
- Après le solve, le temps réel écoulé (elapsed) est chargé via yield self.env.timeout(elapsed) — ça fait avancer l'horloge simulée à T+elapsed.
- Mais rien ne re-capture l'état de l'infra à T+elapsed avant de committer le plan. Le plan a été décidé sur l'état à T, pas sur l'état "T + temps de scheduling".

Donc le temps de scheduling est bien facturé comme un coût (ça retarde tout ce qui suit dans la simulation), mais le solveur ne voit jamais l'état réel de l'infra tel qu'il sera une fois ce temps écoulé — il décide comme si c'était instantané à T, puis on paie le délai après coup.

Autre point que j'ai vérifié en creusant ta question : l'exécution réelle des tâches (compute_node.py:processTasks) n'utilise même pas les timestamps absolus (start_time/end_time) que le solveur calcule — elle dépile la queue FIFO et utilise self.env.now réel au moment où chaque tâche démarre vraiment. Donc les timestamps du solveur servent à la décision de placement (quel job sur quel nœud, dans quel ordre) et à l'affichage/logging, mais pas à piloter directement le timing réel de la simulation.

C'est le même comportement pour les 3 approches — donc pas de biais entre elles sur ce point précis, mais c'est une approximation (photo figée à T) présente partout.