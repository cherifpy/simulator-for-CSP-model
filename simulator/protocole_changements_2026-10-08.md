# Changements au protocole expérimental — session du 2026-10-08

## 0. Contraintes de dégradation — récapitulatif par approche

Vérifié ligne par ligne dans le code (pas supposé) :

| approche | plafond dur par job existant ? |
| --- | --- |
| Incremental | sans objet (ne replanifie jamais les jobs existants) |
| Online (classique, `objective_choice=1`) | **non** |
| Online_biobj (epsilon-constraint) | **non** (plafond agrégé sur le max du batch via `epsilon_fraction`, pas par job) |
| Online_newjob | **oui** — `flow_time_caps` posé comme contrainte dure dans le solveur |
| Hybrid-n_j (toutes variantes, y compris `nofreeze`) | **oui** — même mécanisme, appliqué uniformément à toutes les variantes via `run_variant` dans `_timedParallelEscalation` |

## 1. Réduction à 8 variantes d'escalade

Sur demande explicite, retrait de `avoid_empty_slow_nodes`, `restrict_powerful_nodes`,
`freeze_random_half` — pas lié à une régression mesurée spécifique. 8 variantes restantes :
`freeze_below_mean`, `freeze_above_mean`, `nofreeze`, `warm_nofreeze`, `freeze_ongoing_transfer`,
`greedy_hints`, `hint_from_f1`, `hint_slow_to_fast`.

## 2. Investigation : `nofreeze` vs un appel `online_newjob` dédié

**Constat initial** (Grid5000, nancy.g5k, 8 cœurs) : sur un scénario donné (Phase A mixte x2,
seed=202), la variante `nofreeze` (dans la course à 11 puis 8 variantes) donne systématiquement
**11180.0**, alors qu'un appel `run_online_newjob_style` dédié (même budget 30s, même
`objective_choice=2`, même `flow_time_caps`) donne **9984.0** — un écart de ~12%, parfaitement
reproductible (valeurs identiques à chaque répétition), ce qui exclut d'emblée un simple bruit de
contention aléatoire.

**Pistes explorées, dans l'ordre, chacune invalidée par un test empirique :**

1. *Différence de contrainte de dégradation* — invalidée : les deux chemins appliquent
   exactement le même plafond (vérifié dans le code, §0 ci-dessus).
2. *Classe Java différente* (`nofreeze` tournait sur `MainOnlineMultiObj.java`, l'appel dédié sur
   `MainOnline.java`) avec une variable d'objectif énergie construite inutilement en mode
   mono-objectif, pouvant perturber l'ordre d'enregistrement des variables Choco — corrigé
   (`MainOnlineMultiObj(WarmStart).java` ne construit plus cette variable hors mode
   bi-objectif) — **testé sur Grid5000, aucun changement** (toujours 11180.0).
3. *Swap direct* : `nofreeze` basculé pour utiliser `MainOnline.java` directement (plus de
   classe différente du tout) — **testé sur Grid5000, toujours aucun changement** (11180.0).
4. *Fichiers de run isolés périmés* (`parallel_nofreeze_escalation/`, réutilisé entre plusieurs
   runs successifs) — dossier supprimé avant un nouveau test — **aucun changement**.
5. *Nombre de variantes concurrentes* (11 → 8) — **aucun changement** (11180.0 dans les deux cas).
6. **Test décisif : `nofreeze` seul dans la course (0 concurrence)** — résultat = **9771.0**,
   proche du dédié (9984.0) et même légèrement meilleur. Confirme que la contention CPU est bien
   la cause, mais de façon quasi binaire : passer de 11 à 8 variantes ne desserre rien sur un nœud
   à seulement 8 cœurs (toujours sursouscrit), il fallait l'extrême (1 seule variante) pour voir
   l'effet.
7. **Test final, confirmation propre** : même scénario, même config (8 variantes concurrentes),
   relancé sur un nœud à **28 cœurs** (cluster `ecotaxe`, nantes.g5k, via `-t exotic -p
   "cluster='ecotaxe'"`) — `nofreeze` = **9984.0**, **match exact** avec l'appel dédié. Avec
   assez de cœurs pour les 8 JVM concurrentes, la contention disparaît complètement.

**Conclusion** : la contention CPU est la cause réelle et confirmée — ni la contrainte de
dégradation, ni la classe Java, ni des fichiers périmés n'y jouaient de rôle. Elle se comporte de
façon quasi binaire (saturée dès que le nombre de JVM concurrentes dépasse le nombre de cœurs
disponibles) plutôt que progressive.

**Recommandation retenue** : pour tout run Hybrid sur Grid5000, réserver un nœud avec au moins
autant de cœurs que de variantes d'escalade (8 minimum avec la config actuelle) via `-p
"cluster='<nom>'"` ciblant un cluster à gros cœur, plutôt que `host=1` sans discrimination qui
peut atterrir sur un petit nœud. Clusters identifiés avec beaucoup de cœurs : `ecotaxe` (28,
nantes, `-t exotic` requis), `grdix` (128, nancy), `roazhon4`/`roazhon15` (64, rennes) — ces deux
derniers non testés (ressources indisponibles au moment du test).

## 3. Re-run complet des 16 scénarios Phase A/B/C sur `ecotaxe` (28 cœurs)

`exps/xp_validation_comparison.py --objective new_job` relancé intégralement sur `ecotaxe` au
lieu du Mac local, pour voir l'effet à l'échelle de toute la validation (pas seulement le
scénario isolé du §2). Résultats dans
`results-validation-2026-10-07/phaseABC_full_metrics_ecotaxe.csv`.

**Biais à signaler** : `Incremental` (qui ne dépend jamais de la concurrence) donne lui-même des
résultats différents entre le Mac local et `ecotaxe` (ex: palier petit 2106.0 local vs 2161.0
ecotaxe) — preuve qu'une partie de l'écart entre les deux jeux de résultats vient simplement de
la **vitesse CPU différente** entre les deux machines (budget de 30s = plus ou moins
d'itérations de recherche selon le matériel), pas de la contention. Cette comparaison à 16
scénarios est donc plus bruitée que le test isolé du §2 (qui comparait deux nœuds Grid5000 de la
même famille matérielle).

Gain Hybrid vs Incremental (métrique relative, moins sensible au biais matériel) :

| | local (Mac) | ecotaxe (28 cœurs) |
| --- | --- | --- |
| gain moyen | 6.0% | 7.8% |
| gain médian | 5.3% | 6.6% |

Légèrement meilleur sur `ecotaxe`, cohérent avec la direction attendue, mais pas de
bouleversement radical à cette échelle — la plupart des 16 scénarios ne tombent apparemment pas
dans la situation spécifiquement sensible à la contention qu'on avait isolée au §2 (où `nofreeze`
gagne la course). Hybrid ne perd jamais contre Incremental sur aucun des 16 scénarios, dans les
deux environnements ; Online seul perd parfois nettement (ex: B_n15 sur ecotaxe : 13363 vs 4557).

Tableau complet des 16 scénarios (ecotaxe) :

| scénario | Incremental | Online | Hybrid | variante gagnante |
| --- | --- | --- | --- | --- |
| A_palier_petit | 2161.0 | 2162.0 | 2161.0 | freeze_below_mean |
| A_palier_grand | 10862.0 | 10124.0 | **9640.0** | hint_slow_to_fast |
| A_palier_mixte | 7413.0 | 7958.0 | **6425.0** | freeze_ongoing_transfer |
| A_echelle_x0.5 | 1398.0 | 1269.0 | 1269.0 | freeze_below_mean |
| A_echelle_x2 | 7413.0 | 7958.0 | **6425.0** | freeze_ongoing_transfer |
| A_echelle_x4 | 21335.0 | 23147.0 | 21335.0 | freeze_above_mean |
| A_echelle_x6 | 17222.0 | 19673.0 | **14890.0** | freeze_above_mean |
| B_n2 | 4730.0 | 4731.0 | 4730.0 | freeze_below_mean |
| B_n5 | 7068.0 | 7958.0 | **6425.0** | freeze_ongoing_transfer |
| B_n10 | 5480.0 | 5064.0 | **5064.0** | nofreeze |
| B_n15 | 4557.0 | 13363.0 | 4557.0 | warm_nofreeze |
| C_s0 | 8906.0 | 9503.0 | 8906.0 | nofreeze |
| C_s1 | 4062.0 | 4196.0 | 4062.0 | freeze_below_mean |
| C_s2 | 7851.0 | 9447.0 | **7544.0** | greedy_hints |
| C_s3 | 4396.0 | 4152.0 | **4152.0** | freeze_below_mean |
| C_s4 | 11665.0 | 7725.0 | **7157.0** | freeze_ongoing_transfer |

0 violation de stockage sur les 48 lignes (16 scénarios × 3 approches).
