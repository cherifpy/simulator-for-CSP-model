# Rapport : rendre theta dynamique selon la situation de l'infrastructure

Date : 2026-10-08. Rapport d'analyse uniquement : **aucun fichier de code ou de configuration n'a été modifié.**

## 1. Résumé

- Le paramètre theta existe dans le modèle CSP sous le nom `factor`. Il vaut `1` en dur dans 6 fichiers Java et rien côté Python ne le pilote.
- Le sens de `factor` dans le Java est l'inverse de la définition voulue (theta grand = plus de réplication). Sans effet tant qu'il vaut 1, à corriger dès qu'il varie.
- `factor` est un entier : impossible aujourd'hui de tester 0.5 ou 1.5.
- Les clés `threshold`, `adapting_theta`, `max_theta_user_preferences` de `config.json` et `tracker.log_threshold()` existent déjà mais ne sont branchées sur rien.
- Le rendre dynamique demande trois changements : theta devient une entrée du modèle, le master le recalcule à chaque arrivée de job, et son historique est exporté par run.
- La règle proposée baisse theta quand l'infrastructure est saturée. Cette hypothèse doit être validée par un sweep statique avant de s'y fier.

## 2. État actuel

### 2.1 Où se trouve theta

`int factor = 1;` dans un bloc identique de ces 6 fichiers (`utils/model/src/main/`) :

| Fichier | Ligne | Utilisé par |
|---|---|---|
| `MainIncremental.java` | 706 | Incremental, étape F1 du Hybrid |
| `MainOnline.java` | 849 | Online mono-objectif |
| `MainOnlineMultiObj.java` | 870 | Online bi-objectif, escalade Hybrid |
| `MainOnlineMultiObjWarmStart.java` | 858 | escalade Hybrid (warm start) |
| `MainOnlineWarmStart.java` | 810 | Online warm start |
| `Main.java` | 658 | aucune classe du master actuel |

`MainOnlineThreeStep.java` n'a pas cette contrainte. Le modèle MiniZinc a `int: factor = 1;` (`utils/minizincModel/scheduler.mzn:13`).

### 2.2 Ce que fait la contrainte

Extrait de `MainOnline.java:864-874` :

```java
// Transfer_time <= factor * sum(execution_time)
for (int j = 0; j < nb_nodes && factor > 0; j++) {
    ...
    BoolVar h = transfers_counter.gt(1).and(transferHeights[j][i]).boolVar();
    int transferTime = (int) Math.ceil((double) data_sizes[i] / bandwidths[j]);
    model.sum(executions, ">=", model.intView(transferTime * factor, h, 0)).post();
}
```

Pour un job `i` et un nœud `j`, si le job a plus d'un réplica et qu'un réplica est placé sur `j` :

```
somme(durées des tâches de i exécutées sur j) >= transferTime(i, j) * factor
```

- Le premier réplica d'un job n'est jamais contraint ; seuls les réplicas supplémentaires doivent être rentabilisés.
- `factor = 0` désactive la contrainte (réplication libre).
- `transferTime` est calculé comme `taille / bande passante` du nœud destination.

### 2.3 Le problème de sens

| Source | Forme | Effet d'un `factor` plus grand |
|---|---|---|
| Code Java | `exec >= transfer * factor` | moins de réplication |
| Commentaire Java | `transfer <= factor * exec` | plus de réplication |
| MiniZinc (`scheduler.mzn:157`) | `transfer <= factor * exec` | plus de réplication |
| Définition voulue de theta | | plus de réplication |

Les deux formes sont équivalentes à `factor = 1`, donc tous les résultats obtenus jusqu'ici restent valables. Le code Java est le seul à diverger.

### 2.4 Ce qui est mort côté Python

- `self.threshold = config['threshold']` est lu dans `master_node_with_heterogeneous_nodes_csp.py:62` et `:1999`, puis jamais utilisé.
- Son seul usage est commenté dans l'ancien modèle Python (`utils/modelCSP.py:229-231`), avec `evaluateUtility` qui calcule le même ratio transfert / travail a posteriori.
- `adapting_theta` et `max_theta_user_preferences` ne sont lus nulle part.
- `tracker.log_threshold(theta)` (`classes/tracker.py:147`) n'est jamais appelé ; `threshold_history` est donc toujours vide.
- C'était déjà le cas au premier commit (2025-11-14).

### 2.5 Autre levier existant, à ne pas confondre

`epsilon_fraction` (Online bi-objectif et Hybrid uniquement) influence aussi la réplication, par un autre mécanisme : la phase 2 minimise l'énergie de transfert en tolérant une dégradation du flow time. Il agit en plus de `factor`, pas à sa place. Ce rapport ne propose pas d'y toucher.

## 3. Proposition

### 3.1 Étape 1 : theta devient une entrée du modèle

Comportement inchangé à theta = 1.

**Java**, même modification dans les 6 fichiers :

- Remplacer `int factor = 1;` par la lecture de `inputs/theta.txt`, sur le motif de `solver_time_limit.txt` (`MainOnline.java:1260`). Fichier absent ou vide : `1.0`.
- `theta <= 0` : contrainte désactivée (reprend le rôle de l'ancien `factor = 0`).
- Adopter le sens `transfer <= theta * exec` avec un theta réel, sans mise à l'échelle des entiers : remplacer `transferTime * factor` par `(int) Math.ceil(transferTime / theta)`.
- Afficher la valeur lue dans le log du solveur.

Aucun build manuel n'est nécessaire : `javac` est relancé à chaque solve (`utils/modelCSP.py:725`).

**Python**, dans `_schedulingUsingJavaCSP_impl` (`utils/modelCSP.py`, près de la ligne 596) : écrire `inputs/theta.txt` depuis `master_node.theta`. Les proxys d'escalade parallèle du Hybrid sont des `copy.copy(self)` (`master_node...py:1502`) avec leur propre dossier `inputs/`, donc ils héritent de la valeur sans code supplémentaire.

### 3.2 Étape 2 : calcul de theta dans le master

Dans `SchedulingUsingCSPOnline`, dont héritent les 3 approches :

- `self.theta = config['threshold']` à l'initialisation, à la place de l'attribut mort `self.threshold`.
- Une méthode `_updateTheta()` appelée une fois par arrivée de job, juste avant `schedulingNewJob()` dans la boucle `scheduling()` (ligne ~210). Un seul point d'appel garantit que tous les solves d'une même arrivée (F1, escalade, variantes parallèles) utilisent la même valeur.
- Si `adapting_theta` est faux, theta reste à `threshold` : c'est le mode statique, nécessaire pour les sweeps.
- Dans tous les cas, appel à `tracker.log_threshold()`.

**Règle proposée.** Deux signaux de congestion, tous deux dans [0, 1] :

| Signal | Définition | Source |
|---|---|---|
| Charge `rho` | part des nœuds occupés à l'instant du solve | logique de `_idleNodeIds()` (ligne 810), à remonter dans la classe de base |
| Flow time observé `w` | moyenne sur les K derniers jobs terminés de `(starting_time - arriving_time) / (finishing_time - arriving_time)` | `tracker.stats_on_jobs` |

Puis :

```
p     = beta * rho + (1 - beta) * w            # pression
cible = theta_min^p * theta_max^(1 - p)        # interpolation géométrique
theta = (1 - gamma) * theta + gamma * cible    # lissage
```

- Infrastructure libre (`p = 0`) : `theta_max`, réplication agressive.
- Infrastructure saturée (`p = 1`) : `theta_min`, réplication restreinte.
- Avec `theta_min = 1 / theta_max`, `p = 0.5` redonne theta = 1, le comportement actuel.
- L'interpolation est géométrique parce que theta est un ratio : 0.5 et 2 sont symétriques autour de 1.
- Tant qu'aucun job n'est terminé, `w = rho`.

**Configuration.** Trois clés existent déjà : `threshold` (valeur statique ou initiale), `adapting_theta`, `max_theta_user_preferences` (`theta_max`). Quatre à ajouter : `min_theta` (défaut `1 / theta_max`), `theta_load_weight` (beta, 0.5), `theta_smoothing` (gamma, 0.5), `theta_flow_window` (K, 5). Avec la config actuelle (`threshold = 1`, `adapting_theta = false`), rien ne change.

### 3.3 Étape 3 : export

`threshold_history` n'est exporté que par `simulator.py:244`, pas par l'export par run des scripts d'expériences (`utils/run_export.py`). À ajouter : un `theta_history.csv` par run avec `rho`, `w`, `p` et `theta` à chaque arrivée, et les nouvelles clés dans `run_params.json`.

## 4. Validation prévue

1. **Non-régression.** Sur une petite instance (`exps/xp_test_online_only.py`), theta statique à 1 doit donner les mêmes `works.csv` et `transfers.csv` qu'avant la modification, pour Incremental, Online et Hybrid.
2. **Sens.** Même instance avec theta statique à 0.25 puis 4 : le nombre de réplicas par job doit croître avec theta.
3. **Sweep statique.** theta ∈ {0.25, 0.5, 1, 2, 4} croisé avec charge faible et charge forte.
4. **Dynamique.** Un run Poisson avec `adapting_theta = true` et `theta_max = 4` : vérifier dans `theta_history.csv` que theta baisse quand `rho` et `w` montent, puis comparer flow time et énergie au meilleur theta statique du sweep.

## 5. Points ouverts

- **Sens de la règle.** Elle suppose que répliquer davantage nuit quand l'infrastructure est saturée. Le sweep de l'étape 3 de la validation doit le confirmer. Si c'est l'inverse dans certains régimes, seul le sens de l'interpolation change ; les étapes 1 et 3 de la proposition restent valables.
- **Portée de la contrainte.** Elle ne concerne que les jobs à plus d'un réplica. Theta ne peut donc pas empêcher le premier transfert d'un job, seulement limiter la parallélisation sur plusieurs nœuds.
- **Replanification.** En Online, la contrainte compare le temps de transfert complet au travail restant du job. À vérifier lors de l'implémentation : un réplica déjà résident ne doit pas être pénalisé comme s'il fallait le retransférer.
- **Interaction avec `epsilon_fraction`.** En bi-objectif, les deux leviers s'empilent. Pour isoler l'effet de theta, garder `epsilon_fraction` fixe pendant les sweeps.
- **Theta par job.** Écarté pour l'instant (theta global par appel). La boucle Java étant déjà par job, l'extension resterait simple.
