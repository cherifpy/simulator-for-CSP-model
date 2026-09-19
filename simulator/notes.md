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