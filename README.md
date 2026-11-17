# Prédire la fréquence de sinistres en assurance auto : GLM, réseau de neurones et gradient boosting

## Objectif
Prédire le nombre de sinistres par an d'un contrat d'assurance automobile, et comparer rigoureusement trois modèles : un GLM de Poisson (la référence en tarification), un réseau de neurones et un gradient boosting.

## Données
- Jeu public **freMTPL2freq** (OpenML 41214) : 678 013 contrats français.
- Cible : `ClaimNb` (nombre de sinistres), plafonné à 4. Exposition `Exposure` (années) plafonnée à 1.
- Découpage : 64 % entraînement, 16 % validation (choix des modèles), 20 % test (utilisé une seule fois, à la fin).

## Méthode
- **Perte** : déviance de Poisson pondérée par l'exposition. On modélise la fréquence N/E avec un poids E, ce qui équivaut à un offset log(E).
- **GLM de Poisson** : log-densité, bonus-malus, tranches d'âge du conducteur et du véhicule, zone.
- **Gradient boosting** (`HistGradientBoostingRegressor`, perte de Poisson) : variables brutes, 15 feuilles par arbre (choisi parmi 7, 15, 31, 63 sur la validation), arrêt anticipé.
- **Réseau de neurones** (PyTorch) : une couche cachée de 8 neurones (ReLU), variables numériques normalisées sur l'entraînement seul, catégorielles en indicatrices. Nombre d'époques et de neurones choisis sur la courbe de validation. Le niveau global est recalibré par un facteur multiplicatif estimé sur l'entraînement (le réseau n'a pas la propriété d'équilibre du GLM).

## Résultats sur le jeu de test

| Modèle | Déviance de Poisson | Part de déviance expliquée |
|---|---|---|
| Naïf (fréquence moyenne) | 0,6274 | 0 % |
| GLM de Poisson | 0,5994 | 4,5 % |
| Réseau de neurones | 0,5910 | 5,8 % |
| Gradient boosting | 0,5799 | 7,6 % |

Les scores du test sont très proches de ceux de la validation : les choix faits sur la validation n'ont pas créé d'optimisme visible.

### Les écarts sont-ils significatifs ? Bootstrap apparié
Le jeu de test est rééchantillonné 1 000 fois ; sur chaque tirage, on calcule l'écart de déviance entre deux modèles sur les mêmes contrats.

| Comparaison | IC 95 % de l'écart |
|---|---|
| GLM − réseau | [0,0063 ; 0,0103] |
| Gradient boosting − réseau | [−0,0131 ; −0,0092] |
| Gradient boosting − GLM | [−0,0216 ; −0,0174] |

Aucun intervalle ne contient 0 : **gradient boosting > réseau > GLM**, chaque écart est significatif.

### Calibration par décile de risque
Les contrats sont triés par fréquence prédite et coupés en 10 groupes ; on compare sinistres prédits et observés dans chaque groupe.
- Les trois modèles prédisent le bon niveau global (écart d'environ 1 %, compatible avec le hasard).
- Le gradient boosting discrimine le mieux : son décile le plus risqué concentre 4,6 fois plus de sinistres que le moins risqué (4,0 pour le réseau, 3,1 pour le GLM).
- Avant recalibration, le réseau sous-estimait le nombre total de sinistres de 4,7 % sur la validation (environ 3,6 écarts-types).

### Fuite de données : contrats identiques
28 % des contrats du test ont un jumeau exact (mêmes caractéristiques) dans l'entraînement, probablement le même assuré.

| Modèle | Part expliquée, avec jumeau | Sans jumeau |
|---|---|---|
| GLM | 5,7 % | 4,1 % |
| Réseau | 8,2 % | 5,1 % |
| Gradient boosting | 9,9 % | 6,9 % |

Les modèles flexibles gagnent deux fois plus que le GLM sur ces contrats : signe qu'ils ont en partie mémorisé des assurés déjà vus. Les scores « sans jumeau » sont les plus représentatifs de contrats réellement nouveaux. Le classement ne change pas.

## Limites et pistes
- Corriger la fuite par un découpage par groupe de contrats identiques, puis réentraîner.
- Les véhicules d'âge 0 ont une fréquence observée de 0,31, trois fois plus que les autres : possible artefact de collecte, à vérifier.
- Mise à disposition du modèle par une API (FastAPI) et Docker : en cours.

## Lancer le code
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python modele.py
```
Durée : quelques minutes (entraînement du réseau sur processeur).

## Fichiers
- `modele.py` : pipeline complet, de la préparation des données à l'évaluation finale.
- `exploration.py` : brouillon d'exploration (analyses préliminaires, essais).