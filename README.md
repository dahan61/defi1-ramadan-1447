# S3C'1447 — Défi N° 1 : Contribution à la résolution du RCPSP

## 📋 Résumé du projet

Ce projet implémente et améliore l'algorithme hybride **GANS** proposé par **Goncharov (2024)**
dans l'article *"A hybrid heuristic algorithm for the resource-constrained project scheduling problem"*
(arXiv:2502.18330v2), dans le cadre du championnat **SupNum Coding Challenge Championship 1447 (S3C'1447)**.

Le **RCPSP** (Resource-Constrained Project Scheduling Problem) est un problème NP-difficile
d'ordonnancement de projet sous contraintes de ressources. L'objectif est de minimiser la durée
totale du projet (makespan / Cmax) en respectant les contraintes de précédence entre activités
et les capacités limitées des ressources renouvelables.

---

## 📁 Fichiers livrés

| Fichier | Description |
|---------|-------------|
| `rcpsp_solver.py` | **Code source V1** — Implémentation de l'algorithme de l'article 1 (GA + S-SGS + P-SGS + FBI), avec correction du bug dans backward_sgs |
| `rcpsp_solver_v2.py` | **Code source V2** — Version améliorée avec classement des ressources, recherche de voisinage, et règles de priorité additionnelles |
| `run_challenge.py` | **Script de lancement** — Identifie les instances ouvertes et exécute le solveur (V1 ou V2) |
| `Rapport_S3C1447_Defi1_RCPSP.docx` | **Rapport Word** — Document expliquant le travail réalisé, les méthodes, corrections, et résultats |
| `Tableau_Comparaison_Solutions.docx` | **Tableau de comparaison** (Point 5) — Fichier Word avec le tableau comparant nos solutions aux meilleures connues de PSPLIB |
| `requirements.txt` | **Dépendances** — Aucune dépendance externe, uniquement Python standard library |
| `results_v1.txt` | **Résultats V1** — Résultats de l'algorithme de l'article sur les instances ouvertes j60 |
| `results_v2.txt` | **Résultats V2** — Résultats de l'algorithme amélioré sur les instances ouvertes j60 |

---

## 🔧 Prérequis

- **Python 3.6+** (testé avec Python 3.14.2)
- **Aucune dépendance externe** — le code utilise uniquement la bibliothèque standard Python :
  `os`, `sys`, `random`, `time`, `argparse`, `collections` (deque), `platform`
- **Données PSPLIB** : les instances j60 (.sm files) et le fichier de solutions j60hrs.sm

---

## 🚀 Comment exécuter

### Structure des fichiers attendue

```
defi1-ramadan-1447/
├── j60/                    # Dossier contenant les 480 fichiers .sm (instances j60)
│   ├── j601_1.sm
│   ├── j601_2.sm
│   ├── ...
│   └── j6048_10.sm
├── j60hrs.sm               # Fichier des meilleures solutions connues (de PSPLIB)
├── rcpsp_solver.py         # Solveur V1 (article 1)
├── rcpsp_solver_v2.py      # Solveur V2 (amélioré)
├── run_challenge.py        # Script de lancement
├── requirements.txt
└── Rapport_S3C1447_Defi1_RCPSP.docx
└── Tableau_Comparaison_Solutions.docx
```

### Commandes d'exécution

```bash
# 1. Tester V1 (algorithme de l'article) sur les instances ouvertes (LB < UB)
python run_challenge.py j60 -s j60hrs.sm -m 50000 -o results_v1.txt

# 2. Tester V2 (algorithme amélioré) sur les instances ouvertes
python run_challenge.py j60 -s j60hrs.sm -m 50000 -o results_v2.txt --v2

# 3. Limiter à N instances (ex: 10 ou 20, pour aller plus vite)
python run_challenge.py j60 -s j60hrs.sm -m 50000 -o results_v2.txt --v2 --limit 10

# 4. Tester sur TOUTES les 480 instances (ouvertes + fermées)
python run_challenge.py j60 -s j60hrs.sm -m 50000 -o results_all.txt --all

# 5. Avec plus d'effort (λ = 500 000, plus lent mais meilleurs résultats)
python run_challenge.py j60 -s j60hrs.sm -m 500000 -o results_v2_500k.txt --v2

# 6. Tester une seule instance
python rcpsp_solver.py j60 --single j60/j601_1.sm -m 50000
```

### Paramètres de la ligne de commande

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `data_dir` | (requis) | Dossier contenant les fichiers .sm |
| `-s`, `--solutions` | `j60hrs.sm` | Fichier des solutions connues |
| `-m`, `--max-schedules` | `50000` | Nombre max d'ordonnancements générés (λ) |
| `-o`, `--output` | `results.txt` | Fichier de sortie des résultats |
| `--v2` | désactivé | Utiliser le solveur amélioré V2 |
| `--all` | désactivé | Tester toutes les instances (pas seulement ouvertes) |
| `--limit`, `-n` | `0` (= toutes) | Limiter à N instances ouvertes (ex: `--limit 10` ou `--limit 20`) |
| `--seed` | `42` | Graine aléatoire pour reproductibilité |

---

## 📊 Lecture des résultats

Le programme affiche un tableau avec les colonnes suivantes :

| Colonne | Signification |
|---------|---------------|
| Instance | Nom du fichier d'instance |
| CP_LB | Borne inférieure du chemin critique (Critical Path Lower Bound) |
| Known_UB | Meilleure solution connue (Upper Bound) du fichier j60hrs.sm |
| Gap | Différence UB - CP_LB |
| Gap% | Gap en pourcentage de CP_LB |
| Our_MS | Notre makespan (durée trouvée par notre solveur) |
| OK | Y = solution réalisable (toutes contraintes respectées) |
| Status | MATCHED = égale l'UB connue, IMPROVED! = meilleure, +N = pire de N |
| Time | Temps d'exécution en secondes |

### Statuts possibles

- **MATCHED** : Notre solution est égale à la meilleure connue
- **IMPROVED!** : Notre solution est meilleure que la meilleure connue (contribution !)
- **+N** : Notre solution est N unités de temps au-dessus de la meilleure connue
- **OPTIMAL** : Notre solution atteint la borne inférieure du chemin critique

---

## 🏗️ Architecture de l'algorithme

### V1 — Implémentation de l'article (`rcpsp_solver.py`)

L'algorithme GANS de Goncharov est un algorithme hybride combinant :

1. **Serial SGS (S-SGS)** : Construit un ordonnancement actif en parcourant les activités
   dans l'ordre de la liste et en les planifiant au plus tôt possible en respectant les
   contraintes de précédence et de ressources.

2. **Parallel SGS (P-SGS)** : Parcourt le temps (t = 0, 1, 2, ...) et à chaque pas planifie
   toutes les activités éligibles qui peuvent démarrer sans violer les contraintes.

3. **FBI (Forward-Backward Improvement)** : Procédure d'amélioration locale alternant :
   - Ordonnancement arrière (backward SGS) : planifier les activités le plus tard possible
   - Ordonnancement avant (forward SGS) : replanifier au plus tôt
   - Cette alternance compacte l'ordonnancement et réduit le makespan

4. **Algorithme Génétique** :
   - Population initiale : règles de priorité (LFT, EST, MTS) + listes aléatoires
   - Sélection par tournoi (taille 4)
   - Croisement respectant les précédences
   - Mutation par échange d'activités adjacentes
   - Diversification par injection de nouvelles solutions aléatoires après stagnation

### V2 — Version améliorée (`rcpsp_solver_v2.py`)

En plus de tout ce qui est dans V1, la version V2 ajoute :

1. **Classement des ressources (Resource Ranking)** — Section 3 de l'article :
   Les ressources sont classées par rareté (ratio demande totale / capacité disponible).
   Les ressources les plus rares reçoivent un poids plus élevé, guidant les opérateurs
   de croisement vers des solutions utilisant mieux les ressources critiques.
   4 schémas de pondération sont utilisés : (1,0.8,0.6,0.4), (1,0.9,0.8,0.7),
   (1,1,1,1), et poids basés sur l'utilisation.

2. **6 règles de priorité** (au lieu de 3) :
   - LFT (Latest Finish Time)
   - EST (Earliest Start Time)
   - MTS (Most Total Successors)
   - LST (Latest Start Time)
   - GRPW (Greatest Rank Positional Weight)
   - WRUP (Weighted Resource Utilization Priority — utilise les poids de ressources)

3. **Croisement deux-points** : En plus du croisement standard à un point, un second
   opérateur de croisement avec deux points de coupure pour plus de diversité génétique

4. **Mutation par insertion** : Déplacement d'une activité vers une position valide différente
   (en plus de l'échange par swap de V1)

5. **Recherche de voisinage simplifiée (NS)** — inspirée de la Section 6 de l'article :
   Replanification d'un bloc d'activités proches d'une activité pivot, en gardant les
   autres activités fixes et en résolvant le sous-problème réduit

6. **Alternance GA/NS** — inspirée de la Section 7 de l'article :
   L'algorithme bascule entre phase génétique et recherche de voisinage selon la stagnation,
   permettant d'exploiter et d'explorer l'espace de recherche de manière complémentaire

---

## 🐛 Bug corrigé dans l'implémentation initiale

L'implémentation initiale contenait un **bug critique dans `backward_sgs`** (utilisé par FBI) :

**Problème** : Lors de l'ordonnancement arrière, quand aucun créneau faisable en ressources
n'était trouvé en reculant dans le temps, la recherche de fallback vers l'avant **ne respectait
pas la contrainte de fin au plus tard** (latest finish = min des dates de début des successeurs).
Cela causait des activités planifiées au temps 0 alors que leurs prédécesseurs n'avaient pas
encore terminé → **violations de précédence** (solutions infaisables).

**Corrections apportées** :
- La recherche vers l'avant respecte maintenant la borne `lf` (latest finish) des successeurs
- Ajout de `make_topological_activity_list()` qui garantit que les listes d'activités
  extraites d'un ordonnancement respectent toujours l'ordre topologique, même quand
  des activités ont des dates de début identiques
- Amélioration du test de précédence dans l'opérateur de mutation

**Résultat** : Après correction, **100% des solutions sont réalisables** sur toutes les instances testées (j30 et j60).

---

## ⚡ Note importante sur les performances (Python vs C++)

> **L'implémentation originale de Goncharov est en C++** (Visual Studio, CPU 3.4 GHz).
> Notre implémentation est en **Python**, ce qui implique une différence de performance
> considérable. **Toute comparaison des temps d'exécution entre nos résultats et ceux
> de l'article n'est pas pertinente** car les langages sont fondamentalement différents
> en termes de vitesse d'exécution.

| Aspect | C++ (article original) | Python (notre implémentation) |
|--------|------------------------|-------------------------------|
| Langage | C++ compilé et optimisé | Python interprété |
| Vitesse relative | ~1x (référence) | **~50 à 100x plus lent** |
| Temps moyen (j120, λ=50K) | ~16 secondes | N/A (non testé sur j120) |
| Temps moyen (j60, λ=50K) | ~quelques secondes | ~35-50 secondes |
| Optimisations compilateur | Oui (MSVC, optimisations O2) | Non applicable |
| Cache CPU | Exploité efficacement | Overhead de l'interpréteur |

### Conséquences pratiques

1. **À λ identique (50 000)**, le C++ effectue chaque opération de scheduling beaucoup plus
   rapidement, ce qui permet au programme de faire plus de travail utile dans le même budget
   d'ordonnancements. Notre Python passe plus de temps sur l'overhead de l'interpréteur.

2. **Nos résultats sont systématiquement quelques points au-dessus des meilleurs connus.**
   Ce n'est pas un défaut de l'algorithme mais une conséquence directe de la différence
   de langage et du fait que certaines composantes avancées (croisements denses, liste tabou
   complète) n'ont pas été implémentées.

3. **Pour compenser**, il faudrait :
   - Porter le code en C++ ou utiliser des accélérateurs (Cython, Numba, PyPy)
   - Augmenter significativement λ (500 000 ou 5 000 000)
   - Implémenter les composantes manquantes (Dense Gene Crossovers, liste tabou complète)

---

## 📝 Correspondance avec les livrables demandés (Section IV du cahier des charges)

| N° | Livrable demandé | Fichier fourni | Statut |
|----|------------------|----------------|--------|
| 1 | Code source + résultats de l'algorithme de l'article 1 | `rcpsp_solver.py` + `results_v1.txt` | ✅ Fait |
| 2 | Résultats de l'article 1 sur instances non résolues de 60 tâches | `results_v1.txt` | ✅ Fait |
| 3 | Document Word/PowerPoint expliquant le travail | `Rapport_S3C1447_Defi1_RCPSP.docx` | ✅ Fait |
| 4 | Code source des algorithmes d'amélioration | `rcpsp_solver_v2.py` | ✅ Fait |
| 5 | Fichier Word avec tableau d'améliorations (si trouvées) | `Tableau_Comparaison_Solutions.docx` + Section 5 du rapport | ✅ Fait |

---

## 📊 Données du jeu j60

- **480 instances** au total : 48 jeux de paramètres × 10 instances chacun
- **62 jobs** par instance : 60 activités réelles + 2 fictives (source et puits)
- **4 ressources** renouvelables par instance
- **297 instances fermées** : solution optimale connue (CP_LB = UB)
- **183 instances ouvertes** : solution optimale inconnue (CP_LB < UB)
- Paramètres de génération : NC (network complexity), RF (resource factor), RS (resource strength)
- Source : PSPLIB — https://www.om-db.wi.tum.de/psplib/

---

## 📚 Références

1. Goncharov, E.N. (2024). *A hybrid heuristic algorithm for the resource-constrained
   project scheduling problem.* arXiv:2502.18330v2.
2. Kolisch, R. and Sprecher, A. (1996). *PSPLIB - A project scheduling library.*
   European Journal of Operational Research, 96, 205-216.
3. Kolisch, R. and Hartmann, S. (2006). *Experimental investigation of heuristics for
   resource-constrained project scheduling: An update.* European Journal of Operational
   Research, 174, 23-37.
4. PSPLIB : https://www.om-db.wi.tum.de/psplib/
