# Documentation Technique — API de prédiction d'attrition employés

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Stack technique](#2-stack-technique)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Installation et démarrage](#4-installation-et-démarrage)
5. [Variables d'environnement](#5-variables-denvironnement)
6. [Base de données](#6-base-de-données)
7. [API — Endpoints](#7-api--endpoints)
8. [Modèle de Machine Learning](#8-modèle-de-machine-learning)
9. [Tests](#9-tests)
10. [CI/CD et déploiement](#10-cicd-et-déploiement)
11. [Notebooks d'exploration](#11-notebooks-dexploration)

---

## 1. Vue d'ensemble

Ce projet est une **API REST de prédiction d'attrition** (départ volontaire) d'employés, développée dans le cadre du projet "Déployez un modèle de Machine Learning".

Un modèle LogisticRegression entraîné sur des données RH (SIRH, évaluations, sondages) est exposé via une API FastAPI. Pour chaque employé soumis, l'API retourne :
- une **prédiction binaire** (0 = reste, 1 = risque de départ)
- une **probabilité** associée (0.0 → 1.0)

L'historique de toutes les prédictions est persisté en base PostgreSQL.

---

## 2. Stack technique

| Catégorie | Technologie | Version |
|---|---|---|
| Langage | Python | 3.13 |
| Framework API | FastAPI | ≥ 0.138.1 |
| Machine Learning | scikit-learn | ≥ 1.9.0 |
| Sérialisation modèle | joblib | ≥ 1.5.3 |
| Rééchantillonnage | imbalanced-learn | ≥ 0.0 |
| Explainability | SHAP | ≥ 0.52.0 |
| Manipulation données | pandas | ≥ 3.0.3 |
| Base de données | PostgreSQL | 18 |
| ORM / Connecteur | SQLAlchemy + psycopg | ≥ 2.0.51 / ≥ 3.3.4 |
| Validation données | Pydantic (via FastAPI) | — |
| Client HTTP (tests) | httpx2 | ≥ 2.5.0 |
| Tests | pytest | ≥ 9.1.1 |
| Conteneurisation | Docker + Docker Compose | — |
| Gestionnaire de paquets | uv | — |
| Stockage modèles (Git) | Git LFS | — |
| CI/CD | GitHub Actions | — |
| Hébergement | Hugging Face Spaces | — |
| Notebooks | Jupyter | ≥ 1.1.1 |

---

## 3. Architecture du projet

- .github/workflows/ : pipelines CI/CD GitHub Actions
- data/raw/ : fichiers CSV bruts (SIRH, évaluations, sondages
- database/ : scripts de création et peuplement de la base PostgreSQL
- models/ : code du modèle de ML, pipeline de préprocessing et utilitaires
- notebooks/ : notebooks Jupyter d'exploration et d'analyse des données
- tests/ : tests unitaires et fonctionnels avec Pytest
- .env.example : template des variables d'environnement
- docker-compose.yml : configuration des services Docker (API + BDD)
- Dockerfile : image Docker de l'API FastAPI
- employee.py : modèle Pydantic pour la validation des données d'entrée
- main.py : point d'entrée de l'application FastAPI
- pyproject.toml : déclaration des dépendances et configuration du projet
- README.md : présentation du projet et instructions d'installation
- DOCUMENTATION.md : documentation technique détaillée (ce fichier)

### Flux de données

1. Le client envoie une requête POST à l'endpoint `/predict` avec les données d'un employé. 
2. Les données sont validées par Pydantic (modèle `EmployeeData`). 
3. Les données sont prétraitées (encodage catégoriel, feature engineering, nettoyage de corrélation) via le pipeline scikit-learn. 
4. Le modèle LogisticRegression pré-entraîné effectue la prédiction et calcule la probabilité 
5. La prédiction est sauvegardée en base PostgreSQL (hors production)
6. L'API retourne une réponse JSON contenant l'`employee_id`, la `prediction` et la `probability`.

## 4. Installation et démarrage

### Prérequis

- Docker et Docker Compose installés
- `uv` installé (`pip install uv`) pour le développement local
- Git avec Git LFS activé (`git lfs install`)

### Démarrage avec Docker

```bash
# 1. Cloner le dépôt
git clone https://github.com/AunCly/p5_deploy_ml_model
cd p5_deploy_ml_model

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec les valeurs souhaitées

# 3. Démarrer les services
docker compose up --build

# L'API est disponible sur [http://localhost:8000](http://localhost:8000
# La documentation interactive sur [http://localhost:8000/docs](http://localhost:8000/docs)
```

### Comportement au démarrage

Au lancement (via le lifespan FastAPI), l'application :
1. Crée les tables en base si elles n'existent pas (exécution de `structure.sql`)
2. Peuple la base avec les données CSV (uniquement si `ENVIRONMENT != production`)

---

## 5. Variables d'environnement

Copier `.env.example` en `.env` et renseigner toutes les valeurs.

| Variable | Description | Exemple |
|---|---|---|
| `ENVIRONMENT` | Environnement d'exécution | `local`, `production` |
| `DB_NAME` | Nom de la base de données | `attrition_database` |
| `DB_USER` | Utilisateur PostgreSQL | `user` |
| `DB_PASSWORD` | Mot de passe PostgreSQL | `password` |
| `DB_HOST` | Hôte PostgreSQL | `db` (Docker) ou `localhost` |
| `DB_PORT` | Port PostgreSQL | `5432` |
| `API_KEY` | Clé d'authentification de l'API | `<clé aléatoire sécurisée>` |

## 6. Base de données

La base de données PostgreSQL contient l'historique des prédictions d'attrition. Elle est créée automatiquement au démarrage de l'API si elle n'existe pas.
Si besoin de la recréer.
```
docker compose exec -it api uv run python -m database.init_database
```

### Schéma

La base PostgreSQL contient 4 tables, toutes créées par `database/structure.sql`.
La base de données n'est pas utilisée en production.

![database_schema.png](database/database_schema.png)

### Peuplement (Seeding)

Le script `database/seeding.py` charge les trois fichiers CSV de `data/raw/` et les insère dans les tables correspondantes. 
Il est exécuté automatiquement au démarrage si `ENVIRONMENT != production`.
Ce script est automatiquement appelé au démarrage de l'API par la méthode lifespan FastAPI dans `main.py`.

## 7. API

La documentation interactive Swagger est accessible sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Authentification

Tous les endpoints (sauf `/health`) requièrent l'en-tête HTTP :

```
x-api-key: <valeur de API_KEY dans .env>
```

Une clé invalide retourne `HTTP 403 Forbidden`.

## 8. Modèle de Machine Learning

### Données source

Trois fichiers CSV situés dans `data/raw/` :
- `extrait_sirh.csv` — données RH administratives
- `extrait_eval.csv` — évaluations de performance
- `extrait_sondage.csv` — sondages de satisfaction et comportementaux

**Variable cible :** `a_quitte_l_entreprise` (Oui / Non)

### Pipeline de préprocessing

Le pipeline scikit-learn sérialisé (`classification_preprocessor.joblib`) exécute les étapes suivantes dans l'ordre :

#### 1. Encodage catégoriel (`models/preprocessing.py`)

| Colonne | Encodage |
|---|---|
| `genre` | F → 0, M → 1 |
| `a_quitte_l_entreprise` | Non → 0, Oui → 1 |
| `heure_supplementaires` | Non → 0, Oui → 1 |
| `frequence_deplacement` | Aucun → 1, Occasionnel → 1.5, Frequent → 2 |

Colonnes supprimées avant l'entrée dans le pipeline :
- `id_employee`
- `nombre_heures_travailless`
- `augementation_salaire_precedente`
- `nombre_employee_sous_responsabilite`
- `ayant_enfants`

#### 2. Feature Engineering (`models/technova_features.py`)

Transformer personnalisé `TechnovaFeatures` qui crée les features dérivées :

| Feature | Calcul |
|---|---|
| `ratio_anciennete_total_experence` | `annees_dans_l_entreprise / annee_experience_totale` |
| `ratio_poste_actuel_anciennete` | `annees_dans_le_poste_actuel / annees_dans_l_entreprise` |
| `score_penibilite_transport` | `distance_domicile_travail × frequence_deplacement` |
| `score_bien_etre` | Moyenne des 4 scores de satisfaction |
| `score_performance` | Moyenne des notes d'évaluation précédente et actuelle |
| `evolution_performance` | `note_evaluation_actuelle - note_evaluation_precedente` |
| `score_evolution_hierarchie` | Dérivé du niveau hiérarchique |
| `annees_par_experience` | `annees_dans_l_entreprise / nombre_experiences_precedentes` |
| `salaire_quartile` | Quartile du revenu mensuel par poste |

#### 3. Filtre de corrélation (`models/technova_correlation_cleaning.py`)

Transformer personnalisé `TechnovaCorrelationCleaning` qui supprime les features numériques avec une corrélation > 0.80 pour éviter la multicolinéarité.

### Modèle

- **Algorithme :** LogisticRegression
- **Fichier :** `models/compiled/classification_model.joblib`
- **Version sklearn :** 1.9.0
- **Nombre de features en sortie du pipeline :** 15
- **Métadonnées :** `models/compiled/metadata_20260701_155446.json`

## 9. Tests

### Lancer les tests

```bash
uv run python -m pytest tests/test_api.py
uv run python -m pytest tests/test_training.py
```

### Configuration CI

Les tests s'exécutent automatiquement sur GitHub Actions à chaque push sur `develop` et `main` (voir section CI/CD).

## 10. CI/CD et déploiement

### Branches

| Branche | Pipeline | Action |
|---|---|---|
| `develop` | `test.yml` | Exécute les tests uniquement |
| `main` | `test_and_deploy.yml` | Exécute les tests + déploie sur Hugging Face |

### Secrets et variables GitHub Actions requis

| Nom | Type | Description |
|---|---|---|
| `HF_TOKEN` | Secret | Token d'accès Hugging Face |
| `HF_SPACE_ID` | Variable | Identifiant du Space Hugging Face |

### Déploiement sur Hugging Face Spaces

L'application est déployée sur Hugging Face Spaces via push Git. La plateforme détecte automatiquement le `Dockerfile` et construit l'image.

### Docker

#### `Dockerfile`

```dockerfile
FROM python:3.13-slim
# Installe libpq-dev et gcc pour psycopg
# Installe uv
# Copie les dépendances et le code
# Lance : fastapi run main.py --port 8000
```

#### `docker-compose.yml`

```yaml
services:
  db:           # PostgreSQL 18 avec volume persistant, port 5432
  api:          # Image FastAPI, port 8000, dépend de db
```

**Démarrage complet :**
```bash
docker compose up --build
```

**Arrêt et suppression des volumes :**
```bash
docker compose down -v
```

## 11. Notebooks d'exploration

Les notebooks Jupyter dans `notebooks/` documentent le travail de data science préalable au déploiement. Ils ne sont pas utilisés en production.

| Notebook | Objectif |
|---|---|
| `analyze.ipynb` | Exploration des données (distributions, corrélations, analyse de l'attrition) |
| `cleaning.ipynb` | Nettoyage et préparation des données brutes |
| `training.ipynb` | Entraînement du modèle, sélection des hyperparamètres, évaluation |

```bash
# Lancer Jupyter
uv run jupyter lab
```

