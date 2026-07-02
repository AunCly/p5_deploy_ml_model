---
title: Api de prédiction du turnover
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.13"
---

# Déployez votre modèle de Machine Learning

## Mission
Vous êtes freelance spécialisé en machine learning et vous venez de recevoir une demande de la part de votre client Futurisys, une entreprise innovante qui souhaite rendre ses modèles de machine learning opérationnels et accessibles via une API performante. Vous êtes chargé de déployer un modèle de machine learning en production.
Le directeur technique de Futurisys, Aurélien, vous formule une demande impliquant de : 
- créer une API avec FastAPI (ou équivalent) pour exposer le modèle ;
- écrire des tests unitaires avec Pytest pour garantir sa fiabilité ; 
- et gérer la version du code avec Git pour une collaboration fluide.

L'objectif ? Rendre le modèle utilisable en production tout en respectant les meilleures pratiques de l'ingénierie logicielle. À la fin du projet, vous aurez un Proof of Concept (POC) fonctionnel dont vous pourrez être fier !

Tout en réfléchissant, vous dressez la liste des livrables que vous aurez à construire et présenter :

- **Un dépôt Git structuré contenant** :
    - L'ensemble du code source.
    - Un requirements.txt (ou équivalent).
    - Un historique de commits clair, avec des branches dédiées aux fonctionnalités et l’utilisation de tags pour la gestion des versions.
    - Un README complet présentant le projet, notamment les instructions d’installation, d’utilisation, de déploiement, d’authentification et de sécurisation.
    
- **Une API fonctionnelle et déployée** développée avec FastAPI (ou équivalent) exposant le modèle de machine learning accompagnée d’une documentation intégrée (par exemple via Swagger/OpenAPI) pour décrire les endpoints, les schémas de données et les exemples d’appels.
    - Des scripts de tests unitaires et fonctionnels associés à :
    - Un ensemble de tests écrits en Pytest couvrant les cas critiques et les scénarios d’erreur.
    - Un rapport de couverture de tests (par exemple via pytest-cov) afin de démontrer la robustesse du code.

- **Une base de données PostgreSQL fonctionnelle**:
    - Un script SQL (.sql) ou Python (create_db.py) pour la création de la base de données et des tables.
    - Un modèle de données/de la documentation expliquant la structure des tables (ex: schéma UML).
    - Des exemples d’entrées en base (SQL ou CSV contenant des inputs et outputs du modèle de ML).
    - Des scripts pour interroger les données et interagir avec le modèle ML.
  
- **Une configuration du pipeline CI/CD** capable de gérer les environnements  (dev test, prod) et intégrer la gestion des secrets.
    - Un fichier YAML (par exemple pour GitHub Actions) qui automatise les tests et le déploiement